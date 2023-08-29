from typing import Optional
from torch import Tensor, nn
from torch.nn import functional as F

from .misc import *

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_heads)
        self.attn_ln = LayerNorm(d_model)

        self.cross_attn = (
            MultiHeadAttention(d_model, n_heads) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(d_model) if cross_attention else None

        # Scales by 4x and then back to d_model
        self.mlp = nn.Sequential(
            Linear(d_model, d_model * 4), nn.GELU(), Linear(d_model * 4, d_model)
        )
        self.mlp_ln = LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        
        # Layer norm first (different from original transformer implementation)
        x_ln = self.attn_ln(x)                                              # (batch_size, n_ctx, d_model)
        attn_out = self.attn(x_ln, mask=mask, kv_cache=kv_cache)            # (batch_size, n_ctx, d_model)
        out = x + attn_out                                                  # (batch_size, n_ctx, d_model)

        if self.cross_attn:
            # Layer norm first (different from original transformer implementation)
            cattn_out = self.cross_attn_ln(out)                             # (batch_size, n_ctx, d_model)
            cattn_out = self.cross_attn(cattn_out, xa, kv_cache=kv_cache)   # (batch_size, n_ctx, d_model)
            out = out + cattn_out                                           # (batch_size, n_ctx, d_model)

        out = out + self.mlp(self.mlp_ln(out))                              # (batch_size, n_ctx, d_model)
        return out 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(d_model, d_model)
        self.key = Linear(d_model, d_model, bias=False)
        self.value = Linear(d_model, d_model)
        self.out = Linear(d_model, d_model)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        weighted_v = self.qkv_attention(q, k, v, mask) # (batch_size, n_ctx, d_model)
        out = self.out(weighted_v) # (batch_size, n_ctx, d_model)
        return out 

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, d_model = q.shape

        d_head = d_model // self.n_head
        
        scale = (d_head) ** -0.25 # they changed the scale from usual (d_head)^0.5 to (d_head)^-0.25 

        # Split into heads, permute as necessary, scale q and k
        scaled_q = q.view(*q.shape[:2], self.n_head, d_head).permute(0, 2, 1, 3) * scale    # (batch_size, n_heads, n_ctx, d_head)
        scaled_k_T = k.view(*k.shape[:2], self.n_head, d_head).permute(0, 2, 3, 1) * scale  # (batch_size, n_heads, d_head, n_ctx)
        v = v.view(*v.shape[:2], self.n_head, d_head).permute(0, 2, 1, 3)                   # (batch_size, n_heads, n_ctx, d_head)

        # Compute attention weights
        scaled_attention_weights = scaled_q @ scaled_k_T # (batch_size, n_heads, n_ctx, n_ctx)

        # Apply mask if provided
        if mask is not None:
            scaled_attention_weights = scaled_attention_weights + mask[:n_ctx, :n_ctx] # (batch_size, n_heads, n_ctx, n_ctx)

        scaled_attention_weights = scaled_attention_weights.float()                                 # (batch_size, n_heads, n_ctx, n_ctx)
        scaled_attention_weights = F.softmax(scaled_attention_weights, dim=-1).to(scaled_q.dtype)   # (batch_size, n_heads, n_ctx, n_ctx)

        # Weight the values
        weighted_v = scaled_attention_weights @ v # (batch_size, n_heads, n_ctx, d_head)

        # Merge heads, project back to d_model
        weighted_v = weighted_v.permute(0, 2, 1, 3)     # (batch_size, n_ctx, n_heads, d_head)
        weighted_v = weighted_v.flatten(start_dim=2)    # (batch_size, n_ctx, n_heads*d_head)

        return weighted_v

