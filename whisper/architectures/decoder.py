from typing import  Iterable, Optional
import numpy as np
import torch
from torch import Tensor, nn

from .attention import ResidualAttentionBlock
from .misc import LayerNorm

class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, d_model: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, d_model)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, d_model))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(d_model, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(d_model)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x) # (batch_size, <= n_ctx, d_model )
            + self.positional_embedding[offset : offset + x.shape[-1]] # (<= n_ctx, d_model)
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache) # (batch_size, <= n_ctx, d_model)

        # Final layer norm preserved from original implementation
        x = self.ln(x) # (batch_size, <= n_ctx, d_model)

        # Projecting to logits 
        token_embed_T = torch.transpose(self.token_embedding.weight, 0, 1).to(x.dtype) # (d_model, n_vocab)
        logits = (x @ token_embed_T).float() # (batch_size, <= n_ctx, n_vocab)

        return logits