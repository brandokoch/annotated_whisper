import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import  Iterable

from .misc import Conv1d, LayerNorm
from .attention import ResidualAttentionBlock

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, d_model: int, n_head: int, n_layer: int
    ):
        
        super().__init__()
        self.conv1 = Conv1d(n_mels, d_model, kernel_size=3, padding=1) # e.g for d_model=1024 it runs 1024 80x3 kernels over spectrogram 
        self.conv2 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1) # e.g for d_model=1024 it runs 1024 1024x3 kernels over conv1 output
        self.register_buffer("positional_embedding", sinusoids(n_ctx, d_model))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(d_model, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(d_model)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """

        # x shape is (batch_size, 80, 3000)
        x = F.gelu(self.conv1(x)) # (batch_size, d_model , 3000)
        x = F.gelu(self.conv2(x)) # (batch_size, d_model, n_ctx) 
        x = x.permute(0, 2, 1) # (batch_size, 1500, d_model)

        assert x.shape[1:] == self.positional_embedding.shape, "Incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype) # (batch_size, 1500, d_model)

        for block in self.blocks:
            x = block(x) # (batch_size, 1500, d_model)

        # Final layer norm preserved from original implementation
        x = self.ln_post(x)
        return x
