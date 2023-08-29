import torch
from torch import nn
from typing import Optional, Dict

from .encoder import AudioEncoder
from .decoder import TextDecoder
from .attention import MultiHeadAttention

class AudioToTextTransformer(nn.Module):
    def __init__(self, dims): 
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

    def encode(self, mel: torch.Tensor):
        return self.encoder(mel)
    
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def decode(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))


class AudioToTextTransformerWithKVCaching(AudioToTextTransformer):
    def __init__(self, dims):
        super().__init__(dims)
        self.kv_cache = {}
        self.hooks = []

    def decode(self, tokens, audio_features ): 
        """Perform a forward pass on the decoder and return per-token logits"""

        if not self.kv_cache:
            self.kv_cache, self.hooks = self.install_kv_cache_hooks()

        return self.decoder(tokens, audio_features, kv_cache=self.kv_cache)

    def cleanup_caching(self):
        """Clean up any resources or hooks after decoding is finished"""

        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []

    def rearrange_kv_cache(self, source_indices):
        """Update the key-value cache according to the updated beams"""

        for module, tensor in self.kv_cache.items():
            # update the key/value cache to contain the selected sequences
            self.kv_cache[module] = tensor[source_indices].detach()

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks