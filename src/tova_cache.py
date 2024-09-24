from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache


class TOVACache(DynamicCache):

    def __init__(self, cache_size: int, position_encoding_compression: bool=False):
        super().__init__()
        self.cache_size = cache_size
        self.position_encoding_compression = position_encoding_compression
        self.cached_input_indexes: List[torch.Tensor] = []

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        # We add one because this function is used to determain the attention mask which should be 1 more than the cache size in generation mode.
        return self.key_cache[layer_idx].shape[-2] + 1

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Because the reduce function takes care of it, we don't need to effctivly calculate the usable length.
        # Actually, because of current implementation of models in hf, and given the reduce function we have to return the current use of the cache.
        return 0 if len(self.key_cache) <= layer_idx else self.key_cache[layer_idx].shape[-2]
    
    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        # We add one because this function is used to determain the attention mask which should be 1 more than the cache size in generation mode.
        return self.cache_size + 1

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
            position_ids: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.cached_input_indexes.append(position_ids[0])
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            self.cached_input_indexes[layer_idx] = torch.cat([self.cached_input_indexes[layer_idx], position_ids[0]])

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reduce(
        self,
        layer_idx: int,
        attn_weights: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        bsz, _, _, num_keys = attn_weights.size()
        _, num_kv_heads, _, _ = self.key_cache[layer_idx].size()

        if num_keys <= self.cache_size:
            return
        # Utilize the average attention weights to select the top-k keys and values        
        mean_attn_weights = torch.mean(attn_weights[:, :, -1, :], dim=1).clone().detach()
        vals, ind = torch.topk(mean_attn_weights, k=self.cache_size, dim=-1)
        ind = torch.sort(ind).values        # stabelizes some things for some reason
        expand_ind = ind.unsqueeze(1).unsqueeze(-1).expand(bsz, num_kv_heads, ind.size(-1), self.key_cache[layer_idx].size(-1))

        # Reduce the size of the cache to self.cache_size
        self.key_cache[layer_idx] = torch.gather(self.key_cache[layer_idx], dim=2, index=expand_ind)
        self.value_cache[layer_idx] = torch.gather(self.value_cache[layer_idx], dim=2, index=expand_ind)
        self.cached_input_indexes[layer_idx] = torch.gather(self.cached_input_indexes[layer_idx], dim=0, index=ind[0])
