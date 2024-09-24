import torch
import math
from transformers.models.mistral.modeling_mistral import rotate_half
from ..tova_cache import TOVACache

def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        query_position_indexes: torch.Tensor,
        key_position_indexes: torch.Tensor,
    ):
    # For the query, we only encode the position of the new tokens
    cos_q = cos[query_position_indexes].unsqueeze(1)
    sin_q = sin[query_position_indexes].unsqueeze(1)
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)

    # For the key, we encode the positions of all tokens (new and from cache)
    cos_k = cos[key_position_indexes].unsqueeze(1)
    sin_k = sin[key_position_indexes].unsqueeze(1)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed


def g(x):
    # hyperparameters
    f = lambda x: torch.log(torch.log(x + math.e))
    T = 10

    mask = x <= T
    result = torch.where(mask, x, T + f(x - T))
    return result


def get_compresssed_position_encoding(past_key_value, layer_idx):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cached_indexes = past_key_value.cached_input_indexes[layer_idx]
    diffs = cached_indexes - torch.cat((torch.tensor([0]).to(device), cached_indexes[:-1]))
    compressed_diffs = torch.ceil(g(diffs))
    positions = torch.cumsum(compressed_diffs, dim=0).to(torch.int64)
    return positions.to(device)


def get_positional_encoding_indexes(
        past_key_value: TOVACache,
        position_ids: torch.Tensor,
        layer_idx: int,
        is_input_tokens_round: bool
    ):
    # if first step of the layer, on the original input tokens only, use position ids
    if is_input_tokens_round:
        return position_ids, position_ids

    no_compression = past_key_value.position_encoding_compression is False
    if no_compression:
        key_position_indexes = past_key_value.cached_input_indexes[layer_idx].detach().clone().reshape(1, -1)
        query_position_indexes = key_position_indexes[0, -1].reshape(1, -1)
        return query_position_indexes, key_position_indexes

    key_position_indexes =  get_compresssed_position_encoding(past_key_value, layer_idx).detach().clone().reshape(1, -1)
    query_position_indexes = key_position_indexes[0, -1].reshape(1, -1)
    return query_position_indexes, key_position_indexes
