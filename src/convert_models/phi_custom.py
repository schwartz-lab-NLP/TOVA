from typing import Optional, Tuple
import warnings
import math

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache
from transformers.models.phi.modeling_phi import PhiFlashAttention2
from transformers.models.phi.configuration_phi import PhiConfig
from LongLM import apply_rotary_pos_emb, repeat_kv


def tova_phi_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        config: Optional[PhiConfig] = None,  
        layer_idx: Optional[int] = None,  
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    output_attentions = False
    bsz, q_len, _ = hidden_states.size()

    attention = PhiFlashAttention2(config, layer_idx)

    query_states = attention.q_proj(hidden_states)  
    key_states = attention.k_proj(hidden_states)   
    value_states = attention.v_proj(hidden_states) 

    if attention.qk_layernorm:
        query_states = attention.q_layernorm(query_states)
        key_states = attention.k_layernorm(key_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    query_states = query_states.view(bsz, q_len, attention.num_heads, attention.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)

    
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if attention.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, attention.layer_idx)
    cos, sin = attention.rotary_emb(value_states, seq_len=position_ids[0, -1].item() + 1)

    # Partial rotary embedding
    query_rot, query_pass = (
        query_states[..., : attention.rotary_emb.dim],
        query_states[..., attention.rotary_emb.dim :],
    )
    key_rot, key_pass = (
        key_states[..., : attention.rotary_emb.dim],
        key_states[..., attention.rotary_emb.dim :],
    )
    # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
    query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

    # [batch_size, seq_length, num_heads, head_dim]
    query_states = torch.cat((query_rot, query_pass), dim=-1)
    key_states = torch.cat((key_rot, key_pass), dim=-1)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": attention.rotary_emb.dim}
        key_states, value_states = past_key_value.update(key_states, value_states, attention.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, attention.num_key_value_groups)
    value_states = repeat_kv(value_states, attention.num_key_value_groups)

    attn_weights = torch.matmul(
        query_states.to(torch.float32), key_states.to(torch.float32).transpose(2, 3)
    ) / math.sqrt(attention.head_dim)

    if attn_weights.size() != (bsz, attention.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, attention.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=attention.attention_dropout, training=attention.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, attention.num_heads, q_len, attention.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, attention.num_heads, q_len, attention.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, attention.hidden_size)

    attn_output = attention.dense(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def tova_phi_prepare_inputs_for_generation_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
            and cache_length > 0 # added to the original imp
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = torch.arange(past_length+input_ids.shape[1]).long().unsqueeze(0) # changed from the original imp
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs