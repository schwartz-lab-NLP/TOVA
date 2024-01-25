import types

from torch import nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention
from transformers.models.mistral.modeling_mistral import MistralForCausalLM, MistralAttention
from transformers.models.phi.modeling_phi import PhiForCausalLM, PhiFlashAttention2

from .llama_custom import tova_llama_attention_forward, tova_llama_prepare_inputs_for_generation_generation
from .mistral_custom import tova_mistral_attention_forward, tova_mistral_prepare_inputs_for_generation_generation
from .phi_custom import tova_phi_attention_forward, tova_phi_prepare_inputs_for_generation_generation


def enable_tova_caching(model):
    model_method_mapping = {
        LlamaForCausalLM: tova_llama_prepare_inputs_for_generation_generation,
        MistralForCausalLM: tova_mistral_prepare_inputs_for_generation_generation,
        PhiForCausalLM: tova_phi_prepare_inputs_for_generation_generation
    }

    attention_method_mapping = {
        LlamaAttention: tova_llama_attention_forward,
        MistralAttention: tova_mistral_attention_forward,
        PhiFlashAttention2: tova_phi_attention_forward
    }

    for model_class, method in model_method_mapping.items():
        if isinstance(model, model_class):
            model.prepare_inputs_for_generation = types.MethodType(method, model)

    for name, module in reversed(model._modules.items()):
        for attention_class, method in attention_method_mapping.items():
            if isinstance(module, attention_class):
                module.forward = types.MethodType(method, module)

        if isinstance(module, nn.Module) and len(list(module.children())) > 0:
            enable_tova_caching(module)