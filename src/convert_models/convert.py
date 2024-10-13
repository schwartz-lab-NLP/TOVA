import types

from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention
from transformers.models.mistral.modeling_mistral import MistralForCausalLM, MistralAttention

from .llama_custom import tova_llama_attention_forward, tova_llama_prepare_inputs_for_generation_generation
from .mistral_custom import tova_mistral_attention_forward, tova_mistral_prepare_inputs_for_generation_generation


def enable_tova_caching(model):
    if hasattr(model, "model") and hasattr(model.model, "_use_sdpa"):
        model.model._use_sdpa = False
    
    if isinstance(model, LlamaForCausalLM):
        model.prepare_inputs_for_generation = types.MethodType(
            tova_llama_prepare_inputs_for_generation_generation, model
        )

    if isinstance(model, MistralForCausalLM):
        model.prepare_inputs_for_generation = types.MethodType(
            tova_mistral_prepare_inputs_for_generation_generation, model
        )
    
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_tova_caching(
                module,
            )

        if isinstance(module, LlamaAttention):
            model._modules[name].forward = types.MethodType(
                tova_llama_attention_forward, model._modules[name]
            )

        if isinstance(module, MistralAttention):
            model._modules[name].forward = types.MethodType(
                tova_mistral_attention_forward, model._modules[name]
            )

        
