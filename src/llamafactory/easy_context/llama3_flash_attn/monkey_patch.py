import transformers
from typing import List, Optional, Tuple, Union
import warnings
import torch
import torch.utils.checkpoint
from ring_flash_attn.llama3_flash_attn_varlen import llama3_flash_attn_prepare_cu_seqlens, llama3_flash_attn_varlen_kvpacked_func, llama3_flash_attn_varlen_func
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast
import transformers.models
from transformers.cache_utils import DynamicCache, Cache
from transformers.utils import logging
import torch.distributed as dist
import sys 

def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    is_causal = True,
    dropout=0.0,
    position_ids = None, 
    softmax_scale: Optional[float] = None,
    sliding_window : Optional[int] = None,
    use_top_left_mask=False,
    **kwargs    
):
    # if not self._flash_attn_uses_top_left_mask:
    #     causal = self.is_causal
    # else:
    #     causal = self.is_causal and query_length != 1
    causal = True 
    # Contains at least one padding token in the sequence
    # assert attention_mask is None
    assert causal is True
    # assert use_sliding_windows is False
    print(f"shape is {query_states.shape}")
    local_s = query_states.shape[2] #[b,a,local_s,h]
    cu_seqlens = torch.tensor([i*local_s for i in range(query_states.shape[3]*dist.get_world_size()+1)])
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, locak_k_slice = \
        llama3_flash_attn_prepare_cu_seqlens(cu_seqlens= cu_seqlens,causal=causal, rank=dist.get_rank(),world_size = dist.get_world_size())
    attn_output = llama3_flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        1,
        locak_k_slice,
        dropout,
        softmax_scale,
        causal=causal,
    )
    return attn_output


def new_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    assert isinstance(
        self.self_attn, transformers.models.llama.modeling_llama.LlamaFlashAttention2
    ) or isinstance(
        self.self_attn,
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2,
    ), "Please toggle on the Flash Attention 2 implementation when using zigzag ring attention monkey patch."

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention    
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def apply_llama3_flash_attn_attn_monkey_patch_llama():
    transformers.models.llama.modeling_llama._flash_attention_forward = (
        new_flash_attn_forward
    )
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = (
        new_decoder_forward
    )


