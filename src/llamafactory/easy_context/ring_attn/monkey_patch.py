import transformers
from typing import List, Optional, Tuple, Union
import warnings
import torch
import torch.utils.checkpoint
from ring_flash_attn.ring_flash_attn import ring_flash_attn_func


def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length=None,
    is_causal = True,
    dropout=0.0,
    position_ids = None,
    softmax_scale=None,
    sliding_window : Optional[int] = None,
    use_top_left_mask=False,
    **kwargs 
):
    # if not self._flash_attn_uses_top_left_mask:
    #     causal = self.is_causal
    # else:
    #     causal = self.is_causal and query_length != 1

    # Contains at least one padding token in the sequence
    # assert attention_mask is None
    assert is_causal is True
    # assert use_sliding_windows is False
    assert query_states.dtype == key_states.dtype == value_states.dtype
    # print(f"q dtype is {query_states.dtype}, k dtype is {key_states.dtype}, v dtype is {value_states.dtype}")
    attn_output = ring_flash_attn_func(
        query_states,
        key_states,
        value_states,
        dropout,
        softmax_scale,
        causal=is_causal,
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


def apply_ring_attn_monkey_patch_llama():
    transformers.models.llama.modeling_llama._flash_attention_forward = (
        new_flash_attn_forward
    )
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = (
        new_decoder_forward
    )


def apply_ring_attn_monkey_patch_mistral():
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2._flash_attention_forward = (
        new_flash_attn_forward
    )
    transformers.models.mistral.modeling_mistral.MistralDecoderLayer.forward = (
        new_decoder_forward
    )
