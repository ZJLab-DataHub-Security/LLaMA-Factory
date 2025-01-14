import transformers
from typing import List, Optional, Tuple, Union
import warnings
import torch
import torch.utils.checkpoint
from ring_flash_attn.zigzag_ring_flash_attn import zigzag_ring_flash_attn_func
from ring_flash_attn.zigzag_ring_flash_attn_varlen import zigzag_ring_flash_attn_varlen_func
from functools import partial

SP_GROUPS = {}

def new_flash_attn_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    group=None
):
    if not use_top_left_mask:
        causal = is_causal
    else:
        causal = is_causal and query_length != 1

    # Contains at least one padding token in the sequence
    assert attention_mask is None
    assert causal is True
    assert sliding_window is None
    attn_output = zigzag_ring_flash_attn_func(
        query_states,
        key_states,
        value_states,
        dropout,
        softmax_scale,
        causal=causal,
        group=group
    )
    return attn_output

def new_flash_attn_varlen_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    group=None
):
    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__.
        causal = is_causal and query_length != 1

    assert attention_mask is None
    assert causal is True
    assert sliding_window is None

    attn_output = zigzag_ring_flash_attn_varlen_func(
        query_states.squeeze(0),
        key_states.squeeze(0),
        value_states.squeeze(0),
        cu_seq_lens_q,
        max_length_q,
        dropout_p=dropout,
        causal=causal,
        group=group,
    )
    return attn_output[None,:]

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
        position_embeddings=position_embeddings,
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

def get_sp_process_group(sequence_parallel_size=None):
    if sequence_parallel_size is None:
        return None
    global SP_GROUPS
    assert torch.distributed.is_initialized()
    if sequence_parallel_size in SP_GROUPS:
        return SP_GROUPS[sequence_parallel_size]
    world_size: int = torch.distributed.get_world_size()
    if sequence_parallel_size is None or sequence_parallel_size == -1:
        sequence_parallel_size = world_size
    else:
        assert world_size % sequence_parallel_size == 0
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    rank = torch.distributed.get_rank()

    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        if rank in ranks:
            group = torch.distributed.new_group(ranks)
            SP_GROUPS[sequence_parallel_size] = group
            return group

def apply_zigzag_ring_attn_monkey_patch_llama(sp_size=None):
    sp_group = get_sp_process_group(sp_size)
    transformers.models.llama.modeling_llama._flash_attention_forward = (
        partial(new_flash_attn_forward, group=sp_group)
    )
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = (
        new_decoder_forward
    )

def apply_zigzag_ring_attn_varlen_monkey_patch_llama(sp_size=None):
    sp_group = get_sp_process_group(sp_size)
    transformers.models.llama.modeling_llama._flash_attention_forward = partial(
        new_flash_attn_varlen_forward, group=sp_group
    )
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = (
        new_decoder_forward
    )
