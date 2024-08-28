import transformers
from typing import List, Optional, Tuple, Union
import warnings
import torch
import torch.utils.checkpoint
import transformers.models
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast, apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import DynamicCache, Cache
from transformers.utils import logging
from .async_communication import (is_last_time, is_compute_for_local_query, is_sync_from_remote, is_idle, print_and_reset_comm_stats, 
        launch_async_handles, wait_async_handles, maybe_send_recv_fwd_qkvo, maybe_send_recv_bwd_qkvo, maybe_send_recv_bwd_last_dkv, reset_global_memory_buffer,
        maybe_get_set_global_memory_buffer, maybe_get_set_global_memory_buffer_bwd, initialize_distributed, get_sequence_parallel_size, get_sequence_parallel_rank)
from .prepare_input import extract_local
import torch.distributed as dist
import sys 
from .pydf import all_gather # reference: torch/distributed/nn/functional.py

logger = logging.get_logger(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def cust_apply_rotary_pos_emb(q, k, cos_q, sin_q, cos_k, sin_k, unsqueeze_dim=1):
    cos_q = cos_q.unsqueeze(unsqueeze_dim)
    sin_q = sin_q.unsqueeze(unsqueeze_dim)    
    cos_k = cos_k.unsqueeze(unsqueeze_dim)
    sin_k = sin_k.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed

def lss_layer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        assert output_attentions is False, "no support output attention Now"
    
        seq_rank = dist.get_rank()
        seq_world_size = dist.get_world_size()
        
        # 只在 rank 0 的进程上打印参数形状
        if seq_rank == 0:
            print("LSS Layer Forward函数参数 (仅在 rank 0 上显示):")
            print(f"hidden_states shape: {hidden_states.shape}")
            print(f"attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
            print(f"position_ids shape: {position_ids.shape if position_ids is not None else None}")
            print(f"past_key_value: {type(past_key_value)}")
            print(f"output_attentions: {output_attentions}")
            print(f"use_cache: {use_cache}")
            print(f"cache_position shape: {cache_position.shape if cache_position is not None else None}")
            print(f"position_embeddings: {type(position_embeddings)}")

        # hidden_states(x_i) has been scattered 
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Gather x_i , compute fully K and V, todo: add process_group
        # dist.all_gather(all_hidden_states_list,hidden_states)
        all_hidden_states = all_gather(hidden_states)
        # N * [b,s/N,h] -> [b,s,h]
        all_hidden_states = torch.cat(all_hidden_states, dim=1) 

        if seq_world_size == 1:
            assert torch.allclose(all_hidden_states, hidden_states)
        bsz, q_len, h = all_hidden_states.size()
        _, q_i_len,_ = hidden_states.size()
        # flash attn, Q_i and fully K, V

        query_states = self.self_attn.q_proj(hidden_states).view(bsz, q_i_len, self.self_attn.num_heads, self.self_attn.head_dim).transpose(1, 2)
        key_states = self.self_attn.k_proj(all_hidden_states).view(bsz, q_len, self.self_attn.num_key_value_heads, self.self_attn.head_dim).transpose(1, 2)
        value_states = self.self_attn.v_proj(all_hidden_states).view(bsz, q_len, self.self_attn.num_key_value_heads, self.self_attn.head_dim).transpose(1, 2)

        # modify following code with torch.nn.functional.scaled_dot_product_attention(), 
        # reference: modeling_llama.py/LlamaSdaAttention/forward func

        # position_ids_local = extract_local(position_ids, seq_rank, seq_world_size, query_states.device)
        # cos, sin = self.self_attn.rotary_emb(value_states, position_ids)
        # cos_q, sin_q = self.self_attn.rotary_emb(query_states, position_ids_local)

        # query_states, key_states = cust_apply_rotary_pos_emb(query_states, key_states, cos_q, sin_q, cos, sin)

        assert past_key_value is None,  "past_key_value is not supported"         

        key_states = repeat_kv(key_states, self.self_attn.num_key_value_groups)
        value_states = repeat_kv(value_states, self.self_attn.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]    

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        
        is_causal = True if causal_mask is None and q_len > 1 else False

        # 只在 rank 0 的进程上打印中间结果的形状
        if seq_rank == 0:
            print("\nLSS Layer 中间结果 (仅在 rank 0 上显示):")
            print(f"all_hidden_states shape: {all_hidden_states.shape}")
            print(f"query_states shape: {query_states.shape}")
            print(f"key_states shape: {key_states.shape}")
            print(f"value_states shape: {value_states.shape}")
            print(f"causal_mask shape: {causal_mask.shape if causal_mask is not None else None}")
            print(f"is_causal: {is_causal}")
            print(f'is training: {self.training}')

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.self_attn.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.view(bsz, q_i_len, -1)

        attn_output = self.self_attn.o_proj(attn_output)
        if seq_rank == 0:
            print(f"attn_output shape: {attn_output.shape}")
        hidden_states = attn_output
        present_key_value = past_key_value

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None
) -> Union[Tuple, BaseModelOutputWithPast]:
    '''
    lss的实现,reference: dist_flash_attn, modeling_llama.py/LlamaModel/forward(func)
    '''
    # 只在 rank 0 的进程上打印参数形状
    if dist.get_rank() == 0:
        print("Forward函数参数 (仅在 rank 0 上显示):")
        print(f"input_ids shape: {input_ids.shape if input_ids is not None else None}")
        print(f"attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
        print(f"position_ids shape: {position_ids.shape if position_ids is not None else None}")
        print(f"past_key_values: {type(past_key_values)}")
        print(f"inputs_embeds shape: {inputs_embeds.shape if inputs_embeds is not None else None}")
        print(f"use_cache: {use_cache}")
        print(f"output_attentions: {output_attentions}")
        print(f"output_hidden_states: {output_hidden_states}")
        print(f"return_dict: {return_dict}")
        print(f"cache_position shape: {cache_position.shape if cache_position is not None else None}")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    return_legacy_cache = False
    if (
        use_cache and not isinstance(past_key_values, Cache) and not self.training
    ):  # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        logger.warning_once(
            "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
        )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    
    assert position_ids is not None , "Position_ids should not be None due to prepare_input func"
    # if position_ids is None:
    #     position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds

    # not to sure whether this will be modified 
    # position embeddings
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder 
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )    




def apply_lss_transformer_attn_monkey_patch_llama(sp_size= None):

    # 通过修改模型的结构完成SP的优化
    initialize_distributed(sp_size=sp_size)
    # 重写Model的forward函数，来完成传参上的修改
    transformers.models.llama.modeling_llama.LlamaModel.forward =  forward
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = lss_layer_forward  
    
