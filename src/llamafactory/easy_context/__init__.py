from .dist_flash_attn.prepare_input import prepare_dist_flash_attn_inputs, prepare_dist_flash_attn_sft_inputs
from .dist_flash_attn.monkey_patch import apply_dist_flash_attn_monkey_patch_llama
from .zigzag_ring_attn.prepare_inputs import prepare_zigzag_ring_attn_inputs, prepare_zigzag_ring_attn_sft_inputs, prepare_zigzag_ring_attn_varlen_sft_inputs
from .zigzag_ring_attn.monkey_patch import apply_zigzag_ring_attn_monkey_patch_llama, apply_zigzag_ring_attn_varlen_monkey_patch_llama
from .unsloth_offloaded_gradient_checkpoint.monkey_patch import apply_unsloth_offloaded_gradient_checkpoint_monkey_patch
import torch
import torch.nn.functional as F
import transformers
from transformers.models.llama.modeling_llama import Optional, Union, Cache, List, Unpack, KwargsForCausalLM, Tuple, CausalLMOutputWithPast, FlashAttentionKwargs, BaseModelOutputWithPast, logging, DynamicCache

logger = logging.get_logger(__name__)

def prepare_seq_parallel_inputs(
    seq_algo, input_ids, position_ids, target_ids, rank, world_size, device
):
    if seq_algo == "zigzag_ring_attn":
        return prepare_zigzag_ring_attn_inputs(
            input_ids, position_ids, target_ids, rank, world_size, device
        )
    elif seq_algo == "dist_flash_attn":
        return prepare_dist_flash_attn_inputs(
            input_ids, position_ids, target_ids, rank, world_size, device
        )
    elif seq_algo == "ulysses_attn":
        from .ulysses_attn.prepare_inputs import prepare_ulysses_attn_inputs
        return prepare_ulysses_attn_inputs(
            input_ids, position_ids, target_ids, rank, world_size, device
        )
    elif seq_algo == "data_parallel":
        return {
            "local_input_ids": input_ids.to(device),
            "local_position_ids": position_ids.to(device),
            "local_target_ids": target_ids.to(device),
        }
    else:
        raise ValueError(f"Invalid seq_algo: {seq_algo}")
    
def prepare_seq_parallel_sft_inputs(
    seq_algo, input_ids, attention_mask, position_ids, labels, rank, world_size, device, **kwargs
):
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    shift_labels = F.pad(labels, [0, 1], 'constant', -100)[:, 1:]
    if seq_algo == "zigzag_ring_attn":
        return prepare_zigzag_ring_attn_sft_inputs(
            input_ids, attention_mask, position_ids, shift_labels, rank, world_size, device
        )
    elif seq_algo == "zigzag_ring_attn_varlen":
        return prepare_zigzag_ring_attn_varlen_sft_inputs(
            input_ids, attention_mask, position_ids, shift_labels, rank, world_size, device, **kwargs
        )
    elif seq_algo == "dist_flash_attn":
        return prepare_dist_flash_attn_sft_inputs(
            input_ids, attention_mask, position_ids, shift_labels, rank, world_size, device
        )
    elif seq_algo == "ulysses_attn":
        from .ulysses_attn.prepare_inputs import prepare_ulysses_attn_sft_inputs
        return prepare_ulysses_attn_sft_inputs(
            input_ids, attention_mask, position_ids, shift_labels, rank, world_size, device
        )
    elif seq_algo == "data_parallel":
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "target_ids": labels,
        }
    else:
        raise ValueError(f"Invalid seq_algo: {seq_algo}")
    
def model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # kept for BC (non `Cache` `past_key_values` inputs)
    return_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache):
        return_legacy_cache = True
        if past_key_values is None:
            past_key_values = DynamicCache()
        else:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
            )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
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
                **flash_attn_kwargs,
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
                **flash_attn_kwargs,
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

def causal_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
    **kwargs: Unpack[KwargsForCausalLM],
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    fa_kwargs = dict(kwargs)
    if 'num_items_in_batch' in fa_kwargs:
        fa_kwargs.pop('num_items_in_batch')

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **fa_kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def ForSequenceParallelCausalLMLoss(
    self, logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    from transformers.loss.loss_utils import fixed_cross_entropy
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    logits = logits.contiguous()
    labels = labels.contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)
    # Enable model parallelism
    labels = labels.to(logits.device)
    loss = fixed_cross_entropy(logits, labels, num_items_in_batch, ignore_index, **kwargs)
    return loss

def apply_default_monkey_patch_llama():
    # we rewrite this function because we need to pass `flash_attn_kwargs` to model forward function and pass `num_items_in_batch` to loss_function simultaneously, but there are conflicts in transformers-4.47.0
    # and we also need allow using flash_attn_kwargs whether gradient_checkpointing is True or False which is only allowed when gradient_checkpointing is False in transformers-4.47.0,
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward=causal_model_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward=model_forward

def apply_seq_parallel_monkey_patch(
    args, model
):
    assert args is not None
    seq_algo = args.parallel_mode
    sp_size = args.sp_size
    enable_offload = args.sp_enable_offload
    offload_percent = args.sp_offload_percent
    assert seq_algo in ["zigzag_ring_attn", "zigzag_ring_attn_varlen", "dist_flash_attn", "ulysses_attn", "data_parallel"], f"Invalid seq_algo: {seq_algo}"
    assert model in ["llama", "mistral"], f"Invalid model: {model}"
    apply_default_monkey_patch_llama()
    if args.enable_dynamic_sp:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.loss_function = ForSequenceParallelCausalLMLoss
    if seq_algo == "data_parallel":
        return
    elif seq_algo == "zigzag_ring_attn" and model == "llama":
        apply_zigzag_ring_attn_monkey_patch_llama(sp_size=sp_size)
    elif seq_algo == "zigzag_ring_attn_varlen" and model == "llama":
        apply_zigzag_ring_attn_varlen_monkey_patch_llama(sp_size=sp_size)
    elif seq_algo == "dist_flash_attn" and model == "llama":
        apply_dist_flash_attn_monkey_patch_llama(sp_size=sp_size, enable_offload=enable_offload, offload_percent=offload_percent)
    elif seq_algo == "ulysses_attn" and model == "llama":
        from .ulysses_attn.monkey_patch import apply_ulysses_attn_monkey_patch_llama 
        apply_ulysses_attn_monkey_patch_llama(sp_size=sp_size)
    else:
        raise ValueError(f"Invalid seq_algo: {seq_algo} or model: {model}")
        
def prepare_dataloader(seq_algo, dataloader, acclerator):
    if seq_algo == "data_parallel":
        return acclerator.prepare(dataloader)
    else:
        return dataloader

def prepare_dynamic_sp(seq_algo, sp_size, model):
    if seq_algo == "zigzag_ring_attn":
        apply_zigzag_ring_attn_monkey_patch_llama(sp_size)
    elif seq_algo == "dist_flash_attn":
        from .dist_flash_attn.async_communication import reset_sequence_parallel
        reset_sequence_parallel(sp_size)
    elif seq_algo == "ulysses_attn":
        from .ulysses_attn.monkey_patch import apply_ulysses_attn_monkey_patch_llama 
        apply_ulysses_attn_monkey_patch_llama(sp_size)
