import transformers
from typing import List, Optional, Tuple, Union
import torch
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import KwargsForCausalLM, LlamaRotaryEmbedding
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from torch import nn

def new_CausalLM_forward(
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
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        num_logits_to_keep (`int`, *optional*):
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

"""
    num_items_in_batch = None
    if "num_items_in_batch" in kwargs: # dp时进入该逻辑
        num_items_in_batch = kwargs.pop("num_items_in_batch")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

    loss = None
    if num_items_in_batch:
        kwargs["num_items_in_batch"] = num_items_in_batch
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

def new_LlamaAttention__init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
    super().__init__()
    self.config = config
    self.layer_idx = layer_idx
    if layer_idx is None:
        logger.warning_once(
            f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
            "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
            "when creating this class."
        )

    self.attention_dropout = config.attention_dropout
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.is_causal = True

    self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
    self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
    self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

def apply_data_parallel_monkey_patch_llama():
    transformers.models.llama.modeling_llama.LlamaAttention.__init__ = (
        new_LlamaAttention__init__
    )
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = (
        new_CausalLM_forward
    )