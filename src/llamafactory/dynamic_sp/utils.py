from typing import Dict, Optional, Union
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding, PaddingStrategy

def _nearest_power_of_2(x):
    x=x-1
    x|=(x>>1)
    x|=(x>>2)
    x|=(x>>4)
    x|=(x>>8)
    x|=(x>>16)
    return x+1

def _power_of_2_pad(
    self,
    encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
    max_length: Optional[int] = None,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    pad_to_multiple_of: Optional[int] = None,
    padding_side: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
) -> dict:
    # Load from model defaults
    if return_attention_mask is None:
        return_attention_mask = "attention_mask" in self.model_input_names

    required_input = encoded_inputs[self.model_input_names[0]]

    if padding_strategy == PaddingStrategy.LONGEST:
        max_length = len(required_input)

    if max_length is not None and pad_to_multiple_of is not None:
        max_length = _nearest_power_of_2((max_length-1) // pad_to_multiple_of + 1) * pad_to_multiple_of

    needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length
    # Initialize attention mask if not present.
    if return_attention_mask and "attention_mask" not in encoded_inputs:
        encoded_inputs["attention_mask"] = [1] * len(required_input)
    if needs_to_be_padded:
        difference = max_length - len(required_input)
        padding_side = padding_side if padding_side is not None else self.padding_side

        if padding_side == "right":
            if return_attention_mask:
                encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = (
                    encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                )
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
            encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
        elif padding_side == "left":
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                    "token_type_ids"
                ]
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
        else:
            raise ValueError(f"Invalid padding strategy:{padding_side}")
    return encoded_inputs