from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, default_data_collator, DataCollatorWithFlattening
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from llamafactory.easy_context import prepare_seq_parallel_sft_inputs
import os
import torch.nn.functional as F
from torch import distributed as dist

@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature["{}_input_ids".format(key)],
                    "attention_mask": feature["{}_attention_mask".format(key)],
                    "labels": feature["{}_labels".format(key)],
                }
                if "pixel_values" in feature:
                    target_feature["pixel_values"] = feature["pixel_values"]

                if "{}_token_type_ids".format(key) in feature:
                    target_feature["token_type_ids"] = feature["{}_token_type_ids".format(key)]

                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)


@dataclass
class KTODataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for KTO data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        target_features = []
        kl_features = []
        kto_tags = []
        for feature in features:
            target_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"],
            }
            kl_feature = {
                "input_ids": feature["kl_input_ids"],
                "attention_mask": feature["kl_attention_mask"],
                "labels": feature["kl_labels"],
            }
            if "pixel_values" in feature:
                target_feature["pixel_values"] = feature["pixel_values"]

            if "token_type_ids" in feature:
                target_feature["token_type_ids"] = feature["token_type_ids"]
                kl_feature["token_type_ids"] = feature["kl_token_type_ids"]

            target_features.append(target_feature)
            kl_features.append(kl_feature)
            kto_tags.append(feature["kto_tags"])

        batch = super().__call__(target_features)
        kl_batch = super().__call__(kl_features)
        batch["kl_input_ids"] = kl_batch["input_ids"]
        batch["kl_attention_mask"] = kl_batch["attention_mask"]
        batch["kl_labels"] = kl_batch["labels"]
        if "token_type_ids" in batch:
            batch["kl_token_type_ids"] = kl_batch["token_type_ids"]

        batch["kto_tags"] = torch.tensor(kto_tags)
        return batch

@dataclass
class SeqParallelDataCollator(DataCollatorForSeq2Seq):
    r"""
    Data collator for sequence parallel in supervised finetune(sft) stage.
    """
    seq_algo: str = "data_parallel",
    sp_size: int = -1
    rank: int = 0
    world_size: int = 8
    device: Optional[Any] = None

    def __call__(self, features: Sequence[Dict[str, Any]], return_tensors=None) -> Dict[str, torch.Tensor]:
        batch = super().__call__(features, return_tensors)
        if self.seq_algo == "data_parallel":
            return batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        world_size = self.world_size
        sp_rank = self.rank
        if self.sp_size != -1:
            dp_rank = self.rank // self.sp_size
            sp_rank = self.rank % self.sp_size
            world_size = self.sp_size
            bs = len(input_ids)
            dp_size = self.world_size // self.sp_size
            group_bs = bs // dp_size
            input_ids = input_ids[dp_rank * group_bs: (dp_rank + 1) * group_bs]
            attention_mask = attention_mask[dp_rank * group_bs: (dp_rank + 1) * group_bs]
            labels = labels[dp_rank * group_bs: (dp_rank + 1) * group_bs]
        batch = prepare_seq_parallel_sft_inputs(self.seq_algo,
                                                input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                position_ids=None,
                                                labels=labels,
                                                rank=sp_rank,
                                                world_size=world_size,
                                                device=self.device)
        return batch


@dataclass
class SeqParallelDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    r"""
    Data collator for sequence parallel in pretrain(pt) stage.
    Reuse the sequence parallel distributing function for sft stage.
    """
    seq_algo: str = "data_parallel"
    sp_size: int = -1
    rank: int = 0
    world_size: int = 8
    device: Optional[Any] = None

    def __call__(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().__call__(examples)
        if self.seq_algo == "data_parallel":
            return batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        world_size = self.world_size
        sp_rank = self.rank
        if self.sp_size != -1:
            dp_rank = self.rank // self.sp_size
            sp_rank = self.rank % self.sp_size
            world_size = self.sp_size
            bs = len(input_ids)
            dp_size = self.world_size // self.sp_size
            group_bs = bs // dp_size
            input_ids = input_ids[dp_rank * group_bs: (dp_rank + 1) * group_bs]
            attention_mask = attention_mask[dp_rank * group_bs: (dp_rank + 1) * group_bs]
            labels = labels[dp_rank * group_bs: (dp_rank + 1) * group_bs]
        batch = prepare_seq_parallel_sft_inputs(self.seq_algo,
                                                input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                position_ids=None,
                                                labels=labels,
                                                rank=sp_rank,
                                                world_size=world_size,
                                                device=self.device)
        return batch


@dataclass
class SeqParallelDataCollatorWithFlattening(DataCollatorWithFlattening):
    """
    Data collator used for padding free approach. Does the following:

    - concatate the entire mini batch into single long sequence [1, total_tokens]
    - uses `separator_id` to separate sequences within the concatenated `labels`, default value is -100
    - no padding will be added, returns `input_ids`, `labels` and `position_ids`
    """
    tokenizer: PreTrainedTokenizerBase = None
    seq_algo: str = "data_parallel"
    sp_size: int = -1
    rank: int = 0
    world_size: int = 8
    device: Optional[Any] = None
    return_position_ids: bool = True
    separator_id: int = -100
    


    def __call__(self, features, return_tensors=None, separator_id=None):
        assert self.seq_algo in ["data_parallel", "zigzag_ring_attn_varlen"], "SeqParallelDataCollatorWithFlattening only supports seq_algo of data_parallel or zigzag_ring_attn_varlen"
        assert self.return_position_ids, "return_position_ids should be True"

        if self.seq_algo == "data_parallel":
            return super().__call__(features, return_tensors, separator_id)
        
        if return_tensors is None:
            return_tensors = self.return_tensors
        if separator_id is None:
            separator_id = self.separator_id
        assert self.tokenizer.pad_token_id, "pad_token_id is necessary if you are using zigzag_ring_attn_varlen"
        pad_token_id = self.tokenizer.pad_token_id
        
        is_labels_provided = "labels" in features[0]
        ret = {"input_ids": [], "labels": []}
        seqlens_in_batch = []
        pad_to_multiple_of = self.sp_size*2 if self.sp_size != 1 else self.world_size*2
        
        world_size = self.world_size
        sp_rank = self.rank
        if self.sp_size != -1:
            dp_rank = self.rank // self.sp_size
            sp_rank = self.rank % self.sp_size
            world_size = self.sp_size
            bs = len(features)
            dp_size = self.world_size // self.sp_size
            group_bs = bs // dp_size
            features = features[dp_rank * group_bs: (dp_rank + 1) * group_bs]
        
        if self.return_position_ids:
            ret.update({"position_ids": []})

        for idx in range(0, len(features)):
            token_num = len(features[idx]["input_ids"])
            pad_num = ((token_num // pad_to_multiple_of) + 1) * pad_to_multiple_of - token_num
            features[idx]["input_ids"] += pad_num * [pad_token_id]
            features[idx]["labels"] += pad_num * [separator_id]
            ret["input_ids"] += features[idx]["input_ids"]
            seqlens_in_batch.append(len(features[idx]["input_ids"]))
            if is_labels_provided:
                ret["labels"] += [separator_id] + features[idx]["labels"][1:]
            else:
                ret["labels"] += [separator_id] + features[idx]["input_ids"][1:]
            if self.return_position_ids:
                ret["position_ids"] += list(range(len(features[idx]["input_ids"])))
        
        seqlens_in_batch = torch.tensor(seqlens_in_batch, dtype=torch.int32, device=self.device)
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        max_seqlen = seqlens_in_batch.max().item()
        local_cu_seqlens_tensor = cu_seqlens // world_size
        local_max_seqlen = max_seqlen // world_size
        batch = default_data_collator([ret], return_tensors)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        position_ids = batch["position_ids"]
        batch = prepare_seq_parallel_sft_inputs(self.seq_algo,
                                                input_ids=input_ids,
                                                attention_mask=None,
                                                position_ids=position_ids,
                                                labels=labels,
                                                rank=sp_rank,
                                                world_size=world_size,
                                                device=self.device,
                                                cu_seqlens=cu_seqlens,
                                                cu_seq_lens_q=local_cu_seqlens_tensor,
                                                max_length_q=local_max_seqlen
                                                )
        return batch
