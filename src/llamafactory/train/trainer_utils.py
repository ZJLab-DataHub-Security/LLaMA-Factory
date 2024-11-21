from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union, Iterable
import math, os

import torch
from transformers import Trainer
from transformers.optimization import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from ..extras.constants import IGNORE_INDEX
from ..extras.logging import get_logger
from ..extras.packages import is_galore_available
from ..hparams import FinetuningArguments, ModelArguments
from ..model import find_all_linear_modules, load_model, load_tokenizer, load_valuehead_params

from transformers.tokenization_utils_base import EncodedInput, BatchEncoding, PaddingStrategy
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import T_co, _collate_fn_t, _worker_init_fn_t
from torch.utils.data.dataloader import _BaseDataLoaderIter, _SingleProcessDataLoaderIter
from torch.utils.data import _utils

from llamafactory.easy_context import prepare_seq_parallel_sft_inputs

if is_galore_available():
    from galore_torch import GaLoreAdafactor, GaLoreAdamW, GaLoreAdamW8bit


if TYPE_CHECKING:
    from transformers import PreTrainedModel, Seq2SeqTrainingArguments
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments


logger = get_logger(__name__)


class DummyOptimizer(torch.optim.Optimizer):
    r"""
    A dummy optimizer used for the GaLore algorithm.
    """

    def __init__(
        self, lr: float = 1e-3, optimizer_dict: Optional[Dict["torch.nn.Parameter", "torch.optim.Optimizer"]] = None
    ) -> None:
        dummy_tensor = torch.randn(1, 1)
        self.optimizer_dict = optimizer_dict
        super().__init__([dummy_tensor], {"lr": lr})

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        pass


def create_modelcard_and_push(
    trainer: "Trainer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    kwargs = {
        "tasks": "text-generation",
        "finetuned_from": model_args.model_name_or_path,
        "tags": ["llama-factory", finetuning_args.finetuning_type],
    }
    if data_args.dataset is not None:
        kwargs["dataset"] = [dataset.strip() for dataset in data_args.dataset.split(",")]

    if model_args.use_unsloth:
        kwargs["tags"] = kwargs["tags"] + ["unsloth"]

    if not training_args.do_train:
        pass
    elif training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(license="other", **kwargs)  # prevent from connecting to hub


def create_ref_model(
    model_args: "ModelArguments", finetuning_args: "FinetuningArguments", add_valuehead: bool = False
) -> Optional[Union["PreTrainedModel", "AutoModelForCausalLMWithValueHead"]]:
    r"""
    Creates reference model for PPO/DPO training. Evaluation mode is not supported.

    The valuehead parameter is randomly initialized since it is useless for PPO training.
    """
    if finetuning_args.ref_model is not None:
        ref_model_args = ModelArguments.copyfrom(
            model_args,
            model_name_or_path=finetuning_args.ref_model,
            adapter_name_or_path=finetuning_args.ref_model_adapters,
            quantization_bit=finetuning_args.ref_model_quantization_bit,
        )
        ref_finetuning_args = FinetuningArguments()
        tokenizer = load_tokenizer(ref_model_args)["tokenizer"]
        ref_model = load_model(
            tokenizer, ref_model_args, ref_finetuning_args, is_trainable=False, add_valuehead=add_valuehead
        )
        logger.info("Created reference model from {}".format(finetuning_args.ref_model))
    else:
        if finetuning_args.finetuning_type == "lora":
            ref_model = None
        else:
            ref_model_args = ModelArguments.copyfrom(model_args)
            ref_finetuning_args = FinetuningArguments()
            tokenizer = load_tokenizer(ref_model_args)["tokenizer"]
            ref_model = load_model(
                tokenizer, ref_model_args, ref_finetuning_args, is_trainable=False, add_valuehead=add_valuehead
            )
            logger.info("Created reference model from the model itself.")

    return ref_model


def create_reward_model(
    model: "AutoModelForCausalLMWithValueHead", model_args: "ModelArguments", finetuning_args: "FinetuningArguments"
) -> Optional["AutoModelForCausalLMWithValueHead"]:
    r"""
    Creates reward model for PPO training.
    """
    if finetuning_args.reward_model_type == "api":
        assert finetuning_args.reward_model.startswith("http"), "Please provide full url."
        logger.info("Use reward server {}".format(finetuning_args.reward_model))
        return finetuning_args.reward_model
    elif finetuning_args.reward_model_type == "lora":
        model.pretrained_model.load_adapter(finetuning_args.reward_model, "reward")
        for name, param in model.named_parameters():  # https://github.com/huggingface/peft/issues/1090
            if "default" in name:
                param.data = param.data.to(torch.float32)  # trainable params should in fp32
        vhead_params = load_valuehead_params(finetuning_args.reward_model, model_args)
        assert vhead_params is not None, "Reward model is not correctly loaded."
        model.register_buffer("reward_head_weight", vhead_params["v_head.summary.weight"], persistent=False)
        model.register_buffer("reward_head_bias", vhead_params["v_head.summary.bias"], persistent=False)
        model.register_buffer(
            "default_head_weight", torch.zeros_like(vhead_params["v_head.summary.weight"]), persistent=False
        )
        model.register_buffer(
            "default_head_bias", torch.zeros_like(vhead_params["v_head.summary.bias"]), persistent=False
        )
        logger.info("Loaded adapter weights of reward model from {}".format(finetuning_args.reward_model))
        return None
    else:
        reward_model_args = ModelArguments.copyfrom(
            model_args,
            model_name_or_path=finetuning_args.reward_model,
            adapter_name_or_path=finetuning_args.reward_model_adapters,
            quantization_bit=finetuning_args.reward_model_quantization_bit,
        )
        reward_finetuning_args = FinetuningArguments()
        tokenizer = load_tokenizer(reward_model_args)["tokenizer"]
        reward_model = load_model(
            tokenizer, reward_model_args, reward_finetuning_args, is_trainable=False, add_valuehead=True
        )
        logger.info("Loaded full weights of reward model from {}".format(finetuning_args.reward_model))
        logger.warning("Please ensure the ppo model and reward model share SAME tokenizer and vocabulary.")
        return reward_model


def _get_decay_parameter_names(model: "PreTrainedModel") -> List[str]:
    r"""
    Returns a list of names of parameters with weight decay. (weights in non-layernorm layers)
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def _create_galore_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    if len(finetuning_args.galore_target) == 1 and finetuning_args.galore_target[0] == "all":
        galore_targets = find_all_linear_modules(model)
    else:
        galore_targets = finetuning_args.galore_target

    galore_params: List["torch.nn.Parameter"] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(target in name for target in galore_targets):
            for param in module.parameters():
                if param.requires_grad and len(param.shape) > 1:
                    galore_params.append(param)

    galore_kwargs = {
        "rank": finetuning_args.galore_rank,
        "update_proj_gap": finetuning_args.galore_update_interval,
        "scale": finetuning_args.galore_scale,
        "proj_type": finetuning_args.galore_proj_type,
    }

    id_galore_params = {id(param) for param in galore_params}
    decay_params, nodecay_params = [], []  # they are non-galore parameters
    trainable_params: List["torch.nn.Parameter"] = []  # galore_params + decay_params + nodecay_params
    decay_param_names = _get_decay_parameter_names(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            if id(param) not in id_galore_params:
                if name in decay_param_names:
                    decay_params.append(param)
                else:
                    nodecay_params.append(param)

    _, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)

    if training_args.optim == "adamw_torch":
        optim_class = GaLoreAdamW
    elif training_args.optim in ["adamw_bnb_8bit", "adamw_8bit", "paged_adamw_8bit"]:
        optim_class = GaLoreAdamW8bit
    elif training_args.optim == "adafactor":
        optim_class = GaLoreAdafactor
    else:
        raise NotImplementedError("Unknow optim: {}".format(training_args.optim))

    if finetuning_args.galore_layerwise:
        if training_args.gradient_accumulation_steps != 1:
            raise ValueError("Per-layer GaLore does not support gradient accumulation.")

        optimizer_dict: Dict["torch.Tensor", "torch.optim.Optimizer"] = {}
        for param in nodecay_params:
            param_groups = [dict(params=[param], weight_decay=0.0)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)
        for param in decay_params:
            param_groups = [dict(params=[param], weight_decay=training_args.weight_decay)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)
        for param in galore_params:  # galore params have weight decay
            param_groups = [dict(params=[param], weight_decay=training_args.weight_decay, **galore_kwargs)]
            optimizer_dict[param] = optim_class(param_groups, **optim_kwargs)

        def optimizer_hook(param: "torch.nn.Parameter"):
            if param.grad is not None:
                optimizer_dict[param].step()
                optimizer_dict[param].zero_grad()

        for param in trainable_params:
            param.register_post_accumulate_grad_hook(optimizer_hook)

        optimizer = DummyOptimizer(lr=training_args.learning_rate, optimizer_dict=optimizer_dict)
    else:
        param_groups = [
            dict(params=nodecay_params, weight_decay=0.0),
            dict(params=decay_params, weight_decay=training_args.weight_decay),
            dict(params=galore_params, weight_decay=training_args.weight_decay, **galore_kwargs),
        ]
        optimizer = optim_class(param_groups, **optim_kwargs)

    logger.info("Using GaLore optimizer, may cause hanging at the start of training, wait patiently.")
    return optimizer


def _create_loraplus_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    default_lr = training_args.learning_rate
    loraplus_lr = training_args.learning_rate * finetuning_args.loraplus_lr_ratio
    embedding_lr = finetuning_args.loraplus_lr_embedding

    decay_param_names = _get_decay_parameter_names(model)
    param_dict: Dict[str, List["torch.nn.Parameter"]] = {
        "lora_a": [],
        "lora_b": [],
        "lora_b_nodecay": [],
        "embedding": [],
    }
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_embedding_B" in name:
                param_dict["embedding"].append(param)
            elif "lora_B" in name or param.ndim == 1:
                if name in decay_param_names:
                    param_dict["lora_b"].append(param)
                else:
                    param_dict["lora_b_nodecay"].append(param)
            else:
                param_dict["lora_a"].append(param)

    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    param_groups = [
        dict(params=param_dict["lora_a"], lr=default_lr, weight_decay=training_args.weight_decay),
        dict(params=param_dict["lora_b"], lr=loraplus_lr, weight_decay=training_args.weight_decay),
        dict(params=param_dict["lora_b_nodecay"], lr=loraplus_lr, weight_decay=0.0),
        dict(params=param_dict["embedding"], lr=embedding_lr, weight_decay=training_args.weight_decay),
    ]
    optimizer = optim_class(param_groups, **optim_kwargs)
    logger.info("Using LoRA+ optimizer with loraplus lr ratio {:.2f}.".format(finetuning_args.loraplus_lr_ratio))
    return optimizer


def _create_badam_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    decay_params, nodecay_params = [], []
    decay_param_names = _get_decay_parameter_names(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in decay_param_names:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    param_groups = [
        dict(params=nodecay_params, weight_decay=0.0),
        dict(params=decay_params, weight_decay=training_args.weight_decay),
    ]

    if finetuning_args.badam_mode == "layer":
        from badam import BlockOptimizer

        base_optimizer = optim_class(param_groups, **optim_kwargs)
        optimizer = BlockOptimizer(
            base_optimizer=base_optimizer,
            named_parameters_list=list(model.named_parameters()),
            block_prefix_list=None,
            switch_block_every=finetuning_args.badam_switch_interval,
            start_block=finetuning_args.badam_start_block,
            switch_mode=finetuning_args.badam_switch_mode,
            verbose=finetuning_args.badam_verbose,
        )
        logger.info(
            f"Using BAdam optimizer with layer-wise update, switch mode is {finetuning_args.badam_switch_mode}, "
            f"switch block every {finetuning_args.badam_switch_interval} steps, "
            f"default start block is {finetuning_args.badam_start_block}"
        )

    elif finetuning_args.badam_mode == "ratio":
        from badam import BlockOptimizerRatio

        assert finetuning_args.badam_update_ratio > 1e-6
        optimizer = BlockOptimizerRatio(
            param_groups=param_groups,
            named_parameters_list=list(model.named_parameters()),
            update_ratio=finetuning_args.badam_update_ratio,
            mask_mode=finetuning_args.badam_mask_mode,
            verbose=finetuning_args.badam_verbose,
            include_embedding=False,
            **optim_kwargs,
        )
        logger.info(
            f"Using BAdam optimizer with ratio-wise update, update ratio is {finetuning_args.badam_update_ratio}, "
            f"mask mode is {finetuning_args.badam_mask_mode}"
        )

    return optimizer


def create_custom_optimzer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> Optional["torch.optim.Optimizer"]:
    if finetuning_args.use_galore:
        return _create_galore_optimizer(model, training_args, finetuning_args)

    if finetuning_args.loraplus_lr_ratio is not None:
        return _create_loraplus_optimizer(model, training_args, finetuning_args)

    if finetuning_args.use_badam:
        return _create_badam_optimizer(model, training_args, finetuning_args)


def create_custom_scheduler(
    training_args: "Seq2SeqTrainingArguments",
    num_training_steps: int,
    optimizer: Optional["torch.optim.Optimizer"] = None,
) -> None:
    if optimizer is not None and isinstance(optimizer, DummyOptimizer):
        optimizer_dict = optimizer.optimizer_dict
        scheduler_dict: Dict["torch.nn.Parameter", "torch.optim.lr_scheduler.LRScheduler"] = {}

        for param in optimizer_dict.keys():
            scheduler_dict[param] = get_scheduler(
                training_args.lr_scheduler_type,
                optimizer=optimizer_dict[param],
                num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=training_args.lr_scheduler_kwargs,
            )

        def scheduler_hook(param: "torch.nn.Parameter"):
            scheduler_dict[param].step()

        for param in optimizer_dict.keys():
            param.register_post_accumulate_grad_hook(scheduler_hook)


def get_batch_logps(
    logits: "torch.Tensor", labels: "torch.Tensor", label_pad_token_id: int = IGNORE_INDEX
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    r"""
    Computes the log probabilities of the given labels under the given logits.

    Returns:
        logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
        valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0  # dummy token
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

def _nearest_power_of_2(x):
    x=x-1
    x|=(x>>1)
    x|=(x>>2)
    x|=(x>>4)
    x|=(x>>8)
    x|=(x>>16)
    return x+1

def _new_pad(
    self,
    encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
    max_length: Optional[int] = None,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    pad_to_multiple_of: Optional[int] = None,
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
        if self.padding_side == "right":
            if return_attention_mask:
                encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = (
                    encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                )
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
            encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
        elif self.padding_side == "left":
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
            raise ValueError(f"Invalid padding strategy:{self.padding_side}")
    return encoded_inputs

class _DynamicSequenceParallelDataLoaderIter(_SingleProcessDataLoaderIter):

    def __init__(self, loader):
        super().__init__(loader)
        self.seqlen_per_gpu = loader.seqlen_per_gpu
        self.cutoff_len = loader.cutoff_len
        self.seq_algo = loader.seq_algo
        self.total_batch_size = loader.total_batch_size
        self.mini_batch_size = loader.batch_size
        self.iter_batch = 0
        eps=1e-5
        cache_size = int(math.log2(self.cutoff_len // self.seqlen_per_gpu)+eps)+1
        self.cache = [[] for _ in range(cache_size)]

    def _try_get_data_from_cache(self):
        for i in reversed(range(len(self.cache))):
            batches = self.cache[i]
            min_batches_num = min(self._world_size>>i, (self.total_batch_size-self.iter_batch) // self.mini_batch_size)
            # in this case, dp_size is min_batches_num // mini_batch_size, dp_size * sp_size = world_size. so it should be ensured that world_size should be devided by min_batches_num
            if len(batches) >= min_batches_num and self._world_size % min_batches_num == 0:
                self.iter_batch = (self.iter_batch + min_batches_num * self.mini_batch_size) % self.total_batch_size
                data = batches[:min_batches_num]
                self.cache[i] = batches[min_batches_num:]
                return data
        return None

    def cache_size(self):
        return sum([len(x) for x in self.cache])

    def _append(self, data):
        for i, batches in enumerate(self.cache):
            batches.extend(data[i])

    def _allocate_device(self, data):
        dp_size = len(data)
        sp_size = self._world_size // dp_size
        dp_rank = self._rank // sp_size
        sp_rank = self._rank % sp_size
        batch = data[dp_rank]
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        local_rank = int(os.getenv("LOCAL_RANK"))
        device_data = prepare_seq_parallel_sft_inputs(
            self.seq_algo,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            labels=labels,
            rank=sp_rank,
            world_size=sp_size,
            device=torch.device("cuda", local_rank)
        )
        device_data['sp_size'] = sp_size
        device_data['dp_size'] = dp_size
        return device_data

    def _next_data(self):
        data = self._try_get_data_from_cache()
        while data is None:
            if self.cache_size() > 4*self.total_batch_size:
                batch_len = ', '.join([str(1<<i)+'x'+str(len(batch)) for i, batch in enumerate(self.cache)])
                raise Exception(f"cannot find data meets need, currently need size is {self.total_batch_size-self.iter_batch}, each batch len in cache is {batch_len}")
            index = self._next_index()  # may raise StopIteration
            data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
            self._append(data)
            data = self._try_get_data_from_cache()
        data = self._allocate_device(data)
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data

class DynamicSequenceParallelDataLoader(DataLoader):
    def __init__(self,
                 dataset: Dataset[T_co],
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None,
                 sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
                 num_workers: int = 0,
                 collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 timeout: float = 0,
                 worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None,
                 generator=None,
                 *,
                 prefetch_factor: Optional[int] = None,
                 persistent_workers: bool = False,
                 pin_memory_device: str = "",
                 seqlen_per_gpu: int = 4096,
                 cutoff_len: int = 32768,
                 seq_algo: str = "dist_flash_attn",
                 total_batch_size: int = 1):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device
        )
        self.seqlen_per_gpu = seqlen_per_gpu
        self.cutoff_len = cutoff_len
        self.seq_algo = seq_algo
        self.total_batch_size = total_batch_size

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        return _DynamicSequenceParallelDataLoaderIter(self)
