from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from transformers import Trainer

from ...extras.logging import get_logger
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
import datasets
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import os
from transformers.training_args import OptimizerNames
from transformers.trainer_pt_utils import _get_learning_rate
from transformers.utils import is_apex_available
if TYPE_CHECKING:
    import torch
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments

logger = get_logger(__name__)

if is_apex_available():
    from apex import amp

class CustomTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.processor = processor
        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, "torch.Tensor"]] = None) -> None:
        super()._save(output_dir, state_dict)
        if self.processor is not None:
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            getattr(self.processor, "image_processor").save_pretrained(output_dir)

class CustomSeqParallelTrainer(CustomTrainer):
        # diff from base trainer: always passing `num_items_in_batch` to compute_loss function
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)
            # Finally we need to normalize the loss for reporting
            if num_items_in_batch is None:
                return loss.detach() / self.args.gradient_accumulation_steps
            return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        from transformers.trainer import _is_peft_model, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            if self.finetuning_args.parallel_mode== "data_parallel":
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                if not hasattr(self.args, 'average_tokens_across_devices'):
                    self.args.average_tokens_across_devices = None
                if not hasattr(self, 'model_accepts_loss_kwargs'):
                    self.model_accepts_loss_kwargs= None
                if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
                    loss *= self.accelerator.num_processes
            else:
                if num_items_in_batch is None:
                    sp_size = self.finetuning_args.sp_size
                    loss_fn = CrossEntropyLoss(reduction='sum')
                    labels = inputs.pop("labels")
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
                    valid_label_cnt = (labels!=-100).sum(1)[None, :]
                    valid_label_cnt_gather = self.accelerator.gather(valid_label_cnt)
                    n_gpus = valid_label_cnt_gather.shape[0]
                    if sp_size == -1:
                        sp_size = n_gpus
                    dp_rank = self.accelerator.process_index // sp_size
                    valid_label_cnt_all =valid_label_cnt_gather[dp_rank * sp_size : (dp_rank+1) * sp_size].sum(0).detach()
                    shift_logits = logits.contiguous()
                    shift_labels = labels.contiguous()
                    bs = len(shift_labels)
                    loss = torch.zeros(bs, dtype=shift_logits.dtype, device=shift_labels.device)
                    for b in range(bs):
                        normalizer=valid_label_cnt_all[b].item()
                        loss[b]=loss_fn(shift_logits[b], shift_labels[b])/normalizer
                    loss = loss.mean()*sp_size
                else:
                    assert self.args.average_tokens_across_devices is True, "please set average_tokens_across_devices=True if parallel_mode is not data_parallel"
                    loss_fn = CrossEntropyLoss(reduction='sum')
                    labels = inputs.pop("labels")
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
                    shift_logits = logits.contiguous()
                    shift_labels = labels.contiguous()
                    shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    loss = loss_fn(shift_logits, shift_labels)/num_items_in_batch

                    if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
                        loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": False,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        if hasattr(data_collator, "seq_algo") and data_collator.seq_algo != "data_parallel":
            sp_size = self.finetuning_args.sp_size
            if sp_size != -1:
                world_size = int(os.environ['WORLD_SIZE'])
                assert sp_size != 0 and world_size % sp_size == 0, f"world_size: {world_size} should be devide by seq_parallel_size: {sp_size}"
                dp_size = world_size // sp_size
                dataloader_params["batch_size"] = dataloader_params["batch_size"] * dp_size
            return DataLoader(train_dataset, **dataloader_params)
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    def get_eval_dataloader(self, eval_dataset) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        if hasattr(self, "_eval_dataloader") and self.args.dataloader_persistent_workers:
            return self.accelerator.prepare(self._eval_dataloader)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": False,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            self._eval_dataloader = eval_dataloader

        if hasattr(data_collator, "seq_algo") and data_collator.seq_algo != "data_parallel":
            sp_size = self.finetuning_args.sp_size
            if sp_size != -1:
                world_size = int(os.environ['WORLD_SIZE'])
                assert sp_size != 0 and world_size % sp_size == 0, f"world_size: {world_size} should be devide by seq_parallel_size: {sp_size}"
                dp_size = world_size // sp_size
                dataloader_params["batch_size"] = dataloader_params["batch_size"] * dp_size
            return eval_dataloader
        return self.accelerator.prepare(eval_dataloader)