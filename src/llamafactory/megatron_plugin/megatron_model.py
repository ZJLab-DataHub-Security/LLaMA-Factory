from ..extras.logging import get_logger
from ..data import split_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass
import itertools
import torch
from torch.utils.data import DataLoader
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_micro_batch_size,
    get_num_microbatches,
    reconfigure_num_microbatches_calculator,
)
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import AppState
from omegaconf import DictConfig, OmegaConf, open_dict
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments
    from ..hparams import DataArguments

logging = get_logger(__name__)

## copied from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model
class MegatronModel(MegatronGPTSFTModel):

    def build_train_valid_test_datasets(self, stage):
        return

    def attach_args(self, data_args: "DataArguments", training_args: "Seq2SeqTrainingArguments"):
        self._data_args = data_args
        self._training_args = training_args

    def attach_datasets(self, ds):
        train_val_ds = split_dataset(ds, self._data_args, self._training_args)
        self._train_ds = train_val_ds["train_dataset"]
        self._validation_ds = train_val_ds["eval_dataset"]

    def attach_collate_fn(self, collate_fn):
        self._collate_fn = collate_fn

    def build_data_loader(self, dataset, consumed_samples=0, mode='train'):
        data_parallel_rank = parallel_state.get_data_parallel_rank()
        data_parallel_size=parallel_state.get_data_parallel_world_size()
        micro_batch_size = self._training_args.per_device_train_batch_size if mode == 'train' else self._training_args.per_device_eval_batch_size
        global_batch_size = micro_batch_size * data_parallel_size * self._training_args.gradient_accumulation_steps
        batch_sampler = MegatronPretrainingBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            drop_last=self._training_args.dataloader_drop_last,
            pad_samples_to_global_batch_size=not self._training_args.dataloader_drop_last,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self._collate_fn,
            num_workers=self._training_args.dataloader_num_workers,
            pin_memory=self._training_args.dataloader_pin_memory,
            persistent_workers=True if self._training_args.dataloader_num_workers > 0 else False,
        )

    def setup_training_dataloader(self):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            self._train_dl = self.build_data_loader(
                dataset=self._train_ds,
                consumed_samples=consumed_samples,
                mode='train'
            )

    def setup_eval_dataloader(self):
        self._validation_dl = self.build_data_loader(
            dataset=self._validation_ds,
            consumed_samples=0,
            mode='eval'
        )

    def setup(self, stage=None):
        # NOTE: super().__init__ will try and setup train/val/test datasets, but we sidestep this using a if self._train_ds is not None condition
        # We then set things up for real only once setup() of this class is called.
        resume_checkpoint_path = self.trainer.ckpt_path
        self.setup_complete = True
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples

        if stage == 'predict':
            return

        # If the user wants to manually override train and validation dataloaders before calling `.fit()`
        if self._train_dl is not None and self._validation_dl is not None:
            return
        self.build_train_valid_test_datasets(stage=stage)
        if hasattr(self, '_train_ds'):
            self.setup_training_dataloader()
        if hasattr(self, '_validation_ds'):
            self.setup_eval_dataloader()

        # when using pipeline model parallel the final stage need to initialize word embeddings
        self.initialize_last_rank_embeddings()

        if self.cfg.get('transformer_engine', False) or self.cfg.get('mcore_gpt', False):
            self.setup_transformer_engine_tp_groups()
            self.setup_transformer_engine_cp_groups()
        self.setup_complete = True

    def _reconfigure_and_process_inference_batch(self, batch, data_cfg):
        global_batch_size_per_gpu = batch['tokens'].size(0)
        # This should happen only on the last batch of the dataset.
        if (
            global_batch_size_per_gpu
            != get_current_global_batch_size() // parallel_state.get_data_parallel_world_size()
        ):
            # NOTE: This is reconfiguring to make sure there is no grad-acc for validation batches.
            if (
                global_batch_size_per_gpu
                != data_cfg.global_batch_size // parallel_state.get_data_parallel_world_size()
            ):
                app_state = AppState()
                reconfigure_num_microbatches_calculator(
                    rank=app_state.global_rank,
                    rampup_batch_size=None,
                    global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
                    micro_batch_size=global_batch_size_per_gpu,
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            # NOTE: need to explicitly handle resetting for multi-validation
            else:
                app_state = AppState()
                reconfigure_num_microbatches_calculator(
                    rank=app_state.global_rank,
                    rampup_batch_size=None,
                    global_batch_size=data_cfg.global_batch_size,
                    micro_batch_size=data_cfg.micro_batch_size,
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )

    def sft_validation_step(self, dataloader_iter, dataloader_idx=0):
        """
        Our dataloaders produce a micro-batch and then we fetch
        a number of microbatches depending on the global batch size and model parallel size
        from the dataloader to produce a list of microbatches.
        The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        # Initialize userbuffer communicators.
        if self.initialize_ub:
            self.initialize_ub_func()

        if isinstance(self.model, list):
            for model_module in self.model:
                model_module.eval()
        else:
            self.model.eval()

        first_val_step = None

        with torch.no_grad():
            loss = self.fwd_bwd_step(dataloader_iter, True, first_val_step)

        if isinstance(self.model, list):
            for model_module in self.model:
                model_module.train()
        else:
            self.model.train()

        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(loss)
        else:
            self.validation_step_outputs.append(loss)

        return loss

    def inference_step_validation_call(self, batch, batch_idx, data_cfg, dataloader_idx=0):
        metadata = batch.get('metadata', [{}] * len(batch['tokens']))
        # Pass dataloader_idx, as it's needed in val_step of GPTModel to append the loss correctly to self.val/test_step_outputs
        # in case of multi dataloaders
        loss = super().sft_validation_step(itertools.chain([batch]), dataloader_idx)

        if data_cfg.get("write_predictions_to_file", False) or data_cfg.metric.name != 'loss':
            # We need _inference_config to get generation params
            # add_BOS and tokens_to_generate are set in dataset
            if self.get_inference_config() is None:
                self.set_inference_config(inference_config={})
            self._inference_config['add_BOS'] = data_cfg.add_bos
            self._inference_config['tokens_to_generate'] = data_cfg.get('tokens_to_generate')

            output = self.predict_step(batch, batch_idx, dataloader_idx)
            if output:
                inputs_text = [self.tokenizer.ids_to_text(c.tolist()) for c in batch['contexts']]
                labels_text = [self.tokenizer.ids_to_text(a.tolist()) for a in batch['answers']]
                preds_text = [
                    self.tokenizer.ids_to_text(t[l.item() :][: data_cfg.get('tokens_to_generate')])
                    for t, l in zip(output['token_ids'], batch['context_lengths'])
                ]
            else:
                inputs_text, labels_text, preds_text = [], [], []
        else:
            inputs_text, labels_text, preds_text = [], [], []

        outputs = {
            'loss': loss,
            'preds': preds_text,  # [str]
            'labels': labels_text,  # [str]
            'inputs': inputs_text,  # [str]
            'metadata': metadata,  # [dict]
        }
        return outputs

    def gather_and_maybe_write_predictions(self, output, data_cfg, mode, averaged_metric, dataloader_idx=0):
        # Gather the outputs object from all data parallel ranks since we are using the DistributedSampler which splits data across DDP ranks.
        gathered_outputs = [None for _ in range(parallel_state.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            gathered_outputs,
            [
                {'preds': x['preds'], 'labels': x['labels'], 'inputs': x['inputs'], 'metadata': x['metadata']}
                for x in output
            ],
            group=parallel_state.get_data_parallel_group(),
        )

        # Remove duplicate examples due to distributed sampler.
        deduplicated_outputs = {
            'preds': [],
            'labels': [],
            'inputs': [],
            'metadata': [],
        }
        total_size = 0
        for rank in range(0, parallel_state.get_data_parallel_world_size()):
            for batch in gathered_outputs[rank]:
                for pred, label, input, metadata in zip(
                    batch['preds'], batch['labels'], batch['inputs'], batch['metadata']
                ):
                    total_size += 1
                    if not metadata.get("__AUTOGENERATED__", False):
                        deduplicated_outputs['preds'].append(pred)
                        deduplicated_outputs['labels'].append(label)
                        deduplicated_outputs['inputs'].append(input)
                        deduplicated_outputs['metadata'].append(metadata)
                    else:
                        logging.info(f"skipping autogenerated example example {input} prediction {pred} label {label}")

        # Compute metric score
        metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
        metric_label_key = self.val_metric_label_key if mode == 'validation' else self.test_metric_label_key

        # Write predictions to file
        if self.global_rank == 0 and data_cfg.get("write_predictions_to_file", False):
            logging.info(
                f"Total deduplicated inference data size: {total_size} to {len(deduplicated_outputs['inputs'])}"
            )

            # Check if the user provided a prefix path to the file(s) they want to write.
            if not hasattr(data_cfg, "output_file_path_prefix") or data_cfg.output_file_path_prefix is None:
                raise ValueError(
                    f"Cannot write predictions to file when output_file_path_prefix is not set or present in the yaml config file."
                )
            filename_log_key = "val_loss"
            self.write_predictions_to_file(
                deduplicated_outputs, f"{data_cfg.output_file_path_prefix}_{filename_log_key}"
            )

        return deduplicated_outputs, total_size

    def inference_step(self, dataloader_iter, mode):
        batch, batch_idx, dataloader_idx = next(dataloader_iter)
        data_cfg = self.cfg.data.validation_ds if mode == 'validation' else self.cfg.data.test_ds
        self._reconfigure_and_process_inference_batch(batch, data_cfg)
        # Meta data from dataset
        outputs = self.inference_step_validation_call(batch, batch_idx, data_cfg, dataloader_idx)

        if mode == 'validation':
            if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
                # super().validation_step appends just loss to self.validation_step_outputs, replace the last appended loss with the outputs dict
                self.validation_step_outputs[dataloader_idx][-1] = outputs
            else:
                # super().validation_step appends just loss to self.validation_step_outputs, replace the last appended loss with the outputs dict
                self.validation_step_outputs[-1] = outputs
        else:
            if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
                self.test_step_outputs[dataloader_idx][-1] = outputs
            else:
                self.test_step_outputs[-1] = outputs
        return outputs

    def on_validation_epoch_start(self):
        self._reset_activation_checkpointing_args()
        app_state = AppState()
        app_state.data_parallel_size
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self._training_args.per_device_eval_batch_size * parallel_state.get_data_parallel_world_size() * self._training_args.gradient_accumulation_steps,
            micro_batch_size=self._training_args.per_device_eval_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

    def on_validation_epoch_end(self):
        # TODO: this method should be modularized. It is too long and does too many things. (@adithyare)
        # Parent class will handle logging of the loss.
        outputs = self.validation_step_outputs
        if not outputs or not outputs[0]:
            return

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        averaged_loss = []
        averaged_metric = []
        # Log metrics for each provided validation/test dataset.
        for dataloader_idx, output in enumerate(outputs):
            # Expand on_validation_epoch_end from parent class MegatronGPTModel as on_validation_epoch_end doesnt take outputs arg
            # loss = super().on_validation_epoch_end([x['loss'] for x in output])
            loss_vals = [x['loss'] for x in output]
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss with their batch size
                if self._training_args.dataloader_drop_last:
                    loss = torch.stack(loss_vals).mean()
                else:
                    # Compute the avg loss by total_loss across all samples / total number of samples
                    total_loss_and_total_samples = torch.vstack(loss_vals).sum(axis=0)
                    avg_loss = total_loss_and_total_samples[0] / total_loss_and_total_samples[1]
                    loss = avg_loss.type(torch.float32).cuda()
            else:
                loss = torch.tensor(0.0, dtype=torch.float32).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(loss, get_last_rank())

            self.log('val_loss', loss, prog_bar=True, rank_zero_only=True, batch_size=1)

            # Determine the key used to log the loss based on the user provided name of the dataset or the dataloader index.
            # loss_log_key = self._determine_log_key(self.cfg.data.validation_ds, dataloader_idx, "loss", mode)
            # loss_log_key = "val_loss"
            # self.log(loss_log_key, loss, batch_size=1)
            averaged_loss.append(loss)
            self.gather_and_maybe_write_predictions(output, self.cfg.data.validation_ds, "validation", averaged_metric, dataloader_idx)

            torch.distributed.barrier(group=parallel_state.get_data_parallel_group())
            outputs[dataloader_idx].clear()  # free memory

        # Logging of the averaged metrics:
        averaged_loss = sum(averaged_loss) / len(averaged_loss)
        averaged_metric = sum(averaged_metric) / len(averaged_metric) if len(averaged_metric) >= 1 else None

        # Handle case where metrics can be nan or inf. This can break checkpoint save/load.
        if averaged_metric is not None and (torch.isinf(averaged_metric) or torch.isnan(averaged_metric)):
            app_state = AppState()
            monitor_mode = app_state.checkpoint_callback_params.mode
            assert monitor_mode in ['min', 'max']
            averaged_metric = 0.0 if monitor_mode == 'max' else 1e5

        self.log("validation_loss", averaged_loss, batch_size=1)
        if averaged_metric is not None:
            self.log(f"validation_{self.val_metric_name}", averaged_metric)

        # Merge the functionality of previous on_inference_epoch_end() within inference_epoch_end() func here
        app_state = AppState()
        self._restore_activation_checkpointing_args()
        if hasattr(self, "_train_ds"):
            self._training_args.per_device_train_batch_size
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=self._training_args.per_device_train_batch_size * parallel_state.get_data_parallel_world_size() * self._training_args.gradient_accumulation_steps,
                micro_batch_size=self._training_args.per_device_train_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        # When running `trainer.validate()`, the training dataset is not available.
        else:
            logging.warning('No training data found, reconfiguring microbatches based on validation batch sizes.')
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=self._training_args.per_device_train_batch_size * parallel_state.get_data_parallel_world_size() * self._training_args.gradient_accumulation_steps,
                micro_batch_size=self._training_args.per_device_train_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

        return averaged_loss, averaged_metric

@dataclass
class MegatronDataCollator():

    tokenizer: PreTrainedTokenizerBase
    pad_to_max_length: bool = True
    max_seq_length: Optional[int] = None

    def _collate_item(self, item, max_length, pad_id):
        item = [x + [pad_id] * (max_length - len(x)) for x in item]
        return item

    def _build_loss_mask(self, processed_example):
        """Pad input_ids in batch to max batch length while building loss mask"""
        labels = processed_example['labels']
        loss_mask = [float(x!=-100) for x in labels]
        return loss_mask

    @torch.no_grad()
    def _create_attention_mask(self, max_length):
        attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
        attention_mask = attention_mask < 0.5
        return attention_mask

    def collate_fn(self, batch):
        input_ids = [item['input_ids'][:-1] for item in batch]
        labels = [item['input_ids'][1:] for item in batch]
        # contexts = [item['context_ids'] for item in batch]
        # context_lengths = torch.LongTensor([item['context_length'] for item in batch])
        # answers = [item['answer_ids'] for item in batch]
        loss_mask = [self._build_loss_mask(item)[1:] for item in batch]
        # metadata = [item['metadata'] for item in batch]
        # token_count = [item['token_count'] for item in batch]

        max_length = max([len(x) for x in input_ids])
        # increase max length to nearest multiple of 4 or 8
        if self.pad_to_max_length:
            max_length = self.max_seq_length

        assert max_length <= self.max_seq_length

        attention_mask = [self._create_attention_mask(max_length) for _ in batch]
        attention_mask = torch.stack(attention_mask)
        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(
            self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_token_id)
        )
        labels = torch.LongTensor(self._collate_item(labels, max_length=max_length, pad_id=self.tokenizer.eos_token_id))
        loss_mask = torch.LongTensor(self._collate_item(loss_mask, max_length=max_length, pad_id=0))

        processed_batch = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'attention_mask': attention_mask
        }
        return processed_batch

def load_megatron_model(cfg, trainer) -> MegatronModel:
    model_cfg = MegatronModel.restore_from(cfg.model.restore_from_path, return_config=True)
    OmegaConf.resolve(cfg)
    with open_dict(model_cfg):
        for key, val in cfg.model.items():
            model_cfg[key] = val
        if cfg.get("trainer", None) and cfg.trainer.get("precision"):
            model_cfg.precision = cfg.trainer.precision
    model = MegatronModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)
    peft_cfg_cls = PEFT_CONFIG_MAP[cfg.model.peft.peft_scheme]

    if cfg.model.peft.restore_from_path is not None:
        logging.info("PEFT Weights will be loaded from", cfg.model.peft.restore_from_path)
        model.load_adapters(cfg.model.peft.restore_from_path, peft_cfg_cls(model_cfg))
    elif peft_cfg_cls is not None:
        logging.info("Adding adapter weights to the model for PEFT")
        model.add_adapter(peft_cfg_cls(model_cfg))
    else:
        logging.info(f"Running full finetuning since no peft scheme is given.\n{model.summarize()}")
    return model