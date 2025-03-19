# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, List, Optional
from transformers import DataCollatorForSeq2Seq
from ...data import get_dataset, split_dataset
from ...model import load_tokenizer
from ...extras.constants import IGNORE_INDEX

from omegaconf.omegaconf import OmegaConf
from ...megatron_plugin.megatron_model import load_megatron_model, MegatronDataCollator
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.utils.exp_manager import exp_manager

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_sft_exp(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
):
    cfg = OmegaConf.load(finetuning_args.megatron_cfg_path)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset = get_dataset(model_args, data_args, training_args, stage="sft", **tokenizer_module)
    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)
    model = load_megatron_model(cfg, trainer)
    model.attach_args(data_args, training_args)
    model.attach_datasets(dataset)

    data_collator = MegatronDataCollator(
        tokenizer=tokenizer,
        max_seq_length=data_args.cutoff_len
    )
    model.attach_collate_fn(data_collator.collate_fn)

    trainer.fit(model)
