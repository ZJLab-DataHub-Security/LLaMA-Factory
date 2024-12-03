# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForSeq2Seq

from ...data import get_dataset, split_dataset
from ...data.collator import SeqParallelDataCollator, DynamicSeqParallelDataCollator
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeMetrics
from .trainer import CustomSeq2SeqTrainer, CustomSeqParallelTrainer, DynamicSequenceParallelDataLoader, DynamicSeqParallelTrainer

import torch
import os
from ...easy_context import apply_seq_parallel_monkey_patch
from types import MethodType
from ..trainer_utils import _new_pad


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    num_pad_to_multiple_of = data_args.cutoff_len
    if finetuning_args.enable_dynamic_sp:
        tokenizer._pad = MethodType(_new_pad, tokenizer)
        num_pad_to_multiple_of = finetuning_args.seqlen_per_gpu

    dataset = get_dataset(model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    apply_seq_parallel_monkey_patch(finetuning_args.parallel_mode, "llama", sp_size=finetuning_args.sp_size, enable_offload=finetuning_args.sp_enable_offload, offload_percent=finetuning_args.sp_offload_percent)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    local_rank = int(os.getenv("LOCAL_RANK"))
    world_size = torch.distributed.get_world_size()
    print(f"seq_len: {data_args.cutoff_len}")
    if finetuning_args.enable_dynamic_sp:
        dp_factor = finetuning_args.total_batch_size // training_args.gradient_accumulation_steps // training_args.train_batch_size
        data_collator = DynamicSeqParallelDataCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=num_pad_to_multiple_of,
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            seq_algo=finetuning_args.parallel_mode,
            seqlen_per_gpu=finetuning_args.seqlen_per_gpu,
            cutoff_len=data_args.cutoff_len,
            dp_factor=dp_factor,
            rank=torch.distributed.get_rank(),
            world_size=world_size,
            device=torch.device("cuda", local_rank)
        )
    else:
        data_collator = SeqParallelDataCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=num_pad_to_multiple_of,
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            seq_algo=finetuning_args.parallel_mode,
            sp_size=finetuning_args.sp_size,
            rank=torch.distributed.get_rank(),
            world_size=world_size,
            device=torch.device("cuda", local_rank)
        )
    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False if model_args.visual_inputs else training_args.remove_unused_columns

    # Initialize our Trainer
    if finetuning_args.enable_dynamic_sp:
        trainer = DynamicSeqParallelTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
            **tokenizer_module,
            **split_dataset(dataset, data_args, training_args),
        )
    else:
        trainer = CustomSeqParallelTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
            **tokenizer_module,
            **split_dataset(dataset, data_args, training_args),
        )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # # Evaluation
    # if training_args.do_eval:
    #     metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
    #     if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
    #         metrics.pop("eval_loss", None)
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # # Predict
    # if training_args.do_predict:
    #     predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
    #     if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
    #         predict_results.metrics.pop("predict_loss", None)
    #     trainer.log_metrics("predict", predict_results.metrics)
    #     trainer.save_metrics("predict", predict_results.metrics)
    #     trainer.save_predictions(predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
