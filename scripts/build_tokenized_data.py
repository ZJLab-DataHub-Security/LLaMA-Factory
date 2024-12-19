# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import os, sys
sys.path.append(f"{os.getcwd()}/src")
from llamafactory.data import get_dataset, split_dataset
from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams import get_train_args
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    build_tokenized_data(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)

def build_tokenized_data(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    print("entering build_tokenized_data")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    assert data_args.tokenized_path, "tokenized_path should be given"
    tokenized_path = data_args.tokenized_path
    data_args.tokenized_path = None
    dataset = get_dataset(model_args, data_args, training_args, stage="sft", **tokenizer_module)
    dataset.save_to_disk(tokenized_path)

def main():
    run_exp()


if __name__ == "__main__":
    main()
