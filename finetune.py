import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List
import torch
import pdb
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from src.utils.log import create_logger
from src.utils.extract import extract_config, extract_key
from src.utils.submodule import Architecture
from src.utils.data import get_dataset, get_num_labels, get_sequence_tasks, get_qa_tasks, get_lm_tasks, get_eval_metric
from src.finetune.trainer import get_trainer
from src.finetune.eval import evaluation

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on."},
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "dataset cache directory."},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    model_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    mask_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the save directory of head_mask.pt and neuron_mask.pt"}
    )


def main():
    # Arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
        
    # logger
    logging_path = os.path.join(training_args.output_dir, "finetune.log")
    logger = create_logger(logging_path)
    # logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Model parameters {model_args}")

    # seed
    set_seed(training_args.seed)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.model_cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    # Dataset & Model
    kwargs = {}
    train_dataset = get_dataset(data_args.task_name, tokenizer, subset="train", cache_dir=data_args.data_cache_dir)
    if data_args.task_name in get_sequence_tasks():
        eval_dataset = get_dataset(data_args.task_name, tokenizer, subset="valid", cache_dir=data_args.data_cache_dir)
        test_dataset = eval_dataset
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=get_num_labels(data_args.task_name),
            finetuning_task=data_args.task_name,
            cache_dir=model_args.model_cache_dir,
        )
        model_cls = AutoModelForSequenceClassification
    elif data_args.task_name in get_qa_tasks():
        eval_dataset, eval_examples = get_dataset(data_args.task_name, tokenizer, subset="valid", cache_dir=data_args.data_cache_dir)
        test_dataset = eval_dataset
        kwargs["eval_examples"] = eval_examples
        model_cls = AutoModelForQuestionAnswering
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.model_cache_dir
        )
    elif data_args.task_name in get_lm_tasks():
        eval_dataset = get_dataset(data_args.task_name, tokenizer, subset="valid", cache_dir=data_args.data_cache_dir)
        test_dataset = get_dataset(data_args.task_name, tokenizer, subset="test", cache_dir=data_args.data_cache_dir)
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.model_cache_dir
        )
        model_cls = AutoModelForCausalLM
    
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.model_cache_dir
    )
    if "bert" in model_args.model_name_or_path:
        model_config = extract_config("bert", model)
        model_key = extract_key("bert_classifier")
    elif "gpt2" in model_args.model_name_or_path:
        model_config = extract_config("gpt2", model)
        model_key = extract_key("gpt2_lm")
    arch = Architecture(model, model_key, model_config)
    
    # Prune Model
    head_mask = torch.load(os.path.join(model_args.mask_dir, "head_mask.pt"))
    neuron_mask = torch.load(os.path.join(model_args.mask_dir, "neuron_mask.pt"))
    arch.prune(head_mask, neuron_mask)
    pruned_model = arch.model
    
    # Initialize Trainer
    training_args.metric_for_best_model = "eval_" + get_eval_metric(data_args.task_name)
    training_args.greater_is_better = False if get_eval_metric(data_args.task_name) == "loss" else True
    training_args.load_best_model_at_end = True
    training_args.save_total_limit = 1
    training_args.save_strategy = "epoch"
    training_args.evaluation_strategy = "epoch"
    trainer = get_trainer(data_args.task_name, training_args, pruned_model, tokenizer, train_dataset, eval_dataset, **kwargs)

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    logger.info(metrics)

    # Evaluation
    test_datasets = {data_args.task_name: test_dataset}
    if data_args.task_name == "mnli":
        test_datasets["mnli-mm"] = get_dataset("mnli-mm", tokenizer, subset="test", cache_dir=data_args.data_cache_dir)
    evaluation(data_args.task_name, trainer, test_datasets)
    


if __name__ == "__main__":
    main()
