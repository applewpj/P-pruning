import logging
from datasets import load_dataset
import pdb
from itertools import chain


logger = logging.getLogger()


def get_sequence_tasks():
    SEQUENCE_TASKS = ["stsb", "mrpc", "sst2", "qnli", "qqp", "mnli"]
    return SEQUENCE_TASKS

def get_qa_tasks():
    QA_TASKS = ["squad", "squad_v2"]
    return QA_TASKS

def get_lm_tasks():
    LM_TASKS = ["penn_treebank", "wikitext-2-v1", "wikitext-103-v1"]
    return LM_TASKS

def get_avg_seq_length(task_name):
    # Dev set
    TASK_TO_SEQ_LEN = {
        "stsb": 31.47,
        "mrpc": 53.24,
        "rte": 64.59,
        "sst2": 25.16,
        "qqp": 30.55,
        "qnli": 50.97,
        "cola": 11.67,
        "mnli": 39.05,
        "squad": 170.75,
        "squad_v2": 170.43,
        "penn_treebank": 512.0,
        "wikitext-2-v1": 512.0,
        "wikitext-103-v1": 512.0,
    }
    return TASK_TO_SEQ_LEN[task_name]

def get_task_to_keys(task_name):
    TASK_TO_KEYS = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    return TASK_TO_KEYS[task_name]

def get_max_seq_length(task_name):
    TASK_TO_SEQ_LEN = {
        "stsb": 128,
        "mrpc": 128,
        "rte": 128,
        "sst2": 64,
        "qqp": 128,
        "qnli": 128,
        "cola": 64,
        "mnli": 128,
        "mnli-m": 128,
        "mnli-mm": 128,
        "squad": 384,
        "squad_v2": 384,
    }
    return TASK_TO_SEQ_LEN[task_name]

def get_num_labels(task_name):
    TASK_TO_NUM_LABELS = {
        "stsb": 1,
        "mrpc": 2,
        "rte": 2,
        "sst2": 2,
        "qqp": 2,
        "qnli": 2,
        "cola": 2,
        "mnli": 3,
        "mnli-m": 3,
        "mnli-mm": 3,
    }
    return TASK_TO_NUM_LABELS[task_name]

def get_eval_metric(task_name):
    TASK_TO_DEV_METRIC = {
        "stsb": "spearmanr",
        "mrpc": "accuracy",
        "rte": "accuracy",
        "sst2": "accuracy",
        "qqp": "accuracy",
        "qnli": "accuracy",
        "cola": "matthews_correlation",
        "mnli": "accuracy",
        "mnli-m": "accuracy",
        "mnli-mm": "accuracy",
        "squad": "f1",
        "squad_v2": "f1",
        "penn_treebank": "loss",
        "wikitxt-2-v1": "loss",
        "wikitext-103-v1": "loss",
    }
    return TASK_TO_DEV_METRIC[task_name]

def get_glue(task_name, tokenizer, subset="train", pad_to_max=True, max_seq_len=100, cache_dir=None):
    raw_datasets = load_dataset("glue", task_name, cache_dir=cache_dir)
    
    # Preprocessing the raw_datasets
    sentence1_key, sentence2_key = get_task_to_keys(task_name)
    
    # Padding strategy
    padding = "max_length" if pad_to_max else False

    max_seq_length = min(max_seq_len, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        result["label"] = examples["label"]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        # load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
        remove_columns=raw_datasets["train"].column_names,
    )
    if subset == "train":
        train_dataset = raw_datasets["train"]
        return train_dataset
    else:
        if task_name in ["mnli", "mnli-m"]:
            eval_dataset = raw_datasets["validation_matched"]
        elif task_name == "mnli-mm":
            eval_dataset = raw_datasets["validation_mismatched"]
        else:
            eval_dataset = raw_datasets["validation"]
        return eval_dataset


def get_squad(task_name, tokenizer, subset="train", pad_to_max=True, max_seq_len=384, cache_dir=None):
    raw_datasets = load_dataset(
        task_name,
        cache_dir=cache_dir,
    )
    column_names = raw_datasets["train"].column_names
    column_names = raw_datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    max_seq_length = min(max_seq_len, tokenizer.model_max_length)

    # Training preprocessing
    def prepare_train_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            # truncation=True,
            max_length=max_seq_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if subset == "train":
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=64,
            remove_columns=column_names,
            desc="Running tokenizer on train dataset",
        )
        return train_dataset

    else:
        # Validation preprocessing
        def prepare_validation_features(examples):
            examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                # truncation=True,
                max_length=max_seq_length,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if pad_to_max else False,
            )

            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            tokenized_examples["example_id"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if pad_on_right else 0

                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])

                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]
            return tokenized_examples

        eval_examples = raw_datasets["validation"]
        
        eval_dataset = eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=64,
            remove_columns=column_names,
            desc="Running tokenizer on validation dataset",
        )
        return eval_dataset, eval_examples

def get_lm(task_name, tokenizer, subset="train", max_seq_len=100, cache_dir=None):
    if task_name == "penn_treebank":
        raw_datasets = load_dataset("ptb_text_only", task_name, cache_dir=cache_dir)
    elif "wikitext" in task_name:
        raw_datasets = load_dataset("wikitext", task_name, cache_dir=cache_dir)

    column_names = list(raw_datasets["train"].features)

    text_column_name = "text" if "text" in column_names else column_names[0]


    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        return output

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=64,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    block_size = max_seq_len

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result



    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=64,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    if subset == "train":
        train_dataset = lm_datasets["train"]
        return train_dataset
    elif subset == "valid":
        eval_dataset = lm_datasets["validation"]
        return eval_dataset
    elif subset == "test": 
        test_dataset = lm_datasets["test"]
        return test_dataset

        

def get_dataset(task_name, tokenizer, subset="train", pad_to_max=True, cache_dir=None):
    if task_name in get_sequence_tasks():
        dataset = get_glue(task_name, tokenizer, subset=subset, pad_to_max=pad_to_max, max_seq_len=get_max_seq_length(task_name), cache_dir=cache_dir)
        return dataset
    elif task_name in get_qa_tasks():
        if subset == "train":
            train_dataset = get_squad(task_name, tokenizer, subset=subset, pad_to_max=pad_to_max, max_seq_len=get_max_seq_length(task_name), cache_dir=cache_dir)
            return train_dataset
        else:
            eval_dataset, eval_examples = get_squad(task_name, tokenizer, subset=subset, pad_to_max=pad_to_max, max_seq_len=get_max_seq_length(task_name), cache_dir=cache_dir)
            return eval_dataset, eval_examples
    elif task_name in get_lm_tasks():
        dataset = get_lm(task_name, tokenizer, subset=subset, max_seq_len=512, cache_dir=cache_dir)
        return dataset