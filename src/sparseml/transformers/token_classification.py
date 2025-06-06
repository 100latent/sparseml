#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.\

# Adapted from https://github.com/huggingface/transformers
# neuralmagic: no copyright

"""
Fine-tuning the library models for token classification
"""
# You can also adapt this script on your own token classification task and datasets.
# Pointers for this are left as comments

import logging
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import transformers
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
    load_metric,
)
from torch.nn import Module
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from sparseml.pytorch.utils.distributed import record
from sparseml.transformers.sparsification import (
    SparseAutoModel,
    Trainer,
    TrainingArguments,
)
from sparseml.transformers.sparsification.sparse_model import get_shared_tokenizer_src


require_version(
    "datasets>=1.18.0",
    "To fix: pip install -r examples/pytorch/token-classification/requirements.txt",
)

_LOGGER: logging.Logger = logging.getLogger(__name__)

metadata_args = [
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "fp16",
]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to
    fine-tune from
    """

    model_name_or_path: str = field(
        metadata={
            "help": (
                "Path to pretrained model, sparsezoo stub. or model identifier from "
                "huggingface.co/models"
            )
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained data from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use "
            "(can be a branch name, tag name or commit id)"
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use token generated when running `transformers-cli login` "
            "(necessary to use this script with private models)"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for
    training and eval
    """

    task_name: Optional[str] = field(
        default="ner", metadata={"help": "The name of the task (ner, pos...)."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)"},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": ("The configuration name of the dataset to use"),
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )
    text_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The column name of text to input in the file "
            "(a csv or JSON file)."
        },
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The column name of label to input in the file "
            "(a csv or JSON file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. "
            "If set, sequences longer than this will be truncated, sequences shorter "
            "will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. If False, "
            "will pad the samples dynamically when batching to the maximum length "
            "in the batch (which can be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number "
            "of training examples to this value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number "
            "of evaluation examples to this value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of "
                "prediction examples to this value if set."
            ),
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated "
            "by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether to return all the entity levels during evaluation or "
            "just the overall ones."
        },
    )
    one_shot: bool = field(
        default=False,
        metadata={"help": "Whether to apply recipe in a one shot manner."},
    )
    num_export_samples: int = field(
        default=0,
        metadata={"help": "Number of samples (inputs/outputs) to export during eval."},
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


@record
def main(**kwargs):
    # See all possible arguments in
    # src/sparseml/transformers/sparsification/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif not kwargs:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, training_args = parser.parse_dict(kwargs)
    # Setup logging
    log_level = training_args.get_process_log_level()
    _LOGGER.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    _LOGGER.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )
    _LOGGER.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is "
                "not empty. Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            _LOGGER.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. "
                "To avoid this behavior, change the `--output_dir` or add "
                "`--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = _get_raw_dataset(data_args, model_args.cache_dir)

    column_names, features = _get_column_names_and_features(
        raw_datasets, do_train=training_args.do_train
    )

    text_column_name = _get_text_column_names(column_names, data_args)
    label_column_name = _get_label_column_name(column_names, data_args)

    # If the labels are of type ClassLabel, they are already integers and we have the
    # map stored somewhere. Otherwise, we have to get the list of labels manually.
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = _get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model, teacher = SparseAutoModel.token_classification_from_pretrained_distil(
        model_name_or_path=(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        model_kwargs={
            "config": config,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        },
        teacher_name_or_path=training_args.distill_teacher,
        teacher_kwargs={
            "cache_dir": model_args.cache_dir,
            "use_auth_token": True if model_args.use_auth_token else None,
        },
    )

    if teacher and not isinstance(teacher, str):
        # check whether teacher and student have the corresponding outputs
        label_to_id, label_list = _check_teacher_student_outputs(
            teacher, label_to_id, label_list
        )

    tokenizer_src = (
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else get_shared_tokenizer_src(model, teacher)
    )
    add_prefix_space = config.model_type in {"gpt2", "roberta"}
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_src,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=add_prefix_space,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. "
            "Checkout the big table of models at "
            "https://huggingface.co/transformers/index.html#supported-frameworks "
            "to find the model types that meet this requirement"
        )

    tokenized_dataset = _get_tokenized_dataset(
        data_args=data_args,
        label_list=label_list,
        labels_are_int=labels_are_int,
        model=model,
        num_labels=num_labels,
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        text_column_name=text_column_name,
        label_column_name=label_column_name,
        label_to_id=label_to_id,
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
        do_predict=training_args.do_predict,
        main_process_func=training_args.main_process_first,
    )

    make_eval_dataset = training_args.do_eval or (data_args.num_export_samples > 0)
    train_dataset = tokenized_dataset.get("train")
    eval_dataset = tokenized_dataset.get("validation")
    predict_dataset = tokenized_dataset.get("test")

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Metrics
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[lab] for (_, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        model_state_path=model_args.model_name_or_path,
        recipe=training_args.recipe,
        metadata_args=metadata_args,
        recipe_args=training_args.recipe_args,
        teacher=teacher,
        args=training_args,
        data_args=data_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if make_eval_dataset else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if not trainer.one_shot:
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()
        trainer.save_optimizer_and_scheduler(training_args.output_dir)

    # Evaluation
    if training_args.do_eval and not trainer.one_shot:
        _LOGGER.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict and not trainer.one_shot:
        _LOGGER.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        )
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(
            training_args.output_dir, "predictions.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "token-classification",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    # Exporting Samples

    if data_args.num_export_samples > 0:
        trainer.save_sample_inputs_outputs(
            num_samples_to_export=data_args.num_export_samples
        )


def _check_teacher_student_outputs(
    teacher: Module, label_to_id: Dict[str, int], label_list: List[str]
) -> Tuple[Dict[str, int], List[str]]:
    # Check that the teacher and student have the same labels and if they do,
    # check that the mapping between labels and ids is the same.

    teacher_labels = list(teacher.config.label2id.keys())
    teacher_ids = list(teacher.config.label2id.values())

    student_labels = list(label_to_id.keys())
    student_ids = list(label_to_id.values())

    if set(teacher_labels) != set(student_labels):
        _LOGGER.warning(
            f"Teacher labels {teacher_labels} do not match "
            f"student labels {student_labels}. Ignore this warning "
            "if this is expected behavior."
        )
    else:
        if student_ids != teacher_ids:
            _LOGGER.warning(
                "Teacher and student labels match, but the mapping "
                "between teachers labels and ids does not match the "
                "mapping between student labels and ids. "
                "The student's mapping will be overwritten "
                "by the teacher's mapping."
            )
            label_to_id = teacher.config.label2id
            label_list = teacher_labels

    return label_to_id, label_list


def _get_label_list(labels):
    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go
    # through the dataset to get the unique labels.
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def _get_label_column_name(column_names, data_args):
    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]
    return label_column_name


def _get_text_column_names(column_names, data_args):
    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]
    return text_column_name


def _get_column_names_and_features(
    raw_datasets: Union[Dataset, DatasetDict, IterableDatasetDict, IterableDataset],
    do_train: bool,
):
    if do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features
    return column_names, features


def get_tokenized_token_classification_dataset(
    data_args: DataTrainingArguments,
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: Module,
    cache_dir: Optional[str] = None,
    do_train: bool = False,
    do_eval: bool = True,
    do_predict: bool = False,
    main_process_func=None,
):
    """
    Utility method to get tokenized token classification dataset given at-least
    the tokenizer, model, and data_arguments

    :param data_args: Arguments pertaining to what data we are going to input
        our model for training and eval
    :param tokenizer: The tokenizer to use for tokenizing raw dataset
    :param model: The instantiated torch module to create the dataset for
    :param cache_dir: Local path to store the pretrained data from huggingface.co
    :param do_train: True to create the train dataset
    :param do_predict: True to create test dataset
    :param do_eval: True to create eval dataset
    :param main_process_func: Callable Context Manager to do something on
        the main process
    """
    raw_datasets = _get_raw_dataset(data_args=data_args, cache_dir=cache_dir)
    column_names, features = _get_column_names_and_features(
        raw_datasets, do_train=do_train
    )
    text_column_name = _get_text_column_names(column_names, data_args)
    label_column_name = _get_label_column_name(column_names, data_args)

    # If the labels are of type ClassLabel, they are already integers and we have the
    # map stored somewhere. Otherwise, we have to get the list of labels manually.
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = _get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)
    main_process_func = (
        main_process_func if main_process_func else lambda desc: nullcontext(desc)
    )

    tokenized_dataset = _get_tokenized_dataset(
        data_args=data_args,
        label_list=label_list,
        labels_are_int=labels_are_int,
        model=model,
        num_labels=num_labels,
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        text_column_name=text_column_name,
        label_column_name=label_column_name,
        label_to_id=label_to_id,
        do_train=do_train,
        do_eval=do_eval,
        main_process_func=main_process_func,
        do_predict=do_predict,
    )
    return tokenized_dataset


def _get_tokenized_dataset(
    data_args: DataTrainingArguments,
    label_list: List,
    labels_are_int: bool,
    model: Module,  # model config object can be passed in as well
    num_labels: int,
    raw_datasets: Union[Dataset, DatasetDict, IterableDatasetDict, IterableDataset],
    tokenizer: transformers.PreTrainedTokenizerBase,
    text_column_name: str,
    label_column_name: str,
    label_to_id: Dict[Any, Any],
    do_train: bool,
    do_eval: bool,
    main_process_func,
    do_predict: bool = False,
) -> Dict[str, Any]:
    train_dataset = predict_dataset = eval_dataset = None
    config = model.config if isinstance(model, Module) else model
    # Model has labels -> use them.
    if config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if list(sorted(config.label2id.keys())) == list(sorted(label_list)):
            # Reorganize `label_list` to match the ordering of the model.
            if labels_are_int:
                label_to_id = {
                    i: int(config.label2id[l]) for i, l in enumerate(label_list)
                }
                label_list = [config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            _LOGGER.warning(
                "Your model seems to have been trained with labels, but they don't "
                "match the dataset: ",
                f"model labels: {list(sorted(config.label2id.keys()))}, dataset "
                f"labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    # Set the correspondences label/ID inside the model config
    config.label2id = {l: i for i, l in enumerate(label_list)}
    config.id2label = {i: l for i, l in enumerate(label_list)}
    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if isinstance(label, str) and (
            label.startswith("B-") and label.replace("B-", "I-") in label_list
        ):
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)
    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words
            # (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100
                # so they are automatically ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the
                # current label or -100, depending on the label_all_tokens flag.
                else:
                    if data_args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    if do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with main_process_func(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
    make_eval_dataset = do_eval or data_args.num_export_samples > 0
    if make_eval_dataset:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with main_process_func(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
    if do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(
                range(data_args.max_predict_samples)
            )
        with main_process_func(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
    tokenized_dataset = {
        "train": train_dataset,
        "validation": eval_dataset,
        "test": predict_dataset,
    }
    return tokenized_dataset


def _get_raw_dataset(
    data_args, cache_dir: Optional[str]
) -> Union[Dataset, DatasetDict, IterableDatasetDict, IterableDataset]:
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and
    # evaluation files (see below) or just provide the name of one of the public
    # datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first
    # column if no column called 'text' is found. You can easily tweak this behavior
    # (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one
    # local process can concurrently download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=cache_dir,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension, data_files=data_files, cache_dir=cache_dir
        )
    # See more about loading any type of standard or custom dataset (from files, python
    # dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    return raw_datasets


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
