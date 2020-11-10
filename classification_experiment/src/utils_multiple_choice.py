# coding=utf-8
# Code based on https://github.com/huggingface/transformers/blob/master/examples/multiple-choice/utils_multiple_choice.py

import csv
import glob
import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import numpy as np

import tqdm

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
from transformers.trainer_utils import PredictionOutput


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for
    """

    example_id: str
    summary: str
    masked_description: str
    choices: List[str]
    num_choices_orig: int
    label: int
    label_orig: Optional[int]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]

@dataclass(frozen=True)
class InputFeaturesLongFormer:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    global_attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class MultipleChoiceDataset(Dataset):
        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            max_choices: int,
            max_seq_length: Optional[int] = None,
            max_description_len: Optional[int] = None,
            max_summary_len: Optional[int] = None,
            num_examples: int = None,
            use_unmasked: bool = False,
            use_all_choices: bool = False,
            use_longformer: bool = False,
            mode: Split = Split.train,
        ):
            processor = StoryCharactersProcessor(max_choices, num_examples, max_description_len,
                                                 max_summary_len, use_unmasked=use_unmasked,
                                                 use_all_choices=use_all_choices)
            logger.info(f"Creating features from dataset file at {data_dir}")
            label_list = processor.get_labels()
            if mode == Split.dev:
                examples = processor.get_dev_examples(data_dir)
            elif mode == Split.test:
                examples = processor.get_test_examples(data_dir)
            else:
                examples = processor.get_train_examples(data_dir)
            logger.info("Training examples: %s", len(examples))
            self.examples = examples
            self.features = convert_examples_to_features(
                examples,
                label_list,
                max_seq_length,
                tokenizer,
                add_global_attention_mask=use_longformer
            )

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class StoryCharactersProcessor(DataProcessor):
    """Processor for the char-centric-story data set."""

    def __init__(self, max_choices, num_examples=None, max_description_len=None,
                 max_summary_len=None, use_unmasked=False, use_all_choices=False):
        self.max_choices = max_choices
        self.num_examples = num_examples
        self.max_description_len = max_description_len
        self.max_summary_len = max_summary_len
        self.use_unmasked = use_unmasked
        self.use_all_choices = use_all_choices

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(self.max_choices)]

    def _read_jsonl(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = [json.loads(line.strip()) for line in fin]
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        id_counter = 0
        num_truncated = 0
        examples = []
        for fields in tqdm.tqdm(lines, desc="read data"):
            id_counter += 1
            if self.num_examples is not None and id_counter > self.num_examples:
                break
            choices = fields["multichoice"]['choices']
            num_choices_orig = len(choices)
            label = fields["multichoice"].get('label', None)
            if len(choices) < self.max_choices:
                choices = choices + [""] * (self.max_choices - len(choices))
            elif len(choices) > self.max_choices and not self.use_all_choices:
                # If number of choices is larger than allowed, we "cheat" by truncating the answer choices
                # such that the correct answer is among them. At test time we need to run with a higher
                # max_choices to correct for this!
                num_truncated += 1
                if label is None or label < self.max_choices:
                    choices = choices[:self.max_choices]
                else:
                    # This puts the correct choice last, but the current model is blind to this severe bias
                    start = label - self.max_choices + 1
                    choices = choices[start:start+self.max_choices]
                    label = self.max_choices - 1
            if self.use_unmasked:
                masked_description = fields["description"]
            else:
                masked_description = fields["masked_description"]
            summary = truncate_words(fields['summary'], self.max_summary_len)
            masked_description = truncate_words(masked_description, self.max_description_len)
            if not self.use_all_choices:
                example = InputExample(
                    example_id = f"char-story-{type}-{id_counter}",
                    summary = summary,
                    masked_description = masked_description,
                    choices = choices,
                    num_choices_orig = num_choices_orig,
                    label = label,
                    label_orig = label
                )
                examples.append(example)
            else:
                # If number of choices is larger than max choices, we'll make several max_choice MC questions
                # At eval time we'll combine the logits from each sub question to figure out final prediction
                assert(self.use_all_choices)
                for subq in range(0, (len(choices) - 1) // self.max_choices + 1):
                    sub_choices = choices[subq * self.max_choices: (subq + 1) * self.max_choices]
                    if len(sub_choices) < self.max_choices:
                        sub_choices = sub_choices + [""] * (self.max_choices - len(sub_choices))
                    if subq * self.max_choices <= label < (subq + 1) * self.max_choices:
                        sub_label = label - subq * self.max_choices
                    else:
                        # Aribtrarily pretend the first answer choice is "correct"
                        # We'll ignore this when evaluating later, don't use for training!
                        sub_label = 1
                    example = InputExample(
                        example_id=f"char-story-{type}-{id_counter}-sub{subq}",
                        summary=summary,
                        masked_description=masked_description,
                        choices=sub_choices,
                        num_choices_orig=num_choices_orig,
                        label=sub_label,
                        label_orig = label
                    )
                    examples.append(example)


        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info(f"len examples: {len(examples)}")
        logger.info(f"num truncated choices: {num_truncated}")

        for f in examples[:2]:
            logger.info("*** Example features ***")
            logger.info("example: %s" % str(f)[:50000])

        return examples


# Super-rough word limiter
def truncate_words(text, max_words):
    if max_words is None:
        return text
    words = text.split(" ")[:max_words]
    return " ".join(words)

def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    add_global_attention_mask=None
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    truncated_tokens = 0
    max_truncated_tokens = 0
    total_choices = 0
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 1000 == 0:
            logger.info("Processing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for choice_idx, choice in enumerate(example.choices):
            total_choices += 1
            text_a = choice
            text_b = f"Description: {example.masked_description}  ; Summary: {example.summary}"
            inputs = tokenizer(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True,
            )
            if total_choices <= 2:
                logger.info(f"Sample text_a = {text_a}  ;  text_b = {text_b}")
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                if inputs["num_truncated_tokens"] > max_truncated_tokens:
                    max_truncated_tokens = inputs["num_truncated_tokens"]
                truncated_tokens += inputs["num_truncated_tokens"]
            choices_inputs.append(inputs)

        label = example.label

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        token_type_ids = [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None

        if add_global_attention_mask:
            global_attention_mask = [[1] + [0] * (len(x["input_ids"])-1) for x in choices_inputs]
            feature = InputFeaturesLongFormer(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        else:
            feature = InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )

        features.append(feature)
    if truncated_tokens > 0:
        logger.info(f"Attention! Tokens were truncated (max = {max_truncated_tokens}, average = {truncated_tokens / total_choices} per input).")

    for f in features[:2]:
        logger.info("*** Example features ***")
        logger.info("feature: %s" % str(f)[:50000])

    return features

def combine_subq_predictions(predictions, dataset):
    new_predictions = []
    new_label_ids = []
    all_example_ids = []
    current_id = None
    logits = None
    num_choices = None
    correct_count = 0
    for idx, (prediction, example) in enumerate(zip(predictions.predictions, dataset.examples)):
        example_id = re.sub("-sub\\d+$", "", example.example_id)
        if example_id == current_id:
            logits = np.concatenate([logits, prediction])
        if example_id != current_id or idx == len(dataset.examples) - 1:
            if current_id is not None:
                if len(logits) < num_choices:
                    logger.warning(f"Not enough logits for {current_id}, {len(logits)} < {num_choices}")
                # TODO: compute loss if need be
                logits = logits[:num_choices]
                predicted = np.argmax(logits)
                if predicted == label:
                    correct_count += 1
                new_predictions.append(logits)
                new_label_ids.append(label)
                all_example_ids.append(current_id)
            current_id = example_id
            logits = np.array(prediction)
            label = example.label_orig
            num_choices = example.num_choices_orig
    accuracy = correct_count / len(new_predictions)
    predictions= PredictionOutput(
        predictions=np.array(new_predictions, dtype=object),
        label_ids=np.array(new_label_ids),
        metrics={"acc": accuracy}
    )
    return predictions, all_example_ids

