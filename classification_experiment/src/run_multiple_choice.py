
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet).
    Modified from https://github.com/huggingface/transformers/blob/master/examples/multiple-choice/run_multiple_choice.py"""

import logging
import json
import os
from dataclasses import dataclass, field
import sys
import torch
from typing import Dict, Optional

import numpy as np

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)

from utils_multiple_choice import MultipleChoiceDataset, Split, combine_subq_predictions


logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_info()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_save_dir: Optional[str] = field(
        default=None, metadata={"help": "Optional separate directory for saving the model and tokenizer"}
    )
    use_longformer: bool = field(default=False)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    #task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_choices: int = field(default=10,
                             metadata={"help": "Max number of answer choices, extra choices will be ignored"})
    num_examples: int = field(default=None,
                             metadata={"help": "If set, only process this many examples from each split"})
    max_summary_len: int = field(default=None,
                             metadata={"help": "If set, keep this many words from summary"})
    max_description_len: int = field(default=None,
                             metadata={"help": "If set, keep this many words from description. Use negative for unmasked."})
    use_unmasked: bool = field(default=False, metadata={"help": "Use unmasked descriptions, for sanity check."})
    use_all_choices_eval: bool = field(default=False, metadata={"help": "On eval, use all answer choices even if more than max_choices"})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO # if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info(f"model_args = {model_args}")
    logger.info(f"data_args = {data_args}")
    logger.info(f"training_args = {training_args}")

    # Hack to see if helps with beaker
    os.makedirs(training_args.logging_dir, exist_ok=True)

    # Set seed
    set_seed(training_args.seed)
    num_labels = data_args.max_choices

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="StoryCharacters",
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            max_choices=data_args.max_choices,
            num_examples=data_args.num_examples,
            max_seq_length=data_args.max_seq_length,
            max_summary_len=data_args.max_summary_len,
            max_description_len=data_args.max_description_len,
            use_unmasked=data_args.use_unmasked,
            use_longformer=model_args.use_longformer,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            max_choices=data_args.max_choices,
            num_examples=data_args.num_examples,
            max_seq_length=data_args.max_seq_length,
            max_summary_len=data_args.max_summary_len,
            max_description_len=data_args.max_description_len,
            use_unmasked=data_args.use_unmasked,
            use_longformer=model_args.use_longformer,
            mode=Split.dev,
        )
        if training_args.do_train or (training_args.do_eval and not data_args.use_all_choices_eval)
        else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EvalPrintTrainerCallback]
    )

    # Training
    if training_args.do_train:
        model_output_dir = training_args.output_dir
        if model_args.model_save_dir is not None:
            model_output_dir = model_args.model_save_dir
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model(model_output_dir)
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(model_output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        if data_args.use_all_choices_eval:
            eval_dataset_all_choices = MultipleChoiceDataset(
                    data_dir=data_args.data_dir,
                    tokenizer=tokenizer,
                    max_choices=data_args.max_choices,
                    num_examples=data_args.num_examples,
                    max_seq_length=data_args.max_seq_length,
                    max_summary_len=data_args.max_summary_len,
                    max_description_len=data_args.max_description_len,
                    use_unmasked=data_args.use_unmasked,
                    use_longformer=model_args.use_longformer,
                    use_all_choices=True,
                    mode=Split.dev,
                )
            result_raw = trainer.predict(eval_dataset_all_choices)
            result, eval_ids = combine_subq_predictions(result_raw, eval_dataset_all_choices)
        else:
            result = trainer.predict(eval_dataset)
            eval_ids = [x.example_ids for x in eval_dataset]

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

                results.update(result.metrics)
                with open(os.path.join(training_args.output_dir, "metrics.json"), "w") as writer:
                    writer.write(json.dumps(sanitize(results)))
        output_prediction_file = os.path.join(training_args.output_dir, "eval_predictions.jsonl")
        if trainer.is_world_process_zero():
            with open(output_prediction_file, "w") as writer:
                for example_id, logits, label_id in zip(eval_ids, result.predictions, result.label_ids):
                    gold_label = label_id
                    predicted_label = np.argmax(logits)
                    is_correct = 1 if gold_label == predicted_label else 0
                    pred_json = {"id": example_id, "logits": logits, "predicted_label": predicted_label,
                                 "gold_label": gold_label, "is_correct": is_correct}
                    writer.write(json.dumps(sanitize(pred_json))+"\n")

    return results


def sanitize(x):
    if isinstance(x, (str, float, int, bool)):
        return x
    elif isinstance(x, torch.Tensor):
        return x.cpu().tolist()
    elif isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, np.number):
        return x.item()
    elif isinstance(x, dict):
        return {key: sanitize(value) for key, value in x.items()}
    elif isinstance(x, np.bool_):
        return bool(x)
    elif isinstance(x, (list, tuple)):
        return [sanitize(x_i) for x_i in x]
    return x

class EvalPrintTrainerCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if "metrics" in kwargs:
            logger.info(f"METRICS = {kwargs['metrics']}")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
