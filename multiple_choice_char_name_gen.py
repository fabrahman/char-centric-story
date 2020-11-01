import os
import re
import json
import tqdm
import torch
import logging
import argparse
import numpy as np

from overrides import overrides
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelWithLMHead

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InstanceReader(object):
    def to_uniform_fields(self, fields):
        pass

    def fields_to_instance(self, fields):
        pass

class CharSumInstanceReader(InstanceReader):
    """
    Reads the character summary dataset into a unified format with summary, masked description, label, and choices.
    """
    @overrides
    def to_uniform_fields(self, fields):
        summary = fields['summary']
        masked_description = fields["masked_description"]

        source = f"[sum] {summary} [desc] {masked_description} [name]"
        target = fields["character_name"]
        label = fields["multichoice"].get('label', None)
        choices = fields["multichoice"]['choices']
        return summary, masked_description , label, choices

    @overrides
    def fields_to_instance(self, fields):
        summary, masked_description, label, choices = self.to_uniform_fields(fields)

        source = f"[sum] {summary} [desc] {masked_description} [name]"
        # target = fields["character_name"]
        source_target_with_choices = [(source, choice) for choice in choices]
        return label, choices, source_target_with_choices


# INSTANCE_READERS = {"copa": CopaInstanceReader,
#                     "socialiqa": SocialIQAInstanceReader,
#                     "winogrande": WinograndeInstanceReader,
#                     "piqa": PiqaInstanceReader,
#                     "commonsenseqa":CommonsenseqaInstanceReader,
#                     "mctaco":MCTACOInstanceReader}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="openai-gpt", type=str, required=False, help="language model to use")
    parser.add_argument("--dataset_file", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--out_dir", default=None, type=str, required=True, help="Out directory for the predictions")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device")

    args = parser.parse_args()
    logger.info(args)

    # Load the language model
    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")
    model, tokenizer = init_model(args.model_name_or_path, device)

    # Load the dataset
    instance_reader = CharSumInstanceReader()
    set_name = os.path.basename(args.dataset_file).replace(".jsonl", "")
    out_file = os.path.join(args.out_dir, f"{args.model_name_or_path}_{set_name}_predictions.jsonl")
    gold = []
    predictions = []

    args.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Predict instances
    with open(out_file, "w") as f_out:
        with open(args.dataset_file) as f_in:
            for line in tqdm.tqdm(f_in):
                fields = json.loads(line.strip())
                label, choices, source_target_with_choices = \
                    instance_reader.fields_to_instance(fields)

                gold.append(label)

                # Tokenize and pad
                inputs = [
                    tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ex[0]))
                    for ex in source_target_with_choices
                ]
                outputs = [
                    [inputs[i][-1]]
                    + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ex[1]))
                    for i, ex in enumerate(source_target_with_choices)
                ]

                # Pad
                block_size = model.config.max_position_embeddings
                max_input_length = min(
                    block_size, max([len(ex) for ex in inputs])
                )
                max_output_length = min(
                    block_size, max([len(ex) for ex in outputs])
                )

                input_lengths = [min(len(ex), max_input_length) for ex in inputs]
                output_lengths = [min(len(ex), max_output_length) for ex in outputs]

                inputs = torch.tensor([tokenizer.encode(
                    ex, add_special_tokens=False, max_length=max_input_length, pad_to_max_length=True)
                    for ex in inputs]).long().to(device)

                outputs = torch.tensor([tokenizer.encode(
                    ex, add_special_tokens=False, max_length=max_output_length, pad_to_max_length=True)
                    for ex in outputs]).long().to(device)

                inputs_mask = torch.tensor([[1] * input_len + [0] * (max_input_length - input_len) for input_len in input_lengths]).to(device)
                outputs_mask = torch.tensor([[1] * output_len + [0] * (max_output_length - output_len) for output_len in output_lengths]).to(device)

                score_func = (
                    get_encdec_score
                    if "t5" in args.model_name_or_path or "bart" in args.model_name_or_path
                    else get_lm_score
                )
                prediction = int(np.argmin(score_func(args, model, inputs, outputs, inputs_mask, outputs_mask)))
                fields["prediction"] = prediction
                predictions.append(prediction)
                f_out.write(json.dumps(fields) + "\n")

    # Don't report accuracy if we don't have the labels
    if None not in gold:
        accuracy = accuracy_score(gold, predictions)
        print(f"Accuracy: {accuracy:.3f}")


def get_encdec_score(args, model, input_ids, output_ids, input_mask, output_mask):
    """
    Get the cross entropy loss of the texts in batch using the language model
    """
    # input_ids: [num_choices, max_length]
    with torch.no_grad():
        decoder_input_ids = output_ids[:, :-1].contiguous()
        lm_logits = model(input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids)[0]

        num_choices, max_length, vocab_size = lm_logits.shape
        loss_fct = CrossEntropyLoss(reduction="none")
        lm_labels = output_ids[:, 1:].clone().contiguous()
        lm_labels[output_ids[:, 1:] == args.pad_token_id] = -100
        loss = loss_fct(lm_logits.view(-1, vocab_size), lm_labels.view(-1)).view(num_choices, max_length)
        # Only consider non padded tokens
        loss_mask = output_mask[..., :-1].contiguous()
        loss = torch.mul(loss_mask,loss).view(num_choices, -1).mean(1).cpu().numpy()

    return loss


def get_lm_score(args, model, batch):
    """
    Get the cross entropy loss of the texts in batch using the language model
    """
    # Batch: [num_choices, max_length]
    with torch.no_grad():
        num_choices, max_length = batch.shape
        shift_labels = batch[..., 1:].contiguous().view(-1)
        lm_logits = model(batch)[0]
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(num_choices, -1).mean(1).cpu().numpy()

    return loss


def init_model(model_name: str,
               device: torch.device):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :return: the model and tokenizer
    """
    logger.info(f'Initializing {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


if __name__ == '__main__':
    main()
