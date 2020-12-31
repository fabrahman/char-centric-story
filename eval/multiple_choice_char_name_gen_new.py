import os
import re
import json
import tqdm
import torch
import logging
import argparse
import numpy as np
import glob

from overrides import overrides
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelWithLMHead
from longformer import LongformerEncoderDecoderForConditionalGeneration, LongformerEncoderDecoderConfig
from longformer.sliding_chunks import pad_to_window_size

from pytorch_lightning.utilities.cloud_io import load as pl_load




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
        masked_description = fields["masked_description"].strip(" Read an")

        # source = f"[sum] {summary} [desc] {masked_description} [name]"
        # target = fields["character_name"]
        label = fields["multichoice"].get('label', None)
        choices = fields["multichoice"]['choices']
        return summary, masked_description , label, choices

    @overrides
    def fields_to_instance(self, fields, tokenizer, max_length, format="with-choices", char_len=None):
        summary, masked_description, label, choices = self.to_uniform_fields(fields)
        process = lambda s: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
        if format == "with-choices":
            source = f"[choices] {', '.join(choices)} [desc] {trim_str(masked_description, char_len)} [sum] {summary} [name]"
        else:
            truncated_summary = _truncate(process, summary, masked_description, max_length)
            if truncated_summary is None:
                return label, choices, None
            source = f"[sum] {truncated_summary} [desc] {masked_description} [name]"
            # target = fields["character_name"]
        source_target_with_choices = [(source, choice) for choice in choices]
        return label, choices, source_target_with_choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="openai-gpt", type=str, required=False, help="language model to use")
    parser.add_argument("--dataset_file", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--out_dir", default=None, type=str, required=True, help="Out directory for the predictions")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="attention dropout")
    parser.add_argument("--attention_mode", type=str, default='sliding_chunks', help="Longformer attention mode")
    parser.add_argument("--attention_window", type=int, default=512, help="Attention window")
    parser.add_argument("--temperature", default=1.0, type=float, required=False, help="temperature for sampling")
    parser.add_argument("--beams", default=1, type=int, required=False, help="beams for beam search")
    parser.add_argument("--max_length", default=10, type=int, required=False, help="Maximum text length")
    parser.add_argument("--format", default=None, type=str, required=True, help="what is the input format? with-choices or no-choices")
    parser.add_argument("--char_length", default=None, type=int, required=False, help="Maximum text length")

    args = parser.parse_args()
    logger.info(args)

    # Load the language model
    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")
    if "bart" in args.model_name_or_path:
        model, tokenizer = init_model(args.model_name_or_path, device)
    elif "longformer" in args.model_name_or_path:
        model, tokenizer = init_longformer_model(args, args.model_name_or_path, device)

    # Load the dataset
    instance_reader = CharSumInstanceReader()
    set_name = os.path.basename(args.dataset_file).replace(".jsonl", "")
    out_file = os.path.join(args.out_dir, f"{args.model_name_or_path}_{set_name}_predictions_v2.jsonl")
    gold = []
    predictions = []

    args.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    # print(args.pad_token_id)

    # Predict instances
    with open(out_file, "w") as f_out:
        block_size = model.config.max_encoder_position_embeddings if "longformer" in args.model_name_or_path else model.config.max_position_embeddings
        # print(block_size)
        # examples = load_data_sep(args.in_file, tokenizer, block_size)
        with open(args.dataset_file) as f_in:
            for line in tqdm.tqdm(f_in):
                fields = json.loads(line.strip())
                label, choices, source_target_with_choices = \
                    instance_reader.fields_to_instance(fields, tokenizer, block_size, args.format, args.char_length)

                # when the masked_description is longer than block_size
                if source_target_with_choices is None:
                    continue

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
                    ex[0], add_special_tokens=False, max_length=max_input_length, pad_to_max_length=True, truncation=True)
                    for ex in source_target_with_choices]).long().to(device)

                outputs = torch.tensor([ [inputs[i][-1].item()] +
                    tokenizer.encode(
                    ex[1], add_special_tokens=False, max_length=max_output_length-1, pad_to_max_length=True, truncation=True)
                    for i, ex in enumerate(source_target_with_choices)]).long().to(device)
                # import pdb; pdb.set_trace()
                inputs_mask = torch.tensor([[1] * input_len + [0] * (max_input_length - input_len) for input_len in input_lengths]).to(device)
                outputs_mask = torch.tensor([[1] * output_len + [0] * (max_output_length - output_len) for output_len in output_lengths]).to(device)

                # print("before", inputs.shape)
                if "longformer" in args.model_name_or_path:
                    inputs, inputs_mask = _prepare_input(args, model, tokenizer, inputs)

                # print("after", inputs.shape)
                assert  inputs.size(0)== outputs.size(0), f"input { inputs.size(0)} and output {outputs.size(0)} size mismatch"

                score_func = (
                    get_encdec_score
                    if "longformer" in args.model_name_or_path or "bart" in args.model_name_or_path
                    else get_lm_score
                )
                losses = score_func(args, model, inputs, outputs, inputs_mask, outputs_mask)
                prediction = int(np.argmin(losses))
                fields["prediction"] = prediction
                predictions.append(prediction)

                # ppl_scores = np.exp(losses).tolist()
                fields["losses"] = {i: v for i, v in enumerate(losses.tolist())}

                fields["is_correct"] = bool(prediction == label)

                # gen_func = (
                #     generate_conditional
                #     if "long" in args.model_name_or_path or "bart" in args.model_name_or_path
                #     else generate_regular
                # )
                # gen_input = source_target_with_choices[0][0] # all sources are the same
                # gen_input_mask = inputs_mask[0] if "longformer" in args.model_name_or_path else None
                generated_char = generate_conditional(tokenizer, model, args, inputs[0], inputs_mask[0], device)
                fields["generated_char"] = generated_char


                f_out.write(json.dumps(fields) + "\n")

    # Don't report accuracy if we don't have the labels
    if None not in gold:
        accuracy = accuracy_score(gold, predictions)
        print(f"Accuracy: {accuracy:.4f}")


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

def init_longformer_model(args, model_name: str,
               device: torch.device):
    logger.info(f'Initializing {model_name}')
    # commented out use_fast due to some TypeError
    tokenizer = AutoTokenizer.from_pretrained(model_name) #, use_fast=True
    config = LongformerEncoderDecoderConfig.from_pretrained(model_name)
    config.attention_dropout = args.attention_dropout
    # config.gradient_checkpointing = args.grad_ckpt
    config.attention_mode = args.attention_mode
    config.attention_window = [args.attention_window] * config.encoder_layers
    model = LongformerEncoderDecoderForConditionalGeneration(config)

    checkpoints = list(sorted(
        glob.glob(os.path.join(model_name, "test/*.ckpt"),
                  recursive=True)))
    ckpt = pl_load(checkpoints[0], map_location=lambda storage, loc: storage)
    new_state_dict = {k.split("model.", 1)[1]: v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(new_state_dict)
    # model.load_from_checkpoint(checkpoint_path=os.path.join(model_name,"test/_ckpt_epoch_2.ckpt"))
    model.to(device)
    model.eval()
    return model, tokenizer

def _prepare_input(args, model, tokenizer, input_ids):
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    attention_mask[input_ids == tokenizer.pad_token_id] = 0
    if isinstance(model, LongformerEncoderDecoderForConditionalGeneration):
        attention_mask[:, 0] = 2  # global attention on one token for all model params to be used, which is important for gradient checkpointing to work
        if args.attention_mode == 'sliding_chunks':
            half_padding_mod = model.config.attention_window[0]
        elif args.attention_mode == 'sliding_chunks_no_overlap':
            half_padding_mod = model.config.attention_window[0] / 2
        else:
            raise NotImplementedError
        input_ids, attention_mask = pad_to_window_size(  # ideally, should be moved inside the LongformerModel
            input_ids, attention_mask, half_padding_mod, tokenizer.pad_token_id)
    return input_ids, attention_mask


def get_encdec_score(args, model, input_ids, output_ids, input_mask, output_mask):
    """
    Get the cross entropy loss of the texts in batch using the trained enc-dec model
    """
    # input_ids: [num_choices, max_length]
    with torch.no_grad():

        decoder_input_ids = output_ids[:, :-1].contiguous()
        lm_logits = model(input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids, use_cache=False)[0] # decoder_attention_mask=output_mask,
        # import pdb; pdb.set_trace()
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

def _truncate(process, lst1, lst2, max_length):
    if len(process(f"[sum] [desc] {lst2} [name]")) > max_length:
        return None
#        print(i)
    while True:
        input_seq = "[sum] " + lst1 + " [desc] " + lst2 + " [name]"
        ids = process(input_seq)
#            print(i, len(ids) )
        if len(ids) <= max_length:
            break
        summ_ = lst1.split()
        lst1= " ".join(summ_[:-5])

    return lst1


def generate_conditional(tokenizer, model, args, input_ids, input_mask, device):
    """
    Generate a sequence with models like Bart, longformer and T5
    """
    input_ids = input_ids.unsqueeze(0).to(device)
    input_mask = input_mask.unsqueeze(0).to(device)
    # print(input_ids.shape, input_mask.shape)
    # max_input_length = model.config.max_position_embeddings
    # input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input))
    decoder_start_token_id = input_ids[0, -1].item()
    # input_ids = torch.tensor([input_ids[:max_input_length]]).to(device)
    max_length = args.max_length

    outputs = model.generate(
        input_ids,
        do_sample=args.beams == 0,
        max_length=max_length,
        # min_length=5,
        temperature=args.temperature,
        num_beams=args.beams if args.beams > 0 else 1,
        early_stopping=True,
        no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=decoder_start_token_id,
        num_return_sequences=1,  # max(1, args.beams),
        attention_mask= input_mask,
    )

    preds = [tokenizer.decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=False) for output in outputs]

    # Remove any word that has "]" or "[" in it
    preds = [re.sub(r"(\w*\])", "", pred) for pred in preds]
    preds = [re.sub(r"(\[\w*)", "", pred) for pred in preds]
    preds = [re.sub(" +", " ", pred).strip() for pred in preds]

    # print(preds)

    return preds[0]

def trim_str(txt, max_len):
    max_len = len(txt.split()) if max_len is None else max_len
    return " ".join(txt.split()[:max_len]).strip("Read an ")


if __name__ == '__main__':
    main()
