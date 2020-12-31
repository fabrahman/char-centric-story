"""
Adapted from https://github.com/huggingface/transformers/blob/master/examples/run_generation.py
"""
import re
import json
import tqdm
import torch
import logging
import argparse

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

from common import init_model, load_data, load_data_sep


def main() -> None:
    """
    Generate intensifiers and attenuators
    """
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument(
        "--in_file",
        default=None,
        type=str,
        required=True,
        help="The input json file",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        required=True,
        help="out jsonl file with generations",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2",
        type=str,
        help="LM checkpoint for initialization.",
    )
    parser.add_argument(
        "--max_input_length", default=945, type=int, required=False, help="Maximum input length."
    )

    # Optional
    parser.add_argument(
        "--max_length", default=75, type=int, required=False, help="Maximum text length"
    )
    parser.add_argument(
        "--k", default=0, type=int, required=False, help="k for top k sampling"
    )
    parser.add_argument(
        "--p", default=0, type=float, required=False, help="p for nucleus sampling"
    )
    parser.add_argument(
        "--beams", default=0, type=int, required=False, help="beams for beam search"
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        required=False,
        help="temperature for sampling",
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--char_name_last",
        action="store_true",
        help="character name comes after the orig summary ([sum] summary [char] char_name)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="generative",
        help="Whether generative task (description) or discriminative task (name)"
    )
    parser.add_argument(
        "--truncation_method",
        type=str,
        default="length",
        help="Whether to truncate by length from the end or to use coref truncated summary?"
    )
    args = parser.parse_args()
    logger.debug(args)

    if (
        (args.k == args.p == args.beams == 0)
        or (args.k != 0 and args.p != 0)
        or (args.beams != 0 and args.p != 0)
        or (args.beams != 0 and args.k != 0)
    ):
        raise ValueError(
            "Exactly one of p, k, and beams should be set to a non-zero value."
        )

    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    logger.debug(f"Initializing {args.device}")

    tokenizer, model = init_model(args.model_name_or_path, device)
    max_input_length = model.config.max_position_embeddings if "bart" in args.model_name_or_path else args.max_input_length
    args.max_length = args.max_length if "bart" in args.model_name_or_path else 75

    # eos will be added after truncation for language models
    add_eos = False if "gpt" in args.model_name_or_path else True
    examples = load_data(args.in_file, add_eos=add_eos, truncation_method=args.truncation_method) if not args.char_name_last else load_data_sep(args.in_file, tokenizer, max_input_length, args.task)

    special_tokens = ["[name]", "[sum]", "[desc]", "<eos>", "[MASK]"]

    generate = (
        generate_conditional
        if "t5" in args.model_name_or_path or "bart" in args.model_name_or_path
        else generate_regular
    )

    with open(args.out_file, "w") as f_out:
        for input, output in tqdm.tqdm(examples):
            try:
                preds, trimmed_input = generate(
                    tokenizer,
                    model,
                    args,
                    input,
                    device,
                )

                # For some reason some special tokens are still predicted
                for special_token in special_tokens:
                    preds = [pred.replace(special_token, "") for pred in preds]

                # Remove any word that has "]" or "[" in it
                preds = [re.sub(r"(\w*\])", "", pred) for pred in preds]
                preds = [re.sub(r"(\[\w*)", "", pred) for pred in preds]
                preds = [re.sub(" +", " ", pred).strip() for pred in preds]

            except Exception as exp:
                logger.info(exp)
                preds = []

#            trimmed_input = trim_input_to_max_len(tokenizer, input)
            f_out.write(
                json.dumps({"input": trimmed_input, "gold": output, "predictions": preds})
                + "\n"
            )


def generate_conditional(tokenizer, model, args, input, device):
    """
    Generate a sequence with models like Bart and T5
    """
#    max_input_length = 512 #tokenizer.max_len_single_sentence
    max_input_length = model.config.max_position_embeddings
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input))
    decoder_start_token_id = input_ids[-1]
    input_ids = torch.tensor([input_ids[:max_input_length]]).to(device)
    max_length = args.max_length

    outputs = model.generate(
        input_ids,
        do_sample=args.beams == 0,
        max_length=max_length,
        min_length=5,
        temperature=args.temperature,
        top_p=args.p if args.p > 0 else None,
        top_k=args.k if args.k > 0 else None,
        num_beams=args.beams if args.beams > 0 else None,
        early_stopping=True,
        no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=decoder_start_token_id,
        num_return_sequences=1 #max(1, args.beams)
    )


    preds = [tokenizer.decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=False) for output in outputs]
#    trimmed_input = tokenizer.decode(
#        input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

#    print(preds)

    # truncate input to max_input_length
    trimmed_input = trim_input_to_max_len(tokenizer, input, max_input_length)
    
    return preds, trimmed_input


def generate_regular(tokenizer, model, args, input, device):
    """
    Generate a sequence with models like GPT, GPT2, or XLNet
    """
    # truncate input to max_input_length
    max_input_length = args.max_input_length - 1 # -1 reserved for [desc] special token
    context_tokens = tokenizer.encode(input, max_length=max_input_length, pad_to_max_length=False)
    context_tokens += [tokenizer.convert_tokens_to_ids("[desc]")]
    trimmed_input_text = tokenizer.decode(context_tokens, skip_special_tokens=True)

    max_length = args.max_length + len(context_tokens)
#    print(max_length, len(context_tokens))
    input_ids = torch.tensor(context_tokens, device=device).unsqueeze(0)
#    print(input_ids.shape)


    outputs = model.generate(
        input_ids=input_ids,
        do_sample=args.beams == 0,
        max_length=max_length,
        temperature=args.temperature,
        top_p=args.p if args.p > 0 else None,
        top_k=args.k if args.k > 0 else None,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=args.beams if args.beams > 0 else None,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=1 #max(1, args.beams)
    )

#    print(max_length, input_ids.shape)
#    print(outputs, outputs[0].shape)

    preds = [tokenizer.decode(output, skip_special_tokens=True)[len(trimmed_input_text):].strip() for output in outputs]
#    print(preds)
#    preds = [". ".join(pred.split(".",-1)[:-1]) for pred in preds]
#    preds = [pred.split(".")[0] for pred in preds]
    

    # truncate input to max_input_length
#    trimmed_input = trim_input_to_max_len(tokenizer, input, max_input_length)

    return preds, trimmed_input_text

def trim_input_to_max_len(tokenizer, input, max_length):
    input_tokens = tokenizer.encode(input, max_length=max_length, pad_to_max_length=False)
    input_trimmed = tokenizer.decode(input_tokens, skip_special_tokens=True)

    return input_trimmed.strip()


if __name__ == "__main__":
    main()
