import pandas as pd
import json

from transformers import AutoModelWithLMHead, AutoTokenizer


def init_model(model_name: str, device, do_lower_case: bool = False):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :param do_lower_case: whether the model is lower cased or not
    :return: the model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model

def init_model_from_config(config, device):
    model = AutoModelWithLMHead.from_config(config)
    model.to(device)
    model.eval()
    return model

def _truncate(process, lst1, lst2, target, max_length):
    # lst1_new = []
    # lst2_new = []
    # target_new = []
    new_data = []
    for i in range(len(lst1)):
        if len(process(f"[sum] [name] {lst2[i]} [desc]")) > max_length:
            continue
#        print(i)
        while True:
            input_seq = "[sum] " + lst1[i] + " [name] " + lst2[i] + " [desc]"
            ids = process(input_seq)
#            print(i, len(ids) )
            if len(ids) <= max_length:
                new_data.append((lst1[i], lst2[i], target[i]))
                # lst1_new.append(lst1[i])
                # lst2_new.append(lst2[i])
                # target_new.append(target[i])
                break
            summ_ = lst1[i].split()
            lst1[i] = " ".join(summ_[:-5])
        if i % 500 == 0:
            print(f"{i} processed from {len(lst1)}")
    return new_data #lst1_new, lst2_new, target_new

def load_data(in_file, add_eos=True, truncation_method='length'):
    """
    Loads the dataset file:
    in_file: json file
    Returns a list of tuples (input, output)
    """
    all_lines = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            all_lines.append(json.loads(line))

    # Update: removed <eos> and [desc] from end of output and input for gpt. Will be added after truncation.
    start_of_desc = " [desc]"
    end_of_seq = " <eos>"
    no_end = ""
    examples = [
        (
            f"[name] {line['character_name']} [sum] {line['coref_truncated_summary'] if truncation_method == 'coref' else line['summary']}{start_of_desc if add_eos else no_end}",
            f"{line['description']}{end_of_seq if add_eos else no_end}",
        )
        for line in all_lines
    ]

    return examples

def load_data_sep(in_file, tokenizer, max_length, task='generative'):
    """
    Loads the dataset file and truncate summaries:
    in_file: json file
    Returns a list of tuples (input1, input2, output)
    """
    all_lines = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            all_lines.append(json.loads(line))
    process = lambda s: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
    if task =='generative':
        truncated_data= _truncate(
            process, [line['summary'] for line in all_lines],
            [line['character_name'] for line in all_lines],
            [line['description'] for line in all_lines],
            max_length
        )
    else:
        truncated_data = _truncate(
            process, [line['summary'] for line in all_lines],
            [line['masked_description'] for line in all_lines],
            [line['character_name'] for line in all_lines],
            max_length
        )
    print(len(all_lines), len(truncated_data))
    # # since we skipped some of the long mask_description this assertion is not valid any more
    # assert len(truncated_summaries) == len(all_lines), "truncated summaries should have same size as orig data!"

    # examples = [
    #     (
    #         f"[sum] {summ} [name] {line['character_name']} [desc]" if task=='generative' else f"[sum] {summ} [desc] {line['masked_description']} [name]",
    #         f"{line['description']} <eos>" if task=='generative' else f"{line['character_name']} <eos>", # <eos> here may not work for gpt in tesk=="generative". <eos> may be truncated in Dataset class
    #     )
    #     for summ, line in zip(truncated_summaries, all_lines)
    # ]
    examples = [
        (
            f"[sum] {summ} [name] {char} [desc]" if task=='generative' else f"[sum] {summ} [desc] {char} [name]",
            f"{tgt} <eos>", # <eos> here may not work for gpt in tesk=="generative". <eos> may be truncated in Dataset class
        )
        for summ, char, tgt in truncated_data
    ]

    return examples

def load_data_flexi(in_file, task="generative", char_len= None, add_eos=True, truncation_method='length'):
    """
    Loads the dataset file:
    in_file: json file
    Returns a list of tuples (input, output)
    """
    all_lines = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            all_lines.append(json.loads(line))

    # Update: removed <eos> and [desc] from end of output and input for gpt. Will be added after truncation.
    end_of_src = " [desc]" if task == "generative" else " [name]"
    end_of_trg = " <eos>"
    no_end = ""

    if task == "generative":
        examples = [
            (
                f"[name] {line['character_name']} [sum] {line['coref_truncated_summary'] if truncation_method == 'coref' else line['summary']}{end_of_src if add_eos else no_end}",
                f"{line['description']}{end_of_trg if add_eos else no_end}",
            )
            for line in all_lines
        ]
    else:

        examples = [
            (
                f"[choices] {', '.join(line['multichoice']['choices'])} [desc] {trim_str(line['masked_description'], char_len)} "
                f"[sum] {line['coref_truncated_summary'] if truncation_method == 'coref' else line['summary']}{end_of_src if add_eos else no_end}",
                f"{line['character_name']}{end_of_trg if add_eos else no_end}",
            )
            for line in all_lines
        ]



    return examples

def trim_str(txt, max_len):
    max_len = len(txt.split()) if max_len is None else max_len
    return " ".join(txt.split()[:max_len]).strip("Read an ")
