# use fresh environement

import json
import argparse
import numpy as np

from nltk import bleu
from rouge import Rouge
from collections import defaultdict
from nltk.translate.bleu_score import SmoothingFunction

smoothing = SmoothingFunction().method1
weights = [0.25] * 4
rouge = Rouge()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True, help="The directory of the outputs")
    args = parser.parse_args()

    print("\t".join(["Model", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"]))

    for setup in ["bart-large-xsum_long", "longformer", "gpt2-large", "bart-large-xsum_long_truncated", "longformer_full", "longformer_truncated", "gpt2-large_truncated"]: #["bart-large-xsum_long", "gpt2-large", "longformer",]:
#        for lm in ["bart-large", "gpt2-xl"]:

        # Compute BLEU and ROUGE from the text predictions
        data = [json.loads(line.strip()) for line in open(f"{args.out_dir}/{setup}/generation/test_prediction_beams5_maxlen_{75 if 'gpt2' in setup else 1024}.jsonl")]
        gold = defaultdict(list)
        predictions = defaultdict(set)

        for ex in data:
            curr_gold = ex["gold"].lower().replace("<eos>", "").strip()
            ex["predictions"] = [ex["predictions"]] if isinstance(ex["predictions"], str) else ex["predictions"]
            curr_preds = [pred.lower().replace("<eos>", "").strip() for pred in ex["predictions"]]
            curr_preds = set([pred for pred in curr_preds if len(pred) > 1])

            if len(curr_gold) > 0 and len(curr_preds) > 0:
                gold[ex["input"]].append(curr_gold)
                predictions[ex["input"]] = predictions[ex["input"]].union(curr_preds)

        bleu_scores, rouge1_scores, rouge2_scores, rougel_scores = [], [], [], []

        for input, curr_gold in gold.items():
            curr_predictions = list(predictions[input])

            # The refs and gold must be in the same size
            length = min(len(curr_gold), len(curr_predictions))

            if length > 0:
                hyps = curr_predictions[:length]
                refs = curr_gold[:length]
                all_rouge_scores = rouge.get_scores(hyps, refs)
                rouge1_scores.extend([score["rouge-1"]["f"] for score in all_rouge_scores])
                rouge2_scores.extend([score["rouge-2"]["f"] for score in all_rouge_scores])
                rougel_scores.extend([score["rouge-l"]["f"] for score in all_rouge_scores])

                hyps = [tuple(h.split()) for h in hyps]
                refs = [tuple(r.split()) for r in refs]
                bleu_scores.extend([bleu(
                    refs, pred, weights=weights, smoothing_function=smoothing) for pred in hyps])

        print("\t".join([setup, f"{100.0 * np.mean(bleu_scores):.3f}", f"{100.0 * np.mean(rouge1_scores):.3f}", f"{100.0 * np.mean(rouge2_scores):.3f}", f"{100.0 * np.mean(rougel_scores):.3f}"]))


if __name__ == "__main__":
    main()
