#!/usr/bin/env python

"""
modified by Faeze Brahman to compute BERTScore for generated jsonl files
"""
import os
import argparse
import json
import torch

import bert_score


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    print("Running my modified code ...")
    parser = argparse.ArgumentParser("Calculate BERTScore")
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help='two-letter abbreviation of the language (e.g., en) or "en-sci" for scientific text',
    )
    parser.add_argument(
        "-m", "--model", default=None, help="BERT model name (default: bert-base-uncased) or path to a pretrain model"
    )
    parser.add_argument("-l", "--num_layers", type=int, default=None, help="use first N layer in BERT (default: 8)")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size (default: 64)")
    parser.add_argument("--nthreads", type=int, default=4, help="number of cpu workers (default: 4)")
    parser.add_argument("--idf", action="store_true", help="BERT Score with IDF scaling")
    parser.add_argument(
        "--rescale_with_baseline", action="store_true", help="Rescaling the numerical score with precomputed baselines"
    )
    parser.add_argument("--baseline_path", default=None, type=str, help="path of custom baseline csv file")
    parser.add_argument("-s", "--seg_level", action="store_true", help="show individual score of each pair")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("-r", "--ref", type=str, nargs="+", required=False, help="reference file path(s) or a string")
    parser.add_argument(
        "-c", "--cand", type=str, required=False, help="candidate (system outputs) file path or a string"
    )
    parser.add_argument(
        "-rc", "--out_dir", type=str, required=False, help="directory with models predictions -- json file containing both reference (gold) and candidate (system outputs)"
    )

    args = parser.parse_args()

    assert not ( (args.cand and args.ref) and args.out_dir), "either -r and -c or json with both reference and candidate should be identified."

    # print("\t".join(["Model", "code", "P", "R", "F1"]))
    for setup in ["bart-large-xsum_long", "longformer", "gpt2-large", "bart-large-xsum_long_truncated",
                  "longformer_full", "longformer_truncated", "gpt2-large_truncated"]:  # ["bart-large-xsum_long", "gpt2-large", "longformer",]:
        #        for lm in ["bart-large", "gpt2-xl"]:


        if args.out_dir and os.path.isdir(args.out_dir):
            data = [json.loads(line.strip()) for line in open(
                f"{args.out_dir}/{setup}/generation/test_prediction_beams5_maxlen_{75 if 'gpt2' in setup else 1024}.jsonl")]
            refs = [ex["gold"].lower().replace("<eos>", "").strip() for ex in data]
            cands = [[ex["predictions"]] if isinstance(ex["predictions"], str) else ex["predictions"] for ex in data]
            cands = [c[0].lower().replace("<eos>", "").strip() if c != [] else "" for c in cands]
#            cands = [ex["predictions"][0].lower().replace("<eos>", "").strip() if ex["predictions"] != [] else "" for ex in data]


        elif os.path.isfile(args.cand):
            with open(args.cand) as f:
                cands = [line.strip() for line in f]

            refs = []
            for ref_file in args.ref:
                assert os.path.exists(ref_file), f"reference file {ref_file} doesn't exist"
                with open(ref_file) as f:
                    curr_ref = [line.strip() for line in f]
                    assert len(curr_ref) == len(cands), f"# of sentences in {ref_file} doesn't match the # of candidates"
                    refs.append(curr_ref)
            refs = list(zip(*refs))
        elif os.path.isfile(args.ref[0]):
            assert os.path.exists(args.cand), f"candidate file {args.cand} doesn't exist"
        else:
            cands = [args.cand]
            refs = [args.ref]
            assert not args.idf, "do not support idf mode for a single pair of sentences"

        all_preds, hash_code = bert_score.score(
            cands,
            refs,
            model_type=args.model,
            num_layers=args.num_layers,
            verbose=args.verbose,
            idf=args.idf,
            batch_size=args.batch_size,
            lang=args.lang,
            return_hash=True,
            rescale_with_baseline=args.rescale_with_baseline,
            baseline_path=args.baseline_path,
        )
        avg_scores = [s.mean(dim=0) for s in all_preds]
        P = avg_scores[0].cpu().item()
        R = avg_scores[1].cpu().item()
        F1 = avg_scores[2].cpu().item()
        msg = "\t".join([setup, hash_code , f" P: {P:.6f}", f"R: {R:.6f}", f"F1: {F1:.6f}"])
        print(msg)
        if args.seg_level:
            ps, rs, fs = all_preds
            for p, r, f in zip(ps, rs, fs):
                print("{:.6f}\t{:.6f}\t{:.6f}".format(p, r, f))


if __name__ == "__main__":
    main()
