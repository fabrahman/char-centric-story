### Longformer for Character Description Generation

Please follow the instruction [here](https://github.com/allenai/longformer/tree/encoderdecoder) to install the longformer and its requiremenets in new python environment.

Use the `scripts/convert_bart_to_longformerencoderdecoder.py` in the longformer repo to create a long version of `bart-large-xsum`

To train a longformer model, run:

```bash
sh finetune_orig_lonformer.sh
```

### Longformer for Character Identification

To train a generative classifier using longformer, run:
```bash
sh finetune_disc_orig_lonformer.sh
```
**NOTE**: Add `--char_length 50` when needed (for partial description).


#### Evaluation
run `python compute_acc_from_generated_test.py` inside the directory where generated test file is.
