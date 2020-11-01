### Longformer for generative task

Please follow the instruction [here](https://github.com/allenai/longformer/tree/encoderdecoder) to install the longformer and its requiremenets in new python environment.

Use the `scripts/convert_bart_to_longformerencoderdecoder.py` in the longformer repo to create a long version of `bart-large-xsum`

To train a longformer model, run:

```bash
sh finetune_orig_lonformer.sh
```
