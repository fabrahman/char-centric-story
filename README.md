## "Let Your Characters Tell Their Story": A Dataset for Character-Centric Narrative Understanding

Please follow the instruction [here](https://github.com/huangmeng123/lit_char_data_wayback) to recreate the LiSCU dataset.

Before you start, make sure you have installed all the requiremenets in the `requiremenets.txt`.

Note that the code is still being cleaned.

### Generative: character description generation

To train character description generator given the summary and character name using different LMs (bartlong, gpt2), run:

```bash
sh finetune_[lm].sh
```

```
# display argumenets
python -m encoder_decoder_long --help
```

For gpt2, replace `encoder_decoder_long` with `generative`. Note that for gpt2 we have to limit `max_input_length=945` and `max_output_length=75`.


To generate descriptions for a trained model (e.g. bartlong), run:

```bash
python generate_texts.py \
	--in_file ../data/new/test.jsonl \
	--out_file char_checkpoints/bart-large-xsum_long/generation/test_prediction_beams5_maxlen_1024.jsonl \
	--model_name_or_path char_checkpoints/bart-large-xsum_long \
	--beams 5 \
	--device 0 \
	--max_length 1024
```
For gpt2, `max_length=75`.


### Discriminative: character name identification

Here, we first take a generative approach for character name identification, i.e. given a summary and anonymized character description, we generate the character name.

To train model using bartlong, run:

```bash
sh finetune_disc_bartlong.sh
```
**NOTE**: Add `--char_length 50` when needed (for partial description).

To generate character name, run:

```
python generate_texts.py \
	--in_file ../data/new/discriminative_data/data/v03/test.jsonl \
	--out_file char_checkpoints/bart-large-xsum_long_disc/generation/test_prediction_greedy_maxlen_20.jsonl \
	--model_name_or_path char_checkpoints/bart-large-xsum_long_disc \
	--beams 1 \
	--device 0 \
	--max_length 20 \
	--char_name_last \
	--task discriminative
```
To compute accuracy of character identification using max probable characters in a set of possible choices, run:

```
python -m eval.multiple_choice_char_name_gen_new \
	--model_name_or_path char_checkpoints/bart-large-xsum_long_disc \
	--dataset_file ../data/new/test.jsonl \
	--out_dir char_checkpoints/bart-large-xsum_long_disc \
	--device 0
	--format with-choices
```
#**NOTE** : used env long2 for longformer and py37 for bart baselines.
