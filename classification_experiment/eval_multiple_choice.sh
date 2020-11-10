export STORY_DIR=/Users/tafjord/data/char-centric-story-v2/
python src/run_multiple_choice.py \
  --model_name_or_path tmpout/model \
  --do_eval \
  --data_dir $STORY_DIR \
  --max_seq_length 80 \
  --max_choices 4 \
  --output_dir tmpout/evals \
  --logging_dir tmpout/logs \
  --per_device_eval_batch_size 4 \
  --max_summary_len 50 \
  --max_description_len 50 \
  --use_all_choices_eval \
  --num_examples 30
