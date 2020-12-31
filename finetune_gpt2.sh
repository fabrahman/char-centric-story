#!/bin/bash
#env: py37

DATA_DIR="/home/faezeb/Char_Desc_Gen/modified_scripts_long/data/new"
OUT_DIR=""

mkdir -p ${OUT_DIR}
cp $0 ${OUT_DIR}

python generative.py \
    --train_file ${DATA_DIR}/train.jsonl \
    --eval_data_file ${DATA_DIR}/val.jsonl \
    --out_dir $OUT_DIR \
    --model_name_or_path gpt2-large \
    --device 0 \
    --do_train \
    --do_eval \
    --save_total_limit 1 \
    --overwrite_cache \
    --num_train_epochs 5 \
    --logging_steps 1000 \
    --gradient_accumulation_steps 4 \
    --train_batch_size 2 \
    --eval_batch_size 8 \
    --truncation_method coref \
    $@
#--char_name_last \
