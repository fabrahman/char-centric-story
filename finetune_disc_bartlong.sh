#!/bin/bash


DATA_DIR="/home/faezeb/Char_Desc_Gen/modified_scripts_long/new_dataset/discriminative_data"
OUT_DIR=""

mkdir -p ${OUT_DIR}
cp $0 ${OUT_DIR}

python encoder_decoder_long.py \
    --train_file ${DATA_DIR}/train.jsonl \
    --eval_data_file ${DATA_DIR}/val.jsonl \
    --out_dir $OUT_DIR \
    --model_name_or_path facebook/bart-large \
    --device 0 \
    --do_train \
    --do_eval \
    --save_total_limit 1 \
    --num_train_epochs 5 \
    --logging_steps 1000 \
    --gradient_accumulation_steps 8 \
    --train_batch_size 1 \
    --eval_batch_size 8 \
    --max_input_length 2048 \
    --long \
    --task discriminative \
    $@
#--char_length 50 \
