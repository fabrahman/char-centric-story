#!/bin/bash

# env: long2

DATA_DIR="/home/faezeb/Char_Desc_Gen/modified_scripts_long/data/new/src-trg/"
OUT_DIR="char_checkpoints/longformer_full" #"outputs/longformer"
checkpoint="" # path to converted bart-large-xsum (long version)

mkdir -p ${OUT_DIR}

# #faeze changed warmup from 0 to 1000
# facebook/bart-large-xsum
python character_description_longformer_new.py \
    --model_path $checkpoint \
    --tokenizer facebook/bart-large-xsum \
    --learning_rate 3e-5 \
    --gpus 1 \
    --n_val -1 \
    --val_every 1 \
    --sortish_sampler \
    --max_output_len 1024 \
    --max_input_len 6144 \
    --batch_size 1 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --data_dir $DATA_DIR \
    --save_dir $OUT_DIR \
    --warmup 1000 \
    --attention_window 512 \
    --epochs 5 \
    --num_workers 16 \
    --grad_ckpt \
    $@
