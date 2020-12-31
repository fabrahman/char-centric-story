#!/bin/bash

# env: long2

DATA_DIR="/home/faezeb/Char_Desc_Gen/modified_scripts_long/new_dataset/discriminative_data/" #"/home/faezeb/Char_Desc_Gen/modified_scripts_long/data/new/discriminative_data/" #"/home/faezeb/Char_Desc_Gen/modified_scripts_long/data/new/src-trg/"
OUT_DIR="/net/nfs.corp/alexandria/faezeb/char_checkpoints/new_dataset/longformer_disc_full" #w_choices_50-words" #"outputs/longformer"
checkpoint="/home/faezeb/longformer/checkpoints/converted-bart-large-6144"

mkdir -p ${OUT_DIR}
cp $0 ${OUT_DIR}

# #faeze changed warmup from 0 to 1000
# facebook/bart-large-xsum
python character_identification_gen.py \
    --model_path $checkpoint \
    --tokenizer facebook/bart-large \
    --learning_rate 3e-5 \
    --gpus 1 \
    --n_val -1 \
    --val_every 1 \
    --sortish_sampler \
    --max_output_len 15 \
    --max_input_len 6144 \
    --batch_size 1 \
    --eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --data_dir $DATA_DIR \
    --save_dir $OUT_DIR \
    --warmup 1000 \
    --attention_window 512 \
    --epochs 5 \
    --num_workers 16 \
    --grad_ckpt \
    $@
# --best_checkpoint ${OUT_DIR}/test/_ckpt_epoch_2.ckpt \
# --char_length 50 \
