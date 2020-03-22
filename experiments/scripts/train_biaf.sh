#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -u parsing.py --mode train --config configs/parsing/biaffine.json \
 --num_epochs 400 --batch_size 4 \
 --opt adam_li --schedule step --learning_rate 0.0012 --lr_decay 0.75 --decay_steps 20 \
 --beta1 0.9 --beta2 0.98 --eps 1e-12 --grad_clip 1.0 \
 --loss_type mean --warmup_steps 10 --reset 0 --eval_every 5 --log_every 1 \
 --weight_decay 1e-5 --unk_replace 0 \
 --word_embedding sskip --word_path "data/en.small-100.bin.gz" --char_embedding random \
 --noscreen --basic_word_embedding \
 --punctuation '.' '``' "''" ':' ',' \
 --train "data/ptb_small_auto.conll" \
 --dev "data/ptb_two_auto.conll" \
 --test "data/ptb_two_auto.conll" \
 --model_path "models/parsing/deepbiaf/"
