#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -u parsing.py --mode train --config configs/parsing/biaffine.json \
 --num_epochs 400 --batch_size 32 \
 --opt adam --schedule attention --learning_rate 0.1 --lr_decay 0.75 --decay_steps 50 \
 --beta1 0.9 --beta2 0.98 --eps 1e-12 --grad_clip 1.0 --weight_decay 1e-5 \
 --loss_type token --warmup_steps 100 --reset 0 --eval_every 1 --unk_replace 1.0 \
 --word_embedding sskip --word_path "data/en.small-100.bin.gz" --char_embedding random \
 --noscreen \
 --punctuation '.' '``' "''" ':' ',' \
 --train "data/ptb_two_auto.conll" \
 --dev "data/ptb_two_auto.conll" \
 --test "data/ptb_two_auto.conll" \
 --model_path "models/parsing/deepbiaf/" \
 #--freeze --basic_word_embedding 
