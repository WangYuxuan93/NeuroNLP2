#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -u trans_easyfirst_parsing.py --mode train \
 --config configs/parsing/trans_easyfirst.json --num_epochs 400 --batch_size 32 \
 --update_batch 1 \
 --opt sgd --learning_rate 0.001 --lr_decay 0.999995 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 0 --reset 20 --eval_every 10 --weight_decay 0.0 --unk_replace 0.3 \
 --word_embedding sskip --word_path "data/en.small-100.bin.gz" --char_embedding random \
 --noscreen \
 --punctuation '.' '``' "''" ':' ',' \
 --train "data/ptb_two_auto.conll" \
 --dev "data/ptb_two_auto.conll" \
 --test "data/ptb_two_auto.conll" \
 --model_path "models/parsing/trans_easyfirst/"
