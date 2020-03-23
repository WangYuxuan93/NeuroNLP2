#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -u parsing.py --mode train --config configs/parsing/stackptr.json --num_epochs 500 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 5 \
 --word_embedding sskip --word_path "data/en.small-100.bin.gz" --char_embedding random \
 --eval_every 1 --noscreen \
 --punctuation '.' '``' "''" ':' ',' \
 --train "data/ptb_dev_auto.conll" \
 --dev "data/ptb_dev_auto.conll" \
 --test "data/ptb_dev_auto.conll" \
 --model_path "models/parsing/stackptr/"
