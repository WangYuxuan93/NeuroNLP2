#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -u parsing.py --mode train --config configs/parsing/stackptr.json \
 --num_epochs 500 --patient_epochs 100 --batch_size 32 --beam 5 --seed 0 \
 --opt adamw --schedule step --learning_rate 0.002 --lr_decay 0.75 --decay_steps 5000 \
 --beta1 0.9 --beta2 0.9 --eps 1e-8 --grad_clip 5.0 --weight_decay 0.0 \
 --loss_type token --warmup_steps 40 --reset 20 --eval_every 1  --unk_replace 1.0  \
 --word_embedding sskip --word_path "data/en.small-100.bin.gz" --char_embedding random \
 --eval_every 1 --noscreen \
 --punctuation '.' '``' "''" ':' ',' \
 --train "data/ptb_small_auto.conll" \
 --dev "data/ptb_two_auto_orig.conll" \
 --test "data/ptb_two_auto_orig.conll" \
 --model_path "models/parsing/old_stackptr"
