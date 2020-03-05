#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -u easyfirst_parsing_v2.py --mode train --config configs/parsing/easyfirst.json --num_epochs 400 \
 --batch_size 32 \
 --opt adam --learning_rate 0.0001 --lr_decay 0.999995 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 20 --reset 20 --eval_every 1 --weight_decay 0.0 --unk_replace 0.0 \
 --word_embedding sskip --word_path "data/en.small-100.bin.gz" --char_embedding random \
 --noscreen --sampler random \
 --punctuation '.' '``' "''" ':' ',' \
 --train "data/ptb_two_auto.conll" \
 --dev "data/ptb_two_auto.conll" \
 --test "data/ptb_two_auto.conll" \
 --model_path "models/parsing/easyfirst_v2/" --get_head_by_layer \
 --seed 0 #--symbolic_end False #--explore #--batch_by_arc #--fine_tune #--recomp_prob 0.4 --random_recomp 
