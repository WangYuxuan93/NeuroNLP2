#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -u easyfirst_parsing.py --mode parse --num_epochs 400 --batch_size 32 \
 --noscreen \
 --punctuation '.' '``' "''" ':' ',' \
 --test "data/ptb_two_auto.conll" \
 --model_path "models/parsing/easyfirst/"
