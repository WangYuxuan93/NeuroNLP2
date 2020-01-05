#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -u parsing.py --mode parse --batch_size 32 \
 --noscreen \
 --punctuation '.' '``' "''" ':' ',' \
 --test "data/ptb_two_auto.conll" \
 --model_path "models/parsing/deepbiaf/"
