#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -u easyfirst_parsing_v2.py --mode parse --batch_size 32 \
 --noscreen --get_head_by_layer \
 --punctuation '.' '``' "''" ':' ',' \
 --test "data/ptb_two_auto.conll" \
 --model_path "models/parsing/easyfirst_v2/" --random_recomp --recomp_prob 0.4
