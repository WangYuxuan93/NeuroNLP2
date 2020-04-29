#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -u refinement_parser.py --mode parse \
 --batch_size 32 \
 --noscreen \
 --punctuation '.' '``' "''" ':' ',' \
 --pretrained_lm none \
 --format ud \
 --test "data/ptb_two_auto.conll" \
 --lan_test en \
 --model_path "models/parsing/refine/" \
 --output_filename "models/parsing/refine/ptb_two_auto.pred" \
 # --mix_datasets
