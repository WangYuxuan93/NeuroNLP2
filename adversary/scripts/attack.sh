#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -u adv_attack.py --mode parse \
 --batch_size 32 \
 --noscreen \
 --punctuation '.' '``' "''" ':' ',' \
 --pretrained_lm none \
 --format ud \
 --test "../experiments/data/ptb_two_auto_orig.conll" \
 --lan_test en \
 --model_path "../experiments/models/parsing/robust" \
 --output_filename "ptb_two_auto.pred.conll" \
 --adv_filename "ptb_two_auto.adv.conll" \
 --vocab "data/vocab.json" --cand "data/word_candidates_sense.json" \
 --adv_rel_ratio 0.5 --adv_fluency_ratio 0.2 --max_perp_diff_per_token 0.6
 # --mix_datasets
