#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -u adv_attack.py --mode black \
 --min_word_cos_sim 0.9 --min_sent_cos_sim 0.9 --filters 'embedding' --generators 'synonym:sememe:embedding' \
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
 --syn "data/two_synonyms.json" \
 --adv_rel_ratio 0.5 --adv_fluency_ratio 0.2 --max_mod_percent 0.15 \
 --max_knn_candidates 20 --knn_path "output/embed" \
 --basic_word_embedding 
 #--use_pad
 #--random_sub_if_no_change 
 # --mix_datasets