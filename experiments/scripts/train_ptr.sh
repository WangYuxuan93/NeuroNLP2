#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -u robust_parser.py --mode train --config configs/parsing/robust_stackptr.json \
 --num_epochs 500 --patient_epochs 100 --batch_size 32 --eval_batch_size 32 --seed 0 --beam 5 \
 --opt adamw --schedule step --learning_rate 0.002 --lr_decay 0.75 --decay_steps 5000 \
 --beta1 0.9 --beta2 0.9 --eps 1e-8 --grad_clip 5.0 --weight_decay 0.0 \
 --loss_type token --warmup_steps 40 --reset 20 --eval_every 1 --unk_replace 1.0 \
 --word_embedding sskip --word_path "data/en.small-100.bin.gz" --char_embedding random \
 --pretrained_lm none --lm_path "/mnt/hgfs/share/xlm-roberta-base" --noscreen \
 --punctuation '.' '``' "''" ':' ',' , --pos_idx 3 \
 --format conllx --do_trim --normalize_digits \
 --train "data/ptb_small_auto.conll" \
 --dev "data/ptb_two_auto_orig.conll" \
 --test "data/ptb_two_auto_orig.conll" \
 --lan_train en --lan_dev en --lan_test en \
 --model_path "models/parsing/new_stackptr" \
 #--mix_datasets \
 #--punctuation 'PUNCT' --pos_idx 3 \
 #--freeze --basic_word_embedding 
