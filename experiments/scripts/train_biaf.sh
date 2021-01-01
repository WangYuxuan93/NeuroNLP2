#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=4 python -u parsing.py --mode train \
 --config configs/parsing/biaffine.json \
 --num_epochs 400 --patient_epochs 40 --batch_size 32 --seed 0 \
 --opt adam --schedule attention --learning_rate 0.1 --lr_decay 0.75 --decay_steps 50 \
 --beta1 0.9 --beta2 0.98 --eps 1e-12 --grad_clip 1.0 --weight_decay 1e-5 \
 --loss_type token --warmup_steps 100 --reset 0 --eval_every 1 --unk_replace 1.0 \
 --word_embedding sskip --word_path "../../exp_data/sskip.eng.100.gz" \
 --char_embedding random \
 --noscreen \
 --punctuation '.' '``' "''" ':' ',' \
 --train "../../exp_data/ud-conllu/dm/en.dm.train.udpos.conllu" \
 --dev "../../exp_data/ud-conllu/dm/en.dm.dev.udpos.conllu" \
 --test "../../exp_data/ud-conllu/dm/en.id.dm.udpos.conllu" \
 --model_path "../../exp_data/models/parsing/deepbiaf/" \
 --task_type "sdp"
 #--punctuation 'PUNCT' --pos_idx 3 \
 #--freeze --basic_word_embedding 
