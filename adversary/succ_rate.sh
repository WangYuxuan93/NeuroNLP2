#!/usr/bin/env bash
main=/users7/zllei/NeuroNLP2/adversary/succ_rate.py
dir_name=biaf-glove-v50k-v0-black-ptb_test-0.15-v0
type=$1
orig_parser=$2
attack=$3
model=/users7/zllei/exp_data/models/adv/ptb/${type}
model1=/users7/zllei/exp_data/models/adv/ptb/stack-ptr
adv=/users7/zllei/exp_data/models/parsing/PTB/${type}/$orig_parser/tmp


orig_path=$model/$orig_parser/$dir_name/${orig_parser}@PTB_test_auto.conll.orig
adv_path=${adv}/transferability-same-embedding.txt
gold_path=$model1/$attack/$dir_name/${attack}@PTB_test_auto.conll.adv@black-0.5-0-20.0-0.7-0.95-0.15.gold

python -u $main --orig $orig_path --adv $adv_path --gold $gold_path --p >&1