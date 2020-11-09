#!/usr/bin/env bash
#emb=/users2/yxwang/work/data/ud/embeddings/muse/en_ewt.300.muse.vec.gz
emb=/users2/yxwang/work/data/embeddings/glove/glove.6B.100d.txt.gz
lmdir=/users2/yxwang/work/data/models
#lmpath=$lmdir/bert-base-cased
#dir=/users2/yxwang/work/data/ud/ud-v2.2
#for lc in ${lcs[@]};
#do
#  echo $lc
#  train=$train:$dir/$lc/$lc-ud-train.conllu
#  dev=$dev:$dir/$lc/$lc-ud-dev.conllu
  #test=$test:$dir/$lc/$lc-ud-test.conllu
#done
dir=/users2/yxwang/work/data/ptb/dependency-stanford-chen14
train=$dir/PTB_train_auto.conll
dev=$dir/PTB_dev_auto.conll
test=$dir/PTB_test_auto.conll
lans="en"

tcdir=/users2/yxwang/work/experiments/robust_parser/lm/saves
main=/users2/yxwang/work/codes/NeuroNLP2/experiments/parser.py

seed=666
#seed=888
#seed=777
#seed=999
#seed=555
batch=128
evalbatch=128
epoch=1000
patient=100
lr='0.002'
lm=none
#lm=roberta
lmpath=$lmdir/roberta-base
#lmpath=$lmdir/roberta-large
#lm=electra
#lmpath=$lmdir/electra-large-discriminator
#lmpath=$lmdir/electra-base-discriminator

use_elmo=''
#use_elmo=' --use_elmo '
elmo_path=$lmdir/elmo

random_word=''
#random_word=' --use_random_static '
#pretrain_word=''
pretrain_word=' --use_pretrained_static '
freeze=''
#freeze=' --freeze'
#trim=''
trim=' --do_trim'
#vocab_size=400000
vocab_size=50000

lmlr='2e-5'
#lmlr=0
opt=adamw
#sched=exponential
#decay='0.99999'
sched=step
decay='0.75'
dstep=5000
warmup=500
reset=20
beta1='0.9'
#beta2='0.999'
beta2='0.9'
eps='1e-8'
beam=1
clip='5.0'
l2decay='0'
unk=0
#unk='1.0'
ndigit=''
#ndigit=' --normalize_digits'
losstype=token
evalevery=1
posidx=3
mix=' --mix_datasets'
form=conllx

gpu=$1
save=$2
log=$3

if [ -z $3 ];then
  echo '[gpu] [save] [log]'
  exit
fi

source /users2/yxwang/work/env/py3.6/bin/activate
CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 python -u $main --mode train --config parsers/configs/stackptr.json --seed $seed \
 --num_epochs $epoch --patient_epochs $patient --batch_size $batch --eval_batch_size $evalbatch \
 --opt $opt --schedule $sched --learning_rate $lr --lr_decay $decay --decay_steps $dstep \
 --beta1 $beta1 --beta2 $beta2 --eps $eps --grad_clip $clip --beam $beam \
 --eval_every $evalevery --noscreen ${random_word} ${pretrain_word} $freeze \
 --loss_type $losstype --warmup_steps $warmup --reset $reset --weight_decay $l2decay --unk_replace $unk \
 --word_embedding sskip --word_path $emb --char_embedding random \
 --max_vocab_size ${vocab_size} $trim $ndigit \
 --elmo_path ${elmo_path} ${use_elmo} \
 --pretrained_lm $lm --lm_path $lmpath --lm_lr $lmlr \
 --punctuation '.' '``' "''" ':' ',' --pos_idx $posidx \
 --format $form \
 --train $train \
 --dev $dev \
 --test $test \
 --lan_train $lans --lan_dev $lans --lan_test $lans $mix \
 --model_path $save > $log
