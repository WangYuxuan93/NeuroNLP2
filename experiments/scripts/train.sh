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
dir=/users2/yxwang/work/data/semeval2015-task18/english/conllu/dm
train=$dir/en.dm.train.conllu
dev=$dir/en.dm.dev.conllu
test=$dir/en.id.dm.conllu
lans="en"

#tcdir=/users2/yxwang/work/experiments/robust_parser/lm/saves
main=/users7/zllei/NeuroNLP2/experiments/robust_parser_sdp.py

#seed=666
#seed=888
#seed=777
#seed=999
seed=555
batch=32
evalbatch=$batch
epoch=1000
patient=20
lr='0.001'
lm=none
lmpath=$lmdir/roberta-base
#lmpath=$lmdir/roberta-large
#lm=elmo
#lmpath=$lmdir/elmo
#lm=electra
#lmpath=$lmdir/electra-large-discriminator
#lmpath=$lmdir/electra-base-discriminator
#lm=none
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
beta1='0.0'
#beta2='0.999'
beta2='0.95'
eps='1e-8'
clip='5.0'
l2decay='0'
#unk=0
unk='1.0'
#ndigit=''
ndigit=' --normalize_digits'
losstype=token
evalevery=1
posidx=3
mix=' --mix_datasets'
form=conllx
freeze=''
basic=''
#freeze=' --freeze'
#basic=' --basic_word_embedding'
trim=' --do_trim'
#trim=''

gpu=$1
mode=$2
save=/users7/zllei/exp_data/models/parsing/robust_parser_sdp
log=${save}/log_$(date "+%Y%m%d-%H%M%S").txt

if [ -z $2 ];then
  echo '[gpu] [save] [log]'
  exit
fi

#if [ ! -f "$log" ]; then
#  touch "$log"
#fi

#source /users2/yxwang/work/env/py3.6/bin/activate
CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 python -u $main --mode $mode \
 --config configs/parsing/robust.json --seed $seed \
 --num_epochs $epoch --patient_epochs $patient --batch_size $batch --eval_batch_size $evalbatch \
 --opt $opt --schedule $sched --learning_rate $lr --lr_decay $decay --decay_steps $dstep \
 --beta1 $beta1 --beta2 $beta2 --eps $eps --grad_clip $clip \
 --eval_every $evalevery --noscreen $basic $freeze \
 --loss_type $losstype --warmup_steps $warmup --reset $reset --weight_decay $l2decay --unk_replace $unk \
 --word_embedding glove --word_path $emb --char_embedding random $trim $ndigit \
 --pretrained_lm $lm --lm_path $lmpath --lm_lr $lmlr \
 --punctuation '.' '``' "''" ':' ',' --pos_idx $posidx \
 --format $form \
 --train $train \
 --dev $dev \
 --test $test \
 --lan_train $lans --lan_dev $lans --lan_test $lans $mix \
 --model_path $save
