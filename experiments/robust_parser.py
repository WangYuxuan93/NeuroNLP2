"""
Implementation of Graph-based dependency parsing.
"""

import os
import sys
import gc
import json

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import argparse
import math
import numpy as np
import torch
#from torch.optim.adamw import AdamW
from torch.optim import SGD, Adam, AdamW
from torch.nn.utils import clip_grad_norm_
from neuronlp2.nn.utils import total_grad_norm
from neuronlp2.io import get_logger, conllx_data, ud_data, conllx_stacked_data #, iterate_data
from neuronlp2.models.robust_parsing import RobustParser
from neuronlp2.optim import ExponentialScheduler, StepScheduler, AttentionScheduler
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser
from neuronlp2.nn.utils import freeze_embedding
from neuronlp2.io import common
from transformers import *
from neuronlp2.io.common import PAD, ROOT, END
from neuronlp2.io.batcher import multi_language_iterate_data, iterate_data
from neuronlp2.io import multi_ud_data

def get_optimizer(parameters, optim, learning_rate, lr_decay, betas, eps, amsgrad, weight_decay, 
                  warmup_steps, schedule='step', hidden_size=200, decay_steps=5000):
    if optim == 'sgd':
        optimizer = SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif optim == 'adamw':
        optimizer = AdamW(parameters, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
    elif optim == 'adam':
        optimizer = Adam(parameters, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
    
    init_lr = 1e-7
    if schedule == 'exponential':
        scheduler = ExponentialScheduler(optimizer, lr_decay, warmup_steps, init_lr)
    elif schedule == 'attention':
        scheduler = AttentionScheduler(optimizer, hidden_size, warmup_steps)
    elif schedule == 'step':
        scheduler = StepScheduler(optimizer, lr_decay, decay_steps, init_lr, warmup_steps)
    
    return optimizer, scheduler


def convert_tokens_to_ids(tokenizer, tokens):

    all_wordpiece_list = []
    all_first_index_list = []

    for toks in tokens:
        wordpiece_list = []
        first_index_list = []
        for token in toks:
            if token == PAD:
                token = tokenizer.pad_token
            elif token == ROOT:
                token = tokenizer.cls_token
            elif token == END:
                token = tokenizer.sep_token
            wordpiece = tokenizer.tokenize(token)
            # add 1 for cls_token <s>
            first_index_list.append(len(wordpiece_list)+1)
            wordpiece_list += wordpiece
            #print (wordpiece)
        #print (wordpiece_list)
        #print (first_index_list)
        bpe_ids = tokenizer.convert_tokens_to_ids(wordpiece_list)
        #print (bpe_ids)
        bpe_ids = tokenizer.build_inputs_with_special_tokens(bpe_ids)
        #print (bpe_ids)
        all_wordpiece_list.append(bpe_ids)
        all_first_index_list.append(first_index_list)

    all_wordpiece_max_len = max([len(w) for w in all_wordpiece_list])
    all_wordpiece = np.stack(
          [np.pad(a, (0, all_wordpiece_max_len - len(a)), 'constant', constant_values=tokenizer.pad_token_id) for a in all_wordpiece_list])
    all_first_index_max_len = max([len(i) for i in all_first_index_list])
    all_first_index = np.stack(
          [np.pad(a, (0, all_first_index_max_len - len(a)), 'constant', constant_values=0) for a in all_first_index_list])

    # (batch, max_bpe_len)
    input_ids = torch.from_numpy(all_wordpiece)
    # (batch, seq_len)
    first_indices = torch.from_numpy(all_first_index)

    return input_ids, first_indices


def eval(alg, data, network, pred_writer, gold_writer, punct_set, word_alphabet, pos_alphabet, 
        device, beam=1, batch_size=256, write_to_tmp=True, prev_best_lcorr=0, prev_best_ucorr=0,
        pred_filename=None, tokenizer=None, multi_lan_iter=False):
    network.eval()
    accum_ucorr = 0.0
    accum_lcorr = 0.0
    accum_total = 0
    accum_ucomlpete = 0.0
    accum_lcomplete = 0.0
    accum_ucorr_nopunc = 0.0
    accum_lcorr_nopunc = 0.0
    accum_total_nopunc = 0
    accum_ucomlpete_nopunc = 0.0
    accum_lcomplete_nopunc = 0.0
    accum_root_corr = 0.0
    accum_total_root = 0.0
    accum_total_inst = 0.0
    accum_recomp_freq = 0.0

    all_words = []
    all_postags = []
    all_heads_pred = []
    all_rels_pred = []
    all_lengths = []
    all_src_words = []
    all_heads_by_layer = []

    if multi_lan_iter:
        iterate = multi_language_iterate_data
    else:
        iterate = iterate_data
        lan_id = None

    for data in iterate(data, batch_size):
        if multi_lan_iter:
            lan_id, data = data
            lan_id = torch.LongTensor([lan_id]).to(device)
        if tokenizer:
            bpes, first_idx = convert_tokens_to_ids(tokenizer, data['SRC'])
            bpes = bpes.to(device)
            first_idx = first_idx.to(device)
        else:
            bpes = first_idx = None
        words = data['WORD'].to(device)
        pres = data['PRETRAINED'].to(device)
        chars = data['CHAR'].to(device)
        postags = data['POS'].to(device)
        heads = data['HEAD'].numpy()
        rels = data['TYPE'].numpy()
        lengths = data['LENGTH'].numpy()
        if alg == 'graph':
            masks = data['MASK'].to(device)
            heads_pred, rels_pred = network.decode(words, pres, chars, postags, mask=masks, 
                bpes=bpes, first_idx=first_idx, lan_id=lan_id, leading_symbolic=common.NUM_SYMBOLIC_TAGS)

        words = words.cpu().numpy()
        postags = postags.cpu().numpy()

        if write_to_tmp:
            pred_writer.write(words, postags, heads_pred, rels_pred, lengths, symbolic_root=True, src_words=data['SRC'])
        else:
            all_words.append(words)
            all_postags.append(postags)
            all_heads_pred.append(heads_pred)
            all_rels_pred.append(rels_pred)
            all_lengths.append(lengths)
            all_src_words.append(data['SRC'])

        #gold_writer.write(words, postags, heads, rels, lengths, symbolic_root=True)
        #print ("heads_pred:\n", heads_pred)
        #print ("rels_pred:\n", rels_pred)
        #print ("heads:\n", heads)
        stats, stats_nopunc, stats_root, num_inst = parser.eval(words, postags, heads_pred, rels_pred, heads, rels,
                                                                word_alphabet, pos_alphabet, lengths, punct_set=punct_set, 
                                                                symbolic_root=True)
        ucorr, lcorr, total, ucm, lcm = stats
        ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
        corr_root, total_root = stats_root

        accum_ucorr += ucorr
        accum_lcorr += lcorr
        accum_total += total
        accum_ucomlpete += ucm
        accum_lcomplete += lcm

        accum_ucorr_nopunc += ucorr_nopunc
        accum_lcorr_nopunc += lcorr_nopunc
        accum_total_nopunc += total_nopunc
        accum_ucomlpete_nopunc += ucm_nopunc
        accum_lcomplete_nopunc += lcm_nopunc

        accum_root_corr += corr_root
        accum_total_root += total_root

        accum_total_inst += num_inst

    print('W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        accum_ucorr, accum_lcorr, accum_total, accum_ucorr * 100 / accum_total, accum_lcorr * 100 / accum_total,
        accum_ucomlpete * 100 / accum_total_inst, accum_lcomplete * 100 / accum_total_inst))
    print('Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        accum_ucorr_nopunc, accum_lcorr_nopunc, accum_total_nopunc, accum_ucorr_nopunc * 100 / accum_total_nopunc,
        accum_lcorr_nopunc * 100 / accum_total_nopunc,
        accum_ucomlpete_nopunc * 100 / accum_total_inst, accum_lcomplete_nopunc * 100 / accum_total_inst))
    print('Root: corr: %d, total: %d, acc: %.2f%%' %(accum_root_corr, accum_total_root, accum_root_corr * 100 / accum_total_root))
    
    if not write_to_tmp:
        if prev_best_lcorr < accum_lcorr_nopunc or (prev_best_lcorr == accum_lcorr_nopunc and prev_best_ucorr < accum_ucorr_nopunc):
            print ('### Writing New Best Dev Prediction File ... ###')
            pred_writer.start(pred_filename)
            for i in range(len(all_words)):
                pred_writer.write(all_words[i], all_postags[i], all_heads_pred[i], all_rels_pred[i], 
                                all_lengths[i], symbolic_root=True, src_words=all_src_words[i])
            pred_writer.close()

    return (accum_ucorr, accum_lcorr, accum_ucomlpete, accum_lcomplete, accum_total), \
           (accum_ucorr_nopunc, accum_lcorr_nopunc, accum_ucomlpete_nopunc, accum_lcomplete_nopunc, accum_total_nopunc), \
           (accum_root_corr, accum_total_root, accum_total_inst)


def train(args):
    logger = get_logger("Parsing")

    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    data_format = args.format
    if data_format == 'conllx':
        data_reader = conllx_data
        train_path = args.train
        dev_path = args.dev
        test_path = args.test
    elif data_format == 'ud':
        data_reader = ud_data
        train_path = args.train.split(':')
        dev_path = args.dev.split(':')
        test_path = args.test.split(':')
    else:
        print ("### Unrecognized data formate: %s ###" % data_format)
        exit()

    basic_word_embedding = args.basic_word_embedding
    num_epochs = args.num_epochs
    patient_epochs = args.patient_epochs
    batch_size = args.batch_size
    optim = args.optim
    schedule = args.schedule
    learning_rate = args.learning_rate
    lr_decay = args.lr_decay
    decay_steps = args.decay_steps
    amsgrad = args.amsgrad
    eps = args.eps
    betas = (args.beta1, args.beta2)
    warmup_steps = args.warmup_steps
    weight_decay = args.weight_decay
    grad_clip = args.grad_clip
    eval_every = args.eval_every
    noscreen = args.noscreen

    loss_type_token = args.loss_type == 'token'
    unk_replace = args.unk_replace
    freeze = args.freeze

    model_path = args.model_path
    model_name = os.path.join(model_path, 'model.pt')
    punctuation = args.punctuation

    word_embedding = args.word_embedding
    word_path = args.word_path
    char_embedding = args.char_embedding
    char_path = args.char_path
    pretrained_lm = args.pretrained_lm
    lm_path = args.lm_path

    print(args)

    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)
    char_dict = None
    if char_embedding != 'random':
        char_dict, char_dim = utils.load_embedding_dict(char_embedding, char_path)
    else:
        char_dict = None
        char_dim = None

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(model_path, 'alphabets')
    if data_format == "conllx":
        data_paths=[dev_path, test_path]
    elif data_format == "ud":
        data_paths=dev_path + test_path
    word_alphabet, char_alphabet, pos_alphabet, rel_alphabet = data_reader.create_alphabets(alphabet_path, train_path,
                                                                                             data_paths=data_paths, 
                                                                                             embedd_dict=word_dict, max_vocabulary_size=400000,
                                                                                             normalize_digits=args.normalize_digits,
                                                                                             pos_idx=args.pos_idx)
    pretrained_alphabet = utils.create_alphabet_from_embedding(alphabet_path, word_dict, word_alphabet.instances, max_vocabulary_size=400000)

    num_words = word_alphabet.size()
    num_pretrained = pretrained_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_rels = rel_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Pretrained Alphabet Size: %d" % num_pretrained)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Rel Alphabet Size: %d" % num_rels)

    result_path = os.path.join(model_path, 'tmp')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / word_dim)
        if basic_word_embedding:
            table = np.empty([pretrained_alphabet.size(), word_dim], dtype=np.float32)
            items = pretrained_alphabet.items()
        else:
            table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
            items = word_alphabet.items()
        table[data_reader.UNK_ID, :] = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
        oov = 0
        for word, index in items:
            if word in word_dict:
                embedding = word_dict[word]
            elif word.lower() in word_dict:
                embedding = word_dict[word.lower()]
            else:
                embedding = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('word OOV: %d' % oov)
        return torch.from_numpy(table)

    def construct_char_embedding_table():
        if char_dict is None:
            return None

        scale = np.sqrt(3.0 / char_dim)
        table = np.empty([num_chars, char_dim], dtype=np.float32)
        table[common.UNK_ID, :] = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
        oov = 0
        for char, index, in char_alphabet.items():
            if char in char_dict:
                embedding = char_dict[char]
            else:
                embedding = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('character OOV: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()
    char_table = construct_char_embedding_table()

    logger.info("constructing network...")
    random_seed = args.seed
    if random_seed == -1:
        random_seed = np.random.randint(1e8)
        logger.info("Random Seed (rand): %d" % random_seed)
    else:
        logger.info("Random Seed (set): %d" % random_seed)
    torch.manual_seed(random_seed)

    hyps = json.load(open(args.config, 'r'))
    json.dump(hyps, open(os.path.join(model_path, 'config.json'), 'w'), indent=2)
    model_type = hyps['model']
    assert model_type in ['Robust']
    assert word_dim == hyps['input']['word_dim']
    if char_dim is not None:
        assert char_dim == hyps['input']['char_dim']
    else:
        char_dim = hyps['input']['char_dim']
    loss_interpolation = hyps['biaffine']['loss_interpolation']
    hidden_size = hyps['input_encoder']['hidden_size']
    num_lans = 1
    if not args.mix_datasets:
        lans_train = args.lan_train.split(':')
        lans_dev = args.lan_dev.split(':')
        lans_test = args.lan_test.split(':')
        languages = set(lans_train + lans_dev + lans_test)
        language_alphabet = utils.creat_language_alphabet(alphabet_path, languages)
        num_lans = language_alphabet.size()
        assert len(languages)+1 == num_lans
        data_reader = multi_ud_data
    """
    if pretrained_lm == 'bert':
        tokenizer = BertTokenizer.from_pretrained(lm_path)
    elif pretrained_lm == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(lm_path)
    elif pretrained_lm == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained(lm_path)
    elif pretrained_lm == 'xlm-r':
        tokenizer = XLMRobertaTokenizer.from_pretrained(lm_path)
    else:
        tokenizer = None
    """
    if pretrained_lm == 'none':
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(lm_path)

    alg = 'graph'
    if model_type == 'Robust':
        network = RobustParser(hyps, num_pretrained, num_words, num_chars, num_pos,
                               num_rels, device=device, basic_word_embedding=basic_word_embedding,
                               embedd_word=word_table, embedd_char=char_table, 
                               pretrained_lm=pretrained_lm, lm_path=lm_path,
                               num_lans=num_lans)
    else:
        raise RuntimeError('Unknown model type: %s' % model_type)

    num_gpu = torch.cuda.device_count()
    logger.info("GPU Number: %d" % num_gpu)
    if num_gpu > 1:
        logger.info("Using Data Parallel")
        network = torch.nn.DataParallel(network)
    network.to(device)
    single_network = network if num_gpu <= 1 else network.module

    logger.info("Freeze Pre-trained Emb: %s" % (freeze))
    if freeze:
        if num_gpu > 1:
            freeze_embedding(network.module.word_embed)
        else:
            freeze_embedding(network.word_embed)

    if schedule == 'step':
        logger.info("Scheduler: %s, init lr=%.6f, lr decay=%.6f, decay_steps=%d, warmup_steps=%d" % (schedule, learning_rate, lr_decay, decay_steps, warmup_steps))
    elif schedule == 'attention':
        logger.info("Scheduler: %s, init lr=%.6f, warmup_steps=%d" % (schedule, learning_rate, warmup_steps))
    elif schedule == 'exponential':
        logger.info("Scheduler: %s, init lr=%.6f, lr decay=%.6f, warmup_steps=%d" % (schedule, learning_rate, lr_decay, warmup_steps))
    if not pretrained_lm == 'none':
        optim_parameters = [{'params':single_network._basic_parameters()},
                            {'params':single_network.lm_encoder.parameters(), 'lr':args.lm_lr}]
        logger.info("Language model lr: %.6f" % args.lm_lr)
    else:
        optim_parameters = single_network._basic_parameters() #single_network.parameters()
    optimizer, scheduler = get_optimizer(optim_parameters, optim, learning_rate, lr_decay, 
                                betas, eps, amsgrad, weight_decay, warmup_steps,
                                schedule, hidden_size, decay_steps)


    logger.info("Reading Data")
    if alg == 'graph':
        if not args.mix_datasets:
            data_train = data_reader.read_bucketed_data(train_path, word_alphabet, char_alphabet, pos_alphabet, rel_alphabet, 
                                                        normalize_digits=args.normalize_digits, symbolic_root=True,
                                                        pre_alphabet=pretrained_alphabet, pos_idx=args.pos_idx,
                                                        lans=lans_train, lan_alphabet=language_alphabet)
            data_dev = data_reader.read_data(dev_path, word_alphabet, char_alphabet, pos_alphabet, rel_alphabet,
                                                        normalize_digits=args.normalize_digits, symbolic_root=True,
                                                        pre_alphabet=pretrained_alphabet, pos_idx=args.pos_idx,
                                                        lans=lans_dev, lan_alphabet=language_alphabet)
            data_test = data_reader.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, rel_alphabet,
                                                        normalize_digits=args.normalize_digits, symbolic_root=True,
                                                        pre_alphabet=pretrained_alphabet, pos_idx=args.pos_idx,
                                                        lans=lans_test, lan_alphabet=language_alphabet)
        else:
            data_train = data_reader.read_bucketed_data(train_path, word_alphabet, char_alphabet, pos_alphabet, rel_alphabet, 
                                                        normalize_digits=args.normalize_digits, symbolic_root=True,
                                                        pre_alphabet=pretrained_alphabet, pos_idx=args.pos_idx)
            data_dev = data_reader.read_data(dev_path, word_alphabet, char_alphabet, pos_alphabet, rel_alphabet,
                                                        normalize_digits=args.normalize_digits, symbolic_root=True,
                                                        pre_alphabet=pretrained_alphabet, pos_idx=args.pos_idx)
            data_test = data_reader.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, rel_alphabet,
                                                        normalize_digits=args.normalize_digits, symbolic_root=True,
                                                        pre_alphabet=pretrained_alphabet, pos_idx=args.pos_idx)
    
    if not args.mix_datasets:
        num_data = sum([sum(d) for d in data_train[1]])
    else:
        num_data = sum(data_train[1])
    logger.info("training: #training data: %d, batch: %d, unk replace: %.2f" % (num_data, batch_size, unk_replace))

    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, rel_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, rel_alphabet)
    
    best_ucorrect = 0.0
    best_lcorrect = 0.0
    best_ucomlpete = 0.0
    best_lcomplete = 0.0

    best_ucorrect_nopunc = 0.0
    best_lcorrect_nopunc = 0.0
    best_ucomlpete_nopunc = 0.0
    best_lcomplete_nopunc = 0.0
    best_root_correct = 0.0
    best_total = 0
    best_total_nopunc = 0
    best_total_inst = 0
    best_total_root = 0

    best_epoch = 0

    test_ucorrect = 0.0
    test_lcorrect = 0.0
    test_ucomlpete = 0.0
    test_lcomplete = 0.0

    test_ucorrect_nopunc = 0.0
    test_lcorrect_nopunc = 0.0
    test_ucomlpete_nopunc = 0.0
    test_lcomplete_nopunc = 0.0
    test_root_correct = 0.0
    test_total = 0
    test_total_nopunc = 0
    test_total_inst = 0
    test_total_root = 0

    patient = 0
    num_epochs_without_improvement = 0
    beam = args.beam
    reset = args.reset
    num_batches = num_data // batch_size + 1
    if optim == 'adamw':
        opt_info = 'adamw, betas=(%.1f, %.3f), eps=%.1e, amsgrad=%s' % (betas[0], betas[1], eps, amsgrad)
    elif optim == 'adam':
        opt_info = 'adam, betas=(%.1f, %.3f), eps=%.1e' % (betas[0], betas[1], eps)
    elif optim == 'sgd':
        opt_info = 'sgd, momentum=0.9, nesterov=True'
    if not args.mix_datasets:
        iterate = multi_language_iterate_data
        multi_lan_iter = True
    else:
        iterate = iterate_data
        multi_lan_iter = False
        lan_id = None
    for epoch in range(1, num_epochs + 1):
        num_epochs_without_improvement += 1
        start_time = time.time()
        train_loss = 0.
        train_arc_loss = 0.
        train_rel_loss = 0.
        num_insts = 0
        num_words = 0
        num_back = 0
        num_nans = 0
        overall_arc_correct, overall_rel_correct, overall_total_arcs = 0, 0, 0
        network.train()
        lr = scheduler.get_lr()[0]
        total_step = scheduler.get_total_step()
        print('Epoch %d, Step %d (%s, scheduler: %s, lr=%.6f, lr decay=%.6f, grad clip=%.1f, l2=%.1e): ' % (epoch, total_step, opt_info,  schedule, lr, lr_decay, grad_clip, weight_decay))
        if not pretrained_lm == 'none':
            print ('language model lr=%.6f' % scheduler.get_lr()[1])
        #if args.cuda:
        #    torch.cuda.empty_cache()
        gc.collect()
        #for step, data in enumerate(iterate_data(data_train, batch_size, bucketed=True, unk_replace=unk_replace, shuffle=True)):
        for step, data in enumerate(iterate(data_train, batch_size, bucketed=True, unk_replace=unk_replace, shuffle=True, switch_lan=True)):
            if not args.mix_datasets:
                lan_id, data = data
                lan_id = torch.LongTensor([lan_id]).to(device)
                #print ("lan_id:",lan_id)
            optimizer.zero_grad()
            words = data['WORD'].to(device)
            pres = data['PRETRAINED'].to(device)
            chars = data['CHAR'].to(device)
            postags = data['POS'].to(device)
            heads = data['HEAD'].to(device)
            nbatch = words.size(0)
            if not pretrained_lm == 'none':
                bpes, first_idx = convert_tokens_to_ids(tokenizer, data['SRC'])
                bpes = bpes.to(device)
                first_idx = first_idx.to(device)
                assert first_idx.size() == words.size()
                #print (bpes)
                #print (first_idx)
            else:
                bpes = first_idx = None
            if alg == 'graph':
                
                rels = data['TYPE'].to(device)
                masks = data['MASK'].to(device)
                nwords = masks.sum() - nbatch
                losses, statistics = network(words, pres, chars, postags, heads, rels, 
                            mask=masks, bpes=bpes, first_idx=first_idx, lan_id=lan_id)
            arc_loss, rel_loss = losses
            arc_loss = arc_loss.sum()
            rel_loss = rel_loss.sum()
            loss_total = arc_loss + rel_loss

            arc_correct, rel_correct, total_arcs = statistics
            overall_arc_correct += arc_correct.sum().cpu().numpy()
            overall_rel_correct += rel_correct.sum().cpu().numpy()
            overall_total_arcs += total_arcs.sum().cpu().numpy()
            
            if loss_type_token:
                loss = loss_total.div(nwords)
            else:
                loss = loss_total.div(nbatch)
            loss.backward()
            if grad_clip > 0:
                grad_norm = clip_grad_norm_(network.parameters(), grad_clip)
            else:
                grad_norm = total_grad_norm(network.parameters())

            if math.isnan(grad_norm):
                num_nans += 1
            else:
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    num_insts += nbatch
                    num_words += nwords
                    train_loss += loss_total.item()
                    train_arc_loss += arc_loss.item()
                    train_rel_loss += rel_loss.item()

            # update log
            if step % 100 == 0:
                #torch.cuda.empty_cache()
                if not noscreen: 
                    sys.stdout.write("\b" * num_back)
                    sys.stdout.write(" " * num_back)
                    sys.stdout.write("\b" * num_back)
                    curr_lr = scheduler.get_lr()[0]
                    num_insts = max(num_insts, 1)
                    num_words = max(num_words, 1)
                    log_info = '[%d/%d (%.0f%%) lr=%.6f (%d)] loss: %.4f (%.4f), arc: %.4f (%.4f), rel: %.4f (%.4f)' % (step, num_batches, 100. * step / num_batches, curr_lr, num_nans,
                                                                                                                         train_loss / num_insts, train_loss / num_words,
                                                                                                                         train_arc_loss / num_insts, train_arc_loss / num_words,
                                                                                                                        train_rel_loss / num_insts, train_rel_loss / num_words)
                    sys.stdout.write(log_info)
                    sys.stdout.flush()
                    num_back = len(log_info)
        if not noscreen: 
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)

        train_uas = float(overall_arc_correct) * 100.0 / overall_total_arcs
        train_lacc = float(overall_rel_correct) * 100.0 / overall_total_arcs
        print('total: %d (%d), epochs w/o improve:%d, nans:%d, uas: %.2f%%, lacc: %.2f%%,  loss: %.4f (%.4f), arc: %.4f (%.4f), rel: %.4f (%.4f), time: %.2fs' % (num_insts, num_words,
                                                                                                       num_epochs_without_improvement, num_nans, train_uas, train_lacc,
                                                                                                       train_loss / num_insts, train_loss / num_words,
                                                                                                       train_arc_loss / num_insts, train_arc_loss / num_words,
                                                                                                       train_rel_loss / num_insts, train_rel_loss / num_words,
                                                                                                       time.time() - start_time))
        print('-' * 125)

        if epoch % eval_every == 0:
            # evaluate performance on dev data
            with torch.no_grad():
                pred_filename = os.path.join(result_path, 'pred_dev%d' % epoch)
                #pred_writer.start(pred_filename)
                #gold_filename = os.path.join(result_path, 'gold_dev%d' % epoch)
                #gold_writer.start(gold_filename)

                print('Evaluating dev:')
                dev_stats, dev_stats_nopunct, dev_stats_root = eval(alg, data_dev, single_network, pred_writer, gold_writer, punct_set, word_alphabet, pos_alphabet, device, 
                                                                    beam=beam, batch_size=args.eval_batch_size, write_to_tmp=False, prev_best_lcorr=best_lcorrect_nopunc,
                                                                    prev_best_ucorr=best_ucorrect_nopunc, pred_filename=pred_filename, tokenizer=tokenizer, 
                                                                    multi_lan_iter=multi_lan_iter)

                #pred_writer.close()
                #gold_writer.close()

                dev_ucorr, dev_lcorr, dev_ucomlpete, dev_lcomplete, dev_total = dev_stats
                dev_ucorr_nopunc, dev_lcorr_nopunc, dev_ucomlpete_nopunc, dev_lcomplete_nopunc, dev_total_nopunc = dev_stats_nopunct
                dev_root_corr, dev_total_root, dev_total_inst = dev_stats_root

                if best_lcorrect_nopunc < dev_lcorr_nopunc or (best_lcorrect_nopunc == dev_lcorr_nopunc and best_ucorrect_nopunc < dev_ucorr_nopunc):
                    num_epochs_without_improvement = 0
                    best_ucorrect_nopunc = dev_ucorr_nopunc
                    best_lcorrect_nopunc = dev_lcorr_nopunc
                    best_ucomlpete_nopunc = dev_ucomlpete_nopunc
                    best_lcomplete_nopunc = dev_lcomplete_nopunc

                    best_ucorrect = dev_ucorr
                    best_lcorrect = dev_lcorr
                    best_ucomlpete = dev_ucomlpete
                    best_lcomplete = dev_lcomplete

                    best_root_correct = dev_root_corr
                    best_total = dev_total
                    best_total_nopunc = dev_total_nopunc
                    best_total_root = dev_total_root
                    best_total_inst = dev_total_inst

                    best_epoch = epoch
                    patient = 0
                    torch.save(single_network.state_dict(), model_name)

                    pred_filename = os.path.join(result_path, 'pred_test%d' % epoch)
                    pred_writer.start(pred_filename)
                    #gold_filename = os.path.join(result_path, 'gold_test%d' % epoch)
                    #gold_writer.start(gold_filename)

                    print('Evaluating test:')
                    test_stats, test_stats_nopunct, test_stats_root = eval(alg, data_test, 
                            single_network, pred_writer, gold_writer, punct_set, word_alphabet, 
                            pos_alphabet, device, beam=beam, batch_size=args.eval_batch_size, 
                            tokenizer=tokenizer, multi_lan_iter=multi_lan_iter)

                    test_ucorrect, test_lcorrect, test_ucomlpete, test_lcomplete, test_total = test_stats
                    test_ucorrect_nopunc, test_lcorrect_nopunc, test_ucomlpete_nopunc, test_lcomplete_nopunc, test_total_nopunc = test_stats_nopunct
                    test_root_correct, test_total_root, test_total_inst = test_stats_root

                    pred_writer.close()
                    #gold_writer.close()
                else:
                    patient += 1

                print('-' * 125)
                print('best dev  W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                    best_ucorrect, best_lcorrect, best_total, best_ucorrect * 100 / best_total, best_lcorrect * 100 / best_total,
                    best_ucomlpete * 100 / dev_total_inst, best_lcomplete * 100 / dev_total_inst,
                    best_epoch))
                print('best dev  Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                    best_ucorrect_nopunc, best_lcorrect_nopunc, best_total_nopunc,
                    best_ucorrect_nopunc * 100 / best_total_nopunc, best_lcorrect_nopunc * 100 / best_total_nopunc,
                    best_ucomlpete_nopunc * 100 / best_total_inst, best_lcomplete_nopunc * 100 / best_total_inst,
                    best_epoch))
                print('best dev  Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
                    best_root_correct, best_total_root, best_root_correct * 100 / best_total_root, best_epoch))
                print('-' * 125)
                print('best test W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                    test_ucorrect, test_lcorrect, test_total, test_ucorrect * 100 / test_total, test_lcorrect * 100 / test_total,
                    test_ucomlpete * 100 / test_total_inst, test_lcomplete * 100 / test_total_inst,
                    best_epoch))
                print('best test Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                    test_ucorrect_nopunc, test_lcorrect_nopunc, test_total_nopunc,
                    test_ucorrect_nopunc * 100 / test_total_nopunc, test_lcorrect_nopunc * 100 / test_total_nopunc,
                    test_ucomlpete_nopunc * 100 / test_total_inst, test_lcomplete_nopunc * 100 / test_total_inst,
                    best_epoch))
                print('best test Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
                    test_root_correct, test_total_root, test_root_correct * 100 / test_total_root, best_epoch))
                print('=' * 125)

                if reset > 0 and patient >= reset:
                    print ("### Reset optimizer state ###")
                    single_network.load_state_dict(torch.load(model_name, map_location=device))
                    scheduler.reset_state()
                    patient = 0

        if num_epochs_without_improvement >= patient_epochs:
            logger.info("More than %d epochs without improvement, exit!" % patient_epochs)
            exit()


def parse(args):
    logger = get_logger("Parsing")
    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    data_format = args.format
    if data_format == 'conllx':
        data_reader = conllx_data
        test_path = args.test
    elif data_format == 'ud':
        data_reader = ud_data
        test_path = args.test.split(':')
    else:
        print ("### Unrecognized data formate: %s ###" % data_format)
        exit()

    model_path = args.model_path
    model_name = os.path.join(model_path, 'model.pt')
    punctuation = args.punctuation
    pretrained_lm = args.pretrained_lm
    lm_path = args.lm_path
    print(args)

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(model_path, 'alphabets')
    assert os.path.exists(alphabet_path)
    word_alphabet, char_alphabet, pos_alphabet, rel_alphabet = data_reader.create_alphabets(alphabet_path, None, 
                                    normalize_digits=args.normalize_digits, pos_idx=args.pos_idx)
    pretrained_alphabet = utils.create_alphabet_from_embedding(alphabet_path)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_rels = rel_alphabet.size()
    num_pretrained = pretrained_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Rel Alphabet Size: %d" % num_rels)

    result_path = os.path.join(model_path, 'tmp')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    logger.info("loading network...")
    hyps = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
    model_type = hyps['model']
    assert model_type in ['Robust']

    num_lans = 1
    if not args.mix_datasets:
        lans_train = args.lan_train.split(':')
        lans_dev = args.lan_dev.split(':')
        lans_test = args.lan_test.split(':')
        #languages = set(lans_train + lans_dev + lans_test)
        language_alphabet = utils.creat_language_alphabet(alphabet_path)
        num_lans = language_alphabet.size()
        data_reader = multi_ud_data
    """
    if pretrained_lm == 'bert':
        tokenizer = BertTokenizer.from_pretrained(lm_path)
    elif pretrained_lm == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(lm_path)
    elif pretrained_lm == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained(lm_path)
    elif pretrained_lm == 'xlm-r':
        tokenizer = XLMRobertaTokenizer.from_pretrained(lm_path)
    else:
        tokenizer = None
    """
    if pretrained_lm == 'none':
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(lm_path)

    alg = 'graph'
    if model_type == 'Robust':
        network = RobustParser(hyps, num_pretrained, num_words, num_chars, num_pos,
                               num_rels, device=device, basic_word_embedding=args.basic_word_embedding, 
                               pretrained_lm=args.pretrained_lm, lm_path=args.lm_path,
                               num_lans=num_lans)
    else:
        raise RuntimeError('Unknown model type: %s' % model_type)

    network = network.to(device)
    network.load_state_dict(torch.load(model_name, map_location=device))

    logger.info("Reading Data")
    if alg == 'graph':
        if not args.mix_datasets:
            data_test = data_reader.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, 
                                            rel_alphabet, normalize_digits=args.normalize_digits, 
                                            symbolic_root=True, pre_alphabet=pretrained_alphabet, 
                                            pos_idx=args.pos_idx, lans=lans_test, 
                                            lan_alphabet=language_alphabet)
        else:
            data_test = data_reader.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, 
                                          rel_alphabet, normalize_digits=args.normalize_digits, 
                                          symbolic_root=True, pre_alphabet=pretrained_alphabet, 
                                          pos_idx=args.pos_idx)

    beam = args.beam
    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, rel_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, rel_alphabet)
    if args.output_filename:
        pred_filename = args.output_filename
    else:
        pred_filename = os.path.join(result_path, 'pred.txt')
    pred_writer.start(pred_filename)
    #gold_filename = os.path.join(result_path, 'gold.txt')
    #gold_writer.start(gold_filename)

    if not args.mix_datasets:
        multi_lan_iter = True
    else:
        multi_lan_iter = False
    with torch.no_grad():
        print('Parsing...')
        start_time = time.time()
        eval(alg, data_test, network, pred_writer, gold_writer, punct_set, word_alphabet, 
            pos_alphabet, device, beam, batch_size=args.batch_size, tokenizer=tokenizer, 
            multi_lan_iter=multi_lan_iter)
        print('Time: %.2fs' % (time.time() - start_time))

    pred_writer.close()
    #gold_writer.close()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--mode', choices=['train', 'parse'], required=True, help='processing mode')
    args_parser.add_argument('--seed', type=int, default=-1, help='Random seed for torch and numpy (-1 for random)')
    args_parser.add_argument('--config', type=str, help='config file')
    args_parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    args_parser.add_argument('--eval_batch_size', type=int, default=256, help='Number of sentences in each batch while evaluating')
    args_parser.add_argument('--patient_epochs', type=int, default=100, help='Max number of epochs to exit with no improvement')
    args_parser.add_argument('--loss_type', choices=['sentence', 'token'], default='sentence', help='loss type (default: sentence)')
    args_parser.add_argument('--optim', choices=['sgd', 'adamw', 'adam'], help='type of optimizer')
    args_parser.add_argument('--schedule', choices=['exponential', 'attention', 'step'], help='type of lr scheduler')
    args_parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    args_parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam')
    args_parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam')
    args_parser.add_argument('--eps', type=float, default=1e-8, help='epsilon for adam or adamax')
    args_parser.add_argument('--lr_decay', type=float, default=0.999995, help='Decay rate of learning rate')
    args_parser.add_argument('--decay_steps', type=int, default=5000, help='Number of steps to apply lr decay')
    args_parser.add_argument('--amsgrad', action='store_true', help='AMS Grad')
    args_parser.add_argument('--grad_clip', type=float, default=0, help='max norm for gradient clip (default 0: no clip')
    args_parser.add_argument('--warmup_steps', type=int, default=0, metavar='N', help='number of steps to warm up (default: 0)')
    args_parser.add_argument('--eval_every', type=int, default=100, help='eval every ? epochs')
    args_parser.add_argument('--noscreen', action='store_true', default=True, help='do not print middle log')
    args_parser.add_argument('--reset', type=int, default=10, help='Number of epochs to reset optimizer (default 10)')
    args_parser.add_argument('--weight_decay', type=float, default=0.0, help='weight for l2 norm decay')
    args_parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--pos_idx', type=int, default=4, choices=[3, 4], help='Index in Conll file line for Part-of-speech tags')
    args_parser.add_argument('--beam', type=int, default=1, help='Beam size for decoding')
    args_parser.add_argument('--basic_word_embedding', action='store_true', help='Whether to use extra randomly initialized trainable word embedding.')
    args_parser.add_argument('--word_embedding', choices=['glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words')
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters')
    args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--pretrained_lm', default='none', choices=['none', 'bert', 'roberta', 'xlm-r', 'electra', 'tc_bert', 'tc_roberta', 'tc_electra'], help='Pre-trained language model')
    args_parser.add_argument('--lm_path', help='path for pretrained language model')
    args_parser.add_argument('--lm_lr', type=float, default=2e-5, help='Learning rate of pretrained language model')
    args_parser.add_argument('--normalize_digits', default=False, action='store_true', help='normalize digits to 0 ?')
    args_parser.add_argument('--mix_datasets', default=False, action='store_true', help='Mix dataset from different languages ? (should be False for CPGLSTM)')
    args_parser.add_argument('--format', type=str, choices=['conllx', 'ud'], default='conllx', help='data format')
    args_parser.add_argument('--lan_train', type=str, default='en', help='lc for training files (split with \':\')')
    args_parser.add_argument('--lan_dev', type=str, default='en', help='lc for dev files (split with \':\')')
    args_parser.add_argument('--lan_test', type=str, default='en', help='lc for test files (split with \':\')')
    args_parser.add_argument('--train', help='path for training file.')
    args_parser.add_argument('--dev', help='path for dev file.')
    args_parser.add_argument('--test', help='path for test file.', required=True)
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--output_filename', type=str, help='output filename for parse')

    args = args_parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        parse(args)