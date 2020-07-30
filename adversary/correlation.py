import os
import sys
import gc
import json
import pickle

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

try:
    from allennlp.modules.elmo import batch_to_ids
except:
    print ("can not import batch_to_ids!")

import time
import argparse
import math
import numpy as np
import torch
import random
#from torch.optim.adamw import AdamW
from torch.optim import SGD, Adam, AdamW
from torch.nn.utils import clip_grad_norm_
from neuronlp2.nn.utils import total_grad_norm
from neuronlp2.io import get_logger, conllx_data, ud_data, conllx_stacked_data #, iterate_data
from neuronlp2.io import ud_stacked_data
from neuronlp2.models.robust_parsing import RobustParser
from neuronlp2.models.stack_pointer import StackPtrParser
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
from neuronlp2.io.common import PAD_CHAR, PAD, PAD_POS, PAD_TYPE, PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD
from adversary.blackbox_attacker import BlackBoxAttacker
from adversary.random_attacker import RandomAttacker
from adversary.graybox_attacker import GrayBoxAttacker
from adversary.graybox_single_attacker import GrayBoxSingleAttacker

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


def correlate(alg, orig_data, adv_data, network, punct_set, word_alphabet, pos_alphabet, 
        device, beam=1, batch_size=256, write_to_tmp=True, prev_best_lcorr=0, prev_best_ucorr=0,
        pred_filename=None, tokenizer=None, multi_lan_iter=False, debug=1, pretrained_alphabet=None):
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

    accum_ucorr_err = 0.0
    accum_lcorr_err = 0.0
    accum_total_err = 0
    accum_ucorr_err_nopunc = 0.0
    accum_lcorr_err_nopunc = 0.0
    accum_total_err_nopunc = 0

    accum_total_edit = 0
    accum_total_change_score = 0.0
    accum_total_score = 0.0
    accum_total_perp_diff = 0.0
    accum_success_attack = 0
    accum_total_sent = 0.0
    accum_total_head_change = 0.0
    accum_total_rel_change = 0.0

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

    for o_data, a_data in zip(iterate(orig_data, batch_size), iterate(adv_data, batch_size)):
        if multi_lan_iter:
            lan_id, o_data = o_data
            _, a_data = a_data
            lan_id = torch.LongTensor([lan_id]).to(device)
        orig_words = o_data['WORD'].to(device)
        orig_pres = o_data['PRETRAINED'].to(device)
        orig_chars = o_data['CHAR'].to(device)
        orig_postags = o_data['POS'].to(device)
        heads = o_data['HEAD'].numpy()
        rels = o_data['TYPE'].numpy()
        lengths = o_data['LENGTH'].numpy()
        orig_srcs = o_data['SRC']
        if orig_words.size()[0] == 1 and len(orig_srcs) > 1:
            orig_srcs = [orig_srcs]
        if network.pretrained_lm == 'elmo':
            orig_bpes = batch_to_ids(orig_srcs)
            orig_bpes = orig_bpes.to(device)
            orig_first_idx = None
        elif tokenizer:
            orig_bpes, orig_first_idx = convert_tokens_to_ids(tokenizer, orig_srcs)
            orig_bpes = orig_bpes.to(device)
            orig_first_idx = orig_first_idx.to(device)
        else:
            orig_bpes = orig_first_idx = None

        adv_words = a_data['WORD'].to(device)
        adv_pres = a_data['PRETRAINED'].to(device)
        adv_chars = a_data['CHAR'].to(device)
        adv_postags = a_data['POS'].to(device)
        adv_lengths = a_data['LENGTH'].numpy()
        err_types = a_data['ERR_TYPE']
        adv_srcs = a_data['SRC']
        if adv_words.size()[0] == 1 and len(adv_srcs) > 1:
            adv_srcs = [adv_srcs]
        if network.pretrained_lm == 'elmo':
            adv_bpes = batch_to_ids(adv_srcs)
            adv_bpes = adv_bpes.to(device)
            adv_first_idx = None
        elif tokenizer:
            adv_bpes, adv_first_idx = convert_tokens_to_ids(tokenizer, adv_srcs)
            adv_bpes = adv_bpes.to(device)
            adv_first_idx = adv_first_idx.to(device)
        else:
            adv_bpes = adv_first_idx = None

        if alg == 'graph':
            orig_masks = o_data['MASK'].to(device)
            adv_masks = a_data['MASK'].to(device)
        else:
            orig_masks = o_data['MASK_ENC'].to(device)
            adv_masks = a_data['MASK_ENC'].to(device)

        assert len(lengths) == len(adv_lengths)

        if alg == 'graph':
            orig_heads_pred, orig_rels_pred = network.decode(orig_words, orig_pres, orig_chars, orig_postags, mask=orig_masks, 
                bpes=orig_bpes, first_idx=orig_first_idx, lan_id=lan_id, leading_symbolic=common.NUM_SYMBOLIC_TAGS)
            adv_heads_pred, adv_rels_pred = network.decode(adv_words, adv_pres, adv_chars, adv_postags, mask=adv_masks, 
                bpes=adv_bpes, first_idx=adv_first_idx, lan_id=lan_id, leading_symbolic=common.NUM_SYMBOLIC_TAGS)
        else:
            orig_heads_pred, orig_rels_pred = network.decode(orig_words, orig_pres, orig_chars, orig_postags, mask=orig_masks, 
                bpes=orig_bpes, first_idx=orig_first_idx, lan_id=lan_id, beam=beam, leading_symbolic=common.NUM_SYMBOLIC_TAGS)
            adv_heads_pred, adv_rels_pred = network.decode(adv_words, adv_pres, adv_chars, adv_postags, mask=adv_masks, 
                bpes=adv_bpes, first_idx=adv_first_idx, lan_id=lan_id, beam=beam, leading_symbolic=common.NUM_SYMBOLIC_TAGS)

        for i in range(len(lengths)):
            accum_total_sent += 1
            assert lengths[i] == adv_lengths[i]
            gold_head = heads[i:i+1]
            gold_rel = rels[i:i+1]
            length = lengths[i:i+1]

            orig_w = orig_words[i:i+1]
            orig_t = orig_postags[i:i+1]
            orig_head = orig_heads_pred[i:i+1]
            orig_rel = orig_rels_pred[i:i+1]

            stats, stats_nopunc, err_stats, err_nopunc_stats, stats_root, num_inst = parser.eval(
                                    orig_w, orig_t, orig_head, orig_rel, gold_head, gold_rel,
                                    word_alphabet, pos_alphabet, length, punct_set=punct_set, 
                                    symbolic_root=True)
            orig_ucorr, orig_lcorr, orig_total, orig_ucm, orig_lcm = stats
            orig_ucorr_nopunc, orig_lcorr_nopunc, orig_total_nopunc, orig_ucm_nopunc, orig_lcm_nopunc = stats_nopunc


            adv_w = adv_words[i:i+1]
            adv_t = adv_postags[i:i+1]
            adv_head = adv_heads_pred[i:i+1]
            adv_rel = adv_rels_pred[i:i+1]

            stats, stats_nopunc, err_stats, err_nopunc_stats, stats_root, num_inst = parser.eval(
                                    adv_w, adv_t, adv_head, adv_rel, gold_head, gold_rel,
                                    word_alphabet, pos_alphabet, length, punct_set=punct_set, 
                                    symbolic_root=True)
            adv_ucorr, adv_lcorr, adv_total, adv_ucm, adv_lcm = stats
            adv_ucorr_nopunc, adv_lcorr_nopunc, adv_total_nopunc, adv_ucm_nopunc, adv_lcm_nopunc = stats_nopunc

            print ("orig sent:{}\nadv sent:{}".format(' '.join(orig_srcs[i]), ' '.join(adv_srcs[i])))
            print ("orig uas:{}, las:{}, adv uas:{}, las:{}".format(orig_ucorr/orig_total, orig_lcorr/orig_total, adv_ucorr/adv_total, adv_lcorr/adv_total))

        return 0

        if write_to_tmp:
            pred_writer.write(words, postags, heads_pred, rels_pred, lengths, symbolic_root=True, src_words=data['SRC'] ,adv_words=adv_src)
        else:
            all_words.append(adv_words)
            all_postags.append(postags)
            all_heads_pred.append(heads_pred)
            all_rels_pred.append(rels_pred)
            all_lengths.append(lengths)
            all_src_words.append(adv_src)

        adv_gold_writer.write(words, postags, heads, rels, lengths, symbolic_root=True, src_words=data['SRC'] ,adv_words=adv_src)
        #print ("heads_pred:\n", heads_pred)
        #print ("rels_pred:\n", rels_pred)
        #print ("heads:\n", heads)
        #print ("err_types:\n", err_types)
        stats, stats_nopunc, err_stats, err_nopunc_stats, stats_root, num_inst = parser.eval(
                                    words, postags, heads_pred, rels_pred, heads, rels,
                                    word_alphabet, pos_alphabet, lengths, punct_set=punct_set, 
                                    symbolic_root=True, err_types=err_types)
        ucorr, lcorr, total, ucm, lcm = stats
        ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
        ucorr_err, lcorr_err, total_err = err_stats
        ucorr_err_nopunc, lcorr_err_nopunc, total_err_nopunc = err_nopunc_stats
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

        accum_ucorr_err += ucorr_err
        accum_lcorr_err += lcorr_err
        accum_total_err += total_err
        accum_ucorr_err_nopunc += ucorr_err_nopunc
        accum_lcorr_err_nopunc += lcorr_err_nopunc
        accum_total_err_nopunc += total_err_nopunc

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
    if accum_total_err == 0:
        accum_total_err = 1
    if accum_total_err_nopunc == 0:
        accum_total_err_nopunc = 1
    #print('Error Token: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
    #    accum_ucorr_err, accum_lcorr_err, accum_total_err, accum_ucorr_err * 100 / accum_total_err, accum_lcorr_err * 100 / accum_total_err))
    #print('Error Token Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
    #    accum_ucorr_err_nopunc, accum_lcorr_err_nopunc, accum_total_err_nopunc, 
    #    accum_ucorr_err_nopunc * 100 / accum_total_err_nopunc, accum_lcorr_err_nopunc * 100 / accum_total_err_nopunc))
    if accum_total_edit == 0:
        accum_total_edit = 1
    print('Attack: success/total examples = %d/%d\nTotal head change: %d, rel change: %d, change score: %.2f\nAverage score: %.2f, change score: %.2f, perp diff: %.2f, edit dist: %.2f, change-edit ratio: %.2f' % (
        accum_success_attack, accum_total_sent, 
        accum_total_head_change, accum_total_rel_change, accum_total_change_score,
        accum_total_score/accum_total_sent, 
        accum_total_change_score/accum_total_sent, accum_total_perp_diff/accum_total_sent, 
        accum_total_edit/accum_total_sent, accum_total_change_score/accum_total_edit))

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


def parse(args):

    logger = get_logger("Parsing")
    logger.info("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    data_format = args.format
    if data_format == 'conllx':
        data_reader = conllx_data
        test_path = args.test
        adv_path = args.adv
    elif data_format == 'ud':
        data_reader = ud_data
        test_path = args.test.split(':')
        adv_path = args.adv.split(':')
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

    logger.info("loading network...")
    hyps = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
    model_type = hyps['model']
    assert model_type in ['Robust', 'StackPtr']

    num_lans = 1
    if not args.mix_datasets:
        lans_train = args.lan_train.split(':')
        lans_dev = args.lan_dev.split(':')
        lans_test = args.lan_test.split(':')
        #languages = set(lans_train + lans_dev + lans_test)
        language_alphabet = utils.creat_language_alphabet(alphabet_path)
        num_lans = language_alphabet.size()
        data_reader = multi_ud_data

    if pretrained_lm in ['none','elmo']:
        tokenizer = None 
    else:
        tokenizer = AutoTokenizer.from_pretrained(lm_path)

    logger.info("##### Parser Type: {} #####".format(model_type))
    alg = 'transition' if model_type == 'StackPtr' else 'graph'
    if model_type == 'Robust':
        network = RobustParser(hyps, num_pretrained, num_words, num_chars, num_pos,
                               num_rels, device=device, basic_word_embedding=args.basic_word_embedding, 
                               pretrained_lm=args.pretrained_lm, lm_path=args.lm_path,
                               num_lans=num_lans)
    elif model_type == 'StackPtr':
        network = StackPtrParser(hyps, num_pretrained, num_words, num_chars, num_pos,
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
            orig_data = data_reader.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, 
                                            rel_alphabet, normalize_digits=args.normalize_digits, 
                                            symbolic_root=True, pre_alphabet=pretrained_alphabet, 
                                            pos_idx=args.pos_idx, lans=lans_test, 
                                            lan_alphabet=language_alphabet)
            adv_data = data_reader.read_data(adv_path, word_alphabet, char_alphabet, pos_alphabet, 
                                            rel_alphabet, normalize_digits=args.normalize_digits, 
                                            symbolic_root=True, pre_alphabet=pretrained_alphabet, 
                                            pos_idx=args.pos_idx, lans=lans_test, 
                                            lan_alphabet=language_alphabet)
        else:
            orig_data = data_reader.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, 
                                          rel_alphabet, normalize_digits=args.normalize_digits, 
                                          symbolic_root=True, pre_alphabet=pretrained_alphabet, 
                                          pos_idx=args.pos_idx)
            adv_data = data_reader.read_data(adv_path, word_alphabet, char_alphabet, pos_alphabet, 
                                          rel_alphabet, normalize_digits=args.normalize_digits, 
                                          symbolic_root=True, pre_alphabet=pretrained_alphabet, 
                                          pos_idx=args.pos_idx)
    elif alg == 'transition':
        prior_order = hyps['input']['prior_order']
        orig_data = ud_stacked_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, rel_alphabet,
                                            normalize_digits=args.normalize_digits, symbolic_root=True,
                                            pre_alphabet=pretrained_alphabet, pos_idx=args.pos_idx, 
                                            prior_order=prior_order)
        adv_data = ud_stacked_data.read_data(adv_path, word_alphabet, char_alphabet, pos_alphabet, rel_alphabet,
                                            normalize_digits=args.normalize_digits, symbolic_root=True,
                                            pre_alphabet=pretrained_alphabet, pos_idx=args.pos_idx, 
                                            prior_order=prior_order)

    beam = args.beam
    #pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, rel_alphabet)
    if args.output_filename:
        pred_filename = args.output_filename
    else:
        pred_filename = os.path.join(result_path, 'pred.conll')

    if alg == 'graph' and not args.mix_datasets:
        multi_lan_iter = True
    else:
        multi_lan_iter = False
    with torch.no_grad():
        print('Parsing...')
        start_time = time.time()
        # debug = 1: show orig/adv tokens / debug = 2: show log inside attacker
        correlate(alg, orig_data, adv_data, network, punct_set, word_alphabet, 
            pos_alphabet, device, beam, batch_size=args.batch_size, tokenizer=tokenizer, 
            multi_lan_iter=multi_lan_iter, debug=3, pretrained_alphabet=pretrained_alphabet)
        print('Time: %.2fs' % (time.time() - start_time))

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--seed', type=int, default=666, help='Random seed for torch and numpy (-1 for random)')
    args_parser.add_argument('--config', type=str, help='config file')
    args_parser.add_argument('--vocab', type=str, help='vocab file for attacker')
    args_parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    args_parser.add_argument('--eval_batch_size', type=int, default=256, help='Number of sentences in each batch while evaluating')
    args_parser.add_argument('--noscreen', action='store_true', default=True, help='do not print middle log')
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
    args_parser.add_argument('--pretrained_lm', default='none', choices=['none', 'elmo', 'bert', 'bart', 'roberta', 'xlm-r', 'electra', 'tc_bert', 'tc_bart', 'tc_roberta', 'tc_electra'], help='Pre-trained language model')
    args_parser.add_argument('--lm_path', help='path for pretrained language model')
    args_parser.add_argument('--normalize_digits', default=False, action='store_true', help='normalize digits to 0 ?')
    args_parser.add_argument('--mix_datasets', default=False, action='store_true', help='Mix dataset from different languages ? (should be False for CPGLSTM)')
    args_parser.add_argument('--format', type=str, choices=['conllx', 'ud'], default='conllx', help='data format')
    args_parser.add_argument('--lan_train', type=str, default='en', help='lc for training files (split with \':\')')
    args_parser.add_argument('--lan_dev', type=str, default='en', help='lc for dev files (split with \':\')')
    args_parser.add_argument('--lan_test', type=str, default='en', help='lc for test files (split with \':\')')
    args_parser.add_argument('--train', help='path for training file.')
    args_parser.add_argument('--dev', help='path for dev file.')
    args_parser.add_argument('--test', help='path for test file.', required=True)
    args_parser.add_argument('--adv', help='path for adv file.', required=True)
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--output_filename', type=str, help='output filename for parse')
    args_parser.add_argument('--adv_filename', type=str, help='output adversarial filename')
    args = args_parser.parse_args()
    parse(args)
