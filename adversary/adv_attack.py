"""
Implementation of Graph-based dependency parsing.
"""

import os
import sys
import gc
import json
import pickle
import nltk
from nltk.corpus import wordnet as wn
import spacy
nlp = spacy.load('en_core_web_sm')

from functools import partial

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
from neuronlp2.io.common import PAD_CHAR, PAD, PAD_POS, PAD_TYPE, PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD

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

    accum_ucorr_err = 0.0
    accum_lcorr_err = 0.0
    accum_total_err = 0
    accum_ucorr_err_nopunc = 0.0
    accum_lcorr_err_nopunc = 0.0
    accum_total_err_nopunc = 0

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
        err_types = data['ERR_TYPE']
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

class Attacker(object):
    def __init__(self, model, candidates, vocab, synonyms, adv_lms=None, rel_ratio=0.5, fluency_ratio=0.2,
                max_perp_diff_per_token=0.8, perp_diff_thres=20 ,alphabets=None, tokenizer=None, 
                device=None, lm_device=None, symbolic_root=True, symbolic_end=False, mask_out_root=False, 
                batch_size=32, random_sub_if_no_change=False):
        self.model = model
        self.candidates = candidates
        self.synonyms = synonyms
        self.word2id = vocab
        self.id2word = {i:w for (w,i) in vocab.items()}
        if adv_lms is not None:
            self.adv_tokenizer, self.adv_lm = adv_lms
        else:
            self.adv_tokenizer, self.adv_lm = None, None
        if alphabets is not None:
            self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.rel_alphabet, self.pretrained_alphabet = alphabets
        self.tokenizer = tokenizer
        self.device = device
        self.lm_device = lm_device
        self.symbolic_root = symbolic_root
        self.symbolic_end = symbolic_end
        self.mask_out_root = mask_out_root
        assert rel_ratio >= 0 and rel_ratio <= 1
        self.rel_ratio = rel_ratio
        self.fluency_ratio = fluency_ratio
        self.max_perp_diff_per_token = max_perp_diff_per_token
        self.perp_diff_thres = perp_diff_thres
        self.batch_size = batch_size
        self.stop_words = nltk.corpus.stopwords.words('english')
        self.random_sub_if_no_change = random_sub_if_no_change
        logger = get_logger("Attacker")
        logger.info("Relation ratio:{}, Fluency ratio:{}".format(rel_ratio, fluency_ratio))
        logger.info("Max ppl difference per token:{}, ppl diff threshold:{}".format(max_perp_diff_per_token, perp_diff_thres))
        logger.info("Randomly substitute if no change:{}".format(self.random_sub_if_no_change))

    def _word2id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return None

    def _id2word(self, id):
        if id in self.id2word:
            return self.id2word[id]
        else:
            return None

    def str2id(self, tokens, tags):
        word_ids = [[self.word_alphabet.get_index(x) for x in s] for s in tokens]
        pre_ids = [[self.pretrained_alphabet.get_index(x) for x in s] for s in tokens]
        if self.model.hyps['input']['use_pos']:
            tag_ids = [[self.pos_alphabet.get_index(x) for x in s] for s in tags]
        else:
            tag_ids = None
        if not self.model.hyps['input']['use_char']:
            chars = None
        if not self.model.pretrained_lm == "none":
            bpes, first_idx = convert_tokens_to_ids(self.tokenizer, tokens)
            bpes = bpes.to(self.device)
            first_idx = first_idx.to(self.device)
        else:
            bpes, first_idx = None, None
        if not self.model.lan_emb_as_input:
            lan_id = None

        data_size = len(tokens)
        max_length = max([len(s) for s in tokens])
        wid_inputs = np.empty([data_size, max_length], dtype=np.int64)
        pre_inputs = np.empty([data_size, max_length], dtype=np.int64)
        tid_inputs = np.empty([data_size, max_length], dtype=np.int64)
        pid_inputs = np.empty([data_size, max_length], dtype=np.int64)
        masks = np.zeros([data_size, max_length], dtype=np.float32)

        for i in range(len(word_ids)):
            wids = word_ids[i]
            preids = pre_ids[i]
            inst_size = len(wids)
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            # pretrained ids
            pre_inputs[i, :inst_size] = preids
            pre_inputs[i, inst_size:] = PAD_ID_WORD
            # pos ids
            if tag_ids is not None:
                pids = tag_ids[i]
                pid_inputs[i, :inst_size] = pids
                pid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            if self.symbolic_end:
                # mask out the end token
                masks[i, :inst_size-1] = 1.0
            else:
                masks[i, :inst_size] = 1.0
            #for j, wid in enumerate(wids):
            #    if word_alphabet.is_singleton(wid):
            #        single[i, j] = 1
        if self.mask_out_root:
            masks[:,0] = 0

        words = torch.from_numpy(wid_inputs).to(self.device)
        pos = torch.from_numpy(pid_inputs).to(self.device)
        masks = torch.from_numpy(masks).to(self.device)
        pres = torch.from_numpy(pre_inputs).to(self.device)

        return words, pres, chars, pos, masks, bpes, first_idx, lan_id

    def get_prediction(self, tokens, tags):
        """
        Input:
            tokens: List[List[str]], (batch, seq_len)
            tags: List[List[str]], (batch, seq_len)
        Output:
            heads_pred: (batch, seq_len)
            rels_pred: (batch, seq_len)
        """
        words, pres, chars, pos, masks, bpes, first_idx, lan_id = self.str2id(tokens, tags)
        self.model.eval()
        with torch.no_grad():
            heads_pred, rels_pred = self.model.decode(words, pres, chars, pos, mask=masks, 
                bpes=bpes, first_idx=first_idx, lan_id=lan_id, leading_symbolic=common.NUM_SYMBOLIC_TAGS)
        return heads_pred, rels_pred

    def gen_importance_batch(self, tokens, tags):
        """
        Input:
            tokens: List[str], (seq_len)
            tags: List[str], (seq_len)
        Output:
            batch_tokens: List[List[str]], (batch, seq_len)
            batch_tags: List[List[str]], (batch, seq_len)
        """
        if not self.model.pretrained_lm == "none":
            unk_token = self.tokenizer.unk_token
        else: # this is defined in alphabet.py
            unk_token = '<_UNK>'
        batch_len = len(tokens)+1-self.symbolic_root
        batch_tokens = [tokens.copy() for _ in range(batch_len)]
        batch_tags = [tags.copy() for _ in range(batch_len)]
        unk_id = 1 if self.symbolic_root else 0
        for i in range(1, batch_len):
            batch_tokens[i][unk_id] = unk_token
            unk_id += 1
        return batch_tokens, batch_tags

    def calc_importance(self, batch_tokens, batch_tags, heads, rel_ids, debug=False):
        """
        Input:
            batch_tokens: List[List[str]], (batch, seq_len), the first line should be the original seq
            batch_tags: List[List[str]], (batch, seq_len), the first line should be the original seq
            heads: List[int], (seq_len)
            rel_ids: List[int], (seq_len)
        Output:
            importance: List[int], (seq_len), importance of each seq
            word_rank: List[int], (seq_len), word id ranked by importance
        """
        heads_pred, rels_pred = self.get_prediction(batch_tokens, batch_tags)
        heads_gold = np.tile(np.array(heads), (len(heads_pred),1))
        heads_change_mask = np.where(heads_pred != heads_gold, 1, 0)
        # this should minus the diff between original prediction (line 0) and gold
        heads_change = heads_change_mask.sum(axis=1)
        heads_change = heads_change - heads_change[0]
        if debug:
            print (batch_tokens)
            print ("gold heads:\n", heads_gold)
            print ("pred heads:\n", heads_pred)
            print ("mask:\n", heads_change_mask)
            print ("heads change:\n", heads_change)

        rels_gold = np.tile(np.array(rel_ids), (len(rels_pred),1))
        rels_change_mask = np.where(rels_pred != rels_gold, 1, 0)
        # this should minus the diff between original prediction (line 0) and gold
        rels_change = rels_change_mask.sum(axis=1)
        rels_change = rels_change - rels_change[0]
        if debug:
            print ("gold rels:\n", rel_ids)
            print ("pred rels:\n", rels_pred)
            print ("mask:\n", rels_change_mask)
            print ("rels change:\n", rels_change)
        
        importance = (1-self.rel_ratio) * heads_change + self.rel_ratio * rels_change
        return importance

    def calc_word_rank(self, tokens, tags, heads, rel_ids, debug=False):
        batch_tokens, batch_tags = self.gen_importance_batch(tokens, tags)
        importance = self.calc_importance(batch_tokens, batch_tags, heads, rel_ids, debug)
        word_rank = (-importance).argsort()
        if debug:
            print ("importance:\n", importance)
            print ("word_rank:\n", word_rank)
        return word_rank

    def gen_cand_batch(self, tokens, cands, idx, tags):
        """
        Input:
            tokens: List[str], (seq_len)
            tags: List[str], (seq_len)
        Output:
            batch_tokens: List[List[str]], (batch, seq_len)
            batch_tags: List[List[str]], (batch, seq_len)
        """
        batch_len = len(cands)+1
        batch_tokens = [tokens.copy() for _ in range(batch_len)]
        batch_tags = [tags.copy() for _ in range(batch_len)]
        for i in range(1, batch_len):
            batch_tokens[i][idx] = cands[i-1]
        return batch_tokens, batch_tags

    def get_change_score(self, tokens, cands, idx, tags, heads, rel_ids, debug=False):
        batch_tokens, batch_tags = self.gen_cand_batch(tokens, cands, idx, tags)
        # (cand_size+1), the 1st is the original sentence
        change_score = self.calc_importance(batch_tokens, batch_tags, heads, rel_ids, debug)
        # ignore the original sent
        change_score = change_score[1:]
        if debug:
            #print ("batch tokens:\n", batch_tokens)
            print ("importance:\n", change_score)
            #print ("word_rank:\n", word_rank)
        return change_score

    def get_batch(self, input_ids):
        # (cand_size+1, seq_len)
        data_size = input_ids.size()[0]
        for start_idx in range(0, data_size, self.batch_size):
            excerpt = slice(start_idx, start_idx + self.batch_size)
            yield input_ids[excerpt, :]

    def calc_perplexity(self, tokens):
        if self.symbolic_root:
            lines = [' '.join(t[1:]) for t in tokens]
        else:
            lines = [' '.join(t) for t in tokens]
        batch_encoding = self.adv_tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=128)
        examples = [torch.tensor(b,dtype=torch.long) for b in batch_encoding["input_ids"]]
        input_ids = torch.nn.utils.rnn.pad_sequence(examples, batch_first=True)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        perp_list = []
        for batch in self.get_batch(input_ids):
            batch = batch.to(self.lm_device)
            outputs = self.adv_lm(batch)
            # (batch_size, seq_len, voc_size)
            logits = outputs[0]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch[..., 1:].contiguous()
            # (batch, seq_len)
            loss = loss_fct(shift_logits.transpose(1,2), shift_labels)
            # (batch)
            loss = loss.mean(-1)
            perplexity = torch.exp(loss).cpu().numpy()
            perp_list.append(perplexity)
        perplexity = np.concatenate(perp_list, axis=0)
        return perplexity

    def get_perp_diff(self, tokens, cands, idx, debug=False):
        batch_tokens, _ = self.gen_cand_batch(tokens, cands, idx, tokens)
        # (cand_size+1), the first is the original sentence
        perplexity = self.calc_perplexity(batch_tokens)
        if debug:
            for perp, tokens in zip(perplexity, batch_tokens):
                print ("sent (perp={}):\n".format(perp), " ".join(tokens[:idx])+" </"+tokens[idx]+"/> "+" ".join(tokens[idx+1:]))
        # (cand_size)
        perp_diff = perplexity[1:] - perplexity[0]
        return perp_diff

    def filter_cands(self, tokens, cands, idx, debug=False):
        new_cands, new_perp_diff = [], []
        # (cand_size)
        perp_diff = self.get_perp_diff(tokens, cands, idx)
        for i in range(len(cands)):
            if perp_diff[i] <= self.perp_diff_thres:
                new_cands.append(cands[i])
                new_perp_diff.append(perp_diff[i])
        return new_cands, np.array(new_perp_diff), perp_diff

    def get_best_cand(self, score, change_score):
        cand_rank = (-score).argsort()
        for i in range(len(score)):
            cand_idx = cand_rank[i]
            if change_score[cand_idx] > 0:
                return cand_idx
        return None

    def get_synonyms(self, token, tag):
        if token not in self.synonyms:
            return []
        if tag in self.synonyms[token]:
            return self.synonyms[token][tag]
        else:
            return []

    def get_sememe_cands(self, token, tag):
        tag_list = ['JJ', 'NN', 'RB', 'VB']
        if self._word2id(token) not in range(1, 50000):
            return []
        if token in self.stop_words:
            return []
        if tag[:2] not in tag_list:
            return []
        if tag[:2] == 'JJ':
            pos = 'adj'
        elif tag[:2] == 'NN':
            pos = 'noun'
        elif tag[:2] == 'RB':
            pos = 'adv'
        else:
            pos = 'verb'
        if pos in self.candidates[self._word2id(token)]:
            return [self._id2word(neighbor) for neighbor in self.candidates[self._word2id(token)][pos]]
        else:
            return []

    def get_candidate_set(self, token, tag):
        sememe_cands = self.get_sememe_cands(token, tag)
        #print ("sememe:", sememe_cands)
        synonyms = self.get_synonyms(token, tag)
        #print ("syn:", synonyms)
        candidate_set = sememe_cands
        for syn in synonyms:
            if syn not in candidate_set:
                candidate_set.append(syn)
        return candidate_set
        
    def attack(self, tokens, tags, heads, rel_ids, debug=False):
        """
        Input:
            tokens: List[str], (seq_len)
            tags: List[str], (seq_len)
            heads: List[int], (seq_len)
            rel_ids: List[int], (seq_len)
        Output:
        """
        adv_tokens = tokens.copy()
        word_rank = self.calc_word_rank(tokens, tags, heads, rel_ids, debug==2)
        x_len = len(tokens)
        tag_list = ['JJ', 'NN', 'RB', 'VB']
        neigbhours_list = []
        #stop_words = nltk.corpus.stopwords.words('english')
        for i in range(x_len):
            #print (adv_tokens[i], self._word2id(adv_tokens[i]))
            neigbhours_list.append(self.get_candidate_set(adv_tokens[i], tags[i]))
        neighbours_len = [len(x) for x in neigbhours_list]
        #print (neigbhours_list)
        if np.sum(neighbours_len) == 0:
            return None
        change_edit_ratio = -1
        total_change_score = 0
        total_perp_diff = 0.0
        total_score = 0
        num_edit = 0
        max_perp_diff = x_len * self.max_perp_diff_per_token

        if debug == 3:
            #print ("tokens:\n", adv_tokens)
            print ("importance rank:\n", word_rank)

        for idx in word_rank:
            if neighbours_len[idx] == 0:
                if debug == 3:
                    print ("--------------------------")
                    print ("Idx={}({}), no cands, continue".format(idx, tokens[idx]))
                continue
            # skip the edit for ROOT
            if self.symbolic_root and idx == 0: continue
            cands = neigbhours_list[idx]
            all_cands = cands.copy()
            # filter with language model
            if self.adv_lm is not None:
                cands, perp_diff, all_perp_diff = self.filter_cands(adv_tokens, cands, idx, debug==2)
                if len(cands) == 0:
                    if debug == 3:
                        print ("--------------------------")
                        print ("Idx={}({}), all perp_diff above thres, continue\ncands:{}\nperp_diff:{}".format(idx, tokens[idx], all_cands, all_perp_diff))
                    continue
                blocked_perp_diff = np.where(perp_diff>0, perp_diff, 0)
            # (cand_size)
            change_score = self.get_change_score(adv_tokens, cands, idx, tags, heads, rel_ids, debug==2)
            change_rank = (-change_score).argsort()
            # this means the biggest change is 0
            if change_score[change_rank[0]] <= 0:
                if debug == 3:
                    print ("--------------------------")
                    print ("Idx={}({}), no cand can make change, continue\ncands:{}\nchange_scores:{}".format(idx, tokens[idx], cands, change_score))
                continue
            if self.adv_lm is not None:
                # penalize the score for disfluency substitution
                # if the perplexity of new sent is lower than the original one, no bonus
                score = (1 - self.fluency_ratio) * change_score - self.fluency_ratio * blocked_perp_diff
            else:
                score = change_score
            best_cand_idx = self.get_best_cand(score, change_score)
            if best_cand_idx is None:
                print ("--------------------------")
                print ("Idx={}({}), can't find best cand, continue\ncands:{}\nchange_scores:{}".format(idx, tokens[idx], cands, change_score))
                if self.adv_lm is not None:
                        print ("perp diff: {}\nscores: {}".format(perp_diff, score))
                continue
            #cand_rank = (-score).argsort()
            best_cand = cands[best_cand_idx]
            best_c_score = change_score[best_cand_idx]
            best_score = score[best_cand_idx]
            new_ratio = (total_change_score + best_c_score) / (num_edit + 1)
            if (self.adv_lm is not None and total_perp_diff<=max_perp_diff) or (new_ratio > change_edit_ratio):
                change_edit_ratio = new_ratio
                num_edit += 1
                total_change_score += best_c_score
                total_score += best_score
                adv_tokens[idx] = best_cand
                if self.adv_lm is not None:
                    total_perp_diff += blocked_perp_diff[best_cand_idx]
                if debug == 3:
                    print ("--------------------------")
                    print ("Idx={}({}), chosen cand:{}, total_change_score:{}, change_edit_ratio:{}\ncands: {}\nchange_scores: {}".format(
                            idx, tokens[idx], best_cand, total_change_score, change_edit_ratio, cands, change_score))
                    if self.adv_lm is not None:
                        print ("perp diff: {}\nscores: {}".format(perp_diff, score))
            else:
                if debug == 3:
                    print ("------------Stopping------------")
                    print ("Idx={}({}), chosen cand:{}, total_change_score:{}, change_edit_ratio:{}\ncands: {}\nchange_scores: {}".format(
                            idx, tokens[idx], best_cand, total_change_score, change_edit_ratio, cands, change_score))
                    if self.adv_lm is not None:
                        print ("perp diff: {}\nscores: {}".format(perp_diff, score))
                break
        if adv_tokens == tokens:
            return None
        return adv_tokens, num_edit, total_score, total_change_score, total_perp_diff

def attack(attacker, alg, data, network, pred_writer, punct_set, word_alphabet, pos_alphabet, 
        device, beam=1, batch_size=256, write_to_tmp=True, prev_best_lcorr=0, prev_best_ucorr=0,
        pred_filename=None, tokenizer=None, multi_lan_iter=False, debug=1):
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
        words = data['WORD']
        pres = data['PRETRAINED'].to(device)
        chars = data['CHAR'].to(device)
        postags = data['POS'].to(device)
        heads = data['HEAD'].numpy()
        rels = data['TYPE'].numpy()
        lengths = data['LENGTH'].numpy()
        err_types = data['ERR_TYPE']

        adv_words = words.clone()
        adv_src = []
        for i in range(len(lengths)):
            accum_total_sent += 1
            length = lengths[i]
            #adv_tokens = [word_alphabet.get_instance(w) for w in words[i][:length]]
            adv_tokens = data['SRC'][i]
            adv_postags = [pos_alphabet.get_instance(w) for w in postags[i][:length]]
            adv_heads = heads[i][:length]
            adv_rels = rels[i][:length]
            adv_rels[0] = 0
            if debug == 3: 
                print ("\n###############################")
                print ("Attacking sent-{}".format(int(accum_total_sent)-1))
                print ("tokens:\n", adv_tokens)
            if debug == 1: print ("original sent:", adv_tokens)
            result = attacker.attack(adv_tokens, adv_postags, adv_heads, adv_rels, debug=debug)
            if result is None:
                adv_src.append(adv_tokens)
                continue
            adv_tokens, num_edit, total_score, total_change_score, total_perp_diff = result
            accum_success_attack += 1
            accum_total_edit += num_edit
            accum_total_score += total_score
            accum_total_change_score += total_change_score
            accum_total_perp_diff += total_perp_diff
            if debug == 1: print ("adv sent:", adv_tokens)
            adv_src.append(adv_tokens)
            adv_words[i][:length] = torch.from_numpy(np.array([word_alphabet.get_index(w) for w in adv_tokens]))
        adv_words = adv_words.to(device)
        #print ("orig_words:\n{}\nadv_words:\n{}".format(words, adv_words))

        if tokenizer:
            bpes, first_idx = convert_tokens_to_ids(tokenizer, adv_src)
            bpes = bpes.to(device)
            first_idx = first_idx.to(device)
        else:
            bpes = first_idx = None

        if alg == 'graph':
            masks = data['MASK'].to(device)
            heads_pred, rels_pred = network.decode(adv_words, pres, chars, postags, mask=masks, 
                bpes=bpes, first_idx=first_idx, lan_id=lan_id, leading_symbolic=common.NUM_SYMBOLIC_TAGS)

        adv_words = adv_words.cpu().numpy()
        postags = postags.cpu().numpy()

        if write_to_tmp:
            pred_writer.write(words, postags, heads_pred, rels_pred, lengths, symbolic_root=True, src_words=data['SRC'] ,adv_words=adv_src)
        else:
            all_words.append(adv_words)
            all_postags.append(postags)
            all_heads_pred.append(heads_pred)
            all_rels_pred.append(rels_pred)
            all_lengths.append(lengths)
            all_src_words.append(adv_src)

        #gold_writer.write(words, postags, heads, rels, lengths, symbolic_root=True)
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

    print('Attack: success/total examples = %d/%d, Average score: %.2f, change score: %.2f, perp diff: %.2f, edit dist: %.2f, change-edit ratio: %.2f' % (
        accum_success_attack, accum_total_sent, accum_total_score/accum_total_sent, 
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

    if args.cand.endswith('.json'):
        cands = json.load(open(args.cand, 'r'))
        candidates = {int(i):dic for (i,dic) in cands.items()}
    else:
        candidates = pickle.load(open(args.cand, 'rb'))
    vocab = json.load(open(args.vocab, 'r'))
    synonyms = json.load(open(args.syn, 'r'))
    num_gpu = torch.cuda.device_count()
    if num_gpu >= 2:
        lm_device = torch.device('cuda', 1)
    else:
        lm_device = device
    logger.info("parser device:{}, lm device:{}".format(device, lm_device))
    if args.adv_lm_path is not None:
        adv_tokenizer = AutoTokenizer.from_pretrained(args.adv_lm_path)
        adv_lm = AutoModelWithLMHead.from_pretrained(args.adv_lm_path)
        adv_lm = adv_lm.to(lm_device)
        adv_lms = (adv_tokenizer,adv_lm)
    else:
        adv_lms = None
    alphabets = word_alphabet, char_alphabet, pos_alphabet, rel_alphabet, pretrained_alphabet
    attacker = Attacker(network, candidates, vocab, synonyms, adv_lms=adv_lms, rel_ratio=args.adv_rel_ratio, 
                        fluency_ratio=args.adv_fluency_ratio, max_perp_diff_per_token=args.max_perp_diff_per_token,
                        perp_diff_thres=args.perp_diff_thres,
                        alphabets=alphabets, tokenizer=tokenizer, device=device, lm_device=lm_device,
                        batch_size=args.adv_batch_size, random_sub_if_no_change=args.random_sub_if_no_change)
    #tokens = ["_ROOT", "The", "Dow", "fell", "22.6", "%", "on", "black", "Monday"]#, "."]
    #tags = ["_ROOT_POS", "DT", "NNP", "VBD", "CD", ".", "IN", "NNP", "NNP"]#, "."]
    #heads = [0, 2, 3, 0, 5, 3, 3, 8, 6]#, 3]
    #rels = [0, 3, 4, 5, 6, 7, 8, 9, 10]#, 11]
    #attacker.attack(tokens, tags, heads, rels, True)
    #exit()

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
    adv_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, rel_alphabet)
    if args.output_filename:
        pred_filename = args.output_filename
    else:
        pred_filename = os.path.join(result_path, 'pred.txt')
    pred_writer.start(pred_filename)
    if args.adv_filename:
        adv_filename = args.adv_filename
    else:
        adv_filename = os.path.join(result_path, 'adv.txt')
    adv_writer.start(adv_filename)
    #gold_filename = os.path.join(result_path, 'gold.txt')
    #gold_writer.start(gold_filename)

    if not args.mix_datasets:
        multi_lan_iter = True
    else:
        multi_lan_iter = False
    with torch.no_grad():
        print('Parsing Original Data...')
        start_time = time.time()
        eval(alg, data_test, network, pred_writer, gold_writer, punct_set, word_alphabet, 
            pos_alphabet, device, beam, batch_size=args.batch_size, tokenizer=tokenizer, 
            multi_lan_iter=multi_lan_iter)
        print('Time: %.2fs' % (time.time() - start_time))
    print ('\n------------------\n')
    with torch.no_grad():
        print('Attacking...')
        start_time = time.time()
        # debug = 1: show orig/adv tokens / debug = 2: show log inside attacker
        attack(attacker, alg, data_test, network, adv_writer, punct_set, word_alphabet, 
            pos_alphabet, device, beam, batch_size=args.batch_size, tokenizer=tokenizer, 
            multi_lan_iter=multi_lan_iter, debug=3)
        print('Time: %.2fs' % (time.time() - start_time))
        

    pred_writer.close()
    #gold_writer.close()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    #args_parser.add_argument('--mode', choices=['train', 'parse'], required=True, help='processing mode')
    args_parser.add_argument('--seed', type=int, default=-1, help='Random seed for torch and numpy (-1 for random)')
    args_parser.add_argument('--config', type=str, help='config file')
    args_parser.add_argument('--vocab', type=str, help='vocab file for attacker')
    args_parser.add_argument('--cand', type=str, help='candidate file for attacker')
    args_parser.add_argument('--syn', type=str, help='synonym file for attacker')
    args_parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
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
    args_parser.add_argument('--pretrained_lm', default='none', choices=['none', 'bert', 'bart', 'roberta', 'xlm-r', 'electra', 'tc_bert', 'tc_bart', 'tc_roberta', 'tc_electra'], help='Pre-trained language model')
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
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--adv_lm_path', help='path for pretrained language model (gpt2) for adv filtering')
    args_parser.add_argument('--output_filename', type=str, help='output filename for parse')
    args_parser.add_argument('--adv_filename', type=str, help='output adversarial filename')
    args_parser.add_argument('--adv_rel_ratio', type=float, default=0.5, help='Relation importance in adversarial attack')
    args_parser.add_argument('--adv_fluency_ratio', type=float, default=0.2, help='Fluency importance in adversarial attack')
    args_parser.add_argument('--max_perp_diff_per_token', type=float, default=0.8, help='Maximum allowed perplexity difference per token in adversarial attack')
    args_parser.add_argument('--perp_diff_thres', type=float, default=20.0, help='Perplexity difference threshold in adversarial attack')
    args_parser.add_argument('--adv_batch_size', type=int, default=16, help='Number of sentences in adv lm each batch')
    args_parser.add_argument('--random_sub_if_no_change', action='store_true', default=False, help='randomly substitute if no change')

    args = args_parser.parse_args()
    parse(args)
