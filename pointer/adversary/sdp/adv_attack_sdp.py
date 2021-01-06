# coding=utf-8
import os
import sys
import gc
import json
import pickle
from tqdm import tqdm

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
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

from transformers import *
from pointer.adversary.sdp.blackbox_attacker_sdp import BlackBoxAttacker
from pointer.neuronlp2 import utils
from pointer.neuronlp2.io import *
from pointer.neuronlp2.models import NewStackPtrNet
from pointer.neuronlp2.tasks import parser


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


def eval(network,data_test,batch_size,word_alphabet,lemma_alphabet,pos_alphabet,beam,punct_set,best_epoch,logger):
    network.eval()
    test_ucorrect = 0.0
    test_lcorrect = 0.0
    test_total_pred = 0
    test_total_gold = 0
    test_total_inst = 0
    test_LF1 = 0.0
    test_UF1 = 0.0
    test_Uprecision = 0.0
    test_Lprecision = 0.0
    test_Urecall = 0.0
    test_Lrecall = 0.0
    start_time_test = time.time()
    for batch in tqdm(conllx_stacked_data.iterate_batch_stacked_variable(data_test, batch_size)):
        input_encoder, _ = batch
        word, lemma, char, pos, heads, types, masks, lengths = input_encoder
        heads_pred, types_pred, _, _ = network.decode(word, lemma, char, pos, mask=masks, length=lengths, beam=beam, leading_symbolic=conllx_stacked_data.NUM_SYMBOLIC_TAGS)

        word = word.data.detach()
        lemma = lemma.data.detach()
        pos = pos.data.detach()
        lengths = lengths.detach()
        heads = heads.data.detach()
        types = types.data.detach()

        # pred_writer.write(word, lemma, pos, heads_pred, types_pred, lengths, symbolic_root=True)
        # gold_writer.write(word, lemma, pos, heads, types, lengths, symbolic_root=True)

        ucorr, lcorr, total_gold, total_pred, num_inst = parser.evalF1(word, lemma, pos, heads_pred, types_pred, heads, types, word_alphabet, lemma_alphabet, pos_alphabet, lengths,
                                                                       punct_set=punct_set, symbolic_root=True)

        test_ucorrect += ucorr
        test_lcorrect += lcorr
        test_total_gold += total_gold
        test_total_pred += total_pred

        test_total_inst += num_inst

    end_time_test = time.time()
    lasted_time_test = end_time_test - start_time_test
    # pred_writer.close()
    # gold_writer.close()

    test_Uprecision = 0.
    test_Lprecision = 0.
    if test_total_pred != 0:
        test_Uprecision = test_ucorrect * 100 / test_total_pred
        test_Lprecision = test_lcorrect * 100 / test_total_pred
    test_Urecall = test_ucorrect * 100 / test_total_gold
    test_Lrecall = test_lcorrect * 100 / test_total_gold
    if test_Uprecision == 0. and test_Urecall == 0.:
        test_UF1 = 0
    else:
        test_UF1 = 2 * (test_Uprecision * test_Urecall) / (test_Uprecision + test_Urecall)
    if test_Lprecision == 0. and test_Lrecall == 0.:
        test_LF1 = 0
    else:
        test_LF1 = 2 * (test_Lprecision * test_Lrecall) / (test_Lprecision + test_Lrecall)
    logger.info('------------------------------------------------------------')
    logger.info('TIME TEST: %.2f,NUM SENTS TEST: %d,SPEED TEST: %.2f' % (lasted_time_test, test_total_inst,
                                                                         test_total_inst / lasted_time_test))
    logger.info('TEST: ucorr: %d, lcorr: %d, tot_gold: %d, tot_pred: %d, Uprec: %.2f%%, Urec: %.2f%%, Lprec: %.2f%%, Lrec: %.2f%%, '
                'UF1: %.2f%%, LF1: %.2f%% (epoch: %d)' % (test_ucorrect, test_lcorrect, test_total_gold, test_total_pred,
                                                          test_Uprecision, test_Urecall, test_Lprecision, test_Lrecall,
                                                          test_UF1, test_LF1, best_epoch))
    return (test_ucorrect, test_lcorrect, test_total_gold, test_total_pred, test_Uprecision, test_Urecall,
            test_Lprecision, test_Lrecall, test_UF1, test_LF1, best_epoch),\
           (lasted_time_test, test_total_inst, test_total_inst / lasted_time_test)


def attack(attacker, data, network, punct_set, word_alphabet, pos_alphabet,lemma_alphabet,
           device, beam=1, batch_size=256,  debug=1, cand_cache_path=None, normalize_digits=False,best_epoch=0):

    test_ucorrect = 0.0
    test_lcorrect = 0.0
    test_total_pred = 0
    test_total_gold = 0
    test_total_inst = 0
    accum_total_edit = 0
    accum_total_change_score = 0.0
    accum_total_score = 0.0
    accum_total_perp_diff = 0.0
    accum_success_attack = 0
    accum_total_sent = 0.0
    accum_total_head_change = 0.0
    accum_total_rel_change = 0.0

    start_time_test = time.time()
    if cand_cache_path is not None and attacker.cached_cands is None:
        save_cache = True
        if os.path.exists(cand_cache_path):
            print("Find existing cache file in %s" % cand_cache_path)
            exit()
        all_cand_cache = []
    else:
        save_cache = False
    for batch in tqdm(conllx_stacked_data.iterate_batch_stacked_variable(data, batch_size)):
        input_encoder, _ = batch
        word, lemma, char, pos, heads, types, masks, lengths = input_encoder

        adv_words = word.clone()  # 最终替换后的词语ID
        word_src = word.clone()   # 原始句子
        adv_src = []


        for i in range(len(lengths)):
            accum_total_sent += 1
            length = lengths[i]
            adv_tokens = [word_alphabet.get_instance(w) for w in word_src[i][:length]] # Jeffrey: 原始的word, adv_tokens就是一行句子
            adv_postags = [pos_alphabet.get_instance(w) for w in pos[i][:length]]

            # ******************** sdp ****************
            adv_heads = heads[i, 0:length, 0:length]
            adv_rels = types[i, 0:length, 0:length]
            adv_rels[0][:length] = 0  # root has no heads
            adv_heads[0][:length] = 0

            if debug == 3:
                print("\n###############################")
                print("Attacking sent-{}".format(int(accum_total_sent) - 1))
                print("tokens:\n", adv_tokens)
            if debug == 1: print("original sent:", adv_tokens)
            result, cand_cache = attacker.attack(adv_tokens, adv_postags, adv_heads, adv_rels, sent_id=int(accum_total_sent) - 1, debug=debug, cache=save_cache)
            if save_cache:
                all_cand_cache.append({'sent_id': int(accum_total_sent) - 1, 'tokens': cand_cache})
            if result is None:
                adv_src.append(adv_tokens[:length])  # 替换不成功，则是原始句子
                continue
            adv_tokens, adv_infos = result
            num_edit, total_score, total_change_score, total_perp_diff, total_head_change, total_rel_change = adv_infos
            if total_change_score <= 0:
                adv_src.append(word_src[i])
                continue
            accum_success_attack += 1
            accum_total_edit += num_edit
            accum_total_score += total_score
            accum_total_change_score += total_change_score
            accum_total_perp_diff += total_perp_diff
            accum_total_head_change += total_head_change
            accum_total_rel_change += total_rel_change
            if debug == 1: print("adv sent:", adv_tokens)
            adv_src.append(adv_tokens[:length])
            word_list = []
            for w in adv_tokens:
                w_ = DIGIT_RE.sub("0", w) if normalize_digits else w
                word_list.append(word_alphabet.get_index(w_))

            adv_words[i][:length] = torch.from_numpy(np.array(word_list))

        adv_words = adv_words.to(device)
        pos = pos.cpu().numpy()

        heads_pred, types_pred, _, _ = network.decode(adv_words, lemma, char, pos, mask=masks, length=lengths, beam=beam, leading_symbolic=conllx_stacked_data.NUM_SYMBOLIC_TAGS)

        ucorr, lcorr, total_gold, total_pred, num_inst = parser.evalF1(adv_words, lemma, pos, heads_pred, types_pred, heads, types, word_alphabet, lemma_alphabet, pos_alphabet, lengths,
                                                                       punct_set=punct_set, symbolic_root=True)
        test_ucorrect += ucorr
        test_lcorrect += lcorr
        test_total_gold += total_gold
        test_total_pred += total_pred

        test_total_inst += num_inst

    end_time_test = time.time()
    lasted_time_test = end_time_test - start_time_test

    test_Uprecision = 0.
    test_Lprecision = 0.
    if test_total_pred != 0:
        test_Uprecision = test_ucorrect * 100 / test_total_pred
        test_Lprecision = test_lcorrect * 100 / test_total_pred
    test_Urecall = test_ucorrect * 100 / test_total_gold
    test_Lrecall = test_lcorrect * 100 / test_total_gold
    if test_Uprecision == 0. and test_Urecall == 0.:
        test_UF1 = 0
    else:
        test_UF1 = 2 * (test_Uprecision * test_Urecall) / (test_Uprecision + test_Urecall)
    if test_Lprecision == 0. and test_Lrecall == 0.:
        test_LF1 = 0
    else:
        test_LF1 = 2 * (test_Lprecision * test_Lrecall) / (test_Lprecision + test_Lrecall)
    logger.info('------------------------------------------------------------')
    logger.info('TIME TEST: %.2f,NUM SENTS TEST: %d,SPEED TEST: %.2f' % (lasted_time_test, test_total_inst, test_total_inst / lasted_time_test))
    logger.info('TEST: ucorr: %d, lcorr: %d, tot_gold: %d, tot_pred: %d, Uprec: %.2f%%, Urec: %.2f%%, Lprec: %.2f%%, Lrec: %.2f%%, '
                'UF1: %.2f%%, LF1: %.2f%% (epoch: %d)' % (
                test_ucorrect, test_lcorrect, test_total_gold, test_total_pred, test_Uprecision, test_Urecall, test_Lprecision, test_Lrecall, test_UF1, test_LF1, best_epoch))

    if save_cache:
        print ('Saving candidate cache file to %s' % cand_cache_path)
        with open(cand_cache_path, 'w') as cache_f:
            json.dump(all_cand_cache, cache_f, indent=4)

    return (test_ucorrect, test_lcorrect, test_total_gold, test_total_pred, test_Uprecision, test_Urecall, test_Lprecision, test_Lrecall, test_UF1, test_LF1, best_epoch), (
    lasted_time_test, test_total_inst, test_total_inst / lasted_time_test)


def parse(args):

    logger = get_logger("Parsing")
    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    model_path = args.model_path
    model_name = "models/network.pt"
    model_name = os.path.join(model_path, model_name)
    test_path = args.test
    prior_order = "deep_first"
    alphabet_path = os.path.join(model_path, 'alphabets')
    assert os.path.exists(alphabet_path)
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet, lemma_alphabet = conllx_stacked_data.create_alphabets(alphabet_path, None, data_paths=None,
                                                                                                                     max_vocabulary_size=50000, embedd_dict=None)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()
    num_lemmas = lemma_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)
    logger.info("LEMMA Alphabet Size: %d" % num_lemmas)

    logger.info("Reading Data")
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda', 0) if use_gpu else torch.device('cpu')
    data_test = conllx_stacked_data.read_stacked_data_to_variable(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, lemma_alphabet, use_gpu=use_gpu, prior_order=prior_order)
    num_data = sum(data_test[1])
    logger.info("loading network...")
    window = 3
    freeze = args.freeze
    hyps = json.load(open(os.path.join("/users7/zllei/NeuroNLP2/pointer/adversary/setting", 'pointer.json'), 'r'))
    word_embedding = hyps["word_embedding"]
    word_path = hyps.get("word_path")

    use_char = hyps.get("char")
    char_embedding = hyps.get("char_embedding")
    char_path = hyps.get("char_path")

    use_pos = hyps.get("pos")
    pos_dim = hyps.get("pos_dim")

    use_lemma = hyps.get("lemma")
    lemma_dim = hyps.get("lemma_dim")

    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)
    char_dict = None
    char_dim = hyps.get("char_dim")
    if char_embedding != 'random':
        char_dict, char_dim = utils.load_embedding_dict(char_embedding, char_path)
    num_filters = hyps.get("num_filters")
    mode = hyps.get("mode")
    input_size_decoder = hyps.get("decoder_input_size")
    hidden_size = hyps.get("hidden_size")
    encoder_layers = hyps.get("encoder_layers")
    decoder_layers = hyps.get("decoder_layers")
    arc_space = hyps.get("arc_space")
    type_space =hyps.get("type_space")
    p_in = hyps.get("p_in")
    p_out = hyps.get("p_out")
    p_rnn = hyps.get("p_rnn")
    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / word_dim)
        table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
        table[conllx_stacked_data.UNK_ID, :] = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in word_dict:
                embedding = word_dict[word]
            elif word.lower() in word_dict:
                embedding = word_dict[word.lower()]
            else:
                embedding = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        logger.info('word OOV: %d' % oov)
        logger.info(torch.__version__)
        return torch.from_numpy(table)

    def construct_lemma_embedding_table():
        scale = np.sqrt(3.0 / lemma_dim)
        table = np.empty([lemma_alphabet.size(), lemma_dim], dtype=np.float32)
        table[conllx_stacked_data.UNK_ID, :] = np.zeros([1, lemma_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, lemma_dim]).astype(np.float32)
        oov = 0
        for lemma, index in lemma_alphabet.items():
            if lemma in word_dict:
                embedding = word_dict[lemma]
            elif lemma.lower() in word_dict:
                embedding = word_dict[lemma.lower()]
            else:
                embedding = np.zeros([1, lemma_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, lemma_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        logger.info('LEMMA OOV: %d' % oov)
        logger.info(torch.__version__)
        return torch.from_numpy(table)

    def construct_char_embedding_table():
        if char_dict is None:
            return None

        scale = np.sqrt(3.0 / char_dim)
        table = np.empty([num_chars, char_dim], dtype=np.float32)
        table[conllx_stacked_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
        oov = 0
        for char, index, in char_alphabet.items():
            if char in char_dict:
                embedding = char_dict[char]
            else:
                embedding = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        logger.info('character OOV: %d' % oov)
        return torch.from_numpy(table)
    word_table = construct_word_embedding_table()
    char_table = construct_char_embedding_table()
    lemma_table = construct_lemma_embedding_table()
    skipConnect = False
    grandPar = False
    sibling = False
    punctuation = args.punctuation
    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
    network = NewStackPtrNet(word_dim, num_words, lemma_dim, num_lemmas, char_dim, num_chars, pos_dim, num_pos, num_filters, window, mode, input_size_decoder, hidden_size, encoder_layers,
                             decoder_layers, num_types, arc_space, type_space, embedd_word=word_table, embedd_char=char_table, embedd_lemma=lemma_table, p_in=p_in, p_out=p_out, p_rnn=p_rnn,
                             biaffine=True, pos=use_pos, char=use_char, lemma=use_lemma, prior_order=prior_order, skipConnect=skipConnect, grandPar=grandPar, sibling=sibling)

    network = network.to(device)
    network.load_state_dict(torch.load(model_name, map_location=device))
    if args.cand.endswith('.json'):
        cands = json.load(open(args.cand, 'r'))
        candidates = {int(i):dic for (i,dic) in cands.items()}
    else:
        candidates = pickle.load(open(args.cand, 'rb')) # Jeffrey: sememe candidate
    vocab = json.load(open(args.vocab, 'r'))
    synonyms = pickle.load(open(args.syn, 'rb'))
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
    filters = args.filters.split(':')
    generators = args.generators.split(':')
    # ====================================================================

    alphabets = word_alphabet, char_alphabet, pos_alphabet, type_alphabet, lemma_alphabet
    beam = args.beam
    tokenizer = None
    if args.mode == 'black':
        attacker = BlackBoxAttacker(network, candidates, vocab, synonyms, filters=filters, generators=generators, max_mod_percent=args.max_mod_percent, tagger=args.tagger, punct_set=punct_set,
                                    beam=beam, normalize_digits=args.normalize_digits, cached_path=args.cached_path, train_vocab=args.train_vocab, knn_path=args.knn_path,
                                    max_knn_candidates=args.max_knn_candidates, sent_encoder_path=args.sent_encoder_path, min_word_cos_sim=args.min_word_cos_sim,
                                    min_sent_cos_sim=args.min_sent_cos_sim, cand_mlm=args.cand_mlm, dynamic_mlm_cand=args.dynamic_mlm_cand, temperature=args.temp, top_k=args.top_k, top_p=args.top_p,
                                    n_mlm_cands=args.n_mlm_cands, mlm_cand_file=args.mlm_cand_file, adv_lms=adv_lms, rel_ratio=args.adv_rel_ratio, fluency_ratio=args.adv_fluency_ratio,
                                    ppl_inc_thres=args.ppl_inc_thres, alphabets=alphabets, tokenizer=tokenizer, device=device, lm_device=lm_device, batch_size=args.adv_batch_size,
                                    random_backoff=args.random_backoff, wordpiece_backoff=args.wordpiece_backoff)
    else:
        attacker = None

    # with torch.no_grad():
    #     logger.info('Parsing Original Data...')
    #     eval(network=network, data_test=data_test, batch_size=args.batch_size, word_alphabet=word_alphabet, lemma_alphabet=lemma_alphabet, pos_alphabet=pos_alphabet, beam=args.beam,
    #          punct_set=punct_set, best_epoch=0, logger=logger)

    logger.info('\n----------------------------------------\n')

    with torch.no_grad():
        print('Attacking...')
        logger.info("use pad in input to attacker: {}".format(args.use_pad))
        start_time = time.time()
        # debug = 1: show orig/adv tokens / debug = 2: show log inside attacker
        attack(attacker, data_test, network, punct_set, word_alphabet, pos_alphabet, lemma_alphabet, device, beam=1,
               batch_size=args.batch_size, debug=1, cand_cache_path=None, normalize_digits=False, best_epoch=0)
        print('Time: %.2fs' % (time.time() - start_time))

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--mode', choices=['black', 'random', 'gray', 'gray_single'], required=True, help='processing mode')
    args_parser.add_argument('--seed', type=int, default=666, help='Random seed for torch and numpy (-1 for random)')
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
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--adv_lm_path', help='path for pretrained language model (gpt2) for adv filtering')
    args_parser.add_argument('--cand_mlm', help='path for mlm candidate generating')
    args_parser.add_argument('--mlm_cand_file', help='path for mlm candidate preprocessed json file')
    args_parser.add_argument('--dynamic_mlm_cand', action='store_true', default=False, help='Whether generate MLM candidates dynamically')
    args_parser.add_argument('--temp', type=float, default=1.0, help='Temperature for mlm candidate generating')
    args_parser.add_argument('--n_mlm_cands', type=int, default=50, help='Select candidate number for mlm candidate generating')
    args_parser.add_argument('--top_k', type=int, default=100, help='Top candidate number for filtering mlm candidate generating')
    args_parser.add_argument('--top_p', type=float, default=None, help='Top proportion for filtering mlm candidate generating')
    args_parser.add_argument('--output_filename', type=str, help='output filename for parse')
    args_parser.add_argument('--adv_filename', type=str, help='output adversarial filename')
    args_parser.add_argument('--adv_gold_filename', type=str, help='output adversarial text with gold heads & rels')
    args_parser.add_argument('--adv_rel_ratio', type=float, default=0.5, help='Relation importance in adversarial attack')
    args_parser.add_argument('--adv_fluency_ratio', type=float, default=0.2, help='Fluency importance in adversarial attack')
    #args_parser.add_argument('--max_perp_diff_per_token', type=float, default=0.8, help='Maximum allowed perplexity difference per token in adversarial attack')
    args_parser.add_argument('--ppl_inc_thres', type=float, default=20.0, help='Perplexity difference threshold in adversarial attack')
    args_parser.add_argument('--max_mod_percent', type=float, default=0.05, help='Maximum modification percentage of words')
    args_parser.add_argument('--adv_batch_size', type=int, default=16, help='Number of sentences in adv lm each batch')
    args_parser.add_argument('--random_backoff', action='store_true', default=False, help='randomly substitute if no change')
    args_parser.add_argument('--wordpiece_backoff', action='store_true', default=False, help='choose longest wordpiece substitute if no change')
    args_parser.add_argument('--knn_path', type=str, help='knn embedding path for adversarial attack')
    args_parser.add_argument('--max_knn_candidates', type=int, default=50, help='max knn candidate number')
    args_parser.add_argument('--min_word_cos_sim', type=float, default=0.9, help='Min word cos similarity')
    args_parser.add_argument('--min_sent_cos_sim', type=float, default=0.9, help='Min sent cos similarity')
    args_parser.add_argument('--sent_encoder_path', type=str, help='universal sentence encoder path for sent cos sim')
    args_parser.add_argument('--train_vocab', type=str, help='Training set vocab file (json) for train filter')
    args_parser.add_argument('--filters', type=str, default='word_sim:sent_sim:lm', help='filters for word substitution')
    args_parser.add_argument('--generators', type=str, default='synonym:sememe:embedding', help='generators for word substitution')
    args_parser.add_argument('--tagger', choices=['nltk', 'spacy', 'stanford'], default='nltk', help='POS tagger for POS checking in KNN embedding candidates')
    args_parser.add_argument('--use_pad', action='store_true', default=False, help='use PAD in input to attacker')
    args_parser.add_argument('--cached_path', type=str, default=None, help='input cached file for preprocessed candidate cache file')
    args_parser.add_argument('--cand_cache_path', type=str, default=None, help='output filename for candidate cache file')
    args_parser.add_argument('--ensemble', action='store_true', default=False, help='ensemble multiple parsers for predicting')
    args_parser.add_argument('--merge_by', type=str, choices=['logits', 'probs'], default='logits', help='ensemble policy')

    args = args_parser.parse_args()
    parse(args)
