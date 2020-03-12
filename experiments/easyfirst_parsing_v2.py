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
from torch.optim.adamw import AdamW
from torch.optim import SGD
from torch.nn.utils import clip_grad_norm_
from neuronlp2.nn.utils import total_grad_norm
from neuronlp2.io import get_logger, conllx_data, iterate_data, iterate_data_and_sample, sample_from_model
from neuronlp2.io import random_sample, from_model_sample, iterate_bucketed_data
from neuronlp2.io import get_order_mask
from neuronlp2.models import EasyFirst, EasyFirstV2
from neuronlp2.optim import ExponentialScheduler
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser
from neuronlp2.nn.utils import freeze_embedding
from neuronlp2.io.common import END

def get_optimizer(parameters, optim, learning_rate, lr_decay, betas, eps, amsgrad, weight_decay, warmup_steps):
    if optim == 'sgd':
        optimizer = SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        optimizer = AdamW(parameters, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad, weight_decay=weight_decay)
    init_lr = 1e-7
    scheduler = ExponentialScheduler(optimizer, lr_decay, warmup_steps, init_lr)
    return optimizer, scheduler


def eval(data, network, pred_writer, gold_writer, punct_set, word_alphabet, pos_alphabet, 
        device, beam=1, batch_size=256, get_head_by_layer=False, random_recomp=False, 
        recomp_prob=0.25, is_parse=False, write_to_tmp=True, prev_best_lcorr=0, prev_best_ucorr=0,
        pred_filename=None, symbolic_end=True):
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
    n_step = 0

    all_words = []
    all_postags = []
    all_heads_pred = []
    all_types_pred = []
    all_lengths = []
    all_src_words = []
    all_heads_by_layer = []

    for data in iterate_data(data, batch_size):
        if is_parse or torch.cuda.device_count() > 1:
            words = data['WORD'].to(device)
            chars = data['CHAR'].to(device)
            postags = data['POS'].to(device)
        else:
            words = data['WORD']
            chars = data['CHAR']
            postags = data['POS']
        heads = data['HEAD'].numpy()
        types = data['TYPE'].numpy()
        lengths = data['LENGTH'].numpy()
        masks = data['MASK'].to(device)
        #print (words)
        heads_pred, types_pred, recomp_freq, heads_by_layer = network.decode(words, chars, postags, mask=masks, 
                                                device=device, get_head_by_layer=get_head_by_layer, 
                                                random_recomp=random_recomp, recomp_prob=recomp_prob)
        #print (heads_by_layer)
        n_step += 1
        accum_recomp_freq += recomp_freq

        words = words.cpu().numpy()
        postags = postags.cpu().numpy()

        if write_to_tmp:
            pred_writer.write(words, postags, heads_pred, types_pred, lengths, symbolic_root=True, src_words=data['SRC'], 
                            heads_by_layer=heads_by_layer, symbolic_end=symbolic_end)
        else:
            all_words.append(words)
            all_postags.append(postags)
            all_heads_pred.append(heads_pred)
            all_types_pred.append(types_pred)
            all_lengths.append(lengths)
            all_src_words.append(data['SRC'])
            all_heads_by_layer.append(heads_by_layer)

        
        #gold_writer.write(words, postags, heads, types, lengths, symbolic_root=True)

        stats, stats_nopunc, stats_root, num_inst = parser.eval(words, postags, heads_pred, types_pred, heads, types,
                                                                word_alphabet, pos_alphabet, lengths, punct_set=punct_set, 
                                                                symbolic_root=True, symbolic_end=symbolic_end)
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
    print ('AVG Recompute frequency: %.2f' % (accum_recomp_freq / n_step))
    
    if not write_to_tmp:
        if prev_best_lcorr < accum_lcorr_nopunc or (prev_best_lcorr == accum_lcorr_nopunc and prev_best_ucorr < accum_ucorr_nopunc):
            print ('### Saving New Best File ... ###')
            pred_writer.start(pred_filename)
            #words = np.concatenate(all_words, axis=0)
            #postags = np.concatenate(all_postags, axis=0)
            #heads_pred = np.concatenate(all_heads_pred, axis=0)
            #types_pred = np.concatenate(all_types_pred, axis=0)
            #lengths = np.concatenate(all_lengths, axis=0)
            #src_words = np.concatenate(all_src_words, axis=0)
            #if get_head_by_layer:
            #    heads_by_layer = np.concatenate(all_heads_by_layer, axis=0)
            #else:
            #    heads_by_layer = None
            for i in range(len(all_words)):
                if get_head_by_layer:
                    heads_by_layer = all_heads_by_layer[i]
                else:
                    heads_by_layer = None
                pred_writer.write(all_words[i], all_postags[i], all_heads_pred[i], all_types_pred[i], 
                                all_lengths[i], symbolic_root=True, src_words=all_src_words[i],
                                heads_by_layer=heads_by_layer, symbolic_end=symbolic_end)
            pred_writer.close()

    return (accum_ucorr, accum_lcorr, accum_ucomlpete, accum_lcomplete, accum_total, accum_recomp_freq/n_step), \
           (accum_ucorr_nopunc, accum_lcorr_nopunc, accum_ucomlpete_nopunc, accum_lcomplete_nopunc, accum_total_nopunc), \
           (accum_root_corr, accum_total_root, accum_total_inst)


def train(args):
    logger = get_logger("Parsing")

    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')

    train_path = args.train
    dev_path = args.dev
    test_path = args.test

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    step_batch_size = args.step_batch_size
    optim = args.optim
    learning_rate = args.learning_rate
    lr_decay = args.lr_decay
    amsgrad = args.amsgrad
    eps = args.eps
    betas = (args.beta1, args.beta2)
    warmup_steps = args.warmup_steps
    weight_decay = args.weight_decay
    grad_clip = args.grad_clip
    eval_every = args.eval_every
    noscreen = args.noscreen
    fine_tune = args.fine_tune
    explore = args.explore

    loss_ty_token = args.loss_type == 'token'
    unk_replace = args.unk_replace
    freeze = args.freeze
    sampler = args.sampler

    model_path = args.model_path
    model_name = os.path.join(model_path, 'model.pt')
    punctuation = args.punctuation

    word_embedding = args.word_embedding
    word_path = args.word_path
    char_embedding = args.char_embedding
    char_path = args.char_path

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
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_data.create_alphabets(alphabet_path, train_path,
                                                                                             data_paths=[dev_path, test_path],
                                                                                             embedd_dict=word_dict, max_vocabulary_size=200000)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    result_path = os.path.join(model_path, 'tmp')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / word_dim)
        table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
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
        print('word OOV: %d' % oov)
        return torch.from_numpy(table)

    def construct_char_embedding_table():
        if char_dict is None:
            return None

        scale = np.sqrt(3.0 / char_dim)
        table = np.empty([num_chars, char_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
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

    if fine_tune:
        word_table = None
        char_table = None
    else:
        word_table = construct_word_embedding_table()
        char_table = construct_char_embedding_table()

    logger.info("Constructing Network...")
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
    assert model_type in ['EasyFirst', 'EasyFirstV2']
    assert word_dim == hyps['word_dim']
    if char_dim is not None:
        assert char_dim == hyps['char_dim']
    else:
        char_dim = hyps['char_dim']
    mode = hyps['transformer_mode']
    target_recomp_prob = hyps['target_recomp_prob']
    use_pos = hyps['pos']
    use_char = hyps['use_char']
    use_chosen_head = hyps['use_chosen_head']
    use_whole_seq = hyps['use_whole_seq']
    pos_dim = hyps['pos_dim']
    hidden_size = hyps['hidden_size']
    arc_space = hyps['arc_space']
    type_space = hyps['type_space']
    p_in = hyps['p_in']
    p_out = hyps['p_out']
    activation = hyps['activation']
    loss_interpolation = hyps['loss_interpolation']
    recomp_ratio = hyps['recomp_ratio']
    always_recompute = hyps['always_recompute']
    use_hard_concrete_dist = hyps['use_hard_concrete_dist']
    hc_temp = hyps['hard_concrete_temp']
    hc_eps = hyps['hard_concrete_eps']
    apply_recomp_prob_first = hyps['apply_recomp_prob_first']

    num_attention_heads = hyps['num_attention_heads']
    intermediate_size = hyps['intermediate_size']
    p_hid = hyps['hidden_dropout_prob']
    p_att = hyps['attention_probs_dropout_prob']
    p_graph_hid = hyps['graph_attention_hidden_dropout_prob']
    p_graph_att = hyps['graph_attention_probs_dropout_prob']
    recomp_att_dim = hyps['recomp_att_dim']
    dep_prob_depend_on_head = hyps['dep_prob_depend_on_head']
    use_top2_margin = hyps['use_top2_margin']
    extra_self_attention_layer = hyps['extra_self_attention_layer']
    input_self_attention_layer = hyps['input_self_attention_layer']
    num_input_attention_layers = hyps['num_input_attention_layers']
    num_attention_heads = hyps['num_attention_heads']
    input_encoder = hyps['input_encoder']
    num_layers = hyps['num_layers']
    p_rnn = hyps['p_rnn']
    maximize_unencoded_arcs_for_norc = hyps['maximize_unencoded_arcs_for_norc']
    encode_all_arc_for_rel = hyps['encode_all_arc_for_rel']
    use_input_encode_for_rel = hyps['use_input_encode_for_rel']
    num_graph_attention_layers = hyps['num_graph_attention_layers']
    share_params = hyps['share_params']
    residual_from_input = hyps['residual_from_input']
    transformer_drop_prob = hyps['transformer_drop_prob']
    num_graph_attention_heads = hyps['num_graph_attention_heads']
    only_value_weight = hyps['only_value_weight']
    encode_rel_type = hyps['encode_rel_type']
    rel_dim = hyps['rel_dim']
    use_null_att_pos = hyps['use_null_att_pos']
    num_arcs_per_pred = hyps['num_arcs_per_pred']
    use_input_layer = hyps['use_input_layer']
    use_sin_position_embedding = hyps['use_sin_position_embedding']
    if use_null_att_pos:
        end_word_id = word_alphabet.get_index(END)
    else:
        end_word_id = None

    if always_recompute:
        target_recomp_prob = 1

    num_gpu = torch.cuda.device_count()

    if model_type == 'EasyFirst':
        network = EasyFirst(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                           hidden_size, num_types, arc_space, type_space,
                           intermediate_size,
                           device=device, 
                           hidden_dropout_prob=p_hid,
                           attention_probs_dropout_prob=p_att,
                           graph_attention_hidden_dropout_prob=p_graph_hid,
                           graph_attention_probs_dropout_prob=p_graph_att,
                           embedd_word=word_table, embedd_char=char_table,
                           p_in=p_in, p_out=p_out, pos=use_pos, use_char=use_char, 
                           activation=activation, dep_prob_depend_on_head=dep_prob_depend_on_head, 
                           use_top2_margin=use_top2_margin, target_recomp_prob=target_recomp_prob,
                           extra_self_attention_layer=extra_self_attention_layer,
                           num_attention_heads=num_attention_heads,
                           input_encoder=input_encoder, num_layers=num_layers, p_rnn=p_rnn,
                           input_self_attention_layer=input_self_attention_layer,
                           num_input_attention_layers=num_input_attention_layers,
                           maximize_unencoded_arcs_for_norc=maximize_unencoded_arcs_for_norc,
                           encode_all_arc_for_rel=encode_all_arc_for_rel,
                           use_input_encode_for_rel=use_input_encode_for_rel,
                           always_recompute=always_recompute,
                           use_hard_concrete_dist=use_hard_concrete_dist, 
                           hard_concrete_temp=hc_temp, hard_concrete_eps=hc_eps,
                           apply_recomp_prob_first=apply_recomp_prob_first,
                           num_graph_attention_layers=num_graph_attention_layers,
                           share_params=share_params, residual_from_input=residual_from_input,
                           transformer_drop_prob=transformer_drop_prob,
                           num_graph_attention_heads=num_graph_attention_heads, 
                           only_value_weight=only_value_weight,
                           encode_rel_type=encode_rel_type, rel_dim=rel_dim,
                           use_null_att_pos=use_null_att_pos, end_word_id=end_word_id,
                           num_arcs_per_pred=num_arcs_per_pred, use_input_layer=use_input_layer, 
                           use_sin_position_embedding=use_sin_position_embedding)
    elif model_type == 'EasyFirstV2':
        network = EasyFirstV2(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                           hidden_size, num_types, arc_space, type_space,
                           intermediate_size,
                           device=device, 
                           hidden_dropout_prob=p_hid,
                           attention_probs_dropout_prob=p_att,
                           graph_attention_hidden_dropout_prob=p_graph_hid,
                           graph_attention_probs_dropout_prob=p_graph_att,
                           embedd_word=word_table, embedd_char=char_table,
                           p_in=p_in, p_out=p_out, pos=use_pos, use_char=use_char, 
                           activation=activation, dep_prob_depend_on_head=dep_prob_depend_on_head, 
                           use_top2_margin=use_top2_margin, target_recomp_prob=target_recomp_prob,
                           extra_self_attention_layer=extra_self_attention_layer,
                           num_attention_heads=num_attention_heads,
                           input_encoder=input_encoder, num_layers=num_layers, p_rnn=p_rnn,
                           input_self_attention_layer=input_self_attention_layer,
                           num_input_attention_layers=num_input_attention_layers,
                           maximize_unencoded_arcs_for_norc=maximize_unencoded_arcs_for_norc,
                           encode_all_arc_for_rel=encode_all_arc_for_rel,
                           use_input_encode_for_rel=use_input_encode_for_rel,
                           always_recompute=always_recompute,
                           use_hard_concrete_dist=use_hard_concrete_dist, 
                           hard_concrete_temp=hc_temp, hard_concrete_eps=hc_eps,
                           apply_recomp_prob_first=apply_recomp_prob_first,
                           num_graph_attention_layers=num_graph_attention_layers,
                           share_params=share_params, residual_from_input=residual_from_input,
                           transformer_drop_prob=transformer_drop_prob,
                           num_graph_attention_heads=num_graph_attention_heads, 
                           only_value_weight=only_value_weight,
                           encode_rel_type=encode_rel_type, rel_dim=rel_dim,
                           use_null_att_pos=use_null_att_pos, end_word_id=end_word_id,
                           num_arcs_per_pred=num_arcs_per_pred, use_input_layer=use_input_layer, 
                           use_sin_position_embedding=use_sin_position_embedding)
    else:
        raise RuntimeError('Unknown model type: %s' % model_type)

    if fine_tune:
        logger.info("Fine-tuning: Loading model from %s" % model_name)
        network.load_state_dict(torch.load(model_name))

    logger.info("GPU Number: %d" % num_gpu)
    if num_gpu > 1:
        logger.info("Using Data Parallel")
        network = torch.nn.DataParallel(network)
        network.to(device)
    single_network = network if num_gpu <= 1 else network.module

    if freeze:
        freeze_embedding(network.word_embed)

    #network = network.to(device)
    model = "{}-{}".format(model_type, mode)
    logger.info("Network: %s, hidden=%d, act=%s" % (model, hidden_size, activation))
    logger.info("Sampler: %s (Explore: %s)" % (sampler, explore))
    logger.info("##### Input Encoder (Type: %s, Layer: %d) ###" % (input_encoder, num_layers))
    logger.info("dropout(in, out, hidden, att): (%.2f, %.2f, %.2f, %.2f)" % (p_in, p_out, p_hid, p_att))
    logger.info("Use POS tag: %s" % use_pos)
    logger.info("Use Char: %s" % use_char)
    logger.info("Use Sin Position Embedding: %s" % use_sin_position_embedding)
    logger.info("Use Input Layer: %s" % use_input_layer)
    logger.info("Residual From Input Layer: %s (transformer dropout: %.2f)" % (residual_from_input, transformer_drop_prob))
    logger.info("##### Graph Encoder (Layers: %s, Share Params:%s) #####"% (num_graph_attention_layers, share_params))
    logger.info("dropout(graph_hid, graph_att): (%.2f, %.2f)" % (p_graph_hid, p_graph_att))
    logger.info("Number of Arcs per Prediction: %d" % num_arcs_per_pred)
    logger.info("Only Use Value Weight: %s" % only_value_weight)
    logger.info("Attend to END if no head: %s" % use_null_att_pos)
    logger.info("Encode Relation Type: %s (rel embed dim: %d)" % (encode_rel_type, rel_dim))
    logger.info("Use Input Self Attention Layer: %s (Layer: %d)" % (input_self_attention_layer, num_input_attention_layers))
    logger.info("Use Top Self Attention Layer: %s" % extra_self_attention_layer)
    logger.info("Use Hard Concrete Distribution: %s (Temperature: %.2f, Epsilon: %.2f, Apply Prob First: %s)" % (use_hard_concrete_dist,
                                                                                        hc_temp, hc_eps, apply_recomp_prob_first))
    logger.info("##### Parser #####")
    logger.info("Always Recompute after Generation: %s" % always_recompute)
    logger.info("Maximize All Unencoded Arcs for No Recompute: %s" % maximize_unencoded_arcs_for_norc)
    logger.info("Encode All Arcs for Relation Prediction: %s" % encode_all_arc_for_rel)
    logger.info("Only Use Input Encoder for Relation Prediction: %s" % use_input_encode_for_rel)
    logger.info('# of Parameters: %d' % (sum([param.numel() for param in network.parameters()])))
    
    symbolic_end = args.symbolic_end
    logger.info("Reading Data (symbolic end: %s)" % symbolic_end)
    if use_null_att_pos and not symbolic_end:
        raise ValueError("Must set symbolic_end to True for use_null_att_pos to work!")
        exit()
    
    data_train = conllx_data.read_bucketed_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True,
                                                mask_out_root=False, symbolic_end=symbolic_end)
    data_dev = conllx_data.read_data(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True,
                                        mask_out_root=False, symbolic_end=symbolic_end)
    data_test = conllx_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True,
                                        mask_out_root=False, symbolic_end=symbolic_end)
    

    num_data = sum(data_train[1])
    logger.info("training: #training data: %d, batch: %d, unk replace: %.2f" % (num_data, batch_size, unk_replace))
    if model_type == 'EasyFirstV2':
        logger.info("Batch by arc: %s" % (args.batch_by_arc))
    if args.random_recomp:
        logger.info("Randomly sample recomputation with prob (for eval): %s" % args.recomp_prob)

    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    optimizer, scheduler = get_optimizer(network.parameters(), optim, learning_rate, lr_decay, betas, eps, amsgrad, weight_decay, warmup_steps)

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
    best_recomp_freq = 0.0

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
    test_recomp_freq = 0.0

    patient = 0
    beam = args.beam
    reset = args.reset
    num_batches = num_data // batch_size + 1
    if optim == 'adam':
        opt_info = 'adam, betas=(%.1f, %.3f), eps=%.1e, amsgrad=%s' % (betas[0], betas[1], eps, amsgrad)
    else:
        opt_info = 'sgd, momentum=0.9, nesterov=True'
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss = 0.
        train_arc_loss = 0.
        train_rel_loss = 0.
        train_recomp_loss = 0.
        num_steps = 0
        num_insts = 0
        num_words = 0
        num_back = 0
        num_nans = 0
        network.train()
        lr = scheduler.get_lr()[0]
        print('Epoch %d (%s, lr=%.6f, lr decay=%.6f, grad clip=%.1f, l2=%.1e): ' % (epoch, opt_info, lr, lr_decay, grad_clip, weight_decay))
        #if args.cuda:
        #    torch.cuda.empty_cache()
        gc.collect()
        if model_type == 'EasyFirst':
            if sampler == 'random':
                data_sampler = random_sample(data_train, batch_size, 
                                step_batch_size=step_batch_size, unk_replace=unk_replace, 
                                shuffle=False, target_recomp_prob=target_recomp_prob)
            elif sampler == 'from_model':
                data_sampler = from_model_sample(single_network, data_train, batch_size, 
                                  unk_replace=unk_replace, shuffle=False, device=device,
                                  explore=explore)

            for step, data in enumerate(data_sampler):
                #print ('number in batch:',len(sub_data['WORD']))
                optimizer.zero_grad()
                if num_gpu > 1:
                    words = data['WORD'].to(device)
                    chars = data['CHAR'].to(device)
                    postags = data['POS'].to(device)
                else:
                    words = data['WORD']
                    chars = data['CHAR']
                    postags = data['POS']
                heads = data['HEAD'].to(device)
                nbatch = words.size(0)

                types = data['TYPE'].to(device)
                masks = data['MASK'].to(device)
                # (batch, seq_len)
                recomp_gen_mask = data['RECOMP_GEN_MASK'].to(device)
                # (batch, seq_len)
                no_recmp_gen_mask = data['NO_RECOMP_GEN_MASK'].to(device)
                ref_mask = data['REF_MASK'].to(device)
                if use_chosen_head:
                    next_head_mask = data['NEXT_HEAD_MASK'].to(device)
                else:
                    next_head_mask = None
                nwords = masks.sum() - nbatch
                network.train()
                loss_arc, loss_rel, loss_recomp = network(words, chars, postags, heads, types, 
                                                    recomp_gen_mask, no_recmp_gen_mask, ref_mask, 
                                                    mask=masks, next_head_mask=next_head_mask, device=device)
                loss_arc = loss_arc.mean()
                loss_rel = loss_rel.mean()
                loss_recomp = loss_recomp.mean()
                loss = 0.5 *((1.0 - loss_interpolation) * loss_arc + loss_interpolation * loss_rel) + recomp_ratio * loss_recomp
                #if loss_ty_token:
                #    loss = loss_total.div(nwords)
                #else:
                #    loss = loss_total.div(nbatch)
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
                        num_steps += 1
                        train_loss += loss.item()
                        train_arc_loss += loss_arc.item()
                        train_rel_loss += loss_rel.item()
                        train_recomp_loss += loss_recomp.item()
                #torch.cuda.empty_cache()
                # update log
                if step % 100 == 0:
                    if not noscreen: 
                        sys.stdout.write("\b" * num_back)
                        sys.stdout.write(" " * num_back)
                        sys.stdout.write("\b" * num_back)
                        curr_lr = scheduler.get_lr()[0]
                        num_insts = max(num_insts, 1)
                        num_words = max(num_words, 1)
                        log_info = '[%d/%d (%.0f%%) lr=%.6f (%d)] loss: %.4f (%.4f), arc: %.4f (%.4f), type: %.4f (%.4f)' % (step, num_batches, 100. * step / num_batches, curr_lr, num_nans,
                                                                                                                             train_loss / num_insts, train_loss / num_words,
                                                                                                                             train_arc_loss / num_insts, train_arc_loss / num_words,
                                                                                                                            train_rel_loss / num_insts, train_rel_loss / num_words)
                        sys.stdout.write(log_info)
                        sys.stdout.flush()
                        num_back = len(log_info)

        elif model_type == 'EasyFirstV2':
            data_sampler = iterate_bucketed_data(data_train, batch_size, unk_replace=unk_replace, 
                            shuffle=True, batch_by_arc=args.batch_by_arc)
            for step, data in enumerate(data_sampler):
                #print ('number in batch:',data['WORD'].size())
                
                if num_gpu > 1:
                    words = data['WORD'].to(device)
                    chars = data['CHAR'].to(device)
                    postags = data['POS'].to(device)
                else:
                    words = data['WORD']
                    chars = data['CHAR']
                    postags = data['POS']
                heads = data['HEAD'].to(device)
                types = data['TYPE'].to(device)
                masks = data['MASK'].to(device)
                if sampler == 'random':
                    # (seq_len, batch, seq_len)
                    order_masks = get_order_mask(data['LENGTH'], symbolic_end=symbolic_end).to(device)
                elif sampler == 'from_model':
                    network.eval()
                    if num_gpu > 1:
                        order_masks = network.module.inference(words, chars, postags, heads, mask=masks)
                    else:
                        order_masks = network.inference(words, chars, postags, heads, mask=masks)
                    order_masks.to(device)
                    network.train()
                nbatch = words.size(0)
                nwords = masks.sum() - nbatch
                network.train()
                cur_batch_size, seq_len = words.size()
                # (batch, seq_len, seq_len)
                gen_heads_onehot = torch.zeros((cur_batch_size, seq_len, seq_len), dtype=torch.int32, device=heads.device)
                encode_heads_onehot = torch.zeros_like(gen_heads_onehot)
                n_state_step = -1
                # (batch, seq_len), 1 represent the token whose head is to be generated at this step
                for i in range(len(order_masks)):
                    # if has predicted k arcs, update the arcs to be encoded with GAT
                    n_state_step += 1
                    if n_state_step == num_arcs_per_pred:
                        encode_heads_onehot = gen_heads_onehot
                        n_state_step = 0

                    order_mask = order_masks[i]
                    optimizer.zero_grad()
                    #if num_gpu > 1: 
                        # (batch, seq_len, hidden_size)
                    #    input_encoder_output = network.module._get_input_encoder_output(words, chars, postags, masks)
                    #else:
                    #    input_encoder_output = network._get_input_encoder_output(words, chars, postags, masks)
                    loss_arc, loss_rel, loss_recomp, gen_heads_onehot = network(words, chars, postags, 
                            gen_heads_onehot, encode_heads_onehot, heads, types, order_mask, mask=masks, explore=explore)
                    #print ("errors: ", errs)
                    loss_arc = loss_arc.mean()
                    loss_rel = loss_rel.mean()
                    loss_recomp = loss_recomp.mean()
                    loss = 0.5 *((1.0 - loss_interpolation) * loss_arc + loss_interpolation * loss_rel) + recomp_ratio * loss_recomp
                    #if loss_ty_token:
                    #    loss = loss_total.div(nwords)
                    #else:
                    #    loss = loss_total.div(nbatch)

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
                            num_steps += 1
                            train_loss += loss.item()
                            train_arc_loss += loss_arc.item()
                            train_rel_loss += loss_rel.item()
                            train_recomp_loss += loss_recomp.item()
                    #torch.cuda.empty_cache()
                    # update log
                    if step % 100 == 0:
                        if not noscreen: 
                            sys.stdout.write("\b" * num_back)
                            sys.stdout.write(" " * num_back)
                            sys.stdout.write("\b" * num_back)
                            curr_lr = scheduler.get_lr()[0]
                            num_insts = max(num_insts, 1)
                            num_words = max(num_words, 1)
                            log_info = '[%d/%d (%.0f%%) lr=%.6f (%d)] loss: %.4f (%.4f), arc: %.4f (%.4f), type: %.4f (%.4f)' % (step, num_batches, 100. * step / num_batches, curr_lr, num_nans,
                                                                                                                                 train_loss / num_insts, train_loss / num_words,
                                                                                                                                 train_arc_loss / num_insts, train_arc_loss / num_words,
                                                                                                                                train_rel_loss / num_insts, train_rel_loss / num_words)
                            sys.stdout.write(log_info)
                            sys.stdout.flush()
                            num_back = len(log_info)
                del gen_heads_onehot
        
        if not noscreen: 
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
        print('total: %d (%d), steps: %d, loss: %.4f (nans: %d), arc: %.4f, rel: %.4f, recomp: %.4f, time: %.2fs' % (num_insts, num_words, num_steps, train_loss / (num_steps+1e-9),
                                                                                                       num_nans, train_arc_loss / (num_steps+1e-9),
                                                                                                       train_rel_loss / (num_steps+1e-9),
                                                                                                       train_recomp_loss / (num_steps+1e-9),
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
                dev_stats, dev_stats_nopunct, dev_stats_root = eval(data_dev, single_network, pred_writer, gold_writer, punct_set, word_alphabet, pos_alphabet, device, 
                                                                    beam=beam, get_head_by_layer=args.get_head_by_layer,
                                                                    random_recomp=args.random_recomp, recomp_prob=args.recomp_prob,
                                                                    write_to_tmp=False, prev_best_lcorr=best_lcorrect_nopunc,
                                                                    prev_best_ucorr=best_ucorrect_nopunc, pred_filename=pred_filename,
                                                                    symbolic_end=symbolic_end)

                #gold_writer.close()

                dev_ucorr, dev_lcorr, dev_ucomlpete, dev_lcomplete, dev_total, dev_recomp_freq = dev_stats
                dev_ucorr_nopunc, dev_lcorr_nopunc, dev_ucomlpete_nopunc, dev_lcomplete_nopunc, dev_total_nopunc = dev_stats_nopunct
                dev_root_corr, dev_total_root, dev_total_inst = dev_stats_root

                if best_lcorrect_nopunc < dev_lcorr_nopunc or (best_lcorrect_nopunc == dev_lcorr_nopunc and best_ucorrect_nopunc < dev_ucorr_nopunc):
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
                    best_recomp_freq = dev_recomp_freq

                    best_epoch = epoch
                    patient = 0
                    if num_gpu > 1:
                        torch.save(network.module.state_dict(), model_name)
                    else:
                        torch.save(network.state_dict(), model_name)

                    pred_filename = os.path.join(result_path, 'pred_test%d' % epoch)
                    pred_writer.start(pred_filename)
                    #gold_filename = os.path.join(result_path, 'gold_test%d' % epoch)
                    #gold_writer.start(gold_filename)

                    print('Evaluating test:')
                    test_stats, test_stats_nopunct, test_stats_root = eval(data_test, single_network, pred_writer, gold_writer, punct_set, word_alphabet, pos_alphabet, device, 
                                                                        beam=beam, get_head_by_layer=args.get_head_by_layer,
                                                                        random_recomp=args.random_recomp, recomp_prob=args.recomp_prob,
                                                                        symbolic_end=symbolic_end)

                    test_ucorrect, test_lcorrect, test_ucomlpete, test_lcomplete, test_total, test_recomp_freq = test_stats
                    test_ucorrect_nopunc, test_lcorrect_nopunc, test_ucomlpete_nopunc, test_lcomplete_nopunc, test_total_nopunc = test_stats_nopunct
                    test_root_correct, test_total_root, test_total_inst = test_stats_root

                    pred_writer.close()
                    #gold_writer.close()
                else:
                    patient += 1

                print('-' * 125)
                print('best dev  W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                    best_ucorrect, best_lcorrect, best_total, best_ucorrect * 100 / (best_total+1e-9), best_lcorrect * 100 / (best_total+1e-9),
                    best_ucomlpete * 100 / dev_total_inst, best_lcomplete * 100 / dev_total_inst,
                    best_epoch))
                print('best dev  Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                    best_ucorrect_nopunc, best_lcorrect_nopunc, best_total_nopunc,
                    best_ucorrect_nopunc * 100 / (best_total_nopunc+1e-9), best_lcorrect_nopunc * 100 / (best_total_nopunc+1e-9),
                    best_ucomlpete_nopunc * 100 / (best_total_inst+1e-9), best_lcomplete_nopunc * 100 / (best_total_inst+1e-9),
                    best_epoch))
                print('best dev  Root: corr: %d, total: %d, acc: %.2f%%, avg recompute frequency: %.2f (epoch: %d)' % (
                    best_root_correct, best_total_root, best_root_correct * 100 / (best_total_root+1e-9), best_recomp_freq, best_epoch))
                print('-' * 125)
                print('best test W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                    test_ucorrect, test_lcorrect, test_total, test_ucorrect * 100 / (test_total+1e-9), test_lcorrect * 100 / (test_total+1e-9),
                    test_ucomlpete * 100 / (test_total_inst+1e-9), test_lcomplete * 100 / (test_total_inst+1e-9),
                    best_epoch))
                print('best test Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                    test_ucorrect_nopunc, test_lcorrect_nopunc, test_total_nopunc,
                    test_ucorrect_nopunc * 100 / (test_total_nopunc+1e-9), test_lcorrect_nopunc * 100 / (test_total_nopunc+1e-9),
                    test_ucomlpete_nopunc * 100 / (test_total_inst+1e-9), test_lcomplete_nopunc * 100 / (test_total_inst+1e-9),
                    best_epoch))
                print('best test Root: corr: %d, total: %d, acc: %.2f%%, avg recompute frequency: %.2f (epoch: %d)' % (
                    test_root_correct, test_total_root, test_root_correct * 100 / (test_total_root+1e-9), test_recomp_freq, best_epoch))
                print('=' * 125)

                if patient >= reset:
                    logger.info('reset optimizer momentums')
                    single_network.load_state_dict(torch.load(model_name, map_location=device))
                    scheduler.reset_state()
                    patient = 0


def parse(args):
    logger = get_logger("Parsing")
    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    test_path = args.test

    model_path = args.model_path
    model_name = os.path.join(model_path, 'model.pt')
    punctuation = args.punctuation
    print(args)

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(model_path, 'alphabets')
    assert os.path.exists(alphabet_path)
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_data.create_alphabets(alphabet_path, None)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

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
    assert model_type in ['EasyFirst', 'EasyFirstV2']
    word_dim = hyps['word_dim']
    char_dim = hyps['char_dim']
    mode = hyps['transformer_mode']
    target_recomp_prob = hyps['target_recomp_prob']
    use_pos = hyps['pos']
    use_char = hyps['use_char']
    use_chosen_head = hyps['use_chosen_head']
    use_whole_seq = hyps['use_whole_seq']
    pos_dim = hyps['pos_dim']
    hidden_size = hyps['hidden_size']
    arc_space = hyps['arc_space']
    type_space = hyps['type_space']
    p_in = hyps['p_in']
    p_out = hyps['p_out']
    activation = hyps['activation']
    loss_interpolation = hyps['loss_interpolation']
    recomp_ratio = hyps['recomp_ratio']
    always_recompute = hyps['always_recompute']
    use_hard_concrete_dist = hyps['use_hard_concrete_dist']
    hc_temp = hyps['hard_concrete_temp']
    hc_eps = hyps['hard_concrete_eps']
    apply_recomp_prob_first = hyps['apply_recomp_prob_first']

    num_attention_heads = hyps['num_attention_heads']
    intermediate_size = hyps['intermediate_size']
    p_hid = hyps['hidden_dropout_prob']
    p_att = hyps['attention_probs_dropout_prob']
    p_graph_hid = hyps['graph_attention_hidden_dropout_prob']
    p_graph_att = hyps['graph_attention_probs_dropout_prob']
    recomp_att_dim = hyps['recomp_att_dim']
    dep_prob_depend_on_head = hyps['dep_prob_depend_on_head']
    use_top2_margin = hyps['use_top2_margin']
    extra_self_attention_layer = hyps['extra_self_attention_layer']
    input_self_attention_layer = hyps['input_self_attention_layer']
    num_input_attention_layers = hyps['num_input_attention_layers']
    num_attention_heads = hyps['num_attention_heads']
    input_encoder = hyps['input_encoder']
    num_layers = hyps['num_layers']
    p_rnn = hyps['p_rnn']

    maximize_unencoded_arcs_for_norc = hyps['maximize_unencoded_arcs_for_norc']
    encode_all_arc_for_rel = hyps['encode_all_arc_for_rel']
    use_input_encode_for_rel = hyps['use_input_encode_for_rel']
    num_graph_attention_layers = hyps['num_graph_attention_layers']
    share_params = hyps['share_params']
    residual_from_input = hyps['residual_from_input']
    transformer_drop_prob = hyps['transformer_drop_prob']
    num_graph_attention_heads = hyps['num_graph_attention_heads']
    only_value_weight = hyps['only_value_weight']
    encode_rel_type = hyps['encode_rel_type']
    rel_dim = hyps['rel_dim']
    use_null_att_pos = hyps['use_null_att_pos']
    num_arcs_per_pred = hyps['num_arcs_per_pred']
    use_input_layer = hyps['use_input_layer']
    use_sin_position_embedding = hyps['use_sin_position_embedding']

    if use_null_att_pos:
        end_word_id = word_alphabet.get_index(END)
    else:
        end_word_id = None

    if always_recompute:
        target_recomp_prob = 1

    if model_type == 'EasyFirst':
        network = EasyFirst(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                           hidden_size, num_types, arc_space, type_space,
                           intermediate_size,
                           device=device, 
                           hidden_dropout_prob=p_hid,
                           attention_probs_dropout_prob=p_att,
                           graph_attention_hidden_dropout_prob=p_graph_hid,
                           graph_attention_probs_dropout_prob=p_graph_att,
                           p_in=p_in, p_out=p_out, pos=use_pos, use_char=use_char, 
                           activation=activation, dep_prob_depend_on_head=dep_prob_depend_on_head, 
                           use_top2_margin=use_top2_margin, target_recomp_prob=target_recomp_prob,
                           extra_self_attention_layer=extra_self_attention_layer,
                           num_attention_heads=num_attention_heads,
                           input_encoder=input_encoder, num_layers=num_layers, p_rnn=p_rnn,
                           input_self_attention_layer=input_self_attention_layer,
                           num_input_attention_layers=num_input_attention_layers,
                           maximize_unencoded_arcs_for_norc=maximize_unencoded_arcs_for_norc,
                           encode_all_arc_for_rel=encode_all_arc_for_rel,
                           use_input_encode_for_rel=use_input_encode_for_rel,
                           always_recompute=always_recompute,
                           use_hard_concrete_dist=use_hard_concrete_dist, 
                           hard_concrete_temp=hc_temp, hard_concrete_eps=hc_eps,
                           apply_recomp_prob_first=apply_recomp_prob_first,
                           num_graph_attention_layers=num_graph_attention_layers,
                           share_params=share_params, residual_from_input=residual_from_input,
                           transformer_drop_prob=transformer_drop_prob,
                           num_graph_attention_heads=num_graph_attention_heads, 
                           only_value_weight=only_value_weight,
                           encode_rel_type=encode_rel_type, rel_dim=rel_dim,
                           use_null_att_pos=use_null_att_pos, end_word_id=end_word_id,
                           num_arcs_per_pred=num_arcs_per_pred, use_input_layer=use_input_layer, 
                           use_sin_position_embedding=use_sin_position_embedding)
    elif model_type == 'EasyFirstV2':
        network = EasyFirstV2(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                           hidden_size, num_types, arc_space, type_space,
                           intermediate_size,
                           device=device, 
                           hidden_dropout_prob=p_hid,
                           attention_probs_dropout_prob=p_att,
                           graph_attention_hidden_dropout_prob=p_graph_hid,
                           graph_attention_probs_dropout_prob=p_graph_att,
                           p_in=p_in, p_out=p_out, pos=use_pos, use_char=use_char, 
                           activation=activation, dep_prob_depend_on_head=dep_prob_depend_on_head, 
                           use_top2_margin=use_top2_margin, target_recomp_prob=target_recomp_prob,
                           extra_self_attention_layer=extra_self_attention_layer,
                           num_attention_heads=num_attention_heads,
                           input_encoder=input_encoder, num_layers=num_layers, p_rnn=p_rnn,
                           input_self_attention_layer=input_self_attention_layer,
                           num_input_attention_layers=num_input_attention_layers,
                           maximize_unencoded_arcs_for_norc=maximize_unencoded_arcs_for_norc,
                           encode_all_arc_for_rel=encode_all_arc_for_rel,
                           use_input_encode_for_rel=use_input_encode_for_rel,
                           always_recompute=always_recompute,
                           use_hard_concrete_dist=use_hard_concrete_dist, 
                           hard_concrete_temp=hc_temp, hard_concrete_eps=hc_eps,
                           apply_recomp_prob_first=apply_recomp_prob_first,
                           num_graph_attention_layers=num_graph_attention_layers,
                           share_params=share_params, residual_from_input=residual_from_input,
                           transformer_drop_prob=transformer_drop_prob,
                           num_graph_attention_heads=num_graph_attention_heads, 
                           only_value_weight=only_value_weight,
                           encode_rel_type=encode_rel_type, rel_dim=rel_dim,
                           use_null_att_pos=use_null_att_pos, end_word_id=end_word_id,
                           num_arcs_per_pred=num_arcs_per_pred, use_input_layer=use_input_layer, 
                           use_sin_position_embedding=use_sin_position_embedding)
    else:
        raise RuntimeError('Unknown model type: %s' % model_type)

    network = network.to(device)
    network.load_state_dict(torch.load(model_name, map_location=device))
    model = "{}-{}".format(model_type, mode)
    #print ("Recompute Features Weight: [logp(max), sent_lens, new_arcs, (top margin)] ",network.recomp_dense.weight)
    logger.info("Network: %s, hidden=%d, act=%s" % (model, hidden_size, activation))
    
    logger.info("Reading Data")
    data_test = conllx_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, symbolic_root=True,
                                        mask_out_root=False, symbolic_end=args.symbolic_end)

    beam = args.beam
    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    #pred_filename = os.path.join(result_path, 'pred.txt')
    pred_filename = args.output_filename
    pred_writer.start(pred_filename)
    #gold_filename = os.path.join(result_path, 'gold.txt')
    #gold_writer.start(gold_filename)

    if args.random_recomp:
        logger.info("Randomly sample recomputation with prob: %s" % args.recomp_prob)
    with torch.no_grad():
        print('Parsing...')
        start_time = time.time()
        eval(data_test, network, pred_writer, gold_writer, punct_set, word_alphabet, pos_alphabet, 
                device, beam, batch_size=args.batch_size, get_head_by_layer=args.get_head_by_layer,
                random_recomp=args.random_recomp, recomp_prob=args.recomp_prob, is_parse=True,
                symbolic_end=symbolic_end)
        print('Time: %.2fs' % (time.time() - start_time))

    pred_writer.close()
    #gold_writer.close()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--mode', choices=['train', 'parse'], required=True, help='processing mode')
    args_parser.add_argument('--seed', type=int, default=-1, help='Random seed for torch and numpy (-1 for random)')
    args_parser.add_argument('--symbolic_end', choices=['True', 'False'], default='True', help='Whether to add END symbol')
    args_parser.add_argument('--fine_tune', action='store_true', default=False, help='Whether to fine_tune?')
    args_parser.add_argument('--explore', action='store_true', default=False, help='Whether to explore (encode wrong prediction) while sampling from model?')
    args_parser.add_argument('--config', type=str, help='config file')
    args_parser.add_argument('--output_filename', type=str, help='output filename for parse')
    args_parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    args_parser.add_argument('--batch_by_arc', action='store_true', default=False, help='Whether to count batch by number of arcs.')
    args_parser.add_argument('--update_batch', type=int, default=50, help='Number of errors needed to do one update')
    args_parser.add_argument('--step_batch_size', type=int, default=16, help='Number of steps in each batch (for easyfirst parsing)')
    args_parser.add_argument('--loss_type', choices=['sentence', 'token'], default='sentence', help='loss type (default: sentence)')
    args_parser.add_argument('--optim', choices=['sgd', 'adam'], help='type of optimizer')
    args_parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    args_parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam')
    args_parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam')
    args_parser.add_argument('--eps', type=float, default=1e-8, help='epsilon for adam or adamax')
    args_parser.add_argument('--lr_decay', type=float, default=0.999995, help='Decay rate of learning rate')
    args_parser.add_argument('--amsgrad', action='store_true', help='AMS Grad')
    args_parser.add_argument('--grad_clip', type=float, default=0, help='max norm for gradient clip (default 0: no clip')
    args_parser.add_argument('--warmup_steps', type=int, default=0, metavar='N', help='number of steps to warm up (default: 0)')
    args_parser.add_argument('--eval_every', type=int, default=100, help='eval every ? epochs')
    args_parser.add_argument('--noscreen', action='store_true', default=True, help='do not print middle log')
    args_parser.add_argument('--reset', type=int, default=10, help='Number of epochs to reset optimizer (default 10)')
    args_parser.add_argument('--weight_decay', type=float, default=0.0, help='weight for l2 norm decay')
    args_parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_parser.add_argument('--get_head_by_layer', action='store_true', help='whether to print head by graph attention layer.')
    args_parser.add_argument('--random_recomp', action='store_true', default=False, help='Whether to randomly sample recompute at test time.')
    args_parser.add_argument('--recomp_prob', type=float, default=0.25, help='Probability for random sampling of recompute at test time.')
    args_parser.add_argument('--sampler', choices=['random', 'from_model'], help='Sample strategy')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--beam', type=int, default=1, help='Beam size for decoding')
    args_parser.add_argument('--word_embedding', choices=['glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words')
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters')
    args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--train', help='path for training file.')
    args_parser.add_argument('--dev', help='path for dev file.')
    args_parser.add_argument('--test', help='path for test file.', required=True)
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)

    args = args_parser.parse_args()
    if args.symbolic_end == 'True':
        args.symbolic_end = True
    else:
        args.symbolic_end = False
    if args.mode == 'train':
        train(args)
    else:
        parse(args)
