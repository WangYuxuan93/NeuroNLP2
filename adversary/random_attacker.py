import nltk
from nltk.corpus import wordnet as wn
import spacy
nlp = spacy.load('en_core_web_sm')

from functools import partial
import numpy as np
import torch
import random

from neuronlp2.io import get_logger
from neuronlp2.io.common import PAD_CHAR, PAD, PAD_POS, PAD_TYPE, PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD
from neuronlp2.io import common
from adversary.blackbox_attacker import BlackBoxAttacker

class RandomAttacker(BlackBoxAttacker):
    def __init__(self, *args, **kwargs):
        super(RandomAttacker, self).__init__(*args, **kwargs)

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
        x_len = len(tokens)
        tag_list = ['JJ', 'NN', 'RB', 'VB']
        neigbhours_list = []
        #stop_words = nltk.corpus.stopwords.words('english')
        for i in range(x_len):
            #print (adv_tokens[i], self._word2id(adv_tokens[i]))
            neigbhours_list.append(self.get_candidate_set(adv_tokens, tags[i], i))
        neighbours_len = [len(x) for x in neigbhours_list]
        #print (neigbhours_list)
        if np.sum(neighbours_len) == 0:
            return None
        change_edit_ratio = -1
        total_change_score = 0
        total_head_change = 0
        total_rel_change = 0
        total_perp_diff = 0.0
        total_score = 0
        num_edit = 0
        max_perp_diff = x_len * self.max_perp_diff_per_token

        sub_order = np.arange(len(adv_tokens))
        np.random.shuffle(sub_order)
        if debug == 3:
            #print ("tokens:\n", adv_tokens)
            print ("random substitute order:\n", sub_order)

        no_profit_change_track = []
        for idx in sub_order:
            if neighbours_len[idx] == 0:
                if debug == 3:
                    print ("--------------------------")
                    print ("Idx={}({}), no cands, continue".format(idx, tokens[idx]))
                continue
            # skip the edit for ROOT
            if self.symbolic_root and idx == 0: continue
            cands = neigbhours_list[idx]

            cands, perp_diff = self.filter_cands(adv_tokens, cands, idx, debug=debug)
            if len(cands) == 0:
                if debug == 3:
                    print ("--------------------------")
                    print ("Idx={}({}), all cands filtered out, continue".format(idx, tokens[idx]))
                continue
            all_cands = cands.copy()
            if "lm" in self.filters:
                blocked_perp_diff = np.where(perp_diff>0, perp_diff, 0)
            # (cand_size)
            change_score, head_change, rel_change = self.get_change_score(adv_tokens, cands, idx, tags, heads, rel_ids, debug==2)
            if "lm" in self.filters:
                # penalize the score for disfluency substitution
                # if the perplexity of new sent is lower than the original one, no bonus
                score = (1 - self.fluency_ratio) * change_score - self.fluency_ratio * blocked_perp_diff
            else:
                score = change_score
            best_cand_idx = np.random.randint(0, len(cands))
            #cand_rank = (-score).argsort()
            best_cand = cands[best_cand_idx]
            best_c_score = change_score[best_cand_idx]
            best_score = score[best_cand_idx]
            new_ratio = (total_change_score + best_c_score) / (num_edit + 1)
            if ("lm" in self.filters and total_perp_diff<=max_perp_diff) or (new_ratio > change_edit_ratio):
                change_edit_ratio = new_ratio
                num_edit += 1
                total_change_score += best_c_score
                total_score += best_score
                total_head_change += head_change[best_cand_idx]
                total_rel_change += rel_change[best_cand_idx]
                adv_tokens[idx] = best_cand
                if best_c_score > 0:
                    # clear the track once there is profit
                    no_profit_change_track = []
                else:
                    no_profit_change_track.append(idx)
                if "lm" in self.filters:
                    total_perp_diff += blocked_perp_diff[best_cand_idx]
                if debug == 3:
                    print ("--------------------------")
                    print ("Idx={}({}), chosen cand:{}, total_change_score:{}, change_edit_ratio:{}\ncands: {}\nchange_scores: {}".format(
                            idx, tokens[idx], best_cand, total_change_score, change_edit_ratio, cands, change_score))
                    if "lm" in self.filters:
                        print ("perp diff: {}\nscores: {}".format(perp_diff, score))
            else:
                if debug == 3:
                    print ("------------Stopping------------")
                    print ("Idx={}({}), chosen cand:{}, total_change_score:{}, change_edit_ratio:{}\ncands: {}\nchange_scores: {}".format(
                            idx, tokens[idx], best_cand, total_change_score, change_edit_ratio, cands, change_score))
                    if "lm" in self.filters:
                        print ("perp diff: {}\nscores: {}".format(perp_diff, score))
                break
        # if there is still changes that has no profit after search, recover them
        if no_profit_change_track:
            for idx in no_profit_change_track:
                adv_tokens[idx] = tokens[idx]
                num_edit -= 1
        if adv_tokens == tokens:
            return None
        sent_str = ""
        for x,y in zip(tokens, adv_tokens):
            if x == y:
                sent_str += x + " "
            else:
                sent_str += y + " [ " + x + " ] "
        print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print ("Success attack (change: head:{}, rel:{}, score:{}), adv sent:\n{}".format(
                total_head_change, total_rel_change, total_change_score, sent_str))
        adv_infos = (num_edit, total_score, total_change_score, total_perp_diff,
                    total_head_change, total_rel_change)
        return adv_tokens, adv_infos