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

class GrayBoxSingleAttacker(BlackBoxAttacker):
    def __init__(self, *args, **kwargs):
        super(GrayBoxSingleAttacker, self).__init__(*args, **kwargs)

    def get_probs(self, tokens, tags):
        """
        Input:
            tokens: List[List[str]], (batch, seq_len)
            tags: List[List[str]], (batch, seq_len)
        Output:
            heads_pred: (batch, seq_len)
            rels_pred: (batch, seq_len)
        """
        self.model.eval()
        heads_prob_list, rels_prob_list = [], []
        with torch.no_grad():
            for words, pres, chars, pos, masks, bpes, first_idx, lan_id in self.str2id(tokens, tags):
                heads_prob, rels_prob = self.model.get_probs(words, pres, chars, pos, mask=masks, 
                    bpes=bpes, first_idx=first_idx, lan_id=lan_id, leading_symbolic=common.NUM_SYMBOLIC_TAGS)
                heads_prob_list.append(heads_prob.detach().cpu())
                rels_prob_list.append(rels_prob.detach().cpu())
        heads_prob = torch.cat(heads_prob_list, dim=0)
        rels_prob = torch.cat(rels_prob_list, dim=0)
        return heads_prob, rels_prob

    def calc_prob_change(self, batch_tokens, batch_tags, heads, rel_ids, debug=False):
        heads_pred, rels_pred = self.get_prediction(batch_tokens, batch_tags)
        heads_prob, rels_prob = self.get_probs(batch_tokens, batch_tags)
        heads_prob = heads_prob.permute(0,2,1)
        heads_gold = torch.from_numpy(np.tile(np.array(heads), (len(heads_pred),1)))
        #print ("heads_pred:\n", heads_pred)
        #print ("heads_prob:\n", heads_prob)
        #print (heads_gold)
        # compute the probs of the gold heads
        probs = torch.gather(heads_prob, -1, heads_gold.unsqueeze(-1)).squeeze(-1)
        #print (probs)
        prob_change = probs[1:, :] - probs[0, :]
        #print (prob_change)
        prob_change = prob_change.sum(-1)
        #print ("rels_pred:\n", rels_pred)
        #print ("rels_prob:\n", rels_prob)
        # (cand_size), how much has the gold prob changed
        return prob_change

    def get_prob_change_score(self, tokens, cands, idx, tags, heads, rel_ids, debug=False):
        batch_tokens, batch_tags = self.gen_cand_batch(tokens, cands, idx, tags)
        # (cand_size+1), the 1st is the original sentence
        prob_change = self.calc_prob_change(batch_tokens, batch_tags, heads, rel_ids, debug)
        if debug:
            #print ("batch tokens:\n", batch_tokens)
            print ("prob_change:\n", prob_change)
            #print ("word_rank:\n", word_rank)
        return prob_change

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

        if debug == 3:
            #print ("tokens:\n", adv_tokens)
            print ("importance rank:\n", word_rank)

        no_profit_change_track = []
        for idx in word_rank:
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
            prob_change = self.get_prob_change_score(adv_tokens, cands, idx, tags, heads, rel_ids, debug==2)
            prob_change_rank = prob_change.argsort()
            #print ("prob_change:", prob_change)
            #print ("prob_change_rank:", prob_change_rank)
            # (cand_size)
            change_score, head_change, rel_change = self.get_change_score(adv_tokens, cands, idx, tags, heads, rel_ids, debug==2)
            change_rank = (-change_score).argsort()
            
            # choose the cand that mostly decrease the gold head prob
            best_prob_idx = prob_change_rank[0]
            # this means the best change can not decrease the gold prob
            if prob_change[best_prob_idx] >= 0:
                if debug == 3:
                    print ("--------------------------")
                    print ("Idx={}({}), no cand decreases prob\ncands:{}\nprob_change:{}".format(idx, tokens[idx], 
                                cands, prob_change))
                    if "lm" in self.filters:
                        print ("perp diff: {}".format(perp_diff))
                continue

            if "lm" in self.filters:
                # penalize the score for disfluency substitution
                # if the perplexity of new sent is lower than the original one, no bonus
                score = (1 - self.fluency_ratio) * (-prob_change) - self.fluency_ratio * blocked_perp_diff
            else:
                score = -prob_change

            best_cand_idx = self.get_best_cand(score, -prob_change)
            if best_cand_idx is None:
                if debug == 3:
                    print ("--------------------------")
                    print ("Idx={}({}), can not find best cand\ncands:{}\nprob_change:{}\nscore:{}".format(idx, tokens[idx], 
                                cands, prob_change, score))
                    if "lm" in self.filters:
                        print ("perp diff: {}".format(perp_diff))
                continue
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
                    # keep track of substitutions that has no profit yet
                    no_profit_change_track.append(idx)
                if "lm" in self.filters:
                    total_perp_diff += blocked_perp_diff[best_cand_idx]
                if debug == 3:
                    print ("--------------------------")
                    print ("Idx={}({}), chosen cand:{}, prob drop:{:.4f}, total_change_score:{}, change_edit_ratio:{}\ncands: {}\nprob_change: {}\nchange_scores: {}".format(
                            idx, tokens[idx], best_cand, prob_change[best_cand_idx], total_change_score, change_edit_ratio, cands, prob_change, change_score))
                    if "lm" in self.filters:
                        print ("perp diff: {}\nscores: {}".format(perp_diff, score))
            else:
                if debug == 3:
                    print ("------------Stopping------------")
                    print ("Idx={}({}), chosen cand:{}, total_change_score:{}, change_edit_ratio:{}\ncands: {}\nprob_change: {}\nchange_scores: {}".format(
                            idx, tokens[idx], best_cand, total_change_score, change_edit_ratio, cands, prob_change, change_score))
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