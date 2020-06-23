import nltk
from nltk.corpus import wordnet as wn
import spacy
nlp = spacy.load('en_core_web_sm')

from functools import partial
import numpy as np
import torch

from neuronlp2.io import get_logger
from neuronlp2.io.common import PAD_CHAR, PAD, PAD_POS, PAD_TYPE, PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD
from neuronlp2.io import common
from adversary.blackbox_attacker import BlackBoxAttacker

class GrayBoxAttacker(BlackBoxAttacker):
    def __init__(self, *args, **kwargs):
        super(GrayBoxAttacker, self).__init__(*args, **kwargs)

    def get_prob_change_score(self, tokens, cands, idx, tags, heads, rel_ids, debug=False):
        batch_tokens, batch_tags = self.gen_cand_batch(tokens, cands, idx, tags)
        # (cand_size+1), the 1st is the original sentence
        change_score = self.calc_prob_change(batch_tokens, batch_tags, heads, rel_ids, debug)
        # ignore the original sent
        change_score = change_score[1:]
        if debug:
            #print ("batch tokens:\n", batch_tokens)
            print ("importance:\n", change_score)
            #print ("word_rank:\n", word_rank)
        return change_score

        
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
                if not self.random_sub_if_no_change or change_score[change_rank[0]] < 0:
                    if debug == 3:
                        print ("--------------------------")
                        print ("Idx={}({}), no cand can make change, continue\ncands:{}\nchange_scores:{}".format(idx, tokens[idx], cands, change_score))
                else:
                    if (self.adv_lm is not None and total_perp_diff>max_perp_diff):
                        continue
                    else:
                        num_nochange_sub = 0
                        for i in range(len(change_rank)):
                            if change_score[change_rank[i]] == 0:
                                num_nochange_sub += 1
                            else:
                                break
                        # only choose the subs that will not reduce the error
                        chosen_rank_idx = random.randint(0, len(num_nochange_sub)-1)
                        chosen_idx = change_rank[chosen_rank_idx]
                        adv_tokens[idx] = cands[chosen_idx]
                        num_edit += 1
                        if self.adv_lm is not None:
                            total_perp_diff += blocked_perp_diff[chosen_idx]
                        if debug == 3:
                            print ("--------------------------")
                            print ("Idx={}({}), randomly chose:{} since no cand makes change\ncands:{}\nchange_scores:{}".format(idx, tokens[idx], cands[chosen_idx], cands, change_score))
                            if self.adv_lm is not None:
                                print ("perp diff: {}".format(perp_diff))  
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