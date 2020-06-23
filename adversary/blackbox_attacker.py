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
#from adversary.adv_attack import convert_tokens_to_ids

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

class BlackBoxAttacker(object):
    def __init__(self, model, candidates, vocab, synonyms, adv_lms=None, rel_ratio=0.5, fluency_ratio=0.2,
                max_perp_diff_per_token=0.8, perp_diff_thres=20 ,alphabets=None, tokenizer=None, 
                device=None, lm_device=None, symbolic_root=True, symbolic_end=False, mask_out_root=False, 
                batch_size=32, random_sub_if_no_change=False):
        super(BlackBoxAttacker, self).__init__()
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
        # if the arc is wrong, the rel must be wrong
        rels_change_mask = np.where(rels_change_mask+heads_change_mask>0, 1, 0)
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
                        chosen_rank_idx = random.randint(0, num_nochange_sub-1)
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