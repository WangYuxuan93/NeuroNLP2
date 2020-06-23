import nltk
from nltk.corpus import wordnet as wn
import spacy
nlp = spacy.load('en_core_web_sm')

from functools import partial
import numpy as np
import torch
import random
import os
import pickle
import tensorflow_hub as hub
import tensorflow as tf

from neuronlp2.io import get_logger
from neuronlp2.io.common import PAD, ROOT, END
from neuronlp2.io.common import PAD_CHAR, PAD, PAD_POS, PAD_TYPE, PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD
from neuronlp2.io import common
#from adversary.adv_attack import convert_tokens_to_ids

stopwords = set(
        [
            "a",
            "about",
            "above",
            "across",
            "after",
            "afterwards",
            "again",
            "against",
            "ain",
            "all",
            "almost",
            "alone",
            "along",
            "already",
            "also",
            "although",
            "am",
            "among",
            "amongst",
            "an",
            "and",
            "another",
            "any",
            "anyhow",
            "anyone",
            "anything",
            "anyway",
            "anywhere",
            "are",
            "aren",
            "aren't",
            "around",
            "as",
            "at",
            "back",
            "been",
            "before",
            "beforehand",
            "behind",
            "being",
            "below",
            "beside",
            "besides",
            "between",
            "beyond",
            "both",
            "but",
            "by",
            "can",
            "cannot",
            "could",
            "couldn",
            "couldn't",
            "d",
            "didn",
            "didn't",
            "doesn",
            "doesn't",
            "don",
            "don't",
            "down",
            "due",
            "during",
            "either",
            "else",
            "elsewhere",
            "empty",
            "enough",
            "even",
            "ever",
            "everyone",
            "everything",
            "everywhere",
            "except",
            "first",
            "for",
            "former",
            "formerly",
            "from",
            "hadn",
            "hadn't",
            "hasn",
            "hasn't",
            "haven",
            "haven't",
            "he",
            "hence",
            "her",
            "here",
            "hereafter",
            "hereby",
            "herein",
            "hereupon",
            "hers",
            "herself",
            "him",
            "himself",
            "his",
            "how",
            "however",
            "hundred",
            "i",
            "if",
            "in",
            "indeed",
            "into",
            "is",
            "isn",
            "isn't",
            "it",
            "it's",
            "its",
            "itself",
            "just",
            "latter",
            "latterly",
            "least",
            "ll",
            "may",
            "me",
            "meanwhile",
            "mightn",
            "mightn't",
            "mine",
            "more",
            "moreover",
            "most",
            "mostly",
            "must",
            "mustn",
            "mustn't",
            "my",
            "myself",
            "namely",
            "needn",
            "needn't",
            "neither",
            "never",
            "nevertheless",
            "next",
            "no",
            "nobody",
            "none",
            "noone",
            "nor",
            "not",
            "nothing",
            "now",
            "nowhere",
            "o",
            "of",
            "off",
            "on",
            "once",
            "one",
            "only",
            "onto",
            "or",
            "other",
            "others",
            "otherwise",
            "our",
            "ours",
            "ourselves",
            "out",
            "over",
            "per",
            "please",
            "s",
            "same",
            "shan",
            "shan't",
            "she",
            "she's",
            "should've",
            "shouldn",
            "shouldn't",
            "somehow",
            "something",
            "sometime",
            "somewhere",
            "such",
            "t",
            "than",
            "that",
            "that'll",
            "the",
            "their",
            "theirs",
            "them",
            "themselves",
            "then",
            "thence",
            "there",
            "thereafter",
            "thereby",
            "therefore",
            "therein",
            "thereupon",
            "these",
            "they",
            "this",
            "those",
            "through",
            "throughout",
            "thru",
            "thus",
            "to",
            "too",
            "toward",
            "towards",
            "under",
            "unless",
            "until",
            "up",
            "upon",
            "used",
            "ve",
            "was",
            "wasn",
            "wasn't",
            "we",
            "were",
            "weren",
            "weren't",
            "what",
            "whatever",
            "when",
            "whence",
            "whenever",
            "where",
            "whereafter",
            "whereas",
            "whereby",
            "wherein",
            "whereupon",
            "wherever",
            "whether",
            "which",
            "while",
            "whither",
            "who",
            "whoever",
            "whole",
            "whom",
            "whose",
            "why",
            "with",
            "within",
            "without",
            "won",
            "won't",
            "would",
            "wouldn",
            "wouldn't",
            "y",
            "yet",
            "you",
            "you'd",
            "you'll",
            "you're",
            "you've",
            "your",
            "yours",
            "yourself",
            "yourselves",
        ]
    )

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

def recover_word_case(word, reference_word):
    """ Makes the case of `word` like the case of `reference_word`. Supports 
        lowercase, UPPERCASE, and Capitalized. """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word

class BlackBoxAttacker(object):
    def __init__(self, model, candidates, vocab, synonyms, filters=['word_sim', 'sent_sim', 'lm'],
                knn_path=None, max_knn_candidates=50, sent_encoder_path=None,
                min_word_cos_sim=0.8, min_sent_cos_sim=0.8,  
                adv_lms=None, rel_ratio=0.5, fluency_ratio=0.2,
                max_perp_diff_per_token=0.8, perp_diff_thres=20 ,alphabets=None, tokenizer=None, 
                device=None, lm_device=None, symbolic_root=True, symbolic_end=False, mask_out_root=False, 
                batch_size=32, random_sub_if_no_change=False):
        super(BlackBoxAttacker, self).__init__()
        logger = get_logger("Attacker")
        self.model = model
        self.candidates = candidates
        self.synonyms = synonyms
        self.word2id = vocab
        self.id2word = {i:w for (w,i) in vocab.items()}
        if knn_path is not None:
            logger.info("Loading knn from: {}".format(knn_path))
            self.load_knn_path(knn_path)
            logger.info("Min word cosine similarity: {}".format(min_word_cos_sim))
        else:
            self.nn = None
        if sent_encoder_path is not None:
            logger.info("Loading sent encoder from: {}".format(sent_encoder_path))
            self.sent_encoder = hub.load(sent_encoder_path)
            logger.info("Min sent cosine similarity: {}".format(min_sent_cos_sim))
        else:
            self.sent_encoder = None
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
        #self.stop_words = nltk.corpus.stopwords.words('english')
        self.stop_words = stopwords
        self.random_sub_if_no_change = random_sub_if_no_change
        
        logger.info("Relation ratio:{}, Fluency ratio:{}".format(rel_ratio, fluency_ratio))
        logger.info("Max ppl difference per token:{}, ppl diff threshold:{}".format(max_perp_diff_per_token, perp_diff_thres))
        logger.info("Randomly substitute if no change:{}".format(self.random_sub_if_no_change))
        self.max_knn_candidates = max_knn_candidates
        self.min_word_cos_sim = min_word_cos_sim
        self.min_sent_cos_sim = min_sent_cos_sim
        
        self.filters = filters
        if 'word_sim' in self.filters and self.nn is None:
            print ("Must input embedding path for word cos sim filter!")
            exit()
        if 'sent_sim' in self.filters and self.sent_encoder is None:
            print ("Must input sentence encoder path for sent cos sim filter!")
            exit()
        if 'lm' in self.filters and self.adv_lm is None:
            print ("Must input language model (gpt2) path for lm filter!")
            exit()
        
        logger.info("Filters: {}".format(filters))

    def load_knn_path(self, path):
        word_embeddings_file = "paragram.npy"
        word_list_file = "wordlist.pickle"
        nn_matrix_file = "nn.npy"
        cos_sim_file = "cos_sim.p"
        word_embeddings_file = os.path.join(path, word_embeddings_file)
        word_list_file = os.path.join(path, word_list_file)
        nn_matrix_file = os.path.join(path, nn_matrix_file)
        cos_sim_file = os.path.join(path, cos_sim_file)

        self.word_embeddings = np.load(word_embeddings_file)
        self.word_embedding_word2index = np.load(word_list_file, allow_pickle=True)
        self.nn = np.load(nn_matrix_file)
        with open(cos_sim_file, "rb") as f:
            self.cos_sim_mat = pickle.load(f)

        # Build glove dict and index.
        self.word_embedding_index2word = {}
        for word, index in self.word_embedding_word2index.items():
            self.word_embedding_index2word[index] = word

    def _get_knn_words(self, word):
        """ Returns a list of possible 'candidate words' to replace a word in a sentence 
            or phrase. Based on nearest neighbors selected word embeddings.
        """
        try:
            word_id = self.word_embedding_word2index[word.lower()]
            nnids = self.nn[word_id][1 : self.max_knn_candidates + 1]
            candidate_words = []
            for i, nbr_id in enumerate(nnids):
                nbr_word = self.word_embedding_index2word[nbr_id]
                candidate_words.append(recover_word_case(nbr_word, word))
            return candidate_words
        except KeyError:
            # This word is not in our word embedding database, so return an empty list.
            return []

    def get_word_cos_sim(self, a, b):
        """ Returns the cosine similarity of words with IDs a and b."""
        if a not in self.word_embedding_word2index or b not in self.word_embedding_word2index:
            return None
        if isinstance(a, str):
            a = self.word_embedding_word2index[a]
        if isinstance(b, str):
            b = self.word_embedding_word2index[b]
        a, b = min(a, b), max(a, b)
        try:
            cos_sim = self.cos_sim_mat[a][b]
        except KeyError:
            e1 = self.word_embeddings[a]
            e2 = self.word_embeddings[b]
            e1 = torch.tensor(e1)
            e2 = torch.tensor(e2)
            cos_sim = torch.nn.CosineSimilarity(dim=0)(e1, e2).numpy()
            self.cos_sim_mat[a][b] = cos_sim
        return cos_sim

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

    def filter_cands_with_lm(self, tokens, cands, idx, debug=False):
        new_cands, new_perp_diff = [], []
        # (cand_size)
        perp_diff = self.get_perp_diff(tokens, cands, idx)
        for i in range(len(cands)):
            if perp_diff[i] <= self.perp_diff_thres:
                new_cands.append(cands[i])
                new_perp_diff.append(perp_diff[i])
        return new_cands, np.array(new_perp_diff), perp_diff

    def filter_cands_with_word_sim(self, token, cands, debug=False):
        new_cands= []
        new_sims, all_sims = [], []
        # (cand_size)
        for i in range(len(cands)):
            sim = self.get_word_cos_sim(token.lower(), cands[i].lower())
            all_sims.append(sim)
            if sim is not None and sim >= self.min_word_cos_sim:
                new_cands.append(cands[i])
                new_sims.append(sim)
        return new_cands, new_sims, all_sims

    def cos_sim(self, e1, e2):
        e1 = torch.tensor(e1)
        e2 = torch.tensor(e2)
        cos_sim = torch.nn.CosineSimilarity(dim=0)(e1, e2)
        return cos_sim.numpy()

    def filter_cands_with_sent_sim(self, tokens, cands, idx, debug=False):
        batch_tokens, _ = self.gen_cand_batch(tokens, cands, idx, tokens)
        sents = [' '.join(toks) for toks in batch_tokens]
        # sent-0 is original sent
        with tf.device('/cpu:0'):
            sent_embeds = self.sent_encoder(sents).numpy()
        new_cands= []
        new_sims, all_sims = [], []
        # (cand_size)
        for i in range(1,len(cands)):
            sim = self.cos_sim(sent_embeds[i], sent_embeds[0])
            all_sims.append(sim)
            if sim >= self.min_sent_cos_sim:
                new_cands.append(cands[i])
                new_sims.append(sim)
        return new_cands, new_sims, all_sims

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
        if token in self.stop_words:
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
        if token.lower() in self.stop_words:
            return []
        sememe_cands = self.get_sememe_cands(token, tag)
        #print ("sememe:", sememe_cands)
        synonyms = self.get_synonyms(token, tag)
        #print ("syn:", synonyms)
        candidate_set = sememe_cands
        for syn in synonyms:
            if syn not in candidate_set:
                candidate_set.append(syn)
        if self.nn is not None:
            knn_cands = self._get_knn_words(token)
            for c in knn_cands:
                if c not in candidate_set:
                    candidate_set.append(c)
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
            if "word_sim" in self.filters:
                cands, w_sims, all_w_sims = self.filter_cands_with_word_sim(adv_tokens[idx], cands)
                if len(cands) == 0:
                    if debug == 3:
                        print ("--------------------------")
                        print ("Idx={}({}), all word_sim less than min, continue\ncands:{}\nword_sims:{}".format(idx, tokens[idx], all_cands, all_w_sims))
                else:
                    print ("--------------------------")
                    print ("Idx={}({})\ncands:{}\nword_sims:{}".format(idx, tokens[idx], all_cands, all_w_sims))
            all_cands = cands.copy()
            if "sent_sim" in self.filters:
                cands, s_sims, all_s_sims = self.filter_cands_with_sent_sim(adv_tokens, cands, idx)
                if len(cands) == 0:
                    if debug == 3:
                        print ("--------------------------")
                        print ("Idx={}({}), all sent_sim less than min, continue\ncands:{}\nsent_sims:{}".format(idx, tokens[idx], all_cands, all_s_sims))
                else:
                    print ("--------------------------")
                    print ("Idx={}({})\ncands:{}\nsent_sims:{}".format(idx, tokens[idx], all_cands, all_s_sims))
            # filter with language model
            all_cands = cands.copy()
            if "lm" in self.filters:
                cands, perp_diff, all_perp_diff = self.filter_cands_with_lm(adv_tokens, cands, idx, debug==2)
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
                    if ("lm" in self.filters and total_perp_diff>max_perp_diff):
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
                        if "lm" in self.filters:
                            total_perp_diff += blocked_perp_diff[chosen_idx]
                        if debug == 3:
                            print ("--------------------------")
                            print ("Idx={}({}), randomly chose:{} since no cand makes change\ncands:{}\nchange_scores:{}".format(idx, tokens[idx], cands[chosen_idx], cands, change_score))
                            if "lm" in self.filters:
                                print ("perp diff: {}".format(perp_diff))  
                continue
            if "lm" in self.filters:
                # penalize the score for disfluency substitution
                # if the perplexity of new sent is lower than the original one, no bonus
                score = (1 - self.fluency_ratio) * change_score - self.fluency_ratio * blocked_perp_diff
            else:
                score = change_score
            best_cand_idx = self.get_best_cand(score, change_score)
            if best_cand_idx is None:
                print ("--------------------------")
                print ("Idx={}({}), can't find best cand, continue\ncands:{}\nchange_scores:{}".format(idx, tokens[idx], cands, change_score))
                if "lm" in self.filters:
                        print ("perp diff: {}\nscores: {}".format(perp_diff, score))
                continue
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
                adv_tokens[idx] = best_cand
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
        if adv_tokens == tokens:
            return None
        return adv_tokens, num_edit, total_score, total_change_score, total_perp_diff
