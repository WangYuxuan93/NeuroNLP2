import numpy as np
import re
import shutil
import json
import string
import random
import sys

SPACE_NORMALIZER = re.compile(r"\s+")
def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class NoiseInjector(object):

    def __init__(self, pos_dict, lem_dict, shuffle_sigma=0.35,
                 rep_pos_mean=0.15, rep_pos_std=0.03,
                 rep_lem_mean=0.3, rep_lem_std=0.03,
                 delete_mean=0.1, delete_std=0.03,
                 add_mean=0.1, add_std=0.03,
                 spell_mean=0.1, spell_std=0.03):
        # READ-ONLY, do not modify
        self.pos_dict = pos_dict
        self.lem_dict = lem_dict
        tokens = {}
        for val in self.pos_dict.values():
            tokens.update(val)
        self.corpus = list(tokens.keys())
        print ("corpus len: %d" % len(self.corpus))
        self.shuffle_sigma = shuffle_sigma
        self.rep_pos_a, self.rep_pos_b = self._solve_ab_given_mean_var(rep_pos_mean, rep_pos_std**2)
        self.rep_lem_a, self.rep_lem_b = self._solve_ab_given_mean_var(rep_lem_mean, rep_lem_std**2)
        self.add_a, self.add_b = self._solve_ab_given_mean_var(add_mean, add_std**2)
        self.spell_a, self.spell_b = self._solve_ab_given_mean_var(spell_mean, spell_std**2)
        self.delete_a, self.delete_b = self._solve_ab_given_mean_var(delete_mean, delete_std**2)
        self.keep_tok = "O"
        self.rep_pos_tok = "R-P"
        self.rep_lem_tok = "R-L"
        self.insert_tok = "I"
        self.spell_tok = "S"
        self.swap_tok = "W"
        #self.rep_pos_tok = "REP-POS"
        #self.rep_lem_tok = "REP-LEM"
        #self.spell_tok = "SPELL"
        
        #self.label2id = {self.keep_tok:0,self.rep_pos_tok:1,self.rep_lem_tok:2,
        #                self.insert_tok:3,self.spell_tok:4,self.swap_tok:5}
        #self.label2id = {self.keep_tok:0,self.rep_tok:1,self.insert_tok:2,self.swap_tok:3}

    @staticmethod
    def _solve_ab_given_mean_var(mean, var):
        a = mean * mean * (1. - mean) / var - mean
        b = (1. - mean) * (mean * (1. - mean) / var - 1.)
        return a, b

    def _shuffle_func(self, tgt, labels):
        if self.shuffle_sigma < 1e-6:
            return tgt

        shuffle_key = [i + np.random.normal(loc=0, scale=self.shuffle_sigma) for i in range(len(tgt))]
        new_idx = np.argsort(shuffle_key)
        res = [tgt[i] for i in new_idx]
        for i in range(len(new_idx)):
            if not i == new_idx[i]:
                labels[i] = self.swap_tok
            
        return res, labels

    def _rep_pos_func(self, tgt, labels):
        replace_ratio = np.random.beta(self.rep_pos_a, self.rep_pos_b)
        ret = []
        new_label = []
        rnd = np.random.random(len(tgt))
        for i, p in enumerate(tgt):
            pos = p[1][2]
            word = p[1][0]
            if rnd[i] < replace_ratio and labels[i] == self.keep_tok and pos in self.pos_dict: 
                #rnd_ex = self.corpus[np.random.randint(len(self.corpus))]
                #rnd_word = rnd_ex[np.random.randint(len(rnd_ex))]
                cand_list = list(self.pos_dict[pos].keys())
                rnd_word = cand_list[np.random.randint(len(cand_list))]
                ret.append((-1, (rnd_word,pos,'-')))
                if not rnd_word == word:
                    new_label.append(self.rep_pos_tok)
                else:
                    new_label.append(labels[i])
            else:
                ret.append(p)
                new_label.append(labels[i])
        return ret, new_label

    def _rep_lem_func(self, tgt, labels):
        replace_ratio = np.random.beta(self.rep_lem_a, self.rep_lem_b)
        ret = []
        new_label = []
        rnd = np.random.random(len(tgt))
        for i, p in enumerate(tgt):
            lemma = p[1][1]
            word = p[1][0]
            if rnd[i] < replace_ratio and labels[i] == self.keep_tok and lemma in self.lem_dict: 
                #rnd_ex = self.corpus[np.random.randint(len(self.corpus))]
                #rnd_word = rnd_ex[np.random.randint(len(rnd_ex))]
                cand_list = list(self.lem_dict[lemma].keys())
                rnd_word = cand_list[np.random.randint(len(cand_list))]
                ret.append((-1, (rnd_word,'-',lemma)))
                if not rnd_word == word:
                    new_label.append(self.rep_lem_tok)
                else:
                    new_label.append(labels[i])
            else:
                ret.append(p)
                new_label.append(labels[i])
        return ret, new_label

    def _spell_func(self, tgt, labels):
        spell_ratio = np.random.beta(self.spell_a, self.spell_b)
        ret = []
        new_label = []
        rnd = np.random.random(len(tgt))
        for i, p in enumerate(tgt):
            if rnd[i] < spell_ratio and labels[i] == self.keep_tok:
                word = p[1][0]
                idx = np.random.randint(len(word))
                add_char = True if len(word) == 1 or np.random.random() > 0.5 else False
                if add_char:
                    letter = random.choice(string.ascii_letters)
                    new_word = word[:idx] + letter + word[idx:]
                else:
                    new_word = word[:idx] + word[idx+1:]
                ret.append((p[0], (new_word,p[1][1],p[1][2])))
                new_label.append(self.spell_tok)
            else:
                ret.append(p)
                new_label.append(labels[i])
        return ret, new_label

    def _delete_func(self, tgt, labels):
        delete_ratio = np.random.beta(self.delete_a, self.delete_b)
        ret = []
        new_label = []
        rnd = np.random.random(len(tgt))
        for i, p in enumerate(tgt):
            if rnd[i] < delete_ratio:
                continue
            ret.append(p)
            new_label.append(labels[i])
        return ret, new_label

    def _add_func(self, tgt, labels):
        add_ratio = np.random.beta(self.add_a, self.add_b)
        ret = []
        new_label = []
        rnd = np.random.random(len(tgt))
        for i, p in enumerate(tgt):
            if rnd[i] < add_ratio:
                rnd_word = self.corpus[np.random.randint(len(self.corpus))]
                ret.append((-1, (rnd_word, '-', '-')))
                new_label.append(self.insert_tok)
            ret.append(p)
            new_label.append(labels[i])

        return ret, new_label

    def _parse(self, pairs):
        align = []
        art = []
        for si in range(len(pairs)):
            ti = pairs[si][0]
            w = pairs[si][1]
            art.append(w)
            if ti >= 0:
                align.append('{}-{}'.format(si, ti))
        return art, align

    def inject_noise(self, tokens):
        # tgt is a vector of integers

        funcs = [self._add_func, self._rep_pos_func, self._rep_lem_func, self._spell_func, 
                    self._delete_func]
        np.random.shuffle(funcs)
        funcs = [self._shuffle_func] + funcs
        
        pairs = [(i, w) for (i, w) in enumerate(tokens)]
        labels = [self.keep_tok for _ in range(len(pairs))]
        for f in funcs:
            pairs, labels = f(pairs, labels)
            #art, align = self._parse(pairs)

        #return self._parse(pairs)
        authentic_data = [w[0] for (_,w) in pairs]
        return authentic_data, labels


def save_file(filename, contents):
    cnt = {"O":0, "R-P":0, "R-L":0, "I":0, "S":0, "W":0}
    with open(filename, 'w') as ofile:
        for (data, label) in contents:
            for token, lab in zip(data, label):
                cnt[lab] += 1
                ofile.write(token+' '+lab+'\n')
            ofile.write('\n')
    print ("\n", cnt)
    total = sum(list(cnt.values()))
    percent = {}
    for key, val in cnt.items():
        percent[key] = float(val) / total
    print ("percent: ", percent)

def read_conllu(filename, pos_type):
    sents = []
    with open(filename, 'r') as fi:
        sent = []
        line = fi.readline()
        while line:
            line = line.strip()
            if (not line or line.startswith('#')):
                if sent:
                    sents.append(sent)
                    sent = []
            else:
                items = line.split('\t')
                pos = items[3] if pos_type == 'upos' else items[4]
                # (word, lemma, pos)
                sent.append((items[1], items[2], pos))
            line = fi.readline()
        if sent:
            sents.append(sent)
    return sents

def load_json(filename):
    data = open(filename, 'r').read().strip()
    return json.loads(data)

import argparse

parser=argparse.ArgumentParser()
parser.add_argument('input', type=str, help='input grammatical file in conllu format')
parser.add_argument('output', type=str, help='output authentic error data file')
parser.add_argument('-l', '--lemma', type=str, help='path to lemma cluster file')
parser.add_argument('-p', '--pos', type=str, help='path to pos cluster file')
parser.add_argument('--pos_type', type=str, default='xpos', choices=['upos', 'xpos'], help='type of pos to use')
parser.add_argument('-s', '--seed', type=int, default=2468)

args = parser.parse_args()
np.random.seed(args.seed)
if __name__ == '__main__':
    print("seed={}".format(args.seed))

    sents = read_conllu(args.input, args.pos_type)
    pos_dict = load_json(args.pos)
    lem_dict = load_json(args.lemma)
    noise_injector = NoiseInjector(pos_dict, lem_dict)
    
    datas = []
    for i, sent in enumerate(sents):
        if i % 100000 == 0:
            print ("%d... "%i, end="")
            sys.stdout.flush()
        data, label = noise_injector.inject_noise(sent)
        datas.append((data,label))
    
    save_file(args.output, datas)

