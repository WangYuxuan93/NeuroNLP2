import numpy as np
import re
import shutil

SPACE_NORMALIZER = re.compile(r"\s+")
def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class NoiseInjector(object):

    def __init__(self, corpus, shuffle_sigma=0.5,
                 replace_mean=0.2, replace_std=0.03,
                 delete_mean=0.1, delete_std=0.03,
                 add_mean=0.1, add_std=0.03):
        # READ-ONLY, do not modify
        self.corpus = corpus
        self.shuffle_sigma = shuffle_sigma
        self.replace_a, self.replace_b = self._solve_ab_given_mean_var(replace_mean, replace_std**2)
        self.delete_a, self.delete_b = self._solve_ab_given_mean_var(delete_mean, delete_std**2)
        self.add_a, self.add_b = self._solve_ab_given_mean_var(add_mean, add_std**2)
        self.keep_tok = "O"
        self.rep_tok = "R"
        self.insert_tok = "I"
        self.swap_tok = "S"
        #self.rep_pos_tok = "REP-POS"
        #self.rep_lem_tok = "REP-LEM"
        #self.spell_tok = "SPELL"
        
        #self.label2id = {self.keep_tok:0,self.rep_pos_tok:1,self.rep_lem_tok:2,
        #                self.insert_tok:3,self.spell_tok:4,self.swap_tok:5}
        self.label2id = {self.keep_tok:0,self.rep_tok:1,self.insert_tok:2,self.swap_tok:3}

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

    def _replace_func(self, tgt, labels):
        replace_ratio = np.random.beta(self.replace_a, self.replace_b)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, p in enumerate(tgt):
            if rnd[i] < replace_ratio and labels[i] == self.keep_tok: 
                rnd_ex = self.corpus[np.random.randint(len(self.corpus))]
                rnd_word = rnd_ex[np.random.randint(len(rnd_ex))]
                ret.append((-1, rnd_word))
                labels[i] = self.rep_tok
            else:
                ret.append(p)
        return ret, labels

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
                rnd_ex = self.corpus[np.random.randint(len(self.corpus))]
                rnd_word = rnd_ex[np.random.randint(len(rnd_ex))]
                ret.append((-1, rnd_word))
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

        funcs = [self._shuffle_func, self._add_func, self._replace_func, self._delete_func]
        #np.random.shuffle(funcs)
        
        pairs = [(i, w) for (i, w) in enumerate(tokens)]
        labels = [self.keep_tok for _ in range(len(pairs))]
        for f in funcs:
            pairs, labels = f(pairs, labels)
            #art, align = self._parse(pairs)

        #return self._parse(pairs)
        authentic_data = [w for (_,w) in pairs]
        return authentic_data, labels


def save_file(filename, contents):
    with open(filename, 'w') as ofile:
        for (data, label) in contents:
            for token, lab in zip(data, label):
                ofile.write(token+' '+lab+'\n')
            ofile.write('\n')

# make noise from filename
def noise(filename, output):
    lines = open(filename).readlines()
    tgts = [tokenize_line(line.strip()) for line in lines]
    noise_injector = NoiseInjector(tgts)
    
    datas = []
    for tgt in tgts:
        data, label = noise_injector.inject_noise(tgt)
        datas.append((data,label))
    
    save_file(output, datas)

import argparse

parser=argparse.ArgumentParser()
parser.add_argument('input', type=str, help='input grammatical file')
parser.add_argument('output', type=str, help='output authentic error data file')
#parser.add_argument('-e', '--epoch', type=int, default=10)
parser.add_argument('-s', '--seed', type=int, default=2468)

args = parser.parse_args()
np.random.seed(args.seed)
if __name__ == '__main__':
    print("seed={}".format( args.seed))

    noise(args.input, args.output)

