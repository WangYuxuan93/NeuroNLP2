import pickle
import argparse
import json
from scipy.stats import entropy
import numpy as np

def load_conll(f):
	data = []
	sents = f.read().strip().split("\n\n")
	for sent in sents:
		data.append([line.strip().split("\t") for line in sent.strip().split("\n")])
	return data

def ent(tag_set):
	cnts = list(tag_set.values())
	tot = sum(cnts)
	probs = np.array(cnts) / float(tot)
	#print (tag_set)
	#print (probs)
	#print (entropy(probs))
	return entropy(probs)

def compute(vocab, tag_cnt):
	n_tot, n_oov = 0, 0
	n_iv = 0
	n_tot_tag_num = 0
	n_tot_ent = 0.0
	for tok, num in vocab.items():
		n_tot += num
		if tok not in tag_cnt:
			n_oov += num
		else:
			n_iv += num
			tag_set = tag_cnt[tok]
			n_tot_tag_num += len(tag_set) * num
			n_tot_ent += ent(tag_set) * num

	avg_tag_num = n_tot_tag_num / float(n_iv)
	avg_ent = n_tot_ent / float(n_iv)
	print ("Total:{},OOV:{},IV:{}".format(n_tot, n_oov, n_iv))
	print ("AVG tag num:{}, entropy:{}".format(avg_tag_num, avg_ent))

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="input vocab file (targets.json or subs.json)")
parser.add_argument("tag_cnt", type=str, help="input path for pos tag cnts from ptb data")
#parser.add_argument("--use_id", action="store_true", help="whether use id to store")
args = parser.parse_args()

vocab = json.load(open(args.input, 'r'))

tag_cnt = json.load(open(args.tag_cnt, 'r'))

compute(vocab, tag_cnt)