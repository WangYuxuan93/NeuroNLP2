import pickle
import argparse
import json

def load_conll(f):
	data = []
	sents = f.read().strip().split("\n\n")
	for sent in sents:
		data.append([line.strip().split("\t") for line in sent.strip().split("\n")])
	return data

def compute_tag(datas):
	tag_cnt = {}
	for data in datas:
		for sent in data:
			for line in sent:
				word = line[1]
				tag = line[3]
				if word not in tag_cnt:
					tag_cnt[word] = {}
				if tag not in tag_cnt[word]:
					tag_cnt[word][tag] = 0
				tag_cnt[word][tag] += 1
	return tag_cnt

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="input conll file")
parser.add_argument("output", type=str, default="vocab.json", help="output path for vocab")
#parser.add_argument("--use_id", action="store_true", help="whether use id to store")
#parser.add_argument("--save_binary", action="store_true", help="whether save in binary")
args = parser.parse_args()

files = args.input.split(":")
datas = []
for file in files:
	with open(file, 'r') as fi:
		datas.append(load_conll(fi))

tag_cnt = compute_tag(datas)

with open(args.output, 'w') as fo:
	fo.write(json.dumps(tag_cnt, indent=4))
