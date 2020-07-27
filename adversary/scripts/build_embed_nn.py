import pickle
import gzip
import numpy as np
import json
import argparse
import os, sys
from collections import OrderedDict, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import torch
import multiprocessing

def load_embedding_dict(path, skip_first=True, normalize_digits=False):

	print("loading embedding from %s" % (path))
	word2id = OrderedDict()
	embedd_dim = -1
	embedd_list = []
	if path.endswith('.gz'):
		file = gzip.open(path, 'rt')
	else:
		file = open(path, 'r')
	# skip the first line
	if skip_first:
		file.readline()
	for line in file:
		line = line.strip()
		try:
			if len(line) == 0:
				continue

			tokens = line.split()
			if len(tokens) < embedd_dim:
				continue

			if embedd_dim < 0:
				embedd_dim = len(tokens) - 1

			embedd = np.empty([embedd_dim], dtype=np.float32)
			start = len(tokens) - embedd_dim
			word = ' '.join(tokens[0:start])
			embedd[:] = tokens[start:]
			word = DIGIT_RE.sub("0", word) if normalize_digits else word
			embedd_list.append(embedd)
			word2id[word] = len(word2id)
		except UnicodeDecodeError:
			continue
	embedding = np.stack(embedd_list, axis=0)
	return embedding, embedd_dim, word2id

class NN(object):
	def __init__(self, embed, k=101, pre_compute=True, n_thread=4):
		self.n_thread = n_thread
		self.embed = embed
		self.num = len(self.embed)
		if k > len(embed):
			print ("Total embed number {} > k ({})".format(len(embed), k))
			self.k = len(embed)
		else:
			self.k = k
		if pre_compute:
			print ("Pre-computing cos sim matrix ...")
			self.cos_sim_mat = self.compute_cos_sim()
		else:
			self.cos_sim_mat = defaultdict(dict)
		self.pre_compute = pre_compute
		#print (self.cos_sim_mat)

	def compute_cos_sim(self):
		emb_sparse = sparse.csr_matrix(self.embed)
		similarities = cosine_similarity(emb_sparse)
		return similarities

	def one_cos_sim(self, a, b):
		dot = np.dot(a, b)
		norma = np.linalg.norm(a)
		normb = np.linalg.norm(b)
		cos = dot / (norma * normb)
		return cos

	def cos_sim(self, a, b):
		try:
			cos_sim = self.cos_sim_mat[a][b]
		except KeyError:
			e1 = self.embed[a]
			e2 = self.embed[b]
			e1 = torch.tensor(e1)
			e2 = torch.tensor(e2)
			cos_sim = torch.nn.CosineSimilarity(dim=0)(e1, e2).numpy()
			#cos_sim = self.one_cos_sim(e1, e2)
			#self.cos_sim_mat[a][b] = cos_sim
		return cos_sim

	def compute_nn_single(self, id):
		top_nn = []
		top_sim = []
		#print ("runing ", id)
		for j in range(self.num):
			sim = self.cos_sim(id, j)
			if len(top_nn) < self.k:
				top_nn.append(j)
				top_sim.append(sim)
			else:
				if sim > sims.min():
					min_id = sims.argmin()
					sims[min_id] = sim
					top_nn[min_id] = j
			if len(top_nn) == self.k:
				sims = np.array(top_sim)

		order = (-sims).argsort()
		nn_single = np.array(top_nn)[order]
		#print (id, nn_single)

		return id, nn_single, sims, top_nn

	def update(self, result):
		#print ("result", result)
		i, nn_single, sims, top_nn = result
		self.nn_list.append(nn_single)

		for sim, j in zip(sims, top_nn):
			a, b = min(i,j), max(i,j)
			self.cos_sim_mat[a][b] = sim

	def compute_nn(self, log_every=1000):
		pool = multiprocessing.Pool(processes=self.n_thread)
		self.nn_list = []
		results = []
		for i in range(self.num):
			if i % log_every == 0:
				print (i,"...", end="")
				sys.stdout.flush()
			res = pool.apply_async(self.compute_nn_single, args=(i,))
			results.append(res)
			"""
			nn_single, sims, top_nn = self.compute_nn_single(i)
			nn_list.append(nn_single)
			for sim, j in zip(sims, top_nn):
				a, b = min(i,j), max(i,j)
				self.cos_sim_mat[a][b] = sim
			"""
		pool.close()
		pool.join()
		for r in results:
			self.update(r.get())

		self.nn = np.stack(self.nn_list, axis=0)
		print ()
		#print (self.cos_sim_mat)
		return self.nn

	def get_cos_sim(self, k):
		trim_cos_sim_mat = defaultdict(dict)
		k = min(self.k, k)
		for a in range(len(self.nn)):
			for j in range(k):
				b = self.nn[a][j]
				a_, b_ = min(a, b), max(a, b)
				trim_cos_sim_mat[a_][b_] = self.cos_sim_mat[a_][b_]
		if self.pre_compute:
			all_cos_sim_mat = defaultdict(dict)
			l = len(self.cos_sim_mat)
			for i in range(l):
				for j in range(i, l):
					all_cos_sim_mat[i][j] = self.cos_sim_mat[i][j]
		else:
			all_cos_sim_mat = self.cos_sim_mat
			
		return trim_cos_sim_mat, all_cos_sim_mat

parser = argparse.ArgumentParser()
parser.add_argument("embedding", type=str, help="input embedding file")
parser.add_argument("out_dir", type=str, help="output path")
#parser.add_argument("--out_vocab", type=str, default="vocab.json", help="output path for vocab")
parser.add_argument("--k", type=int, default=101, help="save top k NN")
parser.add_argument("--n_thread", type=int, default=10, help="number of threads")
parser.add_argument("--nn", action="store_true", help="whether to compute NN")
parser.add_argument("--pre", action="store_true", help="whether to pre-compute NN matrix")
parser.add_argument("--log_every", type=int, default=1000, help="print log every")
args = parser.parse_args()

embed, dim, word2id = load_embedding_dict(args.embedding)
print ("Vocab size = ", len(embed))

id2word = OrderedDict()
for word, id in word2id.items():
	id2word[id] = word
if os.path.exists(args.out_dir):
	print ("{} already exists", args.out_dir)
	exit()
else:
	os.mkdir(args.out_dir)

list_path = os.path.join(args.out_dir, 'wordlist.pickle')
with open(list_path, 'wb') as f:
	pickle.dump(word2id, f)
emb_path = os.path.join(args.out_dir, 'paragram')
np.save(emb_path, embed)

#print (word2id)
#print (embed)

if args.nn:
	nn_computer = NN(embed, k=args.k, pre_compute=args.pre, n_thread=args.n_thread)
	nn = nn_computer.compute_nn(log_every=args.log_every)
	nn_path = os.path.join(args.out_dir, 'nn')
	np.save(nn_path, nn)
	#print (nn)

	trim_cos_sim_mat = nn_computer.cos_sim_mat
	trim_cos_path = os.path.join(args.out_dir, 'cos_sim.p')
	with open(trim_cos_path, 'wb') as f:
		pickle.dump(trim_cos_sim_mat, f)

	exit()
	trim_cos_sim_mat, all_cos_sim_mat = nn_computer.get_cos_sim(k=args.k)
	trim_cos_path = os.path.join(args.out_dir, 'cos_sim.p')
	with open(trim_cos_path, 'wb') as f:
		pickle.dump(trim_cos_sim_mat, f)
	all_cos_path = os.path.join(args.out_dir, 'all_cos_sim.p')
	with open(all_cos_path, 'wb') as f:
		pickle.dump(all_cos_sim_mat, f)
	#print (trim_cos_sim_mat)
	#print (all_cos_sim_mat)

