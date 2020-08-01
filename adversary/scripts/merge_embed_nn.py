import pickle
import gzip
import numpy as np
import json
import argparse
import os, sys
import json
from collections import OrderedDict, defaultdict


def merge(dir, size):
	nn_list = []
	cos_sim_mat = defaultdict(dict)

	files = os.listdir(dir)
	files = [os.path.join(dir, f) for f in files]
	data = {}
	for file in files:
		print ("Loading tmp data from:", file)
		with open(file, 'r') as fi:
			tmp = json.load(fi)
		data.update(tmp)
	#print (data)
	assert str(size) in data
	
	for i in range(size):
		nn_single = np.array(data[str(i)]['nn_order'])
		sims = np.array(data[str(i)]['sims'])
		#i, nn_single, sims = result
		nn_list.append(nn_single)
		nn = np.stack(nn_list, axis=0)

		for sim, j in zip(sims, nn_single):
			a, b = min(i,j), max(i,j)
			cos_sim_mat[a][b] = sim
	return nn, cos_sim_mat

parser = argparse.ArgumentParser()
parser.add_argument("tmp_dir", type=str, help="input tmp directory")
parser.add_argument("out_dir", type=str, help="output path")
parser.add_argument("--size", type=int, help="vocab size")
#parser.add_argument("--out_vocab", type=str, default="vocab.json", help="output path for vocab")
args = parser.parse_args()


nn, cos_sim_mat = merge(args.tmp_dir, args.size)

nn_path = os.path.join(args.out_dir, 'nn')
np.save(nn_path, nn)
#print (nn)
cos_path = os.path.join(args.out_dir, 'cos_sim.p')
with open(cos_path, 'wb') as f:
	pickle.dump(cos_sim_mat, f)


