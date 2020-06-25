import argparse
import json
import os

def load_config(filename):
 	conf = json.loads(open(filename, 'r').read())
 	return conf

class Attacker(object):
	def __init__(self, gpu, config, common_config, output_dir):
		self.tasks = load_config(config)
		self.common_dict = load_config(common_config)
		self.common_dict["gpu"] = gpu
		self.output_dir = output_dir
		self.src = "source "+self.common_dict["src_path"]+" ; "
		#print (self.common_dict)

	def build_cmd(self, dict):
		cmd = "CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=4 python -u {main_path} --mode {mode} --seed {seed} \
--generators {generator} --filters {filter} --batch_size {batch} --adv_batch_size {adv_batch} \
--pretrained_lm {lm} --lm_path {lm_path} --sent_encoder_path {sent_path} \
--adv_lm_path {adv_lm_path} {basic_emb} \
--vocab {vocab_path} --cand {cand_path} --syn {syn_path} --knn_path {knn_path} \
--min_word_cos_sim {wsim} --min_sent_cos_sim {ssim} --max_knn_candidates {max_knn} \
--adv_rel_ratio {rel_ratio} --adv_fluency_ratio {flu_ratio} \
--max_perp_diff_per_token {ppl_diff_per_tok} --perp_diff_thres {ppl_diff_thres} \
--test {test} --model_path {parser} \
--output_filename {orig_output} --adv_filename {adv_output} \
--noscreen --punctuation \'.\' \'``\' \"\'\'\" \':\' \',\' \
--format ud --lan_test en > {log} 2>&1".format(**dict)
		return cmd

	def prepare_dir(self, name, config):
		task_path = os.path.join(self.output_dir, name)
		if os.path.exists(task_path):
			print ("path {} already exists, quit".format(task_path))
			return None
		else:
			os.mkdir(task_path)
		model_name = config['parser'].rstrip('/').rsplit('/')[-1]
		data_name = self.common_dict['test'].rstrip('/').rsplit('/')[-1]
		output = model_name+'@'+data_name
		config['orig_output'] = os.path.join(task_path, output+'.orig')
		params = ['mode','rel_ratio','flu_ratio','ppl_diff_per_tok','ppl_diff_thres','wsim','ssim']
		adv_name = output+'.adv@'+'-'.join([str(config[p]) for p in params])
		config['adv_output'] = os.path.join(task_path, adv_name)
		config['log'] = os.path.join(task_path, name+'.log')
		config.update(self.common_dict)
		conf_path = os.path.join(task_path, 'config.json')
		json.dump(config, open(conf_path, 'w'), indent=4)
		return config

	def run(self, name, cmd):
		cmd = self.src + cmd
		print ("\n##### Running task: {} #####".format(name))
		print (cmd)
		os.system(cmd)

	def start(self):
		for name in self.tasks:
			#print (name)
			#print (self.tasks[name])
			task_config = self.prepare_dir(name, self.tasks[name])
			if task_config is not None:
				#print (task_config)
				cmd = self.build_cmd(task_config)
				self.run(name, cmd)

args_parser = argparse.ArgumentParser(description='attack pipeline')
args_parser.add_argument('--gpu', type=str, help='gpu')
args_parser.add_argument('config', type=str, help='experiment config')
args_parser.add_argument('common_config', type=str, help='common config')
args_parser.add_argument('output_dir', type=str, default='attack_output', help='attack output dir')
args = args_parser.parse_args()

attacker = Attacker(args.gpu, args.config, args.common_config, args.output_dir)
#print (attacker.build_cmd(attacker.paths))
attacker.start()
