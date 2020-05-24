import sys, os
import nltk
import string
import json
import xml.etree.ElementTree as ET
from collections import Counter
import argparse

def get_triple(sub):
  type = sub.get('type')
  i = sub.find('i')
  c = sub.find('c')
  if i is None and c is None:
    sub = sub.find('NS')
    if sub is None:
      return None, None, None
    type = sub.get('type')
    i = sub.find('i')
    c = sub.find('c')
  i_text = i.text if i is not None else None
  c_text = c.text if c is not None else None
  return type, i_text, c_text

def extract(filename):
  n_mismatch = 0
  fi = open(filename, 'r')
  pattern_dict = {}
  tree = ET.parse(filename)
  root = tree.getroot()
  err_seqs = []
  cor_seqs = []
  for answer in root.iter('coded_answer'):
    #print (answer)
    paras = answer.findall('p')
    for para in paras:
      err_tokens = []
      cor_tokens = []
      prefix = para.text
      if prefix:
        prefix = prefix.strip()
      if prefix:
        prefix = nltk.word_tokenize(prefix)
        err_tokens.extend([(t,'O') for t in prefix])
        cor_tokens.extend([(t,'O') for t in prefix])
      else:
        prefix = None
      suffixs = None
      para = list(para)
      for n, sub in enumerate(para):
        type, i_text, c_text = get_triple(sub)
        #if not (i_text is None and c_text is None):
        if i_text is not None:
          err_tokens.extend([(t,type) for t in nltk.word_tokenize(i_text)])
        if c_text is not None:
          cor_tokens.extend([(t,'O') for t in nltk.word_tokenize(c_text)])

        suffixs = sub.tail
        if suffixs:
          suffixs = suffixs.strip()
        if suffixs:
          toks = nltk.word_tokenize(suffixs)
          err_tokens.extend([(t,'O') for t in toks])
          cor_tokens.extend([(t,'O') for t in toks])
      err_seqs.append(err_tokens)
      cor_seqs.append(cor_tokens)
    #print (seqs)
  return err_seqs, cor_seqs

def write_vertical(fo, seqs, two_way_label=False):
  n_tok = 0
  for toks in seqs:
    for tok, tag in toks:
      if two_way_label and tag is not 'O':
        tag = 'E'
      fo.write(tok+' '+tag+'\n')
      n_tok += 1
    fo.write('\n')
  return n_tok

parser = argparse.ArgumentParser()
parser.add_argument("xml_dir", type=str, help="xml dir")
parser.add_argument("output", type=str, help="output filename")
parser.add_argument("--min_occur", type=int, default=5, help="min occurence")
parser.add_argument("--exclude", type=str, help="path to exclude file list")
parser.add_argument("--only_train", action="store_true", help="only use fce training set (2000)?")
parser.add_argument("--two_way_label", action="store_true", help="also output 2way label (O|E)?")
args = parser.parse_args()

if args.exclude:
  exclude_files = json.loads(open(args.exclude, 'r').read())
else:
  exclude_files = None
min_occur = args.min_occur
#extract(sys.argv[1])
#exit()

#tagger = StanfordPOSTagger("/mnt/hgfs/share/stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger")
xml_dir = args.xml_dir
sub_dirs = os.listdir(xml_dir)
pattern_dict = {}
print (sub_dirs)
n_files = 0
n_paras = 0
n_err_tok = 0
n_cor_tok = 0
if args.two_way_label:
  two_way_label = args.output+'.2way'
  fo2 = open(two_way_label, 'w')
cor_filename = args.output+'.cor'
with open(args.output, 'w') as foe, open(cor_filename, 'w') as foc:
  for sub_dir in sub_dirs:
    if args.only_train and not sub_dir.split('_')[1] == "2000": continue
    file_dir = os.path.join(xml_dir, sub_dir)
    files = os.listdir(file_dir)
    #print (files)
    for file in files:
      if exclude_files and sub_dir+'-'+file in exclude_files.keys(): continue
      filename = os.path.join(file_dir, file)
      print ("extracting from file: %s" % filename)
      err_seqs, cor_seqs = extract(filename)
      n_paras += len(err_seqs)
      n_files += 1
      n_err_tok += write_vertical(foe, err_seqs)
      n_cor_tok += write_vertical(foc, cor_seqs)
      if args.two_way_label:
        write_vertical(fo2, err_seqs, two_way_label=True)

print ("total files: {}, paragraphs: {}, error tokens:{}, correct tokens:{}".format(n_files, 
        n_paras, n_err_tok, n_cor_tok))