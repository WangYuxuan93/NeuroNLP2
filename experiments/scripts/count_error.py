import sys, os
import xmltodict
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
  fi = open(filename, 'r')
  n_sent = 0
  n_error = 0
  n_tok = 0
  pattern_dict = {}
  tree = ET.parse(filename)
  root = tree.getroot()
  for answer in root.iter('coded_answer'):
    #print (answer)
    paras = answer.findall('p')
    for para in paras:
      para_text = ""
      prefix = para.text
      if prefix:
        para_text += prefix
      para = list(para)
      n_error += len(para)
      for n, sub in enumerate(para):
        _, i_text, _ = get_triple(sub)
        if i_text is None:
          continue
        else:
          para_text += i_text
        suffix = sub.tail
        if suffix:
          para_text += suffix

      n_tok += len(nltk.word_tokenize(para_text))
      sents = nltk.sent_tokenize(para_text)
      #print (sents)
      n_sent += len(sents)
    #print (pattern_dict)
  return n_error, n_sent, n_tok


parser = argparse.ArgumentParser()
parser.add_argument("xml_dir", type=str, help="xml dir")
#parser.add_argument("output", type=str, help="output filename")
#parser.add_argument("--min_occur", type=int, default=5, help="min occurence")
parser.add_argument("--exclude", type=str, help="path to exclude file list")
parser.add_argument("--only_train", action="store_true", help="only use fce training set (2000)?")
args = parser.parse_args()

if args.exclude:
  exclude_files = json.loads(open(args.exclude, 'r').read())
else:
  exclude_files = None

#extract(sys.argv[1])
#exit()

xml_dir = args.xml_dir
sub_dirs = os.listdir(xml_dir)
pattern_dict = {}
print (sub_dirs)
n_files = 0
total_err, total_sent, total_tok =0, 0, 0
for sub_dir in sub_dirs:
  if args.only_train and not sub_dir.split('_')[1] == "2000": continue
  file_dir = os.path.join(xml_dir, sub_dir)
  files = os.listdir(file_dir)
  #print (files)
  for file in files:
    if exclude_files and sub_dir+'-'+file in exclude_files.keys(): continue
    filename = os.path.join(file_dir, file)
    print ("extracting from file: %s" % filename)
    n_err, n_sent, n_tok = extract(filename)
    total_err += n_err
    total_sent += n_sent
    total_tok += n_tok
    n_files += 1

print ("total files: {}, sents: {}, tokens: {}, errors: {}".format(n_files, total_sent, total_tok, total_err))



