import sys
import xmltodict
import xml.etree.ElementTree as ET
import nltk
import string
from collections import OrderedDict
import argparse

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

def get_triple(sub, id, toks):
  type = sub.get('type')
  i = sub.find('i')
  c = sub.find('c')
  cur_id = id
  # find subs in current ns
  #if i is None and c is None:
  subs_ = sub.findall('ns')
  #print ("ns subs:",len(subs_))
  types = []
  words = []
  txt = sub.text
  if txt is not None:
    txt = txt.strip()
  if txt:
    #words = nltk.word_tokenize(txt)
    #print ("pre match: ", txt)
    sub_words = match_tokenize(cur_id, toks, txt)
    cur_id += len(sub_words)
    words += sub_words
    types += [type]*len(sub_words)

  # find subs in i
  elif i is not None: #and (i.text is None or not i.text.strip()):
    i_subs_ = i.findall('ns')
    #print ("i subs:",len(i_subs_))
    #types = []
    #words = []
    txt = i.text
    if txt is not None:
      txt = txt.strip()
    if txt:
      #print ("i match: ", txt)
      #words = nltk.word_tokenize(txt)
      sub_words = match_tokenize(cur_id, toks, txt)
      cur_id += len(sub_words)
      words += sub_words
      types += [type]*len(sub_words)
    for sub_ in i_subs_:
      if sub_ is not None:
        sub_types, sub_words, _ = get_triple(sub_, cur_id, toks)
        if sub_types is not None:
          cur_id += len(sub_words)
          words += sub_words
          types += sub_types
      
      sub_suffix = sub_.tail
      if sub_suffix is not None:
        sub_suffix = sub_suffix.strip()
      if sub_suffix:
        #print ("sub match: ", txt)
        #sub_sufs = nltk.word_tokenize(sub_suffix)
        sub_sufs = match_tokenize(cur_id, toks, sub_suffix)
        cur_id += len(sub_sufs)
        words.extend(sub_sufs)
        types.extend([type]*len(sub_sufs))  
    #return types, words, None

  for sub_ in subs_:
    if sub_ is not None:
      sub_types, sub_words, _ = get_triple(sub_, cur_id, toks)
      if sub_types is not None:
        cur_id += len(sub_words)
        words += sub_words
        types += sub_types
    
    sub_suffix = sub_.tail
    if sub_suffix is not None:
      sub_suffix = sub_suffix.strip()
    if sub_suffix:
      #sub_sufs = nltk.word_tokenize(sub_suffix)
      sub_sufs = match_tokenize(cur_id, toks, sub_suffix)
      cur_id += len(sub_sufs)
      words.extend(sub_sufs)
      types.extend([type]*len(sub_sufs))
    
  return types, words, None

def match_tokenize(cur_id, toks, txt):
  raw_toks = txt.split(' ')
  str_no_sep = ''.join(raw_toks)
  #use_lower = False
  #if str_no_sep.isupper():
  str_no_sep = str_no_sep.lower()
  use_lower = True
  gold_str = ''
  idx = cur_id
  #print (toks)
  while len(gold_str) < len(str_no_sep):
    if use_lower:
      gold_str += toks[idx].lower()
    else:
      gold_str += toks[idx]
    idx += 1
    #print ("gold_str:{}\nstr_no_sep:{}".format(gold_str, str_no_sep))
    if not gold_str == str_no_sep[:len(gold_str)]:
      print ("### tokneize mismatch:(start:{}, cur:{})\ntoks:{}\ntxt:{}".format(cur_id, idx, toks, txt))
      print ("gold_str:{}\nstr_no_sep:{}".format(gold_str, str_no_sep))
      exit()
  return toks[cur_id:idx]

def get_pattern(i, line, toks, data, debug=False):
  new_data = []
  line = line.replace("=true", "=\"true\"")
  if line.endswith("<ns type=\"RP\"><i>."):
    line = line.replace("<ns type=\"RP\"><i>.", ".")
  if line.endswith("<ns type=\"UP\"><i>."):
    line = line.replace("<ns type=\"UP\"><i>.", ".")
  if debug:
    print (line)
  #try:
  root = ET.fromstring('<data>'+line+'</data>')
  #except:
  #  print ("### Error with line-{}: \n".format(i), line)
  #  exit()
  txt = root.text
  id = 0
  if txt:
    txt = txt.strip()
  if txt:
    #words = nltk.word_tokenize(txt)
    words = match_tokenize(id, toks, txt)
  else:
    words = None
  #print (data)
  #print (words)
  if words is not None:
    for j, word in enumerate(words):
      # deal with nltk tokenize problem
      if word == '``' or word == '\'\'':
        word = "\""
      if j == id and word == data[id][1]:
        new_data.append(data[id])
        id += 1
      else:
        print ("### mismatch(prefix): j:{}, word:{}, id:{}, line:{}".format(j, word, id, data[id]))
        print ("words:", words)
        print ("line:\n", line)
        print ("new data:\n", new_data)
        exit()
    #print (words)
  errs = list(root)
  for ns in errs:
    types, i_texts, c_text = get_triple(ns, id, toks)
    #print ("itext:", types, i_texts)
    if i_texts is not None:
      #words = nltk.word_tokenize(i_text.strip())
      #print ("itext:", types, i_texts)
      start = len(new_data)
      for j, (type,word) in enumerate(zip(types,i_texts)):
        # deal with nltk tokenize problem
        if word == '``' or word == '\'\'':
          word = "\""
        if start + j == id and word == data[id][1]:
          new_data.append(data[id])
          new_data[-1][9] = type
          id += 1
        else:
          print ("### mismatch(error): j:{}, word:{}, id:{}, line:{}".format(j, word, id, data[id]))
          print ("line:\n", line)
          print ("new data:\n", new_data)
          exit()
    # processing correct text after each error
    txt = ns.tail
    if txt:
      txt = txt.strip()
    if txt:
      #words = nltk.word_tokenize(txt)
      words = match_tokenize(id, toks, txt)
    else:
      words = None
    if words is not None:
      start = len(new_data)
      for j, word in enumerate(words):
        # deal with nltk tokenize problem
        if word == '``' or word == '\'\'':
          word = "\""
        if start + j == id and word == data[id][1]:
          new_data.append(data[id])
          id += 1
        else:
          print ("### mismatch: j:{}, word:{}, id:{}, line:{}".format(j, word, id, data[id]))
          print ("words:", words)
          print ("line:\n", line)
          print ("new data:\n", new_data)
          exit()
  #print (new_data)
  return new_data

def read(filename):
  samples = []
  with open(filename, 'r') as fi:
    sents = fi.read().strip().split('\n\n')
    for sent in sents:
      lines = sent.strip().split('\n')
      info = lines[:3]
      xml = lines[2].split(' = ', 1)[1].strip()
      toks = lines[1].split(' = ', 1)[1].split(' ')
      data = [line.strip().split('\t') for line in lines[3:]]
      samples.append((xml, toks, data, info))
  return samples

def write(fo, sent, info, _2way=True):
  fo.write('\n'.join(info)+'\n')
  fo.write('\n'.join(['\t'.join(line) for line in sent])+'\n\n')

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="input filename")
parser.add_argument("output", type=str, help="output filename")
parser.add_argument("--two_way", action="store_true", help="store in 2way format")
args = parser.parse_args()

samples = read(args.input)

n_cor = 0
n_err = 0
cnt = {}
outputs = []
fo = open(args.output, 'w')
for i, (xml, toks, data, info) in enumerate(samples):
  output = get_pattern(i, xml, toks, data)
  write(fo, output, info, args.two_way)
  for line in output:
    if line[9] in cnt:
      cnt[line[9]] += 1
    else:
      cnt[line[9]] = 1
    if line[9] == '_':
      n_cor += 1
    else:
      n_err += 1
n_tot = sum(list(cnt.values()))
print ("error/total tokens= {}/{}, error tok rate={:.2f}%".format(n_err, n_tot, 100*float(n_err)/n_tot))
print ("error type distribution:\n", cnt)