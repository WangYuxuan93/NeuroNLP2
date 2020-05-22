import sys, os
import nltk
import string
import json
import argparse

class TagBucket(object):
  def __init__(self, tag=None, depth=0):
    self.patterns = []
    self.next = {}
    #self.tag = tag
    self.depth = depth
    self.max_depth = depth

  def add(self, tag_seq, pattern):
    cur_tag = tag_seq[0]
    bkt = self.get_or_add_bkt(cur_tag)
    if len(tag_seq) > 1:
      max_depth = bkt.add(tag_seq[1:], pattern)
      if self.max_depth < max_depth:
        self.max_depth = max_depth
      return max_depth
    else:
      bkt.add_pattern(pattern)
      return self.depth + 1

  def add_pattern(self, pattern):
    self.patterns.extend(pattern)

  def get_or_add_bkt(self, tag):
    if tag not in self.next:
      bkt = TagBucket(tag, depth=self.depth+1)
      self.next[tag] = bkt
    return self.next[tag]

  def find_bkt(self, tag):
    return self.next[tag] if tag in self.next else None

  def find(self, tag_seq):
    cur_tag = tag_seq[0]
    bkt = self.find_bkt(cur_tag)
    if bkt is None:
      return None
    if len(tag_seq) == 1:
      return bkt.patterns
    else:
      return bkt.find(tag_seq[1:])

def load(filename):
  pattern_dict = {}
  with open(filename, 'r') as fi:
    pat_dict = json.loads(fi.read())
    for pos_pat, pats in pat_dict.items():
      pattern_dict[pos_pat] = []
      for pat in pats:
        pattern_dict[pos_pat].append(tuple(pat['pattern']))
  return pattern_dict

def bucket(pat_dict, max_len):
  bucket = TagBucket()
  n = 0
  over = 0
  for tag, pats in pat_dict.items():
    n += len(pats)
    tag_seq = tag.split('-')
    if len(tag_seq) > max_len:
      over += len(pats)
      continue
      #print (tag_seq)
      #exit()
    bucket.add(tag_seq, pats)
  print ("Reserved(<=%d)/Total Patterns: %d/%d"%(max_len, n-over, n))
  return bucket

def read_conllu(filename):
  with open(filename, 'r') as fi:
    sents = fi.read().strip().split('\n\n')
    for sent in sents:
      data = []
      lines = sent.strip().split('\n')
      for line in lines:
        if line.startswith('#'): continue
        items = line.strip().split('\t')
        data.append((items[1], items[4]))
      yield data

class ErrorGenerator(object):
  def __init__(self, tag_bucket):
    self.tag_bucket = tag_bucket
    self.max_dep = tag_bucket.max_depth
    print ("max depth: %d" % self.max_dep)

  def gen(self, data):
    print (data)
    tok_seq = [x[0] for x in data]
    tag_seq = [x[1] for x in data]
    candidates = []
    for i in range(len(tok_seq)):
      off = min(i+self.max_dep, len(tok_seq)-1)
      for j in range(i+1, off):
        patterns = self.tag_bucket.find(tag_seq[i:j])
        if patterns:
          #print (tag_seq[i:j], patterns)
          errors = self.match(patterns, tok_seq[i:j])
          if errors:
            candidates.append((i,j,errors))
    print (candidates)
    exit()

  def match(self, patterns, toks):
    errors = []
    for pattern in patterns:
      gold_pats = [p[0] for p in pattern[0]]
      tmp_toks = [tok if t_tok is not None else None for t_tok, tok in zip(gold_pats, toks)]
      if tmp_toks == gold_pats:
        #print (tmp_toks)
        error_pats = [p[0] for p in pattern[1]]
        error = [tok if e_tok is None else e_tok for e_tok, tok in zip(error_pats, toks)]
        if error not in errors:
          errors.append(error)
    return errors


parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="input filename")
parser.add_argument("output", type=str, help="output filename")
parser.add_argument("--pattern", type=str, help="input pattern file")
#parser.add_argument("--min_occur", type=int, default=5, help="min occurence")
parser.add_argument("--max_len", type=int, default=5, help="max length of pattern")
parser.add_argument("--format", type=str, default="txt", help="input format (txt or conllu (with pos tag in 4rd column))")

args = parser.parse_args()

#min_occur = args.min_occur

pat_dict = load(args.pattern)
tag_bucket = bucket(pat_dict, args.max_len)
generator = ErrorGenerator(tag_bucket)

for data in read_conllu(args.input):
  generator.gen(data)

print (tag_bucket.max_depth)#, tag_bucket.next)

#print (tag_bucket.find(["DT","JJ","NN"]))