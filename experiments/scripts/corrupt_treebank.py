import sys, os
import nltk
import string
import json
import argparse
import random

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

def load(filename, min_occur=1):
  print ("Loading patterns, min occurence=%d"%min_occur)
  pattern_dict = {}
  with open(filename, 'r') as fi:
    pat_dict = json.loads(fi.read())
    for pos_pat, pats in pat_dict.items():
      #pattern_dict[pos_pat] = []
      for pat in pats:
        if pat['cnt'] >= min_occur:
          if pos_pat not in pattern_dict:
            pattern_dict[pos_pat] = []
          #pattern_dict[pos_pat].append(tuple(pat['pattern']))
          pattern_dict[pos_pat].append(pat)
  return pattern_dict

def bucket(pat_dict, max_len):
  bucket = TagBucket()
  n = 0
  over = 0
  for tag, pats in pat_dict.items():
    n += len(pats)
    tag_seq = tag.split('-')
    if len(tag_seq) > max_len or len(tag_seq) == 1:
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
        data.append(items)
      yield data

class ErrorGenerator(object):
  def __init__(self, tag_bucket, err_per_tok=.12):
    self.err_per_tok = err_per_tok
    self.tag_bucket = tag_bucket
    self.max_dep = tag_bucket.max_depth
    print ("Max Depth: %d" % self.max_dep)
    self.clear_cnt()

  def clear_cnt(self):
    self.tot_tok = 0
    self.tot_err_tok = 0
    self.tot_err = 0
    self.tot_sent = 0

  def print_cnt(self):
    print ("total tokens=%d/error tokens=%d/error token per tok=%.4f"%(self.tot_tok, 
          self.tot_err_tok, float(self.tot_err_tok)/self.tot_tok))
    print ("total sents=%d/error cnt=%d/error per sent=%.4f"%(self.tot_sent, 
          self.tot_err, float(self.tot_err)/self.tot_sent))

  def valid(self, heads, l, r, error):
    if len(error[0]) > r-l:
      return False
    # if deleting
    if len(error[0]) < r-l:
      field = [x for x in range(l,r)]
      for cor_id, err_id in error[3]:
        # if to delete cor_id
        if err_id == -1:
          if field[cor_id] in heads:
            #print ("can not remove token:",field[cor_id])
            # make sure the token to be deleted has no child
            return False
    return True

  def update_head(self, line_ids, heads):
    # this is the root node
    old2new_map = {-1:-1}
    for new_id, old_id in enumerate(line_ids):
      old2new_map[old_id] = new_id
    new_list = []
    for i in line_ids:
      new_list.append(old2new_map[heads[i]])
    return new_list

  def update_rel(self, line_ids, rels):
    new_list = []
    for i in line_ids:
      new_list.append(rels[i])
    return new_list

  def generate(self, data):
    #print (data)
    self.tot_sent += 1
    tok_seq = [x[1] for x in data]
    tag_seq = [x[4] for x in data]
    #print (tok_seq)
    #print (tag_seq)
    mask = [0 for _ in data]
    line_ids = [i for i in range(len(data))]
    heads = [int(x[6])-1 for x in data]
    rels = [x[7] for x in data]
    num_err = int(len(tok_seq) * self.err_per_tok) + 1
    #print ("num err:", num_err)
    i = 0
    while i < num_err:
      #print ("\n### error-%d"%i)
      cands = self.get_candidates(tok_seq, tag_seq)
      #print (cands)
      if not cands: break
      (l, r, errors) = random.sample(cands, 1)[0]
      #print (l, r, errors)
      error = random.sample(errors, 1)[0]
      if len(error[0]) == 0: continue
      #print (l, r, error)
      if not self.valid(heads, l, r, error):
        #print ("invalid")
        continue

      tok_seq = tok_seq[:l] + error[0] + tok_seq[r:]
      tag_seq = tag_seq[:l] + error[1] + tag_seq[r:]
      #print ("mask:", mask)
      #print ("err2:", error[2])
      #print (error[3])
      err2cor_ = sorted([(y,x) for x,y in error[3]], key=lambda x:x[0])
      #print ("err2cor: ",err2cor_) 
      field = line_ids[l:r]
      new_line_ids = line_ids[:l]
      #print ("field: ", field)
      err2cor = []
      for err_id, cor_id in err2cor_:
        if err_id > -1:
          assert err_id == len(err2cor)
          err2cor.append(cor_id)
      for j, cor_offset in enumerate(err2cor):
        #assert j == err_offset
        #print (j, cor_offset, err_offset)
        new_line_ids.append(field[cor_offset])
      new_line_ids += line_ids[r:]
      #print (new_line_ids)
      
      heads = self.update_head(new_line_ids, heads)
      rels = self.update_rel(new_line_ids, rels)
      #print ("new_line_ids:", new_line_ids)
      #print ("heads:", heads)
      #print ("rels:", rels)
      line_ids = [x for x in range(len(new_line_ids))]
      overlap_mask = error[2]
      overlap_mask[0] = overlap_mask[0] or mask[l]
      overlap_mask[-1] = overlap_mask[-1] or mask[r-1]
      #print ("over:", overlap_mask)
      #print ("mask:", mask)
      mask = mask[:l] + overlap_mask + mask[r:]
      self.tot_err += 1
      #print ("tok:{}\ntag:{}\nmask:{}".format(tok_seq, tag_seq, mask))
      i += 1
      #print (mask)
      #print (list(zip(tok_seq, tag_seq, mask)))
      #exit()
    
    self.tot_err_tok += sum(mask)
    self.tot_tok += len(tok_seq)
    return zip(tok_seq,tag_seq,heads,rels,mask)
    

  def get_candidates(self, tok_seq, tag_seq):
    #print (data)
    #tok_seq = [x[0] for x in data]
    #tag_seq = [x[1] for x in data]
    candidates = []
    for i in range(len(tok_seq)):
      off = min(i+self.max_dep, len(tok_seq)-1)
      for j in range(i+1, off):
        patterns = self.tag_bucket.find(tag_seq[i:j])
        if patterns:
          #err_pats = [p['pattern'] for p in patterns]
          #print (tag_seq[i:j], patterns)
          errors = self.match(patterns, tok_seq[i:j])
          if errors:
            #print (i, j, tag_seq[i:j])
            candidates.append((i,j,errors))
    #print (candidates)
    return candidates

  def match(self, patterns, toks):
    errors = []
    for pattern_ in patterns:
      pattern = pattern_["pattern"]
      cor_pats = [p[0] for p in pattern[0]]
      tmp_toks = [tok if t_tok is not None else None for t_tok, tok in zip(cor_pats, toks)]
      if tmp_toks == cor_pats:
        if cor_pats[0] is None:
          prefix = [toks[0]]
        else:
          prefix = []
        if cor_pats[-1] is None:
          suffix = [toks[-1]]
        else:
          suffix = []
        #print (pattern)
        #print ("toks:{}\n,cor:{}\n,prefix:{}\nsufix:{}".format(toks, cor_pats, prefix, suffix)) 
        #print (tmp_toks)
        error_pats = [p[0] for p in pattern[1]]
        error_tags = [p[1] for p in pattern[1]]
        error = prefix
        for tok in error_pats:
          if tok is not None:
            error.append(tok)
        error.extend(suffix)
        err_mask = [0 if t is None else 1 for t in error_pats]
        #error = [tok if e_tok is None else e_tok for e_tok, tok in zip(error_pats, toks)]
        # make sure no insertion error is added
        if error not in errors and len(error) <= len(toks):
          #print ("cor_pats:{}\nerr_pats:{}\nerror:{}\nmask:{}".format(cor_pats, error_pats, 
          #      error, err_mask))
          errors.append((error,error_tags,err_mask,pattern_["align"],pattern_["type"]))
    return errors

class Writer(object):
  def __init__(self, filename, format='conllu'):
    self.fo = open(filename, 'w')
    self.format = format
    self.id2label = {0:'O',1:'E'}
    print ("Output format:", self.format)

  def write(self, data):
    if self.format == 'conllu':
      self.write_conllu(data)
    elif self.format == 'vertical':
      self.write_vertical(data)
    elif self.format == 'plain':
      self.write_plain(data)
    else:
      print ("Unrecognized format: %s", self.format)
      exit()

  def write_conllu(self, data):
    for i, (tok, tag, head, rel, label) in enumerate(data):
      items = [str(i+1), tok, '_', '_', tag, '_', str(head+1), rel, '_', self.id2label[label]]
      self.fo.write('\t'.join(items)+'\n')
    self.fo.write('\n')

  def write_vertical(self, data):
    for (tok, tag, _, _, label) in data:
      self.fo.write(tok+' '+self.id2label[label]+'\n')
    self.fo.write('\n')

  def write_plain(self, data):
    tokens = [d[0] for d in data]
    self.fo.write(' '.join(tokens)+'\n')

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="input filename")
parser.add_argument("output", type=str, help="output filename")
parser.add_argument("--pattern", type=str, help="input pattern file")
parser.add_argument("--min_occur", type=int, default=1, help="min occurence")
parser.add_argument("--max_len", type=int, default=5, help="max length of pattern")
parser.add_argument("--format", type=str, default="conllu", help="input format (txt or conllu (with pos tag in 4rd column))")
parser.add_argument("--err_per_tok", type=float, default=0.12, help="probability of number of errors per token")
parser.add_argument("--out_format", type=str, default="conllu", help="input format (plain|vertical(with O/E tag)|conllu)")
parser.add_argument('-s', '--seed', type=int, default=2468)
args = parser.parse_args()

random.seed(args.seed)
pat_dict = load(args.pattern, args.min_occur)
tag_bucket = bucket(pat_dict, args.max_len)
#print (tag_bucket.find(["DT","JJ","NN"]))
#exit()
generator = ErrorGenerator(tag_bucket, args.err_per_tok)
writer = Writer(args.output, args.out_format)

for data in read_conllu(args.input):
  err_data = generator.generate(data)
  writer.write(err_data)

generator.print_cnt()

