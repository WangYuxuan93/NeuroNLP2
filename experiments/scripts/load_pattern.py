import sys, os
import nltk
import string
import json
from collections import Counter
import argparse
import edlib

def load(filename):
  pattern_dict = {}
  with open(filename, 'r') as fi:
    pat_dict = json.loads(fi.read())
    for pos_pat, pats in pat_dict.items():
      pattern_dict[pos_pat] = {}
      for pat in pats:
        new_tup = []
        for part in pat['pattern']:
          tup = []
          for t in part:
            tup.append(tuple(t))
          new_tup.append(tuple(tup))
        new_tup = tuple(new_tup)
        pattern_dict[pos_pat][new_tup] = pat['cnt']
  return pattern_dict

def merge(pattern_dict):
  tok_dict = {}
  for pos_pat, pats in pattern_dict.items():
    for pat, cnt in pats.items():
      #print (cnt, pat)
      cor, err = pat
      cor_toks = tuple([x[0] for x in cor])
      err_toks = tuple([x[0] for x in err])
      tok_pat = (cor_toks, err_toks)
      #print (cor_toks, err_toks)
      if tok_pat not in tok_dict:
        tok_dict[tok_pat] = {pat: cnt}
      else:
        tok_dict[tok_pat][pat] = cnt
  print (len(tok_dict))
  for tok_pat, patterns in tok_dict.items():
    if len(patterns) > 1:
      print ("tok_pat:", tok_pat)
      print (patterns)
      #exit()

def align_(cor_field, err_field):
  used_err_id = []
  l2r_align = []
  l2r_tot_dist = 0
  for i in range(len(cor_field)):
    min_edit_dist = 1e5
    min_edit_id = -1
    for j in range(len(err_field)):
      if j in used_err_id: continue
      edit_dist = edlib.align(cor_field[i], err_field[j])['editDistance']
      if edit_dist < min_edit_dist:
        min_edit_dist = edit_dist
        min_edit_id = j
    l2r_align.append((i,min_edit_id))
    l2r_tot_dist += min_edit_dist 
    used_err_id.append(min_edit_id)

  used_err_id = []
  r2l_align = []
  r2l_tot_dist = 0
  for i in range(len(cor_field)-1, -1, -1):
    min_edit_dist = 1e5
    min_edit_id = -1
    for j in range(len(err_field)-1, -1, -1):
      if j in used_err_id: continue
      edit_dist = edlib.align(cor_field[i], err_field[j])['editDistance']
      if edit_dist < min_edit_dist:
        min_edit_dist = edit_dist
        min_edit_id = j
    r2l_align.append((i,min_edit_id))
    r2l_tot_dist += min_edit_dist 
    used_err_id.append(min_edit_id)
  r2l_align.reverse()
  print ("l2r({}): {}\nr2l({}): {}".format(l2r_tot_dist, l2r_align, r2l_tot_dist, r2l_align))
  align = l2r_align if l2r_tot_dist <= r2l_tot_dist else r2l_align
  tot_edit_dist = min(l2r_tot_dist, r2l_tot_dist)
  return align, tot_edit_dist

class Aligner(object):
  def __init__(self, cor_field, err_field):
    self.debug = False
    self.len_cor = len(cor_field)
    self.len_err = len(err_field)
    self.reset(min(len(cor_field),len(err_field)))
    if len(cor_field) <= len(err_field):
      self.cor_first = True
      self.align(cor_field, err_field)
    else:
      self.cor_first = False
      self.align(err_field, cor_field)
    #align = self.output()

  def reset(self, length):
    self.edit_dists = [1e5] * length
    self.aligns_memory = [] * length
    self.alg_err_ids = [-1] * length
    self.alg_rank_ids = [-1] * length
    #self.align = [-1] * length

  def output(self):
    if self.cor_first:
      align = [(i,err_id) for i, err_id in enumerate(self.alg_err_ids)]
      all_err_ids = [x[1] for x in align]
      for err_id in range(self.len_err):
        if err_id not in all_err_ids:
          align.append((-1, err_id))
          all_err_ids.append(err_id)
      return align, sum(self.edit_dists)
    else:
      err2cor_align = [(i,err_id) for i, err_id in enumerate(self.alg_err_ids)]
      cor2err_align = [(j,i) for i,j in err2cor_align]
      cor2err = sorted(cor2err_align, key=lambda x:x[0])
      align = []
      off = 0
      for i in range(self.len_cor):
        if off < len(cor2err) and cor2err[off][0] == i:
          align.append(cor2err[off])
          off += 1
        else:
          align.append((i,-1))
      return align, sum(self.edit_dists)

  def add(self, cor_id, rank_id):
    min_err_id, min_dist = self.aligns_memory[cor_id][rank_id]
    if self.debug:
      print ("min_err_id:{}, min_dist:{}".format(min_err_id, min_dist))
    if min_err_id not in self.alg_err_ids:
      self.alg_rank_ids[cor_id] = rank_id
      self.alg_err_ids[cor_id] = min_err_id
      self.edit_dists[cor_id] = min_dist
      if self.debug:
        print ("align (direct): {}\n".format(self.alg_err_ids))
    else:
      # if current min_err_id has been used
      # cmp the 2nd min dist of each and choose the smaller
      ovlp_cor_id = self.alg_err_ids.index(min_err_id)
      ovlp_rank_id = self.alg_rank_ids[ovlp_cor_id]
      ovlp_err_id, ovlp_dist = self.aligns_memory[ovlp_cor_id][ovlp_rank_id]
      # get next best edit dist for overlap correct word 
      ovlp_next_err_id, ovlp_next_dist = self.aligns_memory[ovlp_cor_id][ovlp_rank_id+1]
      # get next best edit dist for current correct word
      cur_next_err_id, cur_next_dist = self.aligns_memory[cor_id][rank_id+1]
      if self.debug:
        print ("ovlp_err_id={}, ovlp_dist={}".format(ovlp_err_id, ovlp_dist))
        print ("ovlp_next_err_id={}, ovlp_next_dist={}".format(ovlp_next_err_id, ovlp_next_dist))
        print ("cur_next_err_id={}, cur_next_dist={}".format(cur_next_err_id, cur_next_dist))

      # keep overlap correct word's alignment, use second best for current word
      if ovlp_dist + cur_next_dist <= ovlp_next_dist + min_dist:
        self.add(cor_id, rank_id+1)
        #if self.debug:
        #  print ("align: {}\n".format(self.alg_err_ids))
      # keep current word's alignment, use second best for overlap correct word
      else:
        self.alg_err_ids[ovlp_cor_id] = -1
        self.add(ovlp_cor_id, ovlp_rank_id+1)
        self.add(cor_id, rank_id)
        #if self.debug:
        #  print ("align: {}\n".format(self.alg_err_ids))

  def align(self, cor_field, err_field):
    for i in range(len(cor_field)):
      algs = []
      for j in range(len(err_field)):
        edit_dist = edlib.align(cor_field[i], err_field[j])['editDistance']
        algs.append((j, edit_dist))
      algs = sorted(algs, key=lambda x:x[1])
      self.aligns_memory.append(algs)
      if self.debug:
        print (algs)
      self.add(i, 0)
  #exit()

def check(pattern_dict):
  n_rep1 = 0
  n_rep2 = 0
  n_bad_edit_dist = 0
  n_miss = 0
  n_cor_more = 0
  n_cor_less = 0
  tot = 0
  l1 = 0
  output_dict = {}
  for typ, pats in pattern_dict.items():
    for pat, cnt in pats.items():
      tot += 1
      cor, err = pat
      cor_toks = tuple([x[0] for x in cor])
      err_toks = tuple([x[0] for x in err])
      #print ('cor: {}\nerr: {}'.format(cor_toks, err_toks))
      if len(cor_toks) < 2:
        l1 += 1
        continue
      if len(cor_toks) >= 2:
        cor_field = cor_toks
        err_field = err_toks
        if cor_field[0] is None:
          assert err_field[0] is None
          cor_field = cor_field[1:]
          err_field = err_field[1:]
        if cor_field[-1] is None:
          assert err_field[-1] is None
          cor_field = cor_field[:-1]
          err_field = err_field[:-1]

        if len(cor_field) == len(err_field):
          if len(cor_field) == 1:
            # one-to-one mapping, for replace and spelling
            align= [(0,0)]
            n_rep1 += 1
            #print ('rep1 | type:{} ) {} => {}'.format(typ, cor_field, err_field))
          else:
            #print ('rep>1 | type:{} ) {} => {}'.format(typ, cor_field, err_field))
            alg = Aligner(cor_field, err_field)
            align, tot_dist = alg.output()
            n_rep2 += 1
            if tot_dist > 10:
              n_bad_edit_dist += 1
            #print ("dist={}, align: {}".format(tot_dist, align))
            #exit()
        elif len(cor_field) > len(err_field):
          if len(err_field) == 0:
            align = []
            # remove all toks in filed, for missing
            for i in range(len(cor_field)):
              align.append((i,-1))
            n_miss += 1
          else:
            print ('cor_more | type:{} ) {} => {}'.format(typ, cor_field, err_field))
            n_cor_more += 1
            alg = Aligner(cor_field, err_field)
            align, tot_dist = alg.output()
            print ("dist={}, align: {}".format(tot_dist, align))

        elif len(cor_field) < len(err_field):
          #print ('cor_less | type:{} ) {} => {}'.format(typ, cor_field, err_field))
          n_cor_less += 1
          alg = Aligner(cor_field, err_field)
          align, tot_dist = alg.output()
          #print ("dist={}, align: {}".format(tot_dist, align))

      pos_pat = '-'.join([x[1] for x in cor])
      if align:
        if cor_toks[0] is None:
          new_align = [(0,0)]
          for cor_id, err_id in align:
            if err_id == -1:
              new_align.append((cor_id+1,-1))
            elif cor_id == -1:
              new_align.append((-1, err_id+1))
            else:
              new_align.append((cor_id+1,err_id+1))
              
          align = new_align
        if cor_toks[-1] is None:
          max_cor_id = max([x[0] for x in align])
          max_err_id = max([x[1] for x in align])
          align.append((max_cor_id+1,max_err_id+1))
        assert len(align) == max(len(cor_toks), len(err_toks))
        if pos_pat in output_dict:
          output_dict[pos_pat].append({'pattern':pat,'cnt':cnt,'align':align,'type':typ})
        else:
          output_dict[pos_pat] = [{'pattern':pat,'cnt':cnt,'align':align,'type':typ}]

  print ("rep1={}, miss={}, rep2={}, n_cor_more={}, n_cor_less={}, total={}, len1={}".format(
          n_rep1, n_miss,
          n_rep2, n_cor_more, n_cor_less, tot, l1))
  print ("n_bad_edit_dist={}".format(n_bad_edit_dist))
  
  return output_dict


def write(filename, pattern_dict, min_occur=5):
  output_dict = {}
  cnt_by_pos_pat = {}
  pruned_by_pos_pat = {}
  for pos_pat, pats in pattern_dict.items():
    cnt_by_pos_pat[pos_pat] = sum(list(pats.values()))
    #full_dict[pos_pat] = []
    for pat, cnt in pats.items():
      #full_dict[pos_pat].append({'pattern':pat,'cnt':cnt})
      if cnt >= min_occur:
        if pos_pat in output_dict:
          output_dict[pos_pat].append({'pattern':pat,'cnt':cnt})
          #output_dict[pos_pat][pat] = cnt
        else:
          output_dict[pos_pat] = [{'pattern':pat,'cnt':cnt}]
          #output_dict[pos_pat] = {pat: cnt}

        if pos_pat in pruned_by_pos_pat:
          pruned_by_pos_pat[pos_pat] += cnt
        else:
          pruned_by_pos_pat[pos_pat] = cnt
  tot_pat = sum(list(cnt_by_pos_pat.values()))
  pruned_pat = sum(list(pruned_by_pos_pat.values()))
  print ("pruned/total patterns: {}/{}".format(pruned_pat, tot_pat))

  with open(filename, 'w') as fo:
    fo.write(json.dumps(output_dict))

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="input pattern file")
parser.add_argument("output", type=str, help="output filename")
parser.add_argument("--min_occur", type=int, default=5, help="min occurence")
args = parser.parse_args()

min_occur = args.min_occur

pat_dict = load(args.input)
output_dict = check(pat_dict)
with open(args.output, 'w') as fo:
  fo.write(json.dumps(output_dict))
#pat_dict = merge(pat_dict)
#write(args.output, pat_dict, args.min_occur)