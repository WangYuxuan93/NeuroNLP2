import sys, os
import nltk
import string
import json
from collections import Counter
import argparse

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
pat_dict = merge(pat_dict)
write(args.output, pat_dict, args.min_occur)