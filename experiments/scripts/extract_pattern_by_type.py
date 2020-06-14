import sys, os
import xmltodict
import nltk
import string
import json
import xml.etree.ElementTree as ET
from collections import Counter
import argparse
from nltk.tag import StanfordPOSTagger

def tag_and_extract(prefix, x, suffix, tagger):
  s = nltk.word_tokenize(x) if x is not None else []
  x_len = len(s)
  if prefix is not None:
    prefix_list = nltk.word_tokenize(prefix)
    start_id = len(prefix_list)
    end_id = start_id + x_len
    # add one context from left
    prefix_id = start_id - 1
    s = prefix_list + s
  else:
    prefix_id = None
    start_id = 0
    end_id = x_len
  if suffix is not None:
    s = s + nltk.word_tokenize(suffix)
    # add one context from right
    suffix_id = end_id
  else:
    suffix_id = None
  #print (s)
  #print (start_id, x_len, end_id, prefix_id, suffix_id)

  tok_tags = nltk.pos_tag(s)
  #tok_tags = tagger.tag(s)
  pattern = tok_tags[start_id:end_id]
  pref, suff = None, None
  if prefix_id is not None:
    pref = tok_tags[prefix_id]
    pattern = [(None, pref[1])] + pattern
  if suffix_id is not None:
    suff = tok_tags[suffix_id]
    pattern = pattern + [(None, suff[1])]
  #print ("{}/{}-{}/{} ({}): {}\n".format(prefix_id, start_id, end_id, suffix_id, pattern, tok_tags))
  return pattern, pref, suff

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

def extract(filename, tagger):
  n_mismatch = 0
  fi = open(filename, 'r')
  pattern_dict = {}
  tree = ET.parse(filename)
  root = tree.getroot()
  for answer in root.iter('coded_answer'):
    #print (answer)
    paras = answer.findall('p')
    for para in paras:
      prefix = para.text
      flag_ns_as_suf = False
      suffixs = None
      if prefix:
        prefix = prefix.strip()
      if prefix:
        prefix = nltk.sent_tokenize(prefix)[-1]
      else:
        prefix = None
      para = list(para)
      for n, sub in enumerate(para):
        #type = sub.get('type')
        #i = sub.find('i')
        #c = sub.find('c')
        #i_text = i.text if i is not None else None
        #c_text = c.text if c is not None else None
        type, i_text, c_text = get_triple(sub)
        if i_text is None and c_text is None:
          # update prefix and skip this error
          suffixs = sub.tail
          if suffixs:
            suffixs = suffixs.strip()
          if suffixs:
            prefix = nltk.sent_tokenize(suffixs)[-1]
          else:
            prefix = None
          continue

        suffixs = sub.tail
        if suffixs:
          suffixs = suffixs.strip()
        if suffixs:
          flag_ns_as_suf = False
        else:
          # use the correct text from next <ns> as suffix
          if n + 1 < len(para):
            #print ("#### trying ")
            flag_ns_as_suf = True
            _, _, suffixs = get_triple(para[n+1])
            #c_next = para[n+1].find('c')
            #suffixs = c_next.text if c_next is not None else None
        if suffixs:
          suffixs = nltk.sent_tokenize(suffixs)
        suffix = suffixs[0] if suffixs else None
        #print ("{}: {} | {} => {} | {}".format(type, prefix, i_text, c_text, suffix))
        i_pat, i_pref, i_suff = tag_and_extract(prefix, i_text, suffix, tagger)
        c_pat, c_pref, c_suff = tag_and_extract(prefix, c_text, suffix, tagger)
        #print ("i_pat:{}\nc_pat:{}".format(i_pat, c_pat))
        #print ("i_pref:{}, i_suff:{}\nc_pref:{}, c_suff:{}".format(i_pref,i_suff,c_pref,c_suff))
        if not i_pref == c_pref and c_pref is not None:
          n_mismatch += 1
          #print ("\n### mismatch: i_pref:{},c_pref:{}".format(i_pref, c_pref))
          i_pat[0] = (None, c_pref[1])
          #print ("### fix: i_pat:{}\nc_pat:{}".format(i_pat, c_pat))
        if not i_suff == c_suff and c_suff is not None:
          n_mismatch += 1
          #print ("\n### mismatch: i_suff:{},c_suff:{}".format(i_suff, c_suff))
          i_pat[-1] = (None, c_suff[1])
          #print ("### fix: i_pat:{}\nc_pat:{}".format(i_pat, c_pat))
        pos_pat = '-'.join([p for t,p in c_pat])
        err_pat = (tuple(c_pat), tuple(i_pat))
        #print (err_pat)
        if type in pattern_dict:
          if err_pat in pattern_dict[type]:
            pattern_dict[type][err_pat] += 1
          else:
            pattern_dict[type][err_pat] = 1
        else:
          pattern_dict[type] = {err_pat: 1}
          
        if flag_ns_as_suf:
          prefix = c_text
        else:
          prefix = suffixs[-1] if suffixs else None
    #print (pattern_dict)
  return pattern_dict, n_mismatch


parser = argparse.ArgumentParser()
parser.add_argument("xml_dir", type=str, help="xml dir")
parser.add_argument("output", type=str, help="output filename")
parser.add_argument("--min_occur", type=int, default=5, help="min occurence")
parser.add_argument("--exclude", type=str, help="path to exclude file list")
parser.add_argument("--only_train", action="store_true", help="only use fce training set (2000)?")
args = parser.parse_args()

if args.exclude:
  exclude_files = json.loads(open(args.exclude, 'r').read())
else:
  exclude_files = None
min_occur = args.min_occur
#extract(sys.argv[1])
#exit()

#tagger = StanfordPOSTagger("/mnt/hgfs/share/stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger")
tagger = None
xml_dir = args.xml_dir
sub_dirs = os.listdir(xml_dir)
pattern_dict = {}
print (sub_dirs)
n_files = 0
tot_mis = 0
for sub_dir in sub_dirs:
  if args.only_train and not sub_dir.split('_')[1] == "2000": continue
  file_dir = os.path.join(xml_dir, sub_dir)
  files = os.listdir(file_dir)
  #print (files)
  for file in files:
    if exclude_files and sub_dir+'-'+file in exclude_files.keys(): continue
    filename = os.path.join(file_dir, file)
    print ("extracting from file: %s" % filename)
    pats, n_mismatch = extract(filename, tagger)
    tot_mis += n_mismatch
    #pattern_dict.update(pats)
    for pos_pat, p in pats.items():
      if pos_pat not in pattern_dict:
        pattern_dict[pos_pat] = p
      else:
        pattern_dict[pos_pat].update(p)
    n_files += 1
print ("Total mismatch: %d" % tot_mis)
#if n_files >= 3:
output_dict = {}
full_dict = {}
cnt_by_pos_pat = {}
pruned_by_pos_pat = {}
#print (n_files)
#print (pattern_dict)
for pos_pat, pats in pattern_dict.items():
  cnt_by_pos_pat[pos_pat] = sum(list(pats.values()))
  full_dict[pos_pat] = []
  for pat, cnt in pats.items():
    full_dict[pos_pat].append({'pattern':pat,'cnt':cnt})
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
print ("total files: {}, pruned/total patterns: {}/{}".format(n_files, pruned_pat, tot_pat))
#print (output_dict)

print (sorted(cnt_by_pos_pat.items(), key=lambda x:x[1], reverse=True))

with open(args.output, 'w') as fo:
  fo.write(json.dumps(full_dict))
#pruned_file = args.output+'.min-'+str(args.min_occur)
#with open(pruned_file, 'w') as fo:
#  fo.write(json.dumps(output_dict))
#exit()