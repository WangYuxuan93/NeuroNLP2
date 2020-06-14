import sys, os
import json

def add(dic, cls, tok):
  if cls not in dic:
    dic[cls] = {tok: 1}
  else:
    if tok not in dic[cls]:
      dic[cls][tok] = 1
    else:
      dic[cls][tok] += 1
  return dic

if len(sys.argv) < 3:
  print ("usage: %s [input] [output_dir]" % sys.argv[0])
  exit()

n = 0
col = 0
if not os.path.exists(sys.argv[2]):
  os.mkdir(sys.argv[2])
upos_dict, xpos_dict, lemma_dict = {}, {}, {}
upath=os.path.join(sys.argv[2], 'upos.json')
xpath=os.path.join(sys.argv[2], 'xpos.json')
lpath=os.path.join(sys.argv[2], 'lemma.json')
with open(sys.argv[1], 'r') as fi, open(upath, 'w') as fou, open(xpath,'w') as fox, open(lpath,'w') as fol:
  line = fi.readline()
  while line:
    line = line.strip()
    if not line or line.startswith('#'):
      line = fi.readline()
      continue
    n += 1
    items = line.split('\t')
    tok = items[1]
    lem = items[2]
    upos = items[3]
    xpos = items[4]
    add(upos_dict, upos, tok)
    add(xpos_dict, xpos, tok)
    add(lemma_dict, lem, tok)
    line = fi.readline()
  #for key, val in upos_dict.items():
  #  upos_dict[key] = list(val)
  #for key, val in xpos_dict.items():
  #  xpos_dict[key] = list(val)
  #for key, val in lemma_dict.items():
  #  lemma_dict[key] = list(val)
  fou.write(json.dumps(upos_dict))
  fox.write(json.dumps(xpos_dict))
  fol.write(json.dumps(lemma_dict))
print ("\nTotal lemma/upos/xpos: {}/{}/{}".format(len(lemma_dict),len(upos_dict),len(xpos_dict)))
