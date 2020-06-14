import sys
import xmltodict
import nltk
import string
from collections import OrderedDict

if len(sys.argv) < 3:
  #print "usage: %s [input] [output]" % sys.argv[0]
  exit()

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

def get_pattern(line):
  patterns = []
  line = line.replace("=true", "=\"true\"")
  if line.endswith("<ns type=\"RP\"><i>."):
    line = line.replace("<ns type=\"RP\"><i>.", ".")
  if line.endswith("<ns type=\"UP\"><i>."):
    line = line.replace("<ns type=\"UP\"><i>.", ".")
  #print (line)
  result = xmltodict.parse('<data>'+line+'</data>')
  #print (result['data'])
  errs = result['data']['ns']
  if not isinstance(errs, list):
    errs = [errs]
  for err in errs:
    typ = err['@type']
    i_ = err['i'] if 'i' in err else None
    c_ = err['c'] if 'c' in err else None
    if isinstance(i_, OrderedDict):
      i_ = i_['ns']
      if isinstance(i_, list):
        i_ = i_[0]['i']
      else:
        i_ = i_['i']
    if isinstance(c_, OrderedDict):
      c_ = c_['ns']
      if isinstance(i_, list):
        c_ = c_[0]['c']
      else:
        c_ = c_['c']
    patterns.append((typ, i_, c_))
  return patterns

n_xml = 0
n_pat = 0
with open(sys.argv[1], 'r') as fi, open(sys.argv[2], 'w') as fo:
  patterns = {}
  line = fi.readline()
  while line:
    line = line.strip()
    try:
      output = get_pattern(line)
    except:
      #print ("error xml:", line)
      n_xml += 1
    #print (output)
    for (t,i,c) in output:
      try:
        if t not in patterns:
          patterns[t] = set()
        patterns[t].add((i,c))    
      except:
        #print ("error pattern:\n", i, "\n", c)
        n_pat += 1
        #exit()
    line = fi.readline()
  print ("xml/pattern error:{}/{}".format(n_xml,n_pat))
  #print (patterns)
  cnt = [(p,len(x)) for p,x in patterns.items()]
  print (cnt)
  group_by_first = {}
  for p, x in patterns.items():
    if p[0] not in group_by_first:
      group_by_first[p[0]] = x
    else:
      group_by_first[p[0]].update(x)
  print ([(p,len(x)) for p,x in group_by_first.items()])
  print (group_by_first['R'])