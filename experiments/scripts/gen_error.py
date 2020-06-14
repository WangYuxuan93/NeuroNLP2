import sys
import xmltodict
import nltk
import string

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

def _get_pair(line):
  items = line.strip().split()
  print (line)
  #print (items)
  n = 0
  outputs = []
  rp_prefix = "<ns type=\"RP\"><i>"
  rp_punct = [rp_prefix+p for p in string.punctuation]
  up_prefix = "<ns type=\"UP\"><i>"
  up_punct = [up_prefix+p for p in string.punctuation]
  #print (rp_punct)
  while n < len(items):
    if items[n] == "<ns":
      n += 1
      error = "<ns " + items[n]
      if error in rp_punct:
        outputs.append((error[-1],'O'))
        n += 1
        continue
      elif error in up_punct:
        outputs.append((error[-1],"UP"))
        n += 1
        continue
      else:
        n_begin = 1
        n_end = len(list(find_all(error, "</ns>")))
        while not n_begin == n_end:
          n += 1
          error += " "+items[n]
          n_begin = len(list(find_all(error, "<ns")))
          n_end = len(list(find_all(error, "</ns>")))
      print (error)
      type = error.split(">")[0].strip(" new=true").strip(" ua=true").split("=")[1].strip("\"")
      #print (type)
      if str.find(error, "<i>") != -1:
        texts = error.split("<i>")
        #print (texts)
        for text in texts:
          if str.find(text, "</i>") != -1:
            target = text.split("</i>")[0]
        for tok in nltk.word_tokenize(target):
          outputs.append((tok, type))
      n += 1
    else:
      if items[n] == '-</i></ns>':
        outputs.append(('-', 'O'))
        n += 1
        continue
      for tok in nltk.word_tokenize(items[n]):
        outputs.append((tok, 'O'))
      n += 1
  return outputs

def get_pair(line):
  line = line.replace("=true", "=\"true\"")
  if line.endswith("<ns type=\"RP\"><i>."):
    line = line.replace("<ns type=\"RP\"><i>.", ".")
  print (line)
  result = xmltodict.parse('<data>'+line+'</data>')
  print (result['data'])
  errs = result['data']['ns']
  if not isinstance(errs, list):
    errs = [errs]
  #print (errs)
  texts = result['data']['#text'].split('  ')
  texts = [nltk.word_tokenize(t) for t in texts]
  #print (texts)
  outputs = []
  if line.startswith('<ns'):
    for n in range(len(errs)):
      if 'i' in errs[n]:
        typ = errs[n]['@type']
        err_toks = errs[n]['i']
        if not isinstance(err_toks, str):
          err_toks = err_toks['ns']
          if not isinstance(err_toks, list):
            err_toks = err_toks['i']
          else:
            err_toks = err_toks[0]['i']
        for tok in nltk.word_tokenize(err_toks):
          outputs.append((tok, typ))
      if n < len(texts):
        for tok in texts[n]:
          outputs.append((tok, 'O'))
  else:
    for n in range(len(texts)):
      for tok in texts[n]:
        outputs.append((tok, 'O'))
      if n < len(errs) and 'i' in errs[n]:
        typ = errs[n]['@type']
        err_toks = errs[n]['i']
        if not isinstance(err_toks, str):
          err_toks = err_toks['ns']
          if not isinstance(err_toks, list):
            err_toks = err_toks['i']
          else:
            err_toks = err_toks[0]['i']
        for tok in nltk.word_tokenize(err_toks):
          outputs.append((tok, typ))
  return outputs


with open(sys.argv[1], 'r') as fi, open(sys.argv[2], 'w') as fo:
  line = fi.readline()
  while line:
    line = line.strip()
    output = _get_pair(line)
    print (output)
    for tok, typ in output:
      fo.write(tok+' '+typ+'\n')
    fo.write('\n')
    line = fi.readline()
