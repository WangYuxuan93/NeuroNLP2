import sys

if len(sys.argv) < 2:
  print ("usage %s [input]" % sys.argv[0])
  exit()

with open(sys.argv[1], 'r') as fi:
  data = fi.read().strip()
  sents = data.split('\n\n')
  lens = [len(sent.split('\n')) for sent in sents]
  n_tok = sum(lens)
  n_sent = len(sents)
  print ("total sents: %d, tokens: %d" % (n_sent, n_tok))
  print ("max len: %d, min len: %d, avg len: %f" % (max(lens), min(lens), float(n_tok)/n_sent))
  tags = {}
  for sent in sents:
    sent = sent.strip()
    if not sent: continue
    lines = sent.split('\n')
    for line in lines:
      if not line:
        print (lines)
      tag = line.split(' ')[1]
      if tag not in tags:
        tags[tag] = 1
      else:
        tags[tag] += 1
  print (tags)
  tot = sum(list(tags.values()))
  for key, val in tags.items():
    tags[key] = float(val) / tot
  print (' '.join(["{}:{:.2f}%".format(key, val*100) for key, val in tags.items()]))

