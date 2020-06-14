import sys

if len(sys.argv) < 3:
  print ("usage: %s [input] [output]" % sys.argv[0])
  exit()

with open(sys.argv[1], 'r') as fi, open(sys.argv[2], 'w') as fo:
  sents = fi.read().strip().split('\n\n')
  for sent in sents:
    line = sent.strip().split('\n')[2]
    fo.write(line.split(' = ')[1].strip() + '\n')
