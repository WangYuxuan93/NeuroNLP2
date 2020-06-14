import sys
import json

if len(sys.argv) < 3:
	print ("usage: %s [input json] [output json]" % sys.argv[0])
	exit()

with open(sys.argv[1], 'r') as fi:
	dict = json.loads(fi.read().strip())
with open(sys.argv[2], 'w') as fo:
	json.dump(dict["instance2index"])