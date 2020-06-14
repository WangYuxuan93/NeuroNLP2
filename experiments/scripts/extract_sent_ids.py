import sys
import json

if len(sys.argv) < 3:
	print ("usage: %s [input] [output]" % sys.argv[0])
	exit()

sent_map = {}
with open(sys.argv[1], 'r') as fi, open(sys.argv[2], 'w') as fo:
	sents = fi.read().strip().split('\n\n')
	for sent in sents:
		line = sent.strip().split('\n')[0]
		info = line.split(' = ')[1].strip()
		file_path = info.rsplit('-', 1)[0]
		sent_id = info.rsplit('-', 1)[1]
		#print (file_path, sent_id)
		if file_path not in sent_map:
			sent_map[file_path] = [sent_id]
		else:
			sent_map[file_path].append(sent_id)
	fo.write(json.dumps(sent_map))
