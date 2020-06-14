import sys
import json

if len(sys.argv) < 3:
  print ("usage: %s [file1] [file2]" % sys.argv[0])
  exit()

with open(sys.argv[1], 'r') as f1, open(sys.argv[2], 'r') as f2:
  map1 = json.loads(f1.read())
  map2 = json.loads(f2.read())
  cnt = 0
  for key in map1.keys():
  	if key in map2.keys():
  		cnt += 1
  		print (key)
  print ("len f1:{}, f2:{}".format(len(list(map1.keys())), len(list(map2.keys()))))
  print ("overlap:", cnt)
  a = set(list(map1.keys()))
  a.update(list(map2.keys()))
  print ("union:", len(a))