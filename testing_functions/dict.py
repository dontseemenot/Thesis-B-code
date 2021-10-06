import json
import sys
#print(len(sys.argv))
print(sys.argv[1])
print(sys.argv[2])
print(sys.argv[3])
print(sys.argv[4])
print(sys.argv[5])
print(sys.argv[6])
print(sys.argv[7])
hyp = json.loads(sys.argv[8])
print(hyp['C'])
print(hyp['lr'])
print(hyp['epochs'])
