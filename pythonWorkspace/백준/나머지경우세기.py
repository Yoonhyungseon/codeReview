import sys
from collections import Counter
input = sys.stdin.readline
alist = []

for i in range(10):
    alist.append(int(input()))

blist = [i%42 for i in alist if i >= 0]
# b = map(str, blist)

b = Counter(blist)
print(len(b))
