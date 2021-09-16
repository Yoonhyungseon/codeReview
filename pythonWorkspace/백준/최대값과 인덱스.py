import sys
input = sys.stdin.readline

alist = []
for i in range(9):
    alist.append(int(input()))

print(max(alist), alist.index(max(alist))+1, sep="\n")