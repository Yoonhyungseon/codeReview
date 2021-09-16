import sys
input = sys.stdin.readline

x = 1
k="0123456789"
answer = [0 for _ in range(10)]


for i in range(3):
    a = int(input())
    x *= a

for j in str(x):
    idx = k.find(j)
    answer[idx] += 1

answer = [str(i) for i in answer]
print(" " .join(answer))

