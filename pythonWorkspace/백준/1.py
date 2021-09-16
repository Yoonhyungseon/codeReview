import sys
input = sys.stdin.readline

n = int(input())
for i in range(1,n+1):
    check = i//1000 + i//100 + i
    