import sys
input = sys.stdin.readline

a = int(input())
for _ in range(1,a+1,1):
    x, y = map(int, input().split())
    print("Case #", _, ": ", x, " + ", y, " = ", x+y, sep="")