import sys
input = sys.stdin.readline


try:
    n = int(input())
    alist = [int(i) for i in input().strip().split() if i.strip().isdigit()]
    min_ = max_ = alist[0]
    for i in alist:
        if i <= min_:
            min_ = i
        if i >= max_:
            max_ = i
    print(min_, max_)
except:
    print("input 오류")


#print("{} {}" .format(min(alist), max(alist)))