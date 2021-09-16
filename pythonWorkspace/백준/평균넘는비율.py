import sys
input = sys.stdin.readline

N = int(input())

for i in range(N):
    alist = [int(k) for k in input().strip().split(" ")]
    del alist[0]

    sum_ = mean_ = 0
    for j in alist:
        sum_ += j
    mean_ = sum_/len(alist)

    cnt = ratio = 0
    for _ in alist:
        if _ > mean_:
            cnt += 1
    ratio = cnt/len(alist)*100
    print("{0:.3f}%" .format(ratio)) #소수점 아래 3자리까지 반올림

