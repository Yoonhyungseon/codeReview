import sys
from collections import Counter
input = sys.stdin.readline

ex = "OX"

n = int(input())
for i in range(n):
    cnt = total = 0
    txt = [str(i) for i in input().strip()]

    compare = txt[0]
    del txt[0]

    if compare is ex[0]:
        cnt += 1
        total += cnt
    
    for j in txt:
        if compare is ex[0] and j is compare: #oo 경우
            cnt += 1
            total += cnt
        elif j is not compare: #ox 경우
            cnt = 0
            if j is ex[0]: #xo 경우
                cnt += 1
                total += cnt
        compare = j
    
    print(total)

# import sys
# for _ in range(int(sys.stdin.readline())):
#     s=sys.stdin.readline().strip().split('X') #X로 나누어 리스트로 저장
#     print(sum((len(a)+1)*len(a)//2 for a in s)) #리스트 원소의 길이로 등차수열의 합


