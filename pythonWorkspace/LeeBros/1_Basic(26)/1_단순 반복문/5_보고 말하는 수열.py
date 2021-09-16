'''
보고 말하는 수열이란 다음 규칙을 만족하는 수열입니다.
수열의 첫 번째 원소는 1 입니다.
수열의 N 번째 원소는 N - 1번째 원소에 “보고 말하는” 과정을 거쳐 완성됩니다.
첫 번째 원소 1은 ‘1이 1개 있다’고 말할 수 있으므로, 두 번째 원소는 11이 됩니다.
두 번째 원소 11은 ‘1이 2개 있다’고 말할 수 있으므로, 세 번째 원소는 12가 됩니다.
세 번째 원소 12는 ‘1이 1개, 2가 1개 있다’고 말할 수 있으므로, 네 번째 원소는 1121가 됩니다.
네 번째 원소 1121는 ‘1이 2개, 2가 1개, 1이 1개 있다’고 말할 수 있으므로, 다섯 번째 원소는 122111이 됩니다.
위와 같이 순서대로 같은 숫자가 나오지 않는 순간을 기준으로 끊어 해당 숫자와 연속하여 나온 개수를 연달아 적은 결과가 그 다음 원소가 됩니다.
보고 말하는 수열의 N번째 원소를 구하는 프로그램을 작성해보세요.

입력 형식
N이 주어집니다.
1 ≤ N ≤ 20

예제1
입력:
2

출력: 
11

예제2
입력:
5

출력: 
122111
'''

import sys
input = sys.stdin.readline

n = int(input())
num = 0
arr = ['1']

while num != n-1:
    i = str(arr[num])
    i = [_ for _ in i]
    compare = i[0]
    del i[0]

    if len(i) == 0:
        arr.append(compare+str(1))
        num += 1
        continue

    new = ""
    cnt = 1

    for j in range(len(i)):
        if compare != i[j]:
            new += compare
            new += str(cnt) 
            cnt = 1

            if j == len(i)-1:
                new += i[j]
                new += str(cnt)
            else:
                compare = i[j]
        else:
            compare = i[j]
            cnt += 1
       
        if j == len(i)-1 and cnt > 1:
            new += i[j]
            new += str(cnt)

    arr.append(new)
    num += 1

print(int(arr[n-1]))