'''
주어지는 두 수(start, end)에 대해서, start 이상, end 이하의 숫자 중에 약수가 3개인 숫자의 개수를 구하는 코드를 작성해보세요.

입력 형식
첫 번째 줄에 start, end가 공백을 사이에 두고 주어집니다.
1 ≤ start ≤ end ≤ 1000

출력 형식
약수가 3개인 숫자의 개수를 출력합니다.

입출력 예제
예제1
입력:
3 7

출력: 
1
예제 설명
3의 약수의 개수 : 2개 (1, 3)
4의 약수의 개수 : 3개 (1, 2, 4)
5의 약수의 개수 : 2개 (1, 5)
6의 약수의 개수 : 4개 (1, 2, 3, 6)
7의 약수의 개수 : 2개 (1, 7)

따라서 약수가 3개인 건 1개입니다.
'''

import sys
input = sys.stdin.readline

answer = 0

start, end = map(int, input().strip().split())
for i in range(start, end+1):
    cnt = 0
    for j in range(1,i+1):
        if i%j == 0:
            cnt += 1
    
    if cnt == 3:
        answer += 1

print(answer)