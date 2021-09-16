'''
자연수가 입력으로 주어질 때 해당 자연수를 일의 자리부터 거꾸로 출력하는 코드를 작성해보세요. (ex : 1234 → 4321)

입력 형식
첫 번째 줄에 자연수가 주어집니다.
자연수는 1 ~ 100,000 사이의 범위를 갖습니다.

출력 형식
입력 자연수를 거꾸로 출력합니다. 단, 뒤집었을 때 앞이 0인 경우를 제외하고 출력합니다.

입출력 예제
예제1
입력:
1234

출력: 
4321

예제2
입력:
1200

출력: 
21
'''

import sys
input = sys.stdin.readline

x = input().strip()
rev = int(x[::-1])

print(rev)

# a = ""
# for i in range(len(x)-1,-1,-1):
#     a += x[i]

# for i in range(len(a)):
#     if a[0] == "0":
#         a = a[1:]
#     else:
#         break
# print(a)