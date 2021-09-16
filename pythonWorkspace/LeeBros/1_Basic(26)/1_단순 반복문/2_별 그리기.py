"""
가로 세로 2n-1 크기에 해당하는 격자에 다이아몬드 모양을 *로 그리는 코드를 작성해보세요.
규칙은 다음과 같습니다.

n = 2
 *
***
 *

n = 3
  *
 ***
*****
 ***
  *
입력 형식
자연수 n이 주어집니다.
1 ≤ n ≤ 100

출력 형식
공백과 *을 이용하여 가로 세로 2n-1 크기의 격자에 다이아몬드 모양을 출력합니다.

입출력 예제
예제1
입력:
3

출력: 
  *
 ***
*****
 ***
  *
"""

import sys
input = sys.stdin.readline

n = int(input())
for i in range(1,2*n,2):
    print(" "*((2*n-i)//2)+"*"*i)

for j in range(2*n-3,0,-2):
    print(" "*((2*n-j)//2)+"*"*j)

    