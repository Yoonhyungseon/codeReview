'''
n * n크기의 직사각형의 가운데에서 시작하여 오른쪽, 위, 왼쪽, 아래 순서로 더 이상 채울 곳이 없을 때까지 회전하며 숫자를 적어나가려고 합니다. 
숫자는 1부터 시작한다고 했을 때, 다음과 같은 모양으로 숫자들을 쭉 채우는 코드를 작성해보세요.
images/2021-02-16-05-10-44.png

입력 형식
첫 번째 줄에 크기를 나타내는 n이 주어집니다. 주어지는 n은 항상 홀수라고 가정해도 좋습니다.
1 ≤ n ≤ 100

출력 형식
숫자로 채워진 완성된 형태의 n * n 크기의 사각형을 출력합니다.
(숫자끼리는 공백을 사이에 두고 출력합니다.)

입출력 예제
예제1
입력:
3

출력: 
5 4 3
6 1 2
7 8 9

예제2
입력:
5

출력: 
17 16 15 14 13
18 5 4 3 12
19 6 1 2 11
20 7 8 9 10
21 22 23 24 25
'''

import sys
input = sys.stdin.readline

n = int(input().strip())

arr = [[1]*n for i in range(n)]
dx, dy = [1, 0, -1, 0], [0, -1, 0, 1]
x = y = n//2
dir_, num= 0, 2

tempo = [i//2 for i in range(1, n*n+1)][1:]
tmp_idx, timer = 0, 1

while True:
    
    if num > n*n or len(arr) == 1:
        break

    arr[y + dy[dir_%4]][x + dx[dir_%4]] = num

    x += dx[dir_%4]
    y += dy[dir_%4]

    if timer == tempo[tmp_idx]:
        dir_ += 1
        tmp_idx += 1
        timer = 0

    num += 1
    timer += 1


for i in range(len(arr)):
    print(" " .join(map(str, arr[i])))
 