'''
n * m크기의 직사각형에 숫자를 0부터 순서대로 1씩 증가시키며 왼쪽 위에서부터 시작하여
다음과 같이 지그재그 모양으로 숫자들을 쭉 채우는 코드를 작성해보세요.
images/2021-02-16-04-57-23.png

입력 형식
n과 m이 공백을 사이에 두고 주어집니다.
1 ≤ n, m ≤ 100

출력 형식
숫자로 채워진 완성된 형태의 n * m 크기의 사각형을 출력합니다. (숫자끼리는 공백을 사이에 두고 출력합니다.)

입출력 예제
예제1
입력:
4 2

출력: 
0 7 
1 6 
2 5 
3 4 

예제2
입력:
5 5

출력: 
0 9 10 19 20 
1 8 11 18 21 
2 7 12 17 22 
3 6 13 16 23 
4 5 14 15 24
'''

import sys
input = sys.stdin.readline

n, m = map(int, input().strip().split())

x, y = 0, 0
dx, dy = [0, 1, 0, 1], [1, 0, -1, 0]
dir_, num = 0, 1

arr = [[0 for i in range(m)] for j in range(n)]

while num != m*n:

    next_x = x + dx[dir_]
    next_y = y + dy[dir_]

    if next_y > n-1 or next_y < 0:
        dir_ += 1

    arr[y+dy[dir_]][x+dx[dir_]] = num

    x += dx[dir_]
    y += dy[dir_]

    if dir_ == 3:
        dir_ = 0

    if dir_ == 1:
        dir_ += 1

    num += 1


for i in range(len(arr)):
    print(" " .join(map(str, arr[i])))
