'''
빙빙 돌며 사각형 채우기
n * m크기의 직사각형에 대문자 알파벳을 A부터 Z까지 순서대로 증가시키며 달팽이 모양으로 채우는 코드를 작성해보세요.

달팽이 모양이란 왼쪽 위 모서리에서 시작해서, 오른쪽, 아래쪽, 왼쪽, 위쪽 순서로 더 이상 채울 곳이 없을 때까지 회전하는 모양을 의미합니다.
Z 이후에는 다시 A부터 채우기 시작합니다.
n : 행(row), m : 열(column)을 의미합니다.

images/2021-02-16-05-07-02.png


입력 형식
n과 m이 공백을 사이에 두고 주어집니다.
1 ≤ n, m ≤ 100

출력 형식
알파벳으로 채워진 완성된 형태의 n * m 크기의 사각형을 출력합니다.
(알파벳끼리는 공백을 사이에 두고 출력합니다.)

입출력 예제
예제1
입력:
4 4

출력: 
A B C D 
L M N E 
K P O F 
J I H G

예제2
입력:
4 2

출력: 
A B 
H C 
G D 
F E
'''

n ,m = map(int, input().strip().split())

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

arr = [[0 for i in range(m)] for j in range(n)]
x, y = 0, 0
dx, dy = [1, 0, -1, 0], [0, 1, 0, -1]
com = 0

def in_range(x, y):
    return 0 <= x < m and 0 <= y < n

def is_zero(x, y):
    return arr[y][x] == 0
   
    
for i in range(n*m):
    arr[y][x] = alphabet[i%(len(alphabet))]

    new_x = x + dx[com%4]
    new_y = y + dy[com%4]

    if com > 2 and not is_zero(new_x, new_y):
        com += 1
    
    if not in_range(new_x, new_y):
        com += 1
    
    x += dx[com%4]
    y += dy[com%4]
    
for i in range(len(arr)):
    print(" " .join(map(str, arr[i])))
    
        
        
        
    