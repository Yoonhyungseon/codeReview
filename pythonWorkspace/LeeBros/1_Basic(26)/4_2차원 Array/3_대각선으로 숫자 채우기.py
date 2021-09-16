'''
n * m크기의 직사각형에 숫자를 1부터 순서대로 1씩 증가시키며 왼쪽 위에서부터 시작하여 오른쪽 아래 쪽까지 
다음과 같은 방향으로 숫자들을 쭉 채우는 코드를 작성해보세요.
images/2021-02-16-04-59-25.png

입력 형식
n과 m이 공백을 사이에 두고 주어집니다.
1 ≤ n, m ≤ 100

출력 형식
숫자로 채워진 완성된 형태의 n * m 크기의 사각형을 출력합니다.
(숫자끼리는 공백을 사이에 두고 출력합니다.)

입출력 예제
예제1
입력:
3 3

출력: 
1 2 4
3 5 7
6 8 9

예제2
입력:
4 2

출력: 
1 2
3 4
5 6
7 8

예제3
입력:
3 5W
출력: 
1 2 4 7 10
3 5 8 11 13
6 9 12 14 15
'''

n, m = map(int, input().strip().split())
arr = [[0 for i in range(m)] for j in range(n)]
x, y = 0, 0
cnt_x, cnt_y = 1, 1

def in_range(x,y):
    return (0 <= x < m and 0 <= y < n)

for i in range(n*m):
    arr[y][x] = i+1

    new_x = x - 1
    new_y = y + 1

    if not in_range(new_x, new_y):
        x = cnt_x
        y = 0
        cnt_x += 1
    
    else:
        x = new_x
        y = new_y

    if cnt_x > m and not in_range(new_x, new_y):
        x = m-1
        y = cnt_y
        cnt_y += 1    

for i in range(len(arr)):
    print(" " .join(map(str, arr[i])))

        

    

