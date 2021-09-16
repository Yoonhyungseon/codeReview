'''
1~100 사이의 숫자로 구성된 n * n 크기의 격자로 이루어져 있는 박스 정보가 주어집니다.
이 박스를 90' 시계방향으로 회전했을 때의 결과를 구하는 프로그램을 작성해보세요.
다음의 예에서 시계방향으로 90' 회전한 결과는 다음과 같습니다.
images/2021-02-16-04-58-04.png

입력 형식
첫 번째 줄에는 격자의 크기를 나타내는 n이 주어집니다.
두 번째 줄 부터는 n개의 줄에 걸쳐 각 행에 해당하는 n개의 숫자가 공백을 사이에 두고 주어집니다.
1 ≤ n ≤ 200

출력 형식
주어진 박스를 90' 시계방향으로 회전했을 때의 결과를 출력합니다.
n개의 줄에 걸쳐 각 행에 해당하는 n개의 숫자를 공백을 사이에 두고 출력합니다.

입출력 예제
예제1
입력:
4
1 2 4 3
3 2 2 3
3 1 6 2
4 5 4 4

출력: 
4 3 3 1
5 1 2 2
4 6 2 4
4 2 3 3

예제2
입력:
3
1 2 1
4 5 6
1 8 1

출력: 
1 4 1
8 5 2
1 6 1
'''

n = int(input())
arr = [[i for i in input().strip().split()] for j in range(n)]
tmp = []

for j in range(n):
    for i in range(n-1, -1, -1):
        tmp.append(arr[i][j])

cnt = 0

for i in range(len(arr)):
    for j in range(len(arr)):
        arr[i][j] = tmp[cnt]
        cnt += 1

for i in range(len(arr)):
    print(" " .join(map(str, arr[i])))