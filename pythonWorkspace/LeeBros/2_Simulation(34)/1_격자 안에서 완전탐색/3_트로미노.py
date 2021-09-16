'''
n * m크기의 이차원 영역의 각 위치에 자연수가 하나씩 적혀있습니다.
이 때 아래의 그림에 주어진 2가지 종류의 블럭 중 한 개를 블럭이 격자를 벗어나지 않도록 적당히 올려놓아 블럭이 놓인 칸 안에 적힌 숫자의 합이 최대가 될 때의 결과를 출력하는 프로그램을 작성해보세요.
단, 주어진 블럭은 자유롭게 회전하거나 뒤집을 수 있습니다.
images/2021-02-16-18-53-16.png

입력 형식
첫 번째 줄에는 n과 m이 공백을 사이에 두고 주어지고, 두 번째 줄부터 (n+1)번째 줄까지는 각 행의 숫자가 공백을 사이에 두고 주어집니다.

3 ≤ n, m ≤ 200
1 ≤ 자연수 ≤ 1,000

출력 형식
블럭 안에 적힌 숫자합의 최대값을 출력합니다.

입출력 예제
예제1
입력:
3 3
1 2 3
3 2 1
3 1 1

출력: 
8

예제2
입력:
4 5
6 5 4 3 1
3 4 4 14 1
6 1 3 15 5
3 5 1 16 3

출력: 
45

예제 설명
2번째 예제는 다음과 같이 놓았을 때 합이 최대가 됩니다.
images/2021-02-16-18-53-50.png
'''

n, m = list(map(int,input().split()))
arr = [[int(i) for i in input().split()] for j in range(n)]
answer = []

#case1 'ㅡ'
def bar(arr):
    parts = []
    for i in range(len(arr)):
        for j in range(len(arr[i])-2):
            part = [arr[i][j], arr[i][j+1], arr[i][j+2]]
            parts.append(part)
    # print(parts)
    parts = list(map(sum, parts))
    return max(parts)

answer.append(bar(arr))


#case2 'ㅣ'
#transfer
parts = []
for i in range(m):
    part = []
    for j in range(n):
        part.append(arr[j][i])
    parts.append(part)
# print(parts)
answer.append(bar(parts))   


#case3 'ㄴ,ㄱ, r,ㅢ'
parts = []
for i in range(n-1):
    for j in range(m-1):
        part = [arr[i][j], arr[i+1][j], arr[i+1][j+1]] #'ㄴ'
        parts.append(part)
        part = [arr[i][j], arr[i][j+1], arr[i+1][j+1]] #'ㄱ'
        parts.append(part)
        part = [arr[i][j], arr[i][j+1], arr[i+1][j]] #'r'
        parts.append(part)
        part = [arr[i+1][j], arr[i+1][j+1], arr[i][j+1]] #'ㅢ'
        parts.append(part)
# print(parts)
parts = [sum(part) for part in parts]
answer.append(max(parts))  


print(max(answer))





