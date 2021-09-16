'''
1~100 사이의 숫자로만 이루어져 있는 n * n 크기의 격자 정보가 주어집니다.
이때 행복한 수열이라는 것은 다음과 같이 정의됩니다.

행복한 수열 = 연속하여 m개 이상의 동일한 원소가 나오는 순간이 존재하는 수열
n * n 크기의 격자 정보가 주어졌을 때 각 행마다 봤을 때 나오는 n개의 수열과, 각 열마다 봤을 때 나올 수 있는 n개의 수열을 포함하여 총 2n개의 수열 중 행복한 수열의 개수를 세서 출력하는 프로그램을 작성해보세요.

예를 들어, 다음과 같은 경우라면, 첫 번째 행을 골랐을 경우와 첫 번째 열을 골랐을 경우에만 행복한 수열이 되므로, 총 행복한 수열의 수는 2개가 됩니다.
images/2021-02-16-18-51-27.png

입력 형식
첫 번째 줄에는 격자의 크기를 나타내는 n과 연속해야 하는 숫자의 수를 나타내는 m이 공백을 사이에 두고 주어집니다.
두 번째 줄부터는 n개의 줄에 걸쳐 격자에 대한 정보가 주어집니다. 각 줄에는 각각의 행에 대한 정보가 주어지며, 이 정보는 1에서 100사이의 숫자로 각각 공백을 사이에 두고 주어집니다.
1 ≤ m ≤ n ≤ 100

출력 형식
2n개의 수열들 중 행복한 수열의 수를 출력해주세요.

입출력 예제
예제1
입력:
3 2
1 2 2
1 3 4
1 2 3

출력: 
2

예제2
입력:
3 1
1 2 3
4 5 6
7 8 8

출력: 
6
'''



n, m = list(map(int, input().strip().split()))
arr = [list(map(int, input().strip().split())) for i in range(n)]

def satisfy(arr, m):
    compare = arr[0]
    score = 0

    for i in range(len(arr)):
        if compare == arr[i]:
            score += 1

            if score >= m:
                return True

        else:
            compare = arr[i]
            score = 1

        if i == len(arr)-1:
            if score >= m:
                return True
            else:
                return False

num = 0

for i in range(len(arr)):
    col = []

    if satisfy(arr[i], m):
        num += 1

    for j in range(len(arr)):
        col.append(arr[j][i])

    # print(col)

    if satisfy(col, m):
        num += 1

print(num)