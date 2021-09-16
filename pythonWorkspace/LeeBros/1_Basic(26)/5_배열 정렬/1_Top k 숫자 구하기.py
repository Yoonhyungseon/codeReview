'''
N개의 숫자가 주어졌을 때, 오름차순으로 정렬했을 때 k번째 숫자를 출력하는 프로그램을 작성해보세요.

입력 형식
첫 번째 줄에는 원소의 수를 나타내는 N과 구하고자 하는 번째 수를 의미하는 k가 공백을 사이에 두고 주어집니다.
두 번째 줄에는 N개의 숫자가 공백을 사이에 두고 주어집니다.

1 ≤ k ≤ N ≤ 1,000
1 ≤ 원소 ≤ 1,000

출력 형식
오름차순으로 정렬했을 때 k번째 숫자를 출력합니다.

입출력 예제
예제1
입력:
3 2
1 2 1

출력: 
1

예제2
입력:
4 3
1 5 4 2

출력: 
4
'''

n, k = map(int, input().strip().split())
arr = list(map(int, input().strip().split(" ")))
# arr = [int(i) for i in input().strip().split()]

arr.sort(reverse=False)
print(arr[k-1])