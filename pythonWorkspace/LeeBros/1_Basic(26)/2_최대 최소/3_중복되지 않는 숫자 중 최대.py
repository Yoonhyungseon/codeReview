'''
N개의 숫자가 주어졌을 때, 중복하여 등장하지 않는 숫자 중 최댓값을 구하는 프로그램을 작성해보세요.

입력 형식
첫 번째 줄에는 원소의 수를 나타내는 N이 주어집니다.
두 번째 줄에는 N개의 숫자가 공백을 사이에 두고 주어집니다.

1 ≤ N ≤ 1,000
1 ≤ 원소 ≤ 1,000

출력 형식
중복하여 등장하지 않는 숫자 중 최댓값을 출력합니다. 만약 그러한 원소가 존재하지 않는다면, -1을 출력합니다.

입출력 예제
예제1
입력:
3
1 2 1

출력: 
2

예제2
입력:
4
1 2 1 2

출력: 
-1
'''
n = int(input())
arr = [int(i) for i in input().split()]
switch = 0

for i in dict.fromkeys(sorted(arr, reverse=1)):
    if arr.count(i) == 1:
        print(i)
        switch = 1
        break

if switch == 0:
    print(-1)


# arr = [int(i) for i in input().split()]
# arr_dic = dict.fromkeys(sorted(arr, reverse=1))


# for i in arr_dic.keys():
#     arr_dic[i] = arr.count(i) 
# print(arr_dic)
    

# arr_dic_change = {val:key for key, val in arr_dic.items()}
# print(arr_dic_change)