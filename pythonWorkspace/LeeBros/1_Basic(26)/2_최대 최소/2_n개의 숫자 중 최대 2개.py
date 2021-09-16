'''
입력으로 주어진 N개의 정수를 내림차순으로 정렬 했을 때 첫 번째와 두 번째 숫자를 출력하는 코드를 작성해보세요.

입력 형식
첫 번째 줄에는 원소의 개수 N이 주어지고, 두 번째 줄에는 N개의 정수가 공백을 사이에 두고 주어집니다.
정수는 -2^{31} ~ 2^{31} −1 사이의 범위를 갖습니다.
2 ≤ N ≤ 100

출력 형식
내림차순으로 정렬 했을 때 첫 번째와 두 번째 원소를 공백을 사이에 두고 출력합니다.

단, 이 때 두 원소의 값이 같을 수도 있습니다.

입출력 예제
예제1
입력:
10 
6 5 10 2 5 2 8 9 2 3

출력: 
10 9
'''

n = int(input())
arr = [int(i) for i in input().split()]
# arr = list(map(int, input().strip().split(" ")))

# arr.sort(reverse = True)
arr = sorted(arr, reverse= True)
print(" " .join(map(str, arr[:2])))


# 1.selection sort descending
# n = int(input())
# arr = [int(i) for i in input().split()]

# for i in range(len(arr)):
#     max_ = arr[i]

#     for j in range(i,len(arr)):
#         if max_ <= arr[j]:
#             max_ = arr[j]
#             index = j

#     arr[i], arr[index] = arr[index], arr[i]

# print(" " .join(map(str, arr[:2])))


# # 2.insertion sort descending
# n = int(input())
# arr = [int(i) for i in input().split()]

# from random import *
# arr = [randint(1,45) for i in range(10)]
# arr1 = arr
# for i in range(len(arr)-1):
#     j =i

#     while arr[j] < arr[j+1] and j >= 0:
#         arr[j], arr[j+1] = arr[j+1], arr[j]
#         j -= 1

# print(arr)
# print(sorted(arr1, reverse=1))
# print("yes" if arr == sorted(arr1, reverse=1) else "NO")


# 3.buble sort descending
# from random import *
# arr = [randint(1,10) for i in range(1,11)]

# for i in range(len(arr)):
#     for j in range(len(arr)-1-i):
#         if arr[j] < arr[j+1]:
#             arr[j], arr[j+1] = arr[j+1], arr[j]
# print(arr)
