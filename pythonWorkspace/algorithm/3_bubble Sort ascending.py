import random

def bubbleSort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-1-i):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]


arr =  random.sample(range(1,10000), 9999)  
bubbleSort(arr)

print(" " .join(list(map(str, arr))))


#O(N*N) 이지만 swap 연산이 많아서 선택정렬보다 더 느리다