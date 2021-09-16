import random

def insertionSort(arr):
    for i in range(len(arr)-1):
        j = i
        
        while arr[j] > arr[j+1] and j >= 0:
            arr[j], arr[j+1] = arr[j+1], arr[j]
            j -= 1

arr =  random.sample(range(1,10000), 9999)  
insertionSort(arr)

print(arr)

#O(N*N) 이지만 필요한 만큼만 swap이 이루어지기 때문에 O(n*n)중에 가장 빠르다, 거의 정렬된 상태라면 가장 빠른 알고리즘
