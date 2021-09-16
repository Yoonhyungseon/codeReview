import random

def partition(arr, l, h):
    pivot = arr[l]
    i = l
    j = h

    while (i<j):
        while (arr[i] <= pivot) and (i<h):
            i += 1
        while (arr[j] >= pivot) and (j>l):
            j -= 1
        if i<j:
            arr[i], arr[j] = arr[j], arr[i] #큰 값과 작은값 swap
            
    arr[l], arr[j] = arr[j], arr[l] #피벗값과 엇갈린 경우 값을 swap
    return arr, j

def quickSort(arr, l, h):
    if l<h:
        arr, fix = partition(arr, l, h)
        quickSort(arr, l, fix-1)
        quickSort(arr, fix+1, h)
    return arr

arr =  random.sample(range(1,10000), 9999)  
quickSort(arr, 0, len(arr)-1)

print(arr)

# O(N*logN) 이미 정렬이 되어있는 경우(최악의 경우)는 O(N*N)