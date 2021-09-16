import random

def selectionSort(arr):
    for i in range(len(arr)):
        min_ = arr[i]

        for j in range(i, len(arr)):
            if arr[j] <= min_:
                min_ = arr[j]
                index = j

        arr[i], arr[index] = arr[index], arr[i]
        

arr =  random.sample(range(1,10000), 9999)   
selectionSort(arr)

print(" " .join(list(map(str, arr))))


#O(N*N)