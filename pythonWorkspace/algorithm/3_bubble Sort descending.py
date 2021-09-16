import random

def bubbleSortDec(arr):
    for i in range(len(arr)):

        for j in range(len(arr)-1-i):
            if arr[j] < arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

    return arr

arr = random.sample(range(1,50), 49)
bubbleSortDec(arr)

print(" " .join(list(map(str, arr))))