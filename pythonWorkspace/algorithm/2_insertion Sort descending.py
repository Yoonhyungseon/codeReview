import random

def insertionSortDec(arr):
    for i in range(len(arr)-1):
        j = i

        while j>=0 and arr[j] < arr[j+1]:
            arr[j], arr[j+1] = arr[j+1], arr[j]
            j -= 1
    
    return arr

arr = random.sample(range(1, 50), 49)
insertionSortDec(arr)

print(" " .join(list(map(str, arr))))