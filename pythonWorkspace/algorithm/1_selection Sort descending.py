import random

def selectionSortDec(arr):
    for i in range(len(arr)):
        max_ = arr[i]

        for j in range(i, len(arr)):
            if max_ <= arr[j]:
                max_ = arr[j]
                idx = j

        arr[i], arr[idx] = arr[idx], arr[i]
    
    return arr

arr = random.sample(range(1,50), 49)
selectionSortDec(arr)

print(arr)
        

