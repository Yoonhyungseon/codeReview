import random

def quickSortDec(arr, start, end):
    if start >= end:
        return arr

    left = start
    right = end
    key = start

    while left < right:
        while arr[key] >= arr[left] and left < end:
            left += 1
        while arr[key] <= arr[right] and right > start:
            right -= 1
        
        if left >= right:
            arr[key], arr[right] = arr[right], arr[key]

        else:
            arr[left], arr[right] = arr[right], arr[left]
    
    quickSortDec(arr, start, right-1)
    quickSortDec(arr, right+1, end)

    return arr

arr = random.sample(range(1, 500), 499)
quickSortDec(arr, 0, len(arr)-1)

print(arr)
