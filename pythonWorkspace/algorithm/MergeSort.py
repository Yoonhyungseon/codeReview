def merge_sorted(arr):
    if len(arr)>1:
        mid = len(arr)//2
        left = arr[:mid]
        right = arr[mid:]

        l = merge_sorted(left)
        r = merge_sorted(right)
        return merge(l, r)
    else:
        return arr
        
def merge(left, right):
    i = 0
    j = 0
    arr = []
    
    while (i<len(left)) and (j<len(right)):
        if left[i] < right[j]:
            arr.append(left[i])
            i+=1
        else:
            arr.append(right[j])
            j+=1
    
    while (i<len(left)):
        arr.append(left[i])
        i+=1
    
    while (j<len(right)):
        arr.append(right[j])
        j+=1
        
    return arr

arr = [5,7,9,0,3,1,6,2,4,8]
print(merge_sorted(arr))