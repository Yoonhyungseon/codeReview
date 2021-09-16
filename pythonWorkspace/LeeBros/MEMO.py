# a = list(map(int, input().strip().split()))
# x, y = map(int, input().strip().split())
# print(a, x, y)

# import sys
# input = sys.stdln.readline

# a = input().strip()
# print(a)



# a = "1234"
# a = [i for i in a]

# print(a)

# import random from *
# arr = [randint(1,10) for i in range(21)]
# arr = [randrange(1,10) for i in range(21)]

# arr = [0,0,0,0,0]
# arr = {i:value for i, value in enumerate(arr)}
# print(arr)

# arr = [0,0,0,0,0]
# arr = list(dict.fromkeys(arr))
# print(arr)

# count={}
# lists = ["a","a","b",'c','c','d']
# for i in lists:
#     try: count[i] += 1
#     except: count[i]=1
# print(count)
# print(list(set(lists))) #중복제거

# for i in count:
#     print(i,end="")
# for i in count.items():
#     print(i,end="")
# for i in count.keys():
#     print(i,end="")
# for i in count.values():
#     print(i,end="")

# from random import *
# arr1 = [randint(1,5) for i in range(1)]
# arr2 = [randint(0,9) for i in range(6)]
# print(arr1, arr2)

# x = input()
    
# dic = {}
# for i in x:
#     try: dic[i] += 1
#     except: dic[i] = 1
    
# print("YES" if sum(dic.values())%2 == 0 else "NO")

arr= [[1,2],[4,5]]
arr.remove([1,2])
print(arr)