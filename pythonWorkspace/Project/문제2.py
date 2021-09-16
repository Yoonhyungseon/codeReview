import itertools

list1=[5,6,7,8,9]
list2=[1,2,3,4]
list3=[2,4,6,8]


    


ans = list(itertools.product(*[list1,list2,list3]))

print("경우의 수 : {0}가지" .format(len(ans)))
print(ans)

# def combination(arr,n):
#     for i in range(len(arr)):  
#         if n == 1: yield [arr[i]]
#         else:
#             for next in combination(arr[i+1:],n-1): 
#                 yield [arr[i]] + next

# for ans in combination(list1,1):
#     print(ans)
