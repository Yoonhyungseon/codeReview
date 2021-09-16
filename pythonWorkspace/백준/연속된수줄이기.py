arr = [2, 3, 2, 2, 4, 5, 1, 1, 1, 3, 3, 3]
answer=[]
bigcount = 1

while(True) :

    compare=arr[0]
    del arr[0]
    if len(arr) == 0:
        break

    cnt = 1

    for i in arr:
        if compare != i:
           answer.append(cnt)
           cnt = 1
           compare = i
        else:
           cnt += 1
           compare = i
        
    if arr[-1] == compare:
         answer.append(cnt)

    if len(answer) == 1 and answer[0] == 1:
        break
    else:
        arr = answer
        answer=[]
        bigcount += 1


print(bigcount)