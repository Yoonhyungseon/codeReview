alist = [int(k) for k in range(10001)]

for i in range(10001):
    new = i + i//1000 + i%1000//100 + i%100//10 + i%10
    
    for j in alist:
        if j == new:
            del alist[alist.index(j)] #alist.pop(alist.index(j)), 리스트는 index, 문자열은 find, index

alist = map(str, alist)
# alist = [str(i) for i in alist]

print("\n" .join(alist)) #join에 들어가는 리스트는 문자열로 구성되여야 한다!


