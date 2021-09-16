arr = [i for i in input().split()]
n = int(input())
rel = [[i for i in input().split()] for j in range(n)]
dic = {}

temp = []
answer = []

for i in rel:
    for j in i:
        try: dic[j] += 1
        except : dic[j] = 1

# print(dic)


for i in rel:
    for j in i[1:]:
        dic[j] = 0

for i in dic.items():
    if i[1] == max(dic.values()):
        start = i[0] 

# print(start)

for i in rel:
    if i[0] == start:
        temp.append(i)

# print(temp)
cnt = 0
for t in temp:
    for i in rel:
        if i[0] == t[-1]:
            answer.append(t+i[1:])
            cnt = 1
    if cnt == 1: temp.remove(t)

        
answer += temp

for i in answer:
    print(" " .join(i))
