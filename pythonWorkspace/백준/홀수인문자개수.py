from collections import Counter
 
value = input()
 
a = Counter(value)
cnt=0
for i in a.values():
    if i%2 != 0:
        cnt += 1

print(a)
print(cnt)