alph = 'abcdefghijklmnopqrstuvwxyz'
answer = [0 for _ in range(26)]
s = input()
for i in s:
    idx = alph.find(i)
    answer[idx] +=1
answer=[str(k) for k in answer]
print(" ".join(answer))