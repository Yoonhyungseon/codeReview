n = [0]*11000

for i in range(1,10001):
	n[(i + i//1000 + i%1000//100 + i%100//10 + i%10)] = False

for i in range(1,10001):
	if n[i]:
		print(i)