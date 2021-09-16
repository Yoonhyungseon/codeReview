arr = [int(i) for i in input().split()]
cus = [[int(i) for i in input().split()] for j in range(arr[1])]
able = []

ans = [0 for j in range(arr[0])]
# print(arr)
# print(cus)


for i in range(len(cus)):
    if cus[i][1] <= arr[2]:
        able.append(cus[i])


def combination(arr, r):
    # 1.
    arr = sorted(arr)

    # 2.
    def generate(chosen):
        if len(chosen) == r:
            print(chosen)
            return

    	# 3.
        start = arr.index(chosen[-1]) + 1 if chosen else 0
        for nxt in range(start, len(arr)):
            chosen.append(arr[nxt])
            generate(chosen)
            chosen.pop()
            
    generate([])

for i in able:
    news = ""
    for j in range(arr[0]):
        if i[0] == j:
            news += i[1]

print(news)
# combination('ABCDE', 2)
# combination([1, 2, 3, 4, 5], 3)
