#학번: 201602210
#이름: 윤형선
#학과: 컴전학부

def dp(arr):
    n = len(arr)
    cost = [[0 for j in range(n)] for i in range(n)]
    cost[0][0] = arr[0][0]

    for i in range(1,n):
        cost[i][0] = arr[i][0] + cost[i-1][0]
    
    for j in range(1,n):
        cost[0][j] = arr[0][j] + cost[0][j-1]
    
    for i in range(1,n):
        for j in range(1,n):
            cost[i][j] = arr[i][j] + min(cost[i-1][j], cost[i][j-1])
    
    return cost[n-1][n-1], cost

def pathmaker(cost):
    n = len(arr)
    path = [[0 for j in range(n)] for i in range(n)]

    for i in range(0,n):

        for j in range(0,n):
            if i == 0 and j == 0:
                path[i][j] = "-"

            else:
                if cost[i-1][j] < cost[i][j-1]:
                    path[i][j] = "left"
                
                else:
                    path[i][j] = "up"

                if j-1 < 0:
                    path[i][j] = "up"
                
                if i-1 < 0:
                    path[i][j] = "left"
                    
    return tracer(path)

def tracer(path):
    tracerlist = []
    i = j = len(path)-1
    while(path[i][j] != "-"):
        tracerlist.append([i,j])

        if path[i][j] == "left":
            j -= 1

        else:
            i -= 1
    
    tracerlist.append([i,j])
    tracerlist = [tracerlist[k] for k in range(::-1)]

    return tracerlist
    

arr =  [[6, 7, 12,  5],
        [5, 3, 11, 18],
        [7, 17, 3,  3],
        [8, 10, 14, 9]]

print("행렬 최소 경로 값: {0}" .format(dp(arr)[0]))

for i in range(0,len(dp(arr)[1])):
    print(dp(arr)[1][i])

print("경로(행,열): ")
for i in range(0,len(pathmaker(dp(arr)[1]))):   
    print("," .join(list(map(str,pathmaker(dp(arr)[1])[i]))))
