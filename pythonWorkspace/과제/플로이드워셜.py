def floyd_warshall(n, edges):
    # 그래프 정보를 담을 인접행렬, 거리는 무한대로 초기화
    adj = [[float('inf')] * (n+1) for _ in range(n+1)] 

    # 인접행렬에 그래프 정보 저장 (정점 u -> v 거리 w)
    for u, v, w in edges:
        adj[u][v] = w   

        # k : 경유지 (각 정점들을 경유지로 설정)
    for k in range(1, n+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
            	# 정점 i -> j로 갈 때 기존 거리값과 k를 거쳐갈 때의 거리 값 중 작은 값을 저장
                adj[i][j] = min(adj[i][j], adj[i][k] + adj[k][j])
        for i in range(1, n+1):
            print(adj[i][1:]) 
        print()      
    return adj
    
# 정점 개수 & 간선 정보
n = 4  
edges =  [[1, 1, 0],
    [2, 2, 0],
    [3, 3, 0],
    [4, 4, 0],
    [1, 2, 3],
    [1, 4, 5],
    [2, 1, 2],
    [2, 4, 4],
    [3, 2, 1],
    [4, 3, 2]]
    
# 결과 출력
dist = floyd_warshall(n, edges)