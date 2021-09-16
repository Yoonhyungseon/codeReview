#학번: 201602210
#이름: 윤형선
#크루스칼 알고리즘으로 구현하였습니다.


parent = dict()
rank = dict()

#초기화
def make_set(vertice):
    parent[vertice] = vertice
    rank[vertice] = 0

#최상위 정점을 찾는다
def find(vertice):
    if parent[vertice] != vertice:
        parent[vertice] = find(parent[vertice])
    return parent[vertice]

#정점 연결
def union(vertice1, vertice2):
    root1 = find(vertice1)
    root2 = find(vertice2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
            if rank[root1] == rank[root2]: 
                rank[root2] += 1

def kruskal(graph):
    minimum_spanning_tree = []

    #초기화
    for vertice in graph['vertices']:
        make_set(vertice)
        
    #간선 sorting
    edges = graph['edges']
    edges.sort()
    
    #간선 연결 (사이클 없게)
    for edge in edges:
        weight, vertice1, vertice2 = edge
        if find(vertice1) != find(vertice2):
            union(vertice1, vertice2)
            minimum_spanning_tree.append(edge)
            print(minimum_spanning_tree)
	    
    return minimum_spanning_tree

graph = {
    'vertices': ['A', 'B', 'C', 'D', 'E', 'F', 'G','H'],
    'edges': [
        (8, 'A', 'B'),
        (15,'A', 'H'),
        (10,'A', 'G'),
        (11,'B', 'C'),
        (9, 'B', 'H'),
        (3, 'C', 'H'),
        (8, 'C', 'E'),
        (8, 'C', 'D'),
        (7, 'D', 'E'),
        (4, 'D', 'F'),
        (12,'E', 'H'),
        (5, 'E', 'F'),
        (2, 'F', 'G'),
        (1, 'G', 'H')
    ]
}

print("\n", kruskal(graph), sep="")