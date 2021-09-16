'''
N명의 학생에 대해 키, 몸무게 정보가 주어졌을 때, 다음의 규칙에 따라 정렬하려고 합니다.

키가 더 큰 학생이 앞에 와야 합니다.
키가 동일하다면, 몸무게가 더 큰 학생이 앞에 와야 합니다.
키와 몸무게가 동일하다면, 번호가 작은 학생이 앞에 와야 합니다.
번호는 정보가 따로 주어지진 않고 입력된 순서대로 부여됩니다. 즉 첫 번째로 입력된 학생은 1, k번째로 입력된 학생은 k입니다. 주어진 규칙에 따라 정렬한 이후 학생의 키, 몸무게, 번호를 출력하는 프로그램을 작성해보세요.

입력 형식
첫 번째 줄에는 학생의 수를 나타내는 N이 주어집니다.
두 번째 줄부터는 N개의 줄에 걸쳐 학생들의 키와 몸무게 정보가 번호 순서대로 주어집니다. 각 줄마다 각 학생에 대한 키와 몸무게를 나타내는 h, w 값이 공백을 사이에 두고 주어집니다. (1 ≤ h, w ≤ 1000)
1 ≤ N ≤ 1,000

출력 형식
규칙에 따라 정렬한 이후의 결과를 N개의 줄에 걸쳐 출력합니다. 각 줄에는 정렬한 이후 N명의 학생의 정보를 순서대로 한 줄에 하나씩 출력합니다. 
각 줄에는 각 학생의 키, 몸무게, 그리고 번호를 공백을 사이에 두고 출력합니다.

입출력 예제
예제1
입력:
2
1 5
3 6

출력: 
3 6 2
1 5 1

예제2
입력:
4
1 5
1 7
3 6
1 7

출력: 
3 6 3
1 7 2
1 7 4
1 5 1
'''



n = int(input())
arr = [tuple(map(int, input().strip().split(" "))) for i in range(n)]
# arr = [tuple(int(j) for j in input().split()) for i in range(n)]

arr_num = [(h, w, i+1) for i, (h, w) in enumerate(arr)] #enumerate는 뒤의 리스트나 튜플에 번호를 메겨 (번호, 리스트).. 로 구성된 튜플로 반환 

'''
arr = [[int(j) for j in input().split()] for i in range(n)]
arr = [list(int(j) for j in input().split()) for i in range(n)]3
arr = [list(map(int, input().strip().split())) for i in range(n)]
'''

arr_num.sort(key = lambda x: (-x[0], -x[1], x[2])) #lambda는 일회용 함수, 튜플은 정렬할때 key값을 정하면 그것을 우선순위로 정렬함, 우선순위 여러개 가능

for h, w, i in arr_num:
    print(h, w, i)  


