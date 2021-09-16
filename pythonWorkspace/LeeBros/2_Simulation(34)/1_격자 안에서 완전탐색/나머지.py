기울어진 직사각형
1~100 사이의 숫자로만 이루어져 있는 n * n 크기의 격자 정보가 주어집니다.



이때, 이 격자 내에 있는 기울어진 직사각형들을 살펴보려고 합니다.

기울어진 직사각형이란, 격자내에 있는 한 지점으로부터 체스의 비숍처럼 대각선으로 움직이며 반시계 순회를 했을 때 지나왔던 지점들의 집합을 일컫습니다. 이 때 반드시 아래에서 시작해서 1, 2, 3, 4번 방향순으로 순회해야하며 각 방향으로 최소 1번은 움직여야 합니다. 또한, 이동하는 도중 격자 밖으로 넘어가서는 안됩니다.



예를 들어 위의 규칙에 따라 다음과 같은 기울어진 직사각형을 2개 만들어 볼 수 있습니다.





가능한 기울어진 직사각형들 중 해당 직사각형을 이루는 지점에 적힌 숫자들의 합이 최대가 되도록 하는 프로그램을 작성해보세요.

위의 예에서는 다음과 같이 기울어진 직사각형을 잡게 되었을 때 합이 21이 되어 최대가 됩니다.



입력 형식
첫 번째 줄에는 격자의 크기를 나타내는 n이 주어집니다.

두 번째 줄부터는 n개의 줄에 걸쳐 격자에 대한 정보가 주어집니다. 각 줄에는 각각의 행에 대한 정보가 주어지며, 이 정보는 1에서 100사이의 숫자로 각각 공백을 사이에 두고 주어집니다.

3 ≤ n ≤ 20
출력 형식
가능한 기울어진 직사각형들 중 최대의 합을 출력해주세요.

입출력 예제
예제1
입력:
5
1 2 2 2 2
1 3 4 4 4
1 2 3 3 3
1 2 3 3 3
1 2 3 3 3

출력: 
21
예제2
입력:
3
1 2 3
4 5 6
7 8 8

출력: 
20



# 변수 선언 및 입력
n, m = tuple(map(int, input().split()))
grid = [
    list(map(int, input().split()))
    for _ in range(n)
]


# 주어진 k에 대하여 마름모의 넓이를 반환합니다.
def get_area(k):
    return k * k + (k + 1) * (k + 1)


# 주어진 k에 대하여 채굴 가능한 금의 개수를 반환합니다.
def get_num_of_gold(row, col, k):
    return sum([
        grid[i][j]
        for i in range(n)
        for j in range(n)
        if abs(row - i) + abs(col - j) <= k
    ])


max_gold = 0

# 격자의 각 위치가 마름모의 중앙일 때 채굴 가능한 금의 개수를 구합니다.
for row in range(n):
    for col in range(n):
        for k in range(2 * (n - 1) + 1):
            num_of_gold = get_num_of_gold(row, col, k)
            
            # 손해를 보지 않으면서 채굴할 수 있는 최대 금의 개수를 저장합니다.
            if num_of_gold * m >= get_area(k):
                max_gold = max(max_gold, num_of_gold)

print(max_gold)



#include <iostream>
#include <algorithm>

#define MAX_N 20
#define DIR_NUM 4

using namespace std;

int n;
int grid[MAX_N][MAX_N];

bool InRange(int x, int y) {
    return 0 <= x && x < n && 0 <= y && y < n;
}

int GetScore(int x, int y, int k, int l) {
    int dx[DIR_NUM] = {-1, -1, 1, 1};
    int dy[DIR_NUM] = {1, -1, -1, 1};
    int move_num[DIR_NUM] = {k, l, k, l};
    
    int sum_of_nums = 0;

    // 기울어진 직사각형의 경계를 쭉 따라가봅니다.
    for(int d = 0; d < DIR_NUM; d++)
        for(int q = 0; q < move_num[d]; q++) {
            x += dx[d]; y += dy[d];
                
            // 기울어진 직사각형이 경계를 벗어나는 경우라면
            // 불가능하다는 의미로 답이 갱신되지 않도록
            // 0을 반환합니다.
            if(!InRange(x, y))
                return 0;
			
            sum_of_nums += grid[x][y];
        }
    
    return sum_of_nums;
}

int main() {
    cin >> n;
    
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            cin >> grid[i][j];
    
    int ans = 0;
    
    // (i, j)를 시작으로 1, 2, 3, 4 방향
    // 순서대로 길이 [k, l, k, l] 만큼 이동하면 그려지는
    // 기울어진 직사각형을 잡아보는
    // 완전탐색을 진행해봅니다.
    for(int i = 0; i < n; i++)
		for(int j = 0; j < n; j++)
			for(int k = 1; k < n; k++)
				for(int l = 1; l < n; l++)
                    ans = max(ans, GetScore(i, j, k, l));

    cout << ans;
    return 0;
}



import sys

INT_MIN = -sys.maxsize

# 변수 선언 및 입력:
n, m = tuple(map(int, input().split()))
grid = [
    list(map(int, input().split()))
    for _ in range(n)
]
board = [
    [0 for _ in range(m)]
    for _ in range(n)
]


def clear_board():
    for i in range(n):
        for j in range(m):
            board[i][j] = 0

            
def draw(x1, y1, x2, y2):
    for i in range(x1, x2 + 1):
        for j in range(y1, y2 + 1):
            board[i][j] += 1

            
def check_board():
    # 동일한 칸을 2개의 직사각형이 모두 포함한다면
    # 겹치게 됩니다.
    for i in range(n):
        for j in range(m):
            if board[i][j] >= 2:
                return True
    return False


# (x1, y1), (x2, y2) 그리고
# (x3, y3), (x4, y4) 로 이루어져있는
# 두 직사각형이 겹치는지 확인하는 함수
def overlapped(x1, y1, x2, y2, x3, y3, x4, y4):
    clear_board()
    draw(x1, y1, x2, y2)
    draw(x3, y3, x4, y4)
    return check_board()


def rect_sum(x1, y1, x2, y2):
    return sum([
        grid[i][j]
        for i in range(x1, x2 + 1)
        for j in range(y1, y2 + 1)
    ])


# 첫 번째 직사각형이 (x1, y1), (x2, y2)를 양쪽 꼭지점으로 할 때
# 두 번째 직사각형을 겹치지 않게 잘 잡아
# 최대 합을 반환하는 함수
def find_max_sum_with_rect(x1, y1, x2, y2):
    max_sum = INT_MIN
    
    # (i, j), (k, l)을 양쪽 꼭지점으로 하는
    # 두 번째 직사각형을 정하여
    # 겹치지 않았을 때 중
    # 최댓값을 찾아 반환합니다.
    for i in range(n):
        for j in range(m):
            for k in range(i, n):
                for l in range(j, m):
                    if not overlapped(x1, y1, x2, y2, i, j, k, l):
                        max_sum = max(max_sum, 
                                      rect_sum(x1, y1, x2, y2) +
                                      rect_sum(i, j, k, l))
    
    return max_sum


# 두 직사각형을 잘 잡았을 때의 최대 합을 반환하는 함수
def find_max_sum():
    max_sum = INT_MIN
    
	# (i, j), (k, l)을 양쪽 꼭지점으로 하는
    # 첫 번째 직사각형을 정하여
    # 그 중 최댓값을 찾아 반환합니다.
    for i in range(n):
        for j in range(m):
            for k in range(i, n):
                for l in range(j, m):
                    max_sum = max(max_sum,
                                  find_max_sum_with_rect(i, j, k, l))
    return max_sum


ans = find_max_sum()
print(ans)


# 변수 선언 및 입력
n, t = tuple(map(int, input().split()))
u = list(map(int, input().split()))
d = list(map(int, input().split()))

for _ in range(t):
    # Step 1
    # 위에서 가장 오른쪽에 있는 숫자를 따로 temp값에 저장해놓습니다.
    temp = u[n - 1]
    
    # Step 2
    # 위에 있는 숫자들을 완성합니다. 
    # 오른쪽에서부터 채워넣어야 하며, 
    # 맨 왼쪽 숫자는 아래에서 가져와야함에 유의합니다.
    for i in range(n - 1, 0, -1):
        u[i] = u[i - 1]
    u[0] = d[n - 1]
    
    # Step 3
    # 아래에 있는 숫자들을 완성합니다. 
    # 마찬가지로 오른쪽에서부터 채워넣어야 하며, 
    # 맨 왼쪽 숫자는 위에서 미리 저장해놨던 temp값을 가져와야함에 유의합니다.
    for i in range(n - 1, 0, -1):
        d[i] = d[i - 1]
    d[0] = temp

# 출력
for elem in u:
    print(elem, end=" ")
print()

for elem in d:
    print(elem, end=" ")


    # 변수 선언 및 입력
n, t = tuple(map(int, input().split()))
l = list(map(int, input().split()))
r = list(map(int, input().split()))
d = list(map(int, input().split()))

for _ in range(t):
    # Step 1
    # 왼쪽에서 가장 오른쪽에 있는 숫자를 따로 temp값에 저장해놓습니다.
    temp = l[n - 1]
    
    # Step 2
    # 왼쪽에 있는 숫자들을 완성합니다. 
    # 벨트를 기준으로 오른쪽에서부터 채워넣어야 하며, 
    # 맨 왼쪽 숫자는 아래에서 가져와야함에 유의합니다.
    for i in range(n - 1, 0, -1):
        l[i] = l[i - 1]
    l[0] = d[n - 1]
    
    # Step 3
    # 오른쪽에 있는 숫자들을 완성합니다. 
    # 벨트를 기준으로 마찬가지로 오른쪽에서부터 채워넣어야 하며, 
    # 맨 왼쪽 숫자는 이전 단계에서 미리 저장해놨던 temp값을 가져와야함에 유의합니다.
    temp2 = r[n - 1]
    for i in range(n - 1, 0, -1):
        r[i] = r[i - 1]
    r[0] = temp
    
    # Step 4
    # 아래에 있는 숫자들을 완성합니다. 
    # 마찬가지로 벨트를 기준으로 오른쪽에서부터 채워넣어야 하며, 
    # 맨 왼쪽 숫자는 이전 단계에서 미리 저장해놨던 temp값을 가져와야함에 유의합니다.
    for i in range(n - 1, 0, -1):
        d[i] = d[i - 1]
    d[0] = temp2
    
# 출력
for elem in l:
    print(elem, end=" ")
print()

for elem in r:
    print(elem, end=" ")
print()

for elem in d:
    print(elem, end=" ")




    SHIFT_RIGHT = 0
SHIFT_LEFT = 1

# 변수 선언 및 입력
n, m, q = tuple(map(int, input().split()))
a = [
    [0 for _ in range(m + 1)]
    for _ in range(n + 1)
]


# row 줄의 원소들을 dir 방향에 따라 한 칸 밀어줍니다.
# dir이 0인 경우 오른쪽으로
# dir이 1인 경우 왼쪽으로 밀어야 합니다.
def shift(row, curr_dir):
    # 오른쪽으로 밀어야 하는 경우 
    if curr_dir == SHIFT_RIGHT:
        a[row].insert(1, a[row].pop())
    else:
        a[row].insert(m, a[row].pop(1))


# row1, row2 행에 대해 같은 열에 같은 숫자를 갖는 경우가
# 있는지를 찾아줍니다.
def has_same_number(row1, row2):
    return any([
        a[row1][col] == a[row2][col]
        for col in range(1, m + 1)
    ])


# 주어진 방향으로부터 반대 방향의 값을 반환합니다.
def flip(curr_dir):
    return SHIFT_RIGHT if curr_dir == SHIFT_LEFT else SHIFT_LEFT


# 조건에 맞춰 움직여봅니다.
# dir이 SHIFT_RIGHT 인 경우 오른쪽으로
# dir이 SHIFT_LEFT 인 경우 왼쪽으로 밀어야 합니다.
def simulate(start_row, start_dir):
    # Step1
    # 바람이 처음으로 불어 온 행의 숫자들을 해당 방향으로 밀어줍니다.
    shift(start_row, start_dir)
    
    # 그 이후부터는 반전된 방향에 영향을 받으므로, 방향을 미리 반전시켜 줍니다.
    start_dir = flip(start_dir)
    
    # Step2
    # 위 방향으로 전파를 계속 시도해봅니다.
    curr_dir = start_dir
    for row in range(start_row, 1, -1):
        # 인접한 행끼리 같은 숫자를 가지고 있다면
        # 위의 행을 한 칸 shift 하고
        # 방향을 반대로 바꿔 계속 전파를 진행합니다.
        if has_same_number(row, row - 1):
            shift(row - 1, curr_dir)
            curr_dir = flip(curr_dir)
        # 같은 숫자가 없다면 전파를 멈춥니다.
        else:
            break
    
    # Step3
    # 아래 방향으로 전파를 계속 시도해봅니다.
    curr_dir = start_dir
    for row in range(start_row, n):
        # 인접한 행끼리 같은 숫자를 가지고 있다면
        # 아래 행을 한 칸 shift하고
        # 방향을 반대로 바꿔 계속 전파를 진행합니다.
        if has_same_number(row, row + 1):
            shift(row + 1, curr_dir)
            curr_dir = flip(curr_dir)
        # 같은 숫자가 없다면 전파를 멈춥니다.
        else:
            break


for row in range(1, n + 1):
    given_nums = list(map(int, input().split()))
    for col, num in enumerate(given_nums, start = 1):
        a[row][col] = num

for _ in range(q):
    r, d = tuple(input().split())
    r = int(r)
    
    # 조건에 맞춰 움직여봅니다
    simulate(r, SHIFT_RIGHT if d == 'L' else SHIFT_LEFT)

# 출력
for row in range(1, n + 1):
    for col in range(1, m + 1):
        print(a[row][col], end = " ")
    print()




    # 변수 선언 및 입력
n, m, q = tuple(map(int, input().split()))
a = [
    [0 for _ in range(m + 1)]
    for _ in range(n + 1)
]
temp_arr = [
    [0 for _ in range(m + 1)]
    for _ in range(n + 1)
]


# 직사각형의 경계에 있는 숫자들을 시계 방향으로 한 칸씩 회전해줍니다.
def rotate(start_row, start_col, end_row, end_col):
    # Step1-1. 직사각형 가장 왼쪽 위 모서리 값을 temp에 저장합니다.
    temp = a[start_row][start_col]
    
    # Step1-2. 직사각형 가장 왼쪽 열을 위로 한 칸씩 shift 합니다.
    for row in range(start_row, end_row):
        a[row][start_col] = a[row + 1][start_col]
    
    # Step1-3. 직사각형 가장 아래 행을 왼쪽으로 한 칸씩 shift 합니다.
    for col in range(start_col, end_col):
        a[end_row][col] = a[end_row][col + 1]
    
    # Step1-4. 직사각형 가장 오른쪽 열을 아래로 한 칸씩 shift 합니다.
    for row in range(end_row, start_row, -1):
        a[row][end_col] = a[row - 1][end_col]
    
    # Step1-5. 직사각형 가장 위 행을 오른쪽으로 한 칸씩 shift 합니다.
    for col in range(end_col, start_col, -1):
        a[start_row][col] = a[start_row][col - 1]
    
    # Step1-6. temp를 가장 왼쪽 위 모서리를 기준으로 바로 오른쪽 칸에 넣습니다.
    a[start_row][start_col + 1] = temp


# 격자를 벗어나는지 판단합니다.
def in_range(x, y):
    return 1 <= x and x <= n and 1 <= y and y <= m


# x행 y열 (x, y)과 인접한 숫자들과의 평균 값을 계산해줍니다.
# 격자를 벗어나지 않는 숫자들만을 고려해줍니다.
def average(x, y):
    # 자기 자신의 위치를 포함하여 평균을 내야 하므로
    # dx, dy 방향을 5개로 설정하면 한 번에 처리가 가능합니다.
    dxs, dys = [0, 1, -1, 0, 0], [0, 0, 0, 1, -1]
    
    active_numbers = [
        a[x + dx][y + dy]
        for dx, dy in zip(dxs, dys)
        if in_range(x + dx, y + dy)
    ]
    
    return sum(active_numbers) // len(active_numbers)


# 직사각형 내 숫자들을 인접한 숫자들과의 평균값으로 바꿔줍니다.
# 동시에 일어나야 하는 작업이므로, 이미 바뀐 숫자에 주위 숫자들이 영향을 받으면 안되기 때문에
# temp_arr 배열에 평균 값들을 전부 적어 준 다음, 그 값을 다시 복사해 옵니다.
def set_average(start_row, start_col, end_row, end_col):
    # Step2-1. temp_arr에 평균 값을 적습니다.
    for row in range(start_row, end_row + 1):
        for col in range(start_col, end_col + 1):
            temp_arr[row][col] = average(row, col)
    
    # Step2-2. temp_arr 값을 다시 가져옵니다.
    for row in range(start_row, end_row + 1):
        for col in range(start_col, end_col + 1):
            a[row][col] = temp_arr[row][col]


# 조건에 맞춰 값을 바꿔봅니다.
def simulate(start_row, start_col, end_row, end_col):
    # Step1
    # 직사각형 경계에 있는 숫자들을 시계 방향으로 한 칸씩 회전해줍니다.
    rotate(start_row, start_col, end_row, end_col)
    
    # Step2
    # 직사각형 내 각각의 숫자들을 인접한 숫자들과의 평균값으로 바꿔줍니다.
    set_average(start_row, start_col, end_row, end_col)


for row in range(1, n + 1):
    given_nums = list(map(int, input().split()))
    for col, num in enumerate(given_nums, start = 1):
        a[row][col] = num

for _ in range(q):
    r1, c1, r2, c2 = tuple(map(int, input().split()))
    
    # 조건에 맞춰 값을 바꿔봅니다.
    simulate(r1, c1, r2, c2)

# 출력
for row in range(1, n + 1):
    for col in range(1, m + 1):
        print(a[row][col], end = " ")
    print()




    CCW = 0
CW = 1

# 변수 선언 및 입력:
n = int(input())
grid = [
    list(map(int, input().split()))
    for _ in range(n)
]
temp = [
    [0 for _ in range(n)]
    for _ in range(n)
]


def shift(x, y, k, l, move_dir):
    if move_dir == CCW:
        dxs, dys = [-1, -1, 1, 1], [1, -1, -1, 1]
        move_nums = [k, l, k, l]
    else:
        dxs, dys = [-1, -1, 1, 1], [-1, 1, 1, -1]
        move_nums = [l, k, l, k]
    
    # Step1. temp 배열에 grid 값을 복사합니다.
    for i in range(n):
        for j in range(n):
            temp[i][j] = grid[i][j]

    # Step2. 기울어진 직사각형의 경계를 쭉 따라가면서
    #        숫자를 한 칸씩 밀었을 때의 결과를
    #        temp에 저장합니다.
    for dx, dy, move_num in zip(dxs, dys, move_nums):
        for _ in range(move_num):
            nx, ny = x + dx, y + dy
            temp[nx][ny] = grid[x][y]
            x, y = nx, ny
    
    # Step3. temp값을 grid에 옮겨줍니다.
    for i in range(n):
        for j in range(n):
            grid[i][j] = temp[i][j]



x, y, m1, m2, m3, m4, d = tuple(map(int, input().split()))
shift(x - 1, y - 1, m1, m2, d)

for i in range(n):
    for j in range(n):
        print(grid[i][j], end=" ")
    print()



    # 변수 선언 및 입력
n = int(input())
numbers = [
    int(input())
    for _ in range(n)
]
end_of_array = n


# 입력 배열에서 지우고자 하는 부분 수열을 삭제합니다.
def cut_array(start_idx, end_idx):
    global end_of_array
    
    cut_len = end_idx - start_idx + 1;
    for i in range(end_idx + 1, end_of_array):
        numbers[i - cut_len] = numbers[i]
    
    end_of_array -= cut_len


# 두 번에 걸쳐 지우는 과정을 반복합니다.
for _ in range(2):
    s, e = tuple(map(int, input().split()))
    # [s, e] 구간을 삭제합니다.
    cut_array(s - 1, e - 1)

# 출력:
print(end_of_array)
for i in range(end_of_array):
    print(numbers[i])


    # 변수 선언 및 입력
n = int(input())
numbers = [
    int(input())
    for _ in range(n)
]
end_of_array = n


# 입력 배열에서 지우고자 하는 부분 수열을 삭제합니다.
def cut_array(start_idx, end_idx):
    global end_of_array, numbers
    
    temp_arr = [
        numbers[i]
        for i in range(end_of_array)
        if i < start_idx or i > end_idx
    ]
    
    numbers = temp_arr
    end_of_array = len(temp_arr)


# 두 번에 걸쳐 지우는 과정을 반복합니다.
for _ in range(2):
    s, e = tuple(map(int, input().split()))
    # [s, e] 구간을 삭제합니다.
    cut_array(s - 1, e - 1)

# 출력:
print(end_of_array)
for i in range(end_of_array):
    print(numbers[i])



    # 변수 선언 및 입력:

n = int(input())
grid = [
    list(map(int, input().split()))
    for _ in range(n)
]
next_grid = [
    [0 for _ in range(n)]
    for _ in range(n)
]


def in_bomb_range(x, y, center_x, center_y, bomb_range):
    return (x == center_x or y == center_y) and \
           abs(x - center_x) + abs(y - center_y) < bomb_range


def bomb(center_x, center_y):
    bomb_range = grid[center_x][center_y]
    
    # Step1. 폭탄이 터질 위치는 0으로 채워줍니다.
    for i in range(n):
        for j in range(n):
            if in_bomb_range(i, j, center_x, center_y, bomb_range):
                grid[i][j] = 0
	
    # Step2. 폭탄이 터진 이후의 결과를 next_grid에 저장합니다.
    for j in range(n):
        next_row = n - 1
        for i in range(n - 1, -1, -1):
            if grid[i][j]:
                next_grid[next_row][j] = grid[i][j]
                next_row -= 1
                
    # Step3. grid로 다시 값을 옮겨줍니다.
    for i in range(n):
        for j in range(n):
            grid[i][j] = next_grid[i][j]

            
r, c = tuple(map(int, input().split()))
bomb(r - 1, c - 1)

for i in range(n):
    for j in range(n):
        print(grid[i][j], end=" ")
    print()



    # 변수 선언 및 입력
n, m = tuple(map(int, input().split()))
numbers = [
    int(input())
    for _ in range(n)
]


# 주어진 시작점에 대하여
# 부분 수열의 끝 위치를 반환합니다.
def get_end_idx_of_explosion(start_idx, curr_num):
    for end_idx in range(start_idx + 1, len(numbers)):
        if numbers[end_idx] != curr_num:
            return end_idx - 1
        
    return len(numbers) - 1


while True:
    did_explode = False
    
    for curr_idx, number in enumerate(numbers):
        # 각 위치마다 그 뒤로 폭탄이 m개 이상 있는지 확인합니다.
			
		# 이미 터지기로 예정되어있는 폭탄은 패스합니다.
        if number == 0:
            continue
        # curr_idx로부터 연속하여 같은 숫자를 갖는 폭탄 중 
		# 가장 마지막 위치를 찾아 반환합니다.
        end_idx = get_end_idx_of_explosion(curr_idx, number)
        
        if end_idx - curr_idx + 1 >= m:
            # 연속한 숫자의 개수가 m개 이상인 경우 폭탄이 터졌음을 기록해줍니다.
            # 터져야 할 폭탄들에 대해 터졌다는 의미로 0을 채워줍니다.
            numbers[curr_idx:end_idx + 1] = [0] * (end_idx - curr_idx + 1)
            did_explode = True
        
    # 폭탄이 터진 이후의 결과를 계산하여 반영해줍니다.
    numbers = list(filter(lambda x: x > 0, numbers))
    
    # 더 이상 폭탄이 터지지 않는다면 종료합니다.
    if not did_explode:
        break

print(len(numbers))
for number in numbers:
    print(number)


    # 변수 선언 및 입력
n, m = tuple(map(int, input().split()))
numbers = [
    int(input())
    for _ in range(n)
]


# 주어진 시작점에 대하여
# 부분 수열의 끝 위치를 반환합니다.
def get_end_idx_of_explosion(start_idx, curr_num):
    for end_idx in range(start_idx + 1, len(numbers)):
        if numbers[end_idx] != curr_num:
            return end_idx - 1
        
    return len(numbers) - 1


while True:
    did_explode = False
    curr_idx = 0
    
    while curr_idx < len(numbers):
        end_idx = get_end_idx_of_explosion(curr_idx, numbers[curr_idx])
        
        if end_idx - curr_idx + 1 >= m:
            # 연속한 숫자의 개수가 m개 이상이면
            # 폭탄이 터질 수 있는 경우 해당 부분 수열을 잘라내고
            # 폭탄이 터졌음을 기록해줍니다.
            del numbers[curr_idx:end_idx + 1]
            did_explode = True
        else:
            # 주어진 시작 원소에 대하여 폭탄이 터질 수 없는 경우
            # 다음 원소에 대하여 탐색하여 줍니다.
            curr_idx += 1

    if not did_explode:
        break

print(len(numbers))
for number in numbers:
    print(number)


    # 변수 선언 및 입력
n, m = tuple(map(int, input().split()))
numbers = [
    int(input())
    for _ in range(n)
]


# 주어진 시작점에 대하여
# 부분 수열의 끝 위치를 반환합니다.
def get_end_idx_of_explosion(start_idx, curr_num):
    for end_idx in range(start_idx + 1, len(numbers)):
        if numbers[end_idx] != curr_num:
            return end_idx - 1
        
    return len(numbers) - 1


while True:
    did_explode = False
    curr_idx = 0
    
    while curr_idx < len(numbers):
        end_idx = get_end_idx_of_explosion(curr_idx, numbers[curr_idx])
        
        if end_idx - curr_idx + 1 >= m:
            # 연속한 숫자의 개수가 m개 이상이면
            # 폭탄이 터질 수 있는 경우 해당 부분 수열을 잘라내고
            # 폭탄이 터졌음을 기록해줍니다.
            del numbers[curr_idx:end_idx + 1]
            did_explode = True
        else:
            # 주어진 시작 원소에 대하여 폭탄이 터질 수 없는 경우
            # 다음 원소에 대하여 탐색하여 줍니다.
            curr_idx = end_idx + 1

    if not did_explode:
        break

print(len(numbers))
for number in numbers:
    print(number)

    NONE = -1

# 변수 선언 및 입력
n = 4
grid = [
    list(map(int, input().split()))
    for _ in range(n)
]
next_grid = [
    [0 for _ in range(n)]
    for _ in range(n)
]


# grid를 시계방향으로 90' 회전시킵니다.
def rotate():
    # next_grid를 0으로 초기화합니다.
    for i in range(n):
        for j in range(n):
            next_grid[i][j] = 0
    
    # 90' 회전합니다.
    for i in range(n):
        for j in range(n):
            next_grid[i][j] = grid[n - j - 1][i]
    
    # next_grid를 grid에 옮겨줍니다.
    for i in range(n):
        for j in range(n):
            grid[i][j] = next_grid[i][j]


# 아래로 숫자들을 떨어뜨립니다.
def drop():
    # next_grid를 0으로 초기화합니다.
    for i in range(n):
        for j in range(n):
            next_grid[i][j] = 0
    
    # 아래 방향으로 떨어뜨립니다.
    for j in range(n):
        # 같은 숫자끼리 단 한번만
        # 합치기 위해 떨어뜨리기 전에
        # 숫자 하나를 keep해줍니다.
        keep_num, next_row = NONE, n - 1
        
        for i in range(n - 1, -1, -1):
            if not grid[i][j]:
                continue
            
            # 아직 떨어진 숫자가 없다면, 갱신해줍니다.
            if keep_num == NONE:
                keep_num = grid[i][j];
            
            # 가장 최근에 관찰한 숫자가 현재 숫자와 일치한다면
            # 하나로 합쳐주고, keep 값을 비워줍니다.
            elif keep_num == grid[i][j]:
                next_grid[next_row][j] = keep_num * 2
                keep_num = NONE
                
                next_row -= 1
            
            # 가장 최근에 관찰한 숫자와 현재 숫자가 다르다면
            # 최근에 관찰한 숫자를 실제 떨어뜨려주고, keep 값을 갱신해줍니다.
            else:
                next_grid[next_row][j] = keep_num
                keep_num = grid[i][j]
                
                next_row -= 1
        
        # 전부 다 진행했는데도 keep 값이 남아있다면
        # 실제로 한번 떨어뜨려줍니다.
        if keep_num != NONE:
            next_grid[next_row][j] = keep_num
            next_row -= 1
    
    # next_grid를 grid에 옮겨줍니다.
    for i in range(n):
        for j in range(n):
            grid[i][j] = next_grid[i][j]


# move_dir 방향으로 기울이는 것을 진행합니다.
# 회전을 규칙적으로 하기 위해
# 아래, 오른쪽, 위, 왼쪽 순으로 dx, dy 순서를 가져갑니다.
def tilt(move_dir):
    # Step 1.
    # move_dir 횟수만큼 시계방향으로 90'회전하는 것을 반복하여
    # 항상 아래로만 숫자들을 떨어뜨리면 되게끔 합니다.
    for _ in range(move_dir):
        rotate()

    # Step 2.
    # 아래 방향으로 떨어뜨립니다.
    drop()
    
    # Step 3.
    # 4 - move_dir 횟수만큼 시계방향으로 90'회전하는 것을 반복하여
    # 처음 상태로 돌아오게 합니다. (총 360' 회전)
    for _ in range(4 - move_dir):
        rotate()


dir_char = input()

# 아래, 오른쪽, 위, 왼쪽 순으로 
# mapper를 지정합니다.
dir_mapper = {
    'D': 0,
    'R': 1,
    'U': 2,
    'L': 3
}

# 기울입니다.
tilt(dir_mapper[dir_char])

for i in range(n):
    for j in range(n):
        print(grid[i][j], end=" ")
    print()


    OUT_OF_GRID = -1

# 변수 선언 및 입력:
n, m = tuple(map(int, input().split()))
grid = [
    list(map(int, input().split()))
    for _ in range(n)
]
next_grid = [
    [0 for _ in range(n)]
    for _ in range(n)
]


def in_bomb_range(x, y, center_x, center_y, bomb_range):
    return (x == center_x or y == center_y) and \
           abs(x - center_x) + abs(y - center_y) < bomb_range


def bomb(center_x, center_y):
    # Step1. next_grid 값을 0으로 초기화합니다.
    for i in range(n):
        for j in range(n):
            next_grid[i][j] = 0
    
    # Step2. 폭탄이 터질 위치는 0으로 채워줍니다.
    bomb_range = grid[center_x][center_y]
    
    for i in range(n):
        for j in range(n):
            if in_bomb_range(i, j, center_x, center_y, bomb_range):
                grid[i][j] = 0
	
    # Step3. 폭탄이 터진 이후의 결과를 next_grid에 저장합니다.
    for j in range(n):
        next_row = n - 1
        for i in range(n - 1, -1, -1):
            if grid[i][j]:
                next_grid[next_row][j] = grid[i][j]
                next_row -= 1
                
    # Step4. grid로 다시 값을 옮겨줍니다.
    for i in range(n):
        for j in range(n):
            grid[i][j] = next_grid[i][j]


# 해당 col 열에 폭탄이 터질 위치를 구합니다.
# 없다면 OUT_OF_GRID를 반환합니다.
def get_bomb_row(col):
    for row in range(n):
        if grid[row][col] != 0:
            return row
    
    return OUT_OF_GRID

        
# m번에 걸쳐 폭탄이 터집니다.
for _ in range(m):
    bomb_col = int(input()) - 1

    # 폭탄이 터지게 될 위치를 구합니다.
    bomb_row = get_bomb_row(bomb_col)

    if bomb_row == OUT_OF_GRID:
        continue

    bomb(bomb_row, bomb_col)

for i in range(n):
    for j in range(n):
        print(grid[i][j], end=" ")
    print()


    BLANK = -1
WILL_EXPLODE = 0

# 변수 선언 및 입력
n, m, k = tuple(map(int, input().split()))
numbers_2d = [
    list(map(int, input().split()))
    for _ in range(n)
]
numbers_1d = [
    0 for _ in range(n)
]


# 주어진 시작점에 대하여
# 부분 수열의 끝 위치를 반환합니다.
def get_end_idx_of_explosion(start_idx, curr_num):
    for end_idx in range(start_idx + 1, len(numbers_1d)):
        if numbers_1d[end_idx] != curr_num:
            return end_idx - 1
        
    return len(numbers_1d) - 1


def explode():
    while True:
        did_explode = False
        curr_idx = 0
    
        while curr_idx < len(numbers_1d):
            end_idx = get_end_idx_of_explosion(curr_idx, numbers_1d[curr_idx])
        
            if end_idx - curr_idx + 1 >= m:
                # 연속한 숫자의 개수가 m개 이상이면
                # 폭탄이 터질 수 있는 경우 해당 부분 수열을 잘라내고
                # 폭탄이 터졌음을 기록해줍니다.
                del numbers_1d[curr_idx:end_idx + 1]
                did_explode = True
            else:
                # 주어진 시작 원소에 대하여 폭탄이 터질 수 없는 경우
                # 다음 원소에 대하여 탐색하여 줍니다.
                curr_idx = end_idx + 1

        if not did_explode:
            break


##################################################################################
##			이 줄을 기준으로 위에 있는 함수들에 대한 설명은 1차원 폭발 게임을 참조해주세요     	  ##
##################################################################################

        
# 격자의 특정 열을 일차원 배열에 복사해줍니다.
def copy_column(col):
    global numbers_1d
    
    numbers_1d = [
        numbers_2d[row][col]
        for row in range(n)
        if numbers_2d[row][col] != BLANK
    ]


# 폭탄이 터진 결과를 격자의 해당 열에 복사해줍니다.
def copy_result(col):
    for row in range(n - 1, -1, -1):
        numbers_2d[row][col] = numbers_1d.pop() if numbers_1d \
                                                else BLANK


# 폭탄이 터지는 과정을 시뮬레이션 합니다.
def simulate():
    for col in range(n):
        copy_column(col)
        explode()
        copy_result(col)

        
# 반시계 방향으로 90도 회전해줍니다.
def rotate():
    global numbers_2d
    
    # 빈 칸으로 초기화 된 임시 격자를 선언합니다.
    temp_2d = [
        [BLANK for _ in range(n)]
        for _ in range(n)
    ]
    
    # 기존 격자를 반시계 방향으로 90도 회전했을 때의 결과를
    # 임시 격자에 저장해줍니다.
    for i in range(n - 1, -1, -1):
        curr_idx = n - 1
        for j in range(n - 1, -1, -1):
            if numbers_2d[i][j] != BLANK:
                temp_2d[curr_idx][n - i - 1] = numbers_2d[i][j]
                curr_idx -= 1
    
    # 임시 격자에 저장된 값을 기존 격자에 복사합니다.
    numbers_2d = temp_2d

        
# 주어진 입력에 따라 폭탄이 터지는 것을 시뮬레이션 합니다.
simulate()
for _ in range(k):
    rotate()
    simulate()

        
# 격자를 순회하며 남아 있는 폭탄의 개수를 세줍니다.
answer = sum([
    numbers_2d[i][j] != BLANK
    for i in range(n)
    for j in range(n)
])
print(answer)


# 변수 선언 및 입력
n, curr_x, curr_y = tuple(map(int, input().split()))
a = [
    [0 for _ in range(n + 1)]
    for _ in range(n + 1)
]

# 방문하게 되는 숫자들을 담을 곳입니다.
visited_nums = []


# 범위가 격자 안에 들어가는지 확인합니다.
def in_range(x, y):
    return 1 <= x and x <= n and 1 <= y and y <= n


# 범위가 격자 안이고, 해당 위치의 값이 더 큰지 확인합니다.
def can_go(x, y, curr_num):
    return in_range(x, y) and a[x][y] > curr_num


# 조건에 맞춰 움직여봅니다.
# 움직였다면 true를 반환하고
# 만약 움직일 수 있는 곳이 없었다면 false를 반환합니다.
def simulate():
    global curr_x, curr_y
    
    # 코딩의 간결함을 위해 
    # 문제 조건에 맞게 상하좌우 순서로
    # 방향을 정의합니다.
    dxs, dys = [-1, 1, 0, 0], [0, 0, -1, 1]
    
    # 각각의 방향에 대해 나아갈 수 있는 곳이 있는지 확인합니다.
    for dx, dy in zip(dxs, dys):
        next_x, next_y = curr_x + dx, curr_y + dy
        
        # 갈 수 있는 곳이라면
        # 이동하고 true를 반환합니다.
        if can_go(next_x, next_y, a[curr_x][curr_y]):
            curr_x, curr_y = next_x, next_y
            return True
    
    # 움직일 수 있는 곳이 없었다는 의미로
    # false 값을 반환합니다.
    return False


for i in range(1, n + 1):
    given_row = list(map(int, input().split()))
    for j, elem in enumerate(given_row, start = 1):
        a[i][j] = elem

# 초기 위치에 적혀있는 값을 답에 넣어줍니다.
visited_nums.append(a[curr_x][curr_y])
while True:
    # 조건에 맞춰 움직여봅니다.
    greater_number_exist = simulate()
    
    # 인접한 곳에 더 큰 숫자가 없다면 종료합니다.
    if not greater_number_exist:
        break
    
    # 움직이고 난 후의 위치를 답에 넣어줍니다.
    visited_nums.append(a[curr_x][curr_y])

# 출력:
for num in visited_nums:
    print(num, end=' ')


    # 변수 선언 및 입력: 

n, m, k = tuple(map(int, input().split()))
grid = [
    list(map(int, input().split()))
    for _ in range(n)
]

# 해당 row에 [col_s, col_e] 열에
# 전부 블럭이 없는지를 확인합니다.
def all_blank(row, col_s, col_e):
    return all([
        not grid[row][col]
        for col in range(col_s, col_e + 1)
    ])


# 최종적으로 도달하게 될 위치는
# 그 다음 위치에 최초로 블럭이 존재하는 순간임을 이용합니다.
def get_target_row():
    for row in range(n - 1):
        if not all_blank(row + 1, k, k + m - 1):
            return row

    return 0
        

k -= 1

# 최종적으로 멈추게 될 위치를 구합니다.
target_row = get_target_row()

# 최종 위치에 전부 블럭을 표시합니다.
for col in range(k, k + m):
    grid[target_row][col] = 1

for i in range(n):
    for j in range(n):
        print(grid[i][j], end=" ")
    print()


    import sys

DIR_NUM = 4

# 변수 선언 및 입력
n = int(input())
curr_x, curr_y = tuple(map(int, input().split()))
a = [
    [0 for _ in range(n + 1)]
    for _ in range(n + 1)
]

# 미로 탈출이 불가능한지 여부를 판단하기 위해
# 동일한 위치에 동일한 방향으로 진행했던 적이 있는지를
# 표시해주는 배열입니다.
visited = [
    [
        [False for _ in range(DIR_NUM)]
        for _ in range(n + 1)
    ]
    for _ in range(n + 1)
]
elapsed_time = 0

# 처음에는 우측 방향을 바라보고 시작합니다.
curr_dir = 0


# 범위가 격자 안에 들어가는지 확인합니다.
def in_range(x, y):
    return 1 <= x and x <= n and 1 <= y and y <= n


# 해당 위치에 벽이 있으면 이동이 불가합니다.
def wall_exist(x, y):
    return in_range(x, y) and a[x][y] == '#'


# 조건에 맞춰 움직여봅니다.
def simulate():
    global curr_x, curr_y, curr_dir, elapsed_time
    
    # 현재 위치에 같은 방향으로 진행한 적이 이미 있었는지 확인합니다.
    # 이미 한 번 겪었던 상황이라면, 탈출이 불가능 하다는 의미이므로 
    # -1을 출력하고 프로그램을 종료합니다.
    if visited[curr_x][curr_y][curr_dir]:
        print(-1)
        sys.exit(0)
    
    # 현재 상황이 다시 반복되는지를 나중에 확인하기 위해
    # 현재 상황에 해당하는 곳에 visited 값을 True로 설정합니다.
    visited[curr_x][curr_y][curr_dir] = True
    
    dxs, dys = [0, 1, 0, -1], [1, 0, -1, 0]
    
    next_x, next_y = curr_x + dxs[curr_dir], curr_y + dys[curr_dir]
    
    # Step1
    
    # 바라보고 있는 방향으로 이동하는 것이 불가능한 경우에는
    # 반 시계 방향으로 90' 방향을 바꿉니다.
    if wall_exist(next_x, next_y):
        curr_dir = (curr_dir - 1 + 4) % 4
    
    # Step2
    
    # Case1
    # 바라보고 있는 방향으로 이동하는 것이 가능한 경우 중
    # 바로 앞이 격자 밖이라면 탈출합니다.
    elif not in_range(next_x, next_y):
        curr_x, curr_y = next_x, next_y
        elapsed_time += 1
    
    # Case 2 & Case 3
    # 바로 앞이 격자 안에서 이동할 수 있는 곳이라면
    else:
        # 그 방향으로 이동했다 가정헀을 때 바로 오른쪽에 짚을 벽이 있는지 봅니다.
        rx = next_x + dxs[(curr_dir + 1) % 4]
        ry = next_y + dys[(curr_dir + 1) % 4]
        
        # Case2
        # 그대로 이동해도 바로 오른쪽에 짚을 벽이 있다면
        # 해당 방향으로 한 칸 이동합니다.
        if wall_exist(rx, ry):
            curr_x, curr_y = next_x, next_y
            elapsed_time += 1
        
        # Case3
        # 그렇지 않다면 2칸 이동후 방향을 시계방향으로 90' 방향을 바꿉니다.
        else:
            curr_x, curr_y = rx, ry
            curr_dir = (curr_dir + 1) % 4
            elapsed_time += 2


for i in range(1, n + 1):
    given_row = input()
    for j, elem in enumerate(given_row, start = 1):
        a[i][j] = elem

# 격자를 빠져나오기 전까지 계속 반복합니다.
while in_range(curr_x, curr_y):
    # 조건에 맞춰 움직여봅니다.
    simulate()

print(elapsed_time)


OUT_OF_GRID = (-1, -1)

# 변수 선언 및 입력
n, m, x, y = tuple(map(int, input().split()))
grid = [
    [0 for _ in range(n)]
    for _ in range(n)
]
movements = input().split()

# 주사위가 놓여있는 상태 
up, front, right = 1, 2, 3


# 격자 안에 있는지를 확인합니다.
def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < n


# 해당 방향으로 이동했을 때의 다음 위치를 구합니다.
# 이동이 불가능할 경우 OUT_OF_GRID를 반환합니다.
def next_pos(x, y, move_dir):
    dxs, dys = [0, 0, -1, 1], [1, -1, 0, 0]
    nx, ny = x + dxs[move_dir], y + dys[move_dir]
    return (nx, ny) if in_range(nx, ny) else OUT_OF_GRID


def simulate(move_dir):
    global x, y
    global up, front, right
    
    # move_dir 방향으로 굴렸을 때의 격자상의 위치를 구합니다.
    nx, ny = next_pos(x, y, move_dir)
    # 굴리는게 불가능한 경우라면 패스합니다.
    if (nx, ny) == OUT_OF_GRID:
        return
    
    # 위치를 이동합니다.
    x, y = nx, ny
    
    # 주사위가 놓여있는 상태를 조정합니다.
    if move_dir == 0: # 동쪽
        up, front, right = 7 - right, front, up
    elif move_dir == 1: # 서쪽
        up, front, right = right, front, 7 - up
    elif move_dir == 2: # 북쪽
        up, front, right = front, 7 - up, right
    else: # 남쪽
        up, front, right = 7 - front, up, right
    
    # 바닥에 적혀있는 숫자를 변경합니다.
    bottom = 7 - up
    grid[x][y] = bottom


x -= 1
y -= 1

dir_mapper = {
    'R': 0,
    'L': 1,
    'U': 2,
    'D': 3
}

# 시뮬레이션 진행
grid[x][y] = 6
for char_dir in movements:
    simulate(dir_mapper[char_dir])

ans = sum([
    grid[i][j]
    for i in range(n)
    for j in range(n)
])

print(ans)


# 변수 선언 및 입력:
n, m, r, c = tuple(map(int, input().split()))
grid = [
    [0 for _ in range(n)]
    for _ in range(n)
]
next_grid = [
    [0 for _ in range(n)]
    for _ in range(n)
]


def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < n


def expand(x, y, dist):
    dxs, dys = [-1, 1, 0, 0], [0, 0, -1, 1]
    for dx, dy in zip(dxs, dys):
        nx, ny = x + dx * dist, y + dy * dist
        if in_range(nx, ny):
            next_grid[nx][ny] = 1


def simulate(dist):
    # Step1. next_grid 값을 0으로 초기화합니다.
    for i in range(n):
        for j in range(n):
            next_grid[i][j] = 0
    
    # Step2. 폭탄을 던지는 시뮬레이션을 진행합니다.
    for i in range(n):
        for j in range(n):
            if grid[i][j]:
                expand(i, j, dist)

    # Step3. next_grid 값을 grid로 업데이트 해줍니다.
    for i in range(n):
        for j in range(n):
            if next_grid[i][j]:
                grid[i][j] = 1

    
grid[r - 1][c - 1] = 1

# 총 m번 시뮬레이션을 진행합니다.
dist = 1
for _ in range(m):
    simulate(dist)
    dist *= 2

ans = sum([
    grid[i][j]
    for i in range(n)
    for j in range(n)
])

print(ans)


# 변수 선언 및 입력
n, m, K = tuple(map(int, input().split()))
apple = [
    [False for _ in range(n + 1)]
    for _ in range(n + 1)
]
# 뱀은 처음에 (1, 1)에서 길이 1의 상태로 있습니다.
snake = [(1, 1)]

# 입력으로 주어진 방향을 정의한 dx, dy에 맞도록
# 변환하는데 쓰이는 dict를 정의합니다.
mapper = {
    'D': 0,
    'U': 1,
    'R': 2,
    'L': 3
}

ans = 0


# (x, y)가 범위 안에 들어가는지 확인합니다.
def can_go(x, y):
    return 1 <= x and x <= n and 1 <= y and y <= n


# 뱀이 꼬였는지 확인합니다.
def is_twisted(new_head):
    return new_head in snake


# 새로운 머리를 추가합니다.
def push_front(new_head):
    # 몸이 꼬이는 경우
    # false를 반환합니다.
    if is_twisted(new_head):
        return False
    
    # 새로운 머리를 추가합니다.
    snake.insert(0, new_head)
    
    # 정상적으로 머리를 추가했다는 의미로
    # True를 반환합니다.
    return True


# 꼬리를 지웁니다.
def pop_back():
    snake.pop()


# (nx, ny)쪽으로 뱀을 움직입니다.
def move_snake(nx, ny):
    # 머리가 이동할 자리에 사과가 존재하면
    # 사과는 사라지게 되고
    if apple[nx][ny]:
        apple[nx][ny] = False
        # 꼬리는 사라지지 않고 머리만 늘어납니다.
        # 늘어난 머리때문에 몸이 꼬이게 된다면
        # False를 반환합니다.
        if not push_front((nx, ny)):
            return False
    else:
        # 사과가 없으면 꼬리는 사라지게 되고
        pop_back()
        
        # 머리는 늘어나게 됩니다.
        # 늘어난 머리때문에 몸이 꼬이게 된다면
        # False를 반환합니다.
        if not push_front((nx, ny)):
            return False
    
    # 정상적으로 뱀이 움직였으므로
    # True를 반환합니다.
    return True


# 뱀을 move_dir 방향으로 num번 움직입니다.
def move(move_dir, num):
    global ans
    
    dxs, dys = [1, -1, 0, 0], [0, 0, 1, -1]
    
    # num 횟수만큼 뱀을 움직입니다.
    # 한 번 움직일때마다 답을 갱신합니다.
    for _ in range(num):
        ans += 1
        
        # 뱀의 머리가 그다음으로 움직일
        # 위치를 구합니다.
        (head_x, head_y) = snake[0]
        nx = head_x + dxs[move_dir]
        ny = head_y + dys[move_dir]
        
        # 그 다음 위치로 갈 수 없다면
        # 게임을 종료합니다.
        if not can_go(nx, ny):
            return False
        
        # 뱀을 한 칸 움직입니다.
        # 만약 몸이 꼬인다면 False를 반환합니다.
        if not move_snake(nx, ny):
            return False
    
    # 정상적으로 명령을 수행했다는 의미인 True를 반환합니다.
    return True


# 사과가 있는 위치를 표시합니다.
for _ in range(m):
    x, y = tuple(map(int, input().split()))
    apple[x][y] = True

# K개의 명령을 수행합니다.
for _ in range(K):
    # move_dir 방향으로 num 횟수 만큼 움직여야 합니다.
    move_dir, num = tuple(input().split())
    num = int(num)
    
    # 움직이는 도중 게임이 종료되었을 경우
    # 더 이상 진행하지 않습니다.
    if not move(mapper[move_dir], num):
        break

print(ans)

import collections

# 변수 선언 및 입력
n, m, K = tuple(map(int, input().split()))
apple = [
    [False for _ in range(n + 1)]
    for _ in range(n + 1)
]
# 뱀은 처음에 (1, 1)에서 길이 1의 상태로 있습니다.
snake = collections.deque([(1, 1)])

# 입력으로 주어진 방향을 정의한 dx, dy에 맞도록
# 변환하는데 쓰이는 dict를 정의합니다.
mapper = {
    'D': 0,
    'U': 1,
    'R': 2,
    'L': 3
}

ans = 0


# (x, y)가 범위 안에 들어가는지 확인합니다.
def can_go(x, y):
    return 1 <= x and x <= n and 1 <= y and y <= n


# 뱀이 꼬였는지 확인합니다.
def is_twisted(new_head):
    return new_head in snake


# 새로운 머리를 추가합니다.
def push_front(new_head):
    # 몸이 꼬이는 경우
    # false를 반환합니다.
    if is_twisted(new_head):
        return False
    
    # 새로운 머리를 추가합니다.
    snake.appendleft(new_head)
    
    # 정상적으로 머리를 추가했다는 의미로
    # True를 반환합니다.
    return True


# 꼬리를 지웁니다.
def pop_back():
    snake.pop()


# (nx, ny)쪽으로 뱀을 움직입니다.
def move_snake(nx, ny):
    # 머리가 이동할 자리에 사과가 존재하면
    # 사과는 사라지게 되고
    if apple[nx][ny]:
        apple[nx][ny] = False
        # 꼬리는 사라지지 않고 머리만 늘어납니다.
        # 늘어난 머리때문에 몸이 꼬이게 된다면
        # False를 반환합니다.
        if not push_front((nx, ny)):
            return False
    else:
        # 사과가 없으면 꼬리는 사라지게 되고
        pop_back()
        
        # 머리는 늘어나게 됩니다.
        # 늘어난 머리때문에 몸이 꼬이게 된다면
        # False를 반환합니다.
        if not push_front((nx, ny)):
            return False
    
    # 정상적으로 뱀이 움직였으므로
    # True를 반환합니다.
    return True


# 뱀을 move_dir 방향으로 num번 움직입니다.
def move(move_dir, num):
    global ans
    
    dxs, dys = [1, -1, 0, 0], [0, 0, 1, -1]
    
    # num 횟수만큼 뱀을 움직입니다.
    # 한 번 움직일때마다 답을 갱신합니다.
    for _ in range(num):
        ans += 1
        
        # 뱀의 머리가 그다음으로 움직일
        # 위치를 구합니다.
        (head_x, head_y) = snake[0]
        nx = head_x + dxs[move_dir]
        ny = head_y + dys[move_dir]
        
        # 그 다음 위치로 갈 수 없다면
        # 게임을 종료합니다.
        if not can_go(nx, ny):
            return False
        
        # 뱀을 한 칸 움직입니다.
        # 만약 몸이 꼬인다면 False를 반환합니다.
        if not move_snake(nx, ny):
            return False
    
    # 정상적으로 명령을 수행했다는 의미인 True를 반환합니다.
    return True


# 사과가 있는 위치를 표시합니다.
for _ in range(m):
    x, y = tuple(map(int, input().split()))
    apple[x][y] = True

# K개의 명령을 수행합니다.
for _ in range(K):
    # move_dir 방향으로 num 횟수 만큼 움직여야 합니다.
    move_dir, num = tuple(input().split())
    num = int(num)
    
    # 움직이는 도중 게임이 종료되었을 경우
    # 더 이상 진행하지 않습니다.
    if not move(mapper[move_dir], num):
        break

print(ans)


import collections

# 변수 선언 및 입력
n, m, K = tuple(map(int, input().split()))
apple = [
    [False for _ in range(n + 1)]
    for _ in range(n + 1)
]
# 뱀은 처음에 (1, 1)에서 길이 1의 상태로 있습니다.
snake = collections.deque([(1, 1)])
snake_pos = set([(1, 1)])

# 입력으로 주어진 방향을 정의한 dx, dy에 맞도록
# 변환하는데 쓰이는 dict를 정의합니다.
mapper = {
    'D': 0,
    'U': 1,
    'R': 2,
    'L': 3
}

ans = 0


# (x, y)가 범위 안에 들어가는지 확인합니다.
def can_go(x, y):
    return 1 <= x and x <= n and 1 <= y and y <= n


# 뱀이 꼬였는지 확인합니다.
# 몸이 꼬였는지 여부는
# HashSet에 새로 들어온 머리 위치가
# 이미 존재하는지를 확인하면 됩니다.
def is_twisted(new_head):
    return new_head in snake_pos


# 새로운 머리를 추가합니다.
def push_front(new_head):
    # 몸이 꼬이는 경우
    # false를 반환합니다.
    if is_twisted(new_head):
        return False
    
    # 새로운 머리를 추가하고
    snake.appendleft(new_head)
    # HashSet에 새로운 좌표를 기록합니다.
    snake_pos.add(new_head)
            
    # 정상적으로 머리를 추가했다는 의미로
    # True를 반환합니다.
    return True


# 꼬리를 지웁니다.
def pop_back():
    # 머리 부분을 List에서 삭제하고
    tail = snake.pop()
    # HashSet에서도 지웁니다.
    snake_pos.remove(tail)


# (nx, ny)쪽으로 뱀을 움직입니다.
def move_snake(nx, ny):
    # 머리가 이동할 자리에 사과가 존재하면
    # 사과는 사라지게 되고
    if apple[nx][ny]:
        apple[nx][ny] = False
        # 꼬리는 사라지지 않고 머리만 늘어납니다.
        # 늘어난 머리때문에 몸이 꼬이게 된다면
        # False를 반환합니다.
        if not push_front((nx, ny)):
            return False
    else:
        # 사과가 없으면 꼬리는 사라지게 되고
        pop_back()
        
        # 머리는 늘어나게 됩니다.
        # 늘어난 머리때문에 몸이 꼬이게 된다면
        # False를 반환합니다.
        if not push_front((nx, ny)):
            return False
    
    # 정상적으로 뱀이 움직였으므로
    # True를 반환합니다.
    return True


# 뱀을 move_dir 방향으로 num번 움직입니다.
def move(move_dir, num):
    global ans
    
    dxs, dys = [1, -1, 0, 0], [0, 0, 1, -1]
    
    # num 횟수만큼 뱀을 움직입니다.
    # 한 번 움직일때마다 답을 갱신합니다.
    for _ in range(num):
        ans += 1
        
        # 뱀의 머리가 그다음으로 움직일
        # 위치를 구합니다.
        (head_x, head_y) = snake[0]
        nx = head_x + dxs[move_dir]
        ny = head_y + dys[move_dir]
        
        # 그 다음 위치로 갈 수 없다면
        # 게임을 종료합니다.
        if not can_go(nx, ny):
            return False
        
        # 뱀을 한 칸 움직입니다.
        # 만약 몸이 꼬인다면 False를 반환합니다.
        if not move_snake(nx, ny):
            return False
    
    # 정상적으로 명령을 수행했다는 의미인 True를 반환합니다.
    return True


# 사과가 있는 위치를 표시합니다.
for _ in range(m):
    x, y = tuple(map(int, input().split()))
    apple[x][y] = True

# K개의 명령을 수행합니다.
for _ in range(K):
    # move_dir 방향으로 num 횟수 만큼 움직여야 합니다.
    move_dir, num = tuple(input().split())
    num = int(num)
    
    # 움직이는 도중 게임이 종료되었을 경우
    # 더 이상 진행하지 않습니다.
    if not move(mapper[move_dir], num):
        break

print(ans)



# 변수 선언 및 입력
n, m, t = tuple(map(int, input().split()))
a = [
    [0 for _ in range(n + 1)]
    for _ in range(n + 1)
]
count = [
    [0 for _ in range(n + 1)]
    for _ in range(n + 1)
]
next_count = [
    [0 for _ in range(n + 1)]
    for _ in range(n + 1)
]


# 범위가 격자 안에 들어가는지 확인합니다.
def in_range(x, y):
    return 1 <= x and x <= n and 1 <= y and y <= n


# 인접한 곳들 중 가장 값이 큰 위치를 반환합니다.
def get_max_neighbor_pos(curr_x, curr_y):
    # 코딩의 간결함을 위해 
    # 문제 조건에 맞게 상하좌우 순서로
    # 방향을 정의합니다.
    dxs, dys = [-1, 1, 0, 0], [0, 0, -1, 1]
    
    max_num, max_pos = 0, (0, 0)
    
    # 각각의 방향에 대해 나아갈 수 있는 곳이 있는지 확인합니다.
    for dx, dy in zip(dxs, dys):
        next_x, next_y = curr_x + dx, curr_y + dy
        
        # 범위안에 들어오는 격자 중 최댓값을 갱신합니다.
        if in_range(next_x, next_y) and a[next_x][next_y] > max_num:
            max_num = a[next_x][next_y]
            max_pos = (next_x, next_y)
    
    return max_pos


# (x, y) 위치에 있는 구슬을 움직입니다.
def move(x, y):
    # 인접한 곳들 중 가장 값이 큰 위치를 계산합니다.
    next_x, next_y = get_max_neighbor_pos(x, y)
    
    # 그 다음 위치에 구슬의 개수를 1만큼 추가해줍니다.
    next_count[next_x][next_y] += 1


# 구슬을 전부 한 번씩 움직여 봅니다.
def move_all():
    # 그 다음 각 위치에서의 구슬 개수를 전부 초기화해놓습니다.
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            next_count[i][j] = 0
            
    # (i, j) 위치에 구슬이 있는경우 
    # 움직임을 시도해보고, 그 결과를 전부 next_count에 기록합니다.
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if count[i][j] == 1:
                move(i, j)
    
    # next_count 값을 count에 복사합니다.
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            count[i][j] = next_count[i][j]


# 충돌이 일어나는 구슬은 전부 지워줍니다.
def remove_duplicate_marbles():
    # 충돌이 일어난 구슬들이 있는 위치만 빈 곳으로 설정하면 됩니다.
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if count[i][j] >= 2:
                count[i][j] = 0


# 조건에 맞춰 시뮬레이션을 진행합니다.
def simulate():
    # Step1
    # 구슬을 전부 한 번씩 움직여 봅니다.
    move_all()
    
    # Step2
    # 움직임 이후에 충돌이 일어나는 구슬들을 골라 목록에서 지워줍니다.
    remove_duplicate_marbles()


for i in range(1, n + 1):
    given_row = list(map(int, input().split()))
    for j, elem in enumerate(given_row, start = 1):
        a[i][j] = elem
        
# 초기 count 배열을 설정합니다.
# 구슬이 있는 곳에 1을 표시합니다.
for _ in range(m):
    x, y = tuple(map(int, input().split()))
    count[x][y] = 1
    
# t초 동안 시뮬레이션을 진행합니다.
for _ in range(t):
    simulate()

# 출력:
ans = sum([
    count[i][j]
    for i in range(1, n + 1)
    for j in range(1, n + 1)
])

print(ans)


# 변수 선언 및 입력:
n, m = tuple(map(int, input().split()))
grid = [
    list(map(int, input().split()))
    for _ in range(n)
]


def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < n


def find_pos(num):
    for i in range(n):
        for j in range(n):
            if grid[i][j] == num:
                return (i, j)


# 그 다음 위치를 찾아 반환합니다.
def next_pos(pos):
    dxs = [-1, -1, -1,  0, 0,  1, 1, 1]
    dys = [-1,  0,  1, -1, 1, -1, 0, 1]
    
    x, y = pos
    
    # 인접한 8개의 칸 중 가장 값이 큰 위치를 찾아 반환합니다.
    max_val = -1
    max_pos = (-1, -1)
    for dx, dy in zip(dxs, dys):
        nx, ny = x + dx, y + dy
        if in_range(nx, ny) and grid[nx][ny] > max_val:
            max_val, max_pos = grid[nx][ny], (nx, ny)
    
    return max_pos


def swap(pos, next_pos):
    (x, y), (nx, ny) = pos, next_pos
    grid[x][y], grid[nx][ny] = grid[nx][ny], grid[x][y]


def simulate():
    # 번호가 증가하는 순으로
    # 그 다음 위치를 구해
    # 한 칸씩 움직입니다.
    for num in range(1, n * n + 1):
        pos = find_pos(num)
        max_pos = next_pos(pos)
        swap(pos, max_pos)


# m번 시뮬레이션을 진행합니다.
for _ in range(m):
    simulate()

for i in range(n):
    for j in range(n):
        print(grid[i][j], end=" ")
    print()



    # 변수 선언 및 입력
t = int(input())
n, m = 0, 0
marbles = []

# 입력으로 주어진 방향을 정의한 dx, dy에 맞도록
# 변환하는데 쓰이는 dict를 정의합니다.
mapper = {
    'U': 0,
    'R': 1,
    'L': 2,
    'D': 3
}


# 해당 위치가 격자 안에 들어와 있는지 확인합니다.
def in_range(x, y):
    return 1 <= x and x <= n and 1 <= y and y <= n


# 해당 구슬이 1초 후에 어떤 위치에서 어떤 방향을 보고 있는지를 구해
# 그 상태를 반환합니다.
def move(marble):
    # 구슬이 벽에 부딪혔을 때의 처리를 간단히 하기 위해
    # dir 기준 0, 3이 대칭 1, 2가 대칭이 되도록 설정합니다.
    dxs, dys = [-1, 0, 0, 1], [0, 1, -1, 0]
    
    x, y, move_dir = marble
    
    # 바로 앞에 벽이 있는지를 판단합니다.
    nx, ny = x + dxs[move_dir], y + dys[move_dir]
    
    # Case 1 : 벽이 없는 경우에는 그대로 한 칸 전진합니다.
    if in_range(nx, ny):
        return (nx, ny, move_dir)
    # Case 2 : 벽이 있는 경우에는 방향을 반대로 틀어줍니다.
    # 위에서 dx, dy를 move_dir 기준 0, 3이 대칭 1, 2가 대칭이 되도록
    # 설정해놨기 때문에 간단하게 처리가 가능합니다.
    else:
        return (x, y, 3 - move_dir)
    

# 구슬을 전부 한 번씩 움직여봅니다.
def move_all():
    for i, marble in enumerate(marbles):
        marbles[i] = move(marble)


# 해당 구슬과 충돌이 일어나는 구슬이 있는지 확인합니다.
# 자신을 제외한 구슬 중에 위치가 동일한 구슬이 있는지 확인하면 됩니다.
def duplicate_marble_exist(target_idx):
    target_x, target_y, _ = marbles[target_idx]
    
    return any([
        i != target_idx and (x, y) == (target_x, target_y) 
        for i, (x, y, _) in enumerate(marbles)
    ])
    

# 충돌이 일어나는 구슬을 전부 지워줍니다.
def remove_duplicate_marbles():
    global marbles
    
    marbles = [
        marble
        for i, marble in enumerate(marbles)
        if not duplicate_marble_exist(i)
    ]


# 조건에 맞춰 시뮬레이션을 진행합니다.
def simulate():
    # Step1
    # 구슬을 전부 한 번씩 움직여봅니다.
    move_all()
    
    # Step2
    # 움직임 이후에 충돌이 일어나는 구슬들을 골라 목록에서 지워줍니다.
    remove_duplicate_marbles()


for _ in range(t):
    # 새로운 테스트 케이스가 시작될때마다 기존에 사용하던 값들을 초기화해줍니다.
    marbles = []
    
    # 입력
    n, m = tuple(map(int, input().split()))
    for _ in range(m):
        x, y, d = tuple(input().split())
        x, y = int(x), int(y)
        marbles.append((x, y, mapper[d]))
    
    # 2 * n번 이후에는 충돌이 절대 일어날 수 없으므로
    # 시뮬레이션을 총 2 * n번 진행합니다.
    for _ in range(2 * n):
        simulate()
    
    # 출력
    print(len(marbles))


    MAX_N = 50

# 변수 선언 및 입력
t = int(input())
n, m = 0, 0
marbles = []
marble_cnt = [
    [0 for _ in range(MAX_N + 1)]
    for _ in range(MAX_N + 1)
]

# 입력으로 주어진 방향을 정의한 dx, dy에 맞도록
# 변환하는데 쓰이는 dict를 정의합니다.
mapper = {
    'U': 0,
    'R': 1,
    'L': 2,
    'D': 3
}


# 해당 위치가 격자 안에 들어와 있는지 확인합니다.
def in_range(x, y):
    return 1 <= x and x <= n and 1 <= y and y <= n


# 해당 구슬이 1초 후에 어떤 위치에서 어떤 방향을 보고 있는지를 구해
# 그 상태를 반환합니다.
def move(marble):
    # 구슬이 벽에 부딪혔을 때의 처리를 간단히 하기 위해
    # dir 기준 0, 3이 대칭 1, 2가 대칭이 되도록 설정합니다.
    dxs, dys = [-1, 0, 0, 1], [0, 1, -1, 0]
    
    x, y, move_dir = marble
    
    # 바로 앞에 벽이 있는지를 판단합니다.
    nx, ny = x + dxs[move_dir], y + dys[move_dir]
    
    # Case 1 : 벽이 없는 경우에는 그대로 한 칸 전진합니다.
    if in_range(nx, ny):
        return (nx, ny, move_dir)
    # Case 2 : 벽이 있는 경우에는 방향을 반대로 틀어줍니다.
    # 위에서 dx, dy를 move_dir 기준 0, 3이 대칭 1, 2가 대칭이 되도록
    # 설정해놨기 때문에 간단하게 처리가 가능합니다.
    else:
        return (x, y, 3 - move_dir)
    

# 구슬을 전부 한 번씩 움직여봅니다.
def move_all():
    for i, marble in enumerate(marbles):
        marbles[i] = move(marble)


# 해당 구슬과 충돌이 일어나는 구슬이 있는지 확인합니다.
# 이를 위해 자신의 현재 위치에 놓은 구슬의 개수가
# 자신을 포함하여 2개 이상인지 확인합니다.
def duplicate_marble_exist(target_idx):
    target_x, target_y, _ = marbles[target_idx]
    
    return marble_cnt[target_x][target_y] >= 2
    

# 충돌이 일어나는 구슬을 전부 지워줍니다.
def remove_duplicate_marbles():
    global marbles
    
    # Step2-1 : 각 구슬의 위치에 count를 증가 시킵니다.
    for x, y, _ in marbles:
        marble_cnt[x][y] += 1

    # Step2-2 : 충돌이 일어나지 않은 구슬만 전부 기록합니다.
    remaining_marbles = [
        marble
        for i, marble in enumerate(marbles)
        if not duplicate_marble_exist(i)
    ]
    
    # Step2-3 : 나중을 위해 각 구슬의 위치에 적어놓은 count 수를 다시 초기화합니다.
    for x, y, _ in marbles:
        marble_cnt[x][y] -= 1
    
    # Step2-4 : 충돌이 일어나지 않은 구슬들로 다시 채워줍니다.
    marbles = remaining_marbles


# 조건에 맞춰 시뮬레이션을 진행합니다.
def simulate():
    # Step1
    # 구슬을 전부 한 번씩 움직여봅니다.
    move_all()
    
    # Step2
    # 움직임 이후에 충돌이 일어나는 구슬들을 골라 목록에서 지워줍니다.
    remove_duplicate_marbles()


for _ in range(t):
    # 새로운 테스트 케이스가 시작될때마다 기존에 사용하던 값들을 초기화해줍니다.
    marbles = []
    
    # 입력
    n, m = tuple(map(int, input().split()))
    for _ in range(m):
        x, y, d = tuple(input().split())
        x, y = int(x), int(y)
        marbles.append((x, y, mapper[d]))
    
    # 2 * n번 이후에는 충돌이 절대 일어날 수 없으므로
    # 시뮬레이션을 총 2 * n번 진행합니다.
    for _ in range(2 * n):
        simulate()
    
    # 출력
    print(len(marbles))


    BLANK = -1
COLLIDE = -2

# 변수 선언 및 입력
t = int(input())
n, m = 0, 0
curr_dir = list()
next_dir = list()

# 입력으로 주어진 방향을 정의한 dx, dy에 맞도록
# 변환하는데 쓰이는 dict를 정의합니다.
mapper = {
    'U': 0,
    'R': 1,
    'L': 2,
    'D': 3
}


# 해당 위치가 격자 안에 들어와 있는지 확인합니다.
def in_range(x, y):
    return 1 <= x and x <= n and 1 <= y and y <= n


# 해당 위치에 dir 방향을 갖는 구슬이 새롭게 추가되는 경우에 대한
# 처리를 합니다.
def update_next_dir(x, y, move_dir):
    # 빈 곳이었다면 해당 구슬을 넣어주고
    if next_dir[x][y] == BLANK:
        next_dir[x][y] = move_dir
    # 빈 곳이 아니었다면 이미 다른 구슬이 놓여져 있는 것이므로
    # 충돌 표시를 해줍니다.
    else:
        next_dir[x][y] = COLLIDE


def move(x, y, move_dir):
    # 구슬이 벽에 부딪혔을 때의 처리를 간단히 하기 위해
    # dir 기준 0, 3이 대칭 1, 2가 대칭이 되도록 설정합니다.
    dxs, dys = [-1, 0, 0, 1], [0, 1, -1, 0]
    
    # 바로 앞에 벽이 있는지를 판단합니다.
    nx, ny = x + dxs[move_dir], y + dys[move_dir]
    
    # Case 1 : 벽이 없는 경우에는 그대로 한 칸 전진합니다.
    # 따라서 그 다음 위치에 같은 방향을 갖는 구슬이 있게 됩니다.
    if in_range(nx, ny):
        update_next_dir(nx, ny, move_dir)
        
    # Case 2 : 벽이 있는 경우에는 방향을 반대로 틀어줍니다.
    # 따라서 같은 위치에 반대 방향을 갖는 구슬이 있게 됩니다.
    else:
        update_next_dir(x, y, 3 - move_dir)   


# 구슬을 전부 한 번씩 움직여봅니다.
def move_all():
    global next_dir
    
    # 그 다음 각 위치에서의 방향들을 전부 초기화 해놓습니다.
    next_dir = [
        [BLANK for _ in range(n + 1)]
        for _ in range(n + 1)
    ]
    
    # (i, j) 위치에 구슬이 있는경우
    # 움직임을 시도해보고, 그 결과를 전부 next_dir에 기록합니다.
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if curr_dir[i][j] != BLANK:
                move(i, j, curr_dir[i][j])
    
    # next_dir 값을 다시 curr_dir에 복사합니다.
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            curr_dir[i][j] = next_dir[i][j]


# 충돌이 일어나는 구슬을 전부 지워줍니다.
def remove_duplicate_marbles():
    # 충돌이 일어난 구슬들이 있는 위치만 빈 곳으로 설정하면 됩니다.
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if curr_dir[i][j] == COLLIDE:
                curr_dir[i][j] = BLANK


# 조건에 맞춰 시뮬레이션을 진행합니다.
def simulate():
    # Step1
    # 구슬을 전부 한 번씩 움직여봅니다.
    move_all()
    
    # Step2
    # 움직임 이후에 충돌이 일어나는 구슬들을 골라 목록에서 지워줍니다.
    remove_duplicate_marbles()


for _ in range(t):
    # 입력
    n, m = tuple(map(int, input().split()))
    
    # 새로운 테스트 케이스가 시작될때마다 기존에 사용하던 값들을 초기화해줍니다.
    curr_dir = [
        [BLANK for _ in range(n + 1)]
        for _ in range(n + 1)
    ]
    
    for _ in range(m):
        x, y, d = tuple(input().split())
        x, y = int(x), int(y)
        curr_dir[x][y] = mapper[d]
    
    # 2 * n번 이후에는 충돌이 절대 일어날 수 없으므로
    # 시뮬레이션을 총 2 * n번 진행합니다.
    for _ in range(2 * n):
        simulate()
        
    marble_cnt = sum([
        curr_dir[i][j] != BLANK
        for i in range(1, n + 1)
        for j in range(1, n + 1)
    ])
    
    # 출력
    print(marble_cnt)


    OUT_OF_GRID = (-1, -1)

# 변수 선언 및 입력:
n, m = tuple(map(int, input().split()))
grid = [
    [[] for _ in range(n)]
    for _ in range(n)
]


def get_pos(move_num):
    for i in range(n):
        for j in range(n):
            for num in grid[i][j]:
                if num == move_num:
                    return (i, j)


def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < n


# 그 다음 위치를 찾아 반환합니다.
def next_pos(pos):
    dxs = [-1, -1, -1,  0, 0,  1, 1, 1]
    dys = [-1,  0,  1, -1, 1, -1, 0, 1]
    
    x, y = pos
    
    # 인접한 8개의 칸 중 가장 값이 큰 위치를 찾아 반환합니다.
    max_val, max_pos = -1, OUT_OF_GRID
    for dx, dy in zip(dxs, dys):
        nx, ny = x + dx, y + dy
        if in_range(nx, ny):
            for num in grid[nx][ny]:
                if num > max_val:
                    max_val, max_pos = num, (nx, ny)
    
    return max_pos


def move(pos, next_pos, move_num):
    (x, y), (nx, ny) = pos, next_pos
    
    # Step1. (x, y) 위치에 있던 숫자들 중
    # move_num 위에 있는 숫자들을 전부 옆 위치로 옮겨줍니다.
    to_move = False
    for num in grid[x][y]:
        if num == move_num:
            to_move = True
        
        if to_move:
            grid[nx][ny].append(num)
    
    # Step2. (x, y) 위치에 있던 숫자들 중
    # 움직인 숫자들을 전부 비워줍니다.
    while grid[x][y][-1] != move_num:
        grid[x][y].pop()
    grid[x][y].pop()


def simulate(move_num):
    # 그 다음으로 나아가야할 위치를 구해
    # 해당 위치로 숫자들을 옮겨줍니다.
    pos = get_pos(move_num)
    max_pos = next_pos(pos)
    if max_pos != OUT_OF_GRID:
        move(pos, max_pos, move_num)


for i in range(n):
    given_row = list(map(int, input().split()))
    for j, num in enumerate(given_row):
        grid[i][j].append(num)

# m번 시뮬레이션을 진행합니다.
move_nums = list(map(int, input().split()))
for move_num in move_nums:
    simulate(move_num)

for i in range(n):
    for j in range(n):
        if not grid[i][j]:
            print("None", end="")
        else:
            for num in grid[i][j][::-1]:
                print(num, end=" ")
        print()


        # 변수 선언 및 입력:
n, m, t, k = tuple(map(int, input().split()))
grid = [
    [[] for _ in range(n)]
    for _ in range(n)
]
next_grid = [
    [[] for _ in range(n)]
    for _ in range(n)
]


def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < n


def next_pos(x, y, vnum, move_dir):
    dxs, dys = [-1, 0, 0, 1], [0, 1, -1, 0]
    
    # vnum 횟수만큼 이동한 이후의 위치를 반환합니다.
    for _ in range(vnum):
        nx, ny = x + dxs[move_dir], y + dys[move_dir]
        # 벽에 부딪히면
        # 방향을 바꾼 뒤 이동합니다.
        if not in_range(nx, ny):
            move_dir = 3 - move_dir
            nx, ny = x + dxs[move_dir], y + dys[move_dir]
        x, y = nx, ny

    return (x, y, move_dir)


def move_all():
    for x in range(n):
        for y in range(n):
            for v, num, move_dir in grid[x][y]:
                next_x, next_y, next_dir = next_pos(x, y, v, move_dir)
                next_grid[next_x][next_y].append((v, num, next_dir))


def select_marbles():
    for i in range(n):
        for j in range(n):
            if len(next_grid[i][j]) >= k:
                # 우선순위가 높은 k개만 남겨줍니다.
                next_grid[i][j].sort(lambda x: (-x[0], -x[1]))
                while len(next_grid[i][j]) > k:
                    next_grid[i][j].pop()
            

def simulate():
    # Step1. next_grid를 초기화합니다.
    for i in range(n):
        for j in range(n):
            next_grid[i][j] = []
		
    # Step2. 구슬들을 전부 움직입니다.
    move_all()
    
    # Step3. 각 칸마다 구슬이 최대 k개만 있도록 조정합니다.
    select_marbles()
    
    # Step4. next_grid 값을 grid로 옮겨줍니다.
    for i in range(n):
        for j in range(n):
            grid[i][j] = next_grid[i][j]


dir_mapper = {
    "U": 0,
    "R": 1,
    "L": 2,
    "D": 3
}

for i in range(m):
    r, c, d, v = tuple(input().split())
    r, c, v = tuple(map(int, [r, c, v]))

    # 살아남는 구슬의 우선순위가 더 빠른 속도, 더 큰 번호 이므로
    # (속도, 방향, 번호) 순서를 유지합니다.
    grid[r - 1][c - 1].append((v, i + 1, dir_mapper[d]))

# t초에 걸쳐 시뮬레이션을 반복합니다.
for _ in range(t):
    simulate()

ans = sum([
    len(grid[i][j])
    for i in range(n)
    for j in range(n)
])
print(ans)


COORD_SIZE = 4000

# 변수 선언 및 입력
t = int(input())
n = 0
marbles = []
next_marbles = []

curr_time = 0
last_collision_time = -1

mapper = {
    'U': 0,
    'R': 1,
    'L': 2,
    'D': 3
}


# 해당 구슬이 1초 후에 어느 위치에 있는지를 구해 상태를 반환합니다.
def move(marble):
    # 구슬이 벽에 부딪혔을 때의 처리를 간단히 하기 위해
    # dir 기준 0, 3이 대칭 1, 2가 대칭이 되도록 설정합니다.
    dxs, dys = [0, 1, -1, 0], [1, 0, 0, -1]
    
    x, y, weight, move_dir, num = marble
    nx, ny = x + dxs[move_dir], y + dys[move_dir]
    
    return (nx, ny, weight, move_dir, num)


# 해당 구슬과 충돌이 일어나는 구슬이 있는지 확인합니다.
# 있다면 해당 구슬의 index를 반환하고, 없다면 -1을 반환합니다.
def find_duplicate_marble(marble):
    target_x, target_y, _, _, _ = marble
    
    for i, (mx, my, _, _, _) in enumerate(next_marbles):
        if (target_x, target_y) == (mx, my):
            return i
    
    return -1

    
# 두 구슬이 같은 위치에서 충돌했을 경우
# 살아남는 구슬을 반환합니다.
def collide(marble1, marble2):
    _, _, weight1, _, num1 = marble1
    _, _, weight2, _, num2 = marble2
    
    # 첫 번째 구슬을 따라가게 되는 경우는
    # 첫 번째 구슬의 무게가 더 크거나
    # 무게는 같은데 번호가 더 클 경우 입니다.
    if weight1 > weight2 or (weight1 == weight2 and num1 > num2):
        return marble1
    else:
        return marble2


# 그 다음 구슬의 목록에 반영합니다.
def push_next_marble(marble):
    global last_collision_time
    
    index = find_duplicate_marble(marble)
    
    # Case1 : 같은 위치에 있는 구슬이 앚기 없다면 그대로 목록에 추가합니다.
    if index == -1:
        next_marbles.append(marble)
    
    # Case2 :
    # 다음 구슬의 목록 중 같은 위치에 구슬이 이미 있다면
    # 더 영향력 있는 구슬만 남기고
    # 현재 시간을 가장 최근 충돌 시간에 기록합니다.
    else:
        next_marbles[index] = collide(next_marbles[index], marble)
        last_collision_time = curr_time
    

# 모든 구슬들을 한 칸씩 움직이는 시뮬레이션을 진행합니다.
def simulate():
    global marbles, next_marbles
    
    for marble in marbles:
        # Step1 : 각 구슬에 대해 한 칸 움직인 이후의 위치를 받아옵니다.
        next_marble = move(marble)
        
        # Step2 : 그 다음 구슬의 목록에 반영합니다.
        push_next_marble(next_marble)
    
    marbles = next_marbles[:]
    
    # 그 다음 simulation 때 다시 사용해야 하므로
    # 구슬의 목록을 미리 초기화 해줍니다.
    next_marbles = []


for _ in range(t):
    # 새로운 테스트 케이스가 시작될때마다 기존에 사용하던 값들을 초기화해줍니다.
    marbles = []
    last_collision_time = -1
    
    # 입력
    n = int(input())
    for i in range(1, n + 1):
        x, y, weight, d = tuple(input().split())
        x, y, weight = int(x), int(y), int(weight)
        
        # 구슬이 움직이는 도중에 충돌하는 문제를 깔끔하게 처리하기 위해
        # 좌표를 2배로 불려 1초에 한칸 씩 이동하는 문제로 바꿉니다.
        # 이렇게 문제가 바뀌면 따로 구슬이 움직이는 도중 충돌하는 경우를 생각하지
        # 않아도 됩니다.
        x, y = x * 2, y * 2
        marbles.append((x, y, weight, mapper[d], i))
    
    # 처음에 구슬들은 전부
    # (-2000, -2000)에서 (2000, 2000) 사이에 있기 때문에
    # COORD SIZE + 1 (4001)만큼 이동하면
    # 입력으로 주어진 구슬들이 모두 (-2000, -2000) ~ (2000, 2000)
    # 영역 밖으로 벗어나게 되므로 더 이상 충돌이 일어나지 않게 됩니다.
    # 따라서 시뮬레이션을 총 COORD_SIZE번 진행합니다.
    for i in range(1, COORD_SIZE + 1):
        curr_time = i
        simulate()
    
    # 출력
    print(last_collision_time)


    COORD_SIZE = 4000
OFFSET = 2000
BLANK = -1

# 변수 선언 및 입력
t = int(input())
n = 0
marbles = []
next_marbles = []

# 처음에는 구슬이 전혀 놓여있지 않다는 표시로 전부 BLANK로 채워 놓습니다.
next_marble_index = [
    [BLANK for _ in range(COORD_SIZE + 1)]
    for _ in range(COORD_SIZE + 1)
]

curr_time = 0
last_collision_time = -1

mapper = {
    'U': 0,
    'R': 1,
    'L': 2,
    'D': 3
}


# 해당 구슬이 1초 후에 어느 위치에 있는지를 구해 상태를 반환합니다.
def move(marble):
    # 구슬이 벽에 부딪혔을 때의 처리를 간단히 하기 위해
    # dir 기준 0, 3이 대칭 1, 2가 대칭이 되도록 설정합니다.
    dxs, dys = [0, 1, -1, 0], [1, 0, 0, -1]
    
    x, y, weight, move_dir, num = marble
    nx, ny = x + dxs[move_dir], y + dys[move_dir]
    
    return (nx, ny, weight, move_dir, num)


# 해당 구슬과 충돌이 일어나는 구슬이 있는지 확인합니다.
# 있다면 해당 구슬의 index를 반환하고, 없다면 BLANK를 반환합니다.
# 이는 next_marble_index를 활용하면 O(1)에 바로 가능합니다.
def find_duplicate_marble(marble):
    target_x, target_y, _, _, _ = marble
    
    return next_marble_index[target_x][target_y]

    
# 두 구슬이 같은 위치에서 충돌했을 경우
# 살아남는 구슬을 반환합니다.
def collide(marble1, marble2):
    _, _, weight1, _, num1 = marble1
    _, _, weight2, _, num2 = marble2
    
    # 첫 번째 구슬을 따라가게 되는 경우는
    # 첫 번째 구슬의 무게가 더 크거나
    # 무게는 같은데 번호가 더 클 경우 입니다.
    if weight1 > weight2 or (weight1 == weight2 and num1 > num2):
        return marble1
    else:
        return marble2


# 구슬이 이미 (0, 0) ~ (COORD_SIZE, COORD_SIZE) 사이를 벗어났다면
# 더 이상 충돌이 일어나지 않으므로
# Active Coordinate를 벗어났다고 판단합니다.
def out_of_active_coordinate(marble):
    x, y, _, _, _ = marble
    return x < 0 or x > COORD_SIZE or y < 0 or y > COORD_SIZE


# 그 다음 구슬의 목록에 반영합니다.
def push_next_marble(marble):
    global last_collision_time
    
    # 구슬이 이미 (0, 0) ~ (COORD_SIZE, COORD_SIZE) 사이를 벗어났다면
    # 그 이후부터는 절대 충돌이 일어나지 않으므로
    # 그 구슬은 앞으로 더 이상 관찰하지 않습니다. 
    if out_of_active_coordinate(marble):
        return
    
    index = find_duplicate_marble(marble)
    
    # Case1 : 같은 위치에 있는 구슬이 앚기 없다면 그대로 목록에 추가합니다.
    if index == BLANK:
        next_marbles.append(marble)
        
        # 나중에 위치가 겹치는 구슬이 있는지,
        # 만약 있다면 그 구슬이 next_marbles의 어느 index에 있는지를
        # 상수 시간안에 판단하기 위해 해당 위치에
        # 새로 추가되는 구슬의 index를 적어놓습니다.
        x, y, _, _, _ = marble
        next_marble_index[x][y] = len(next_marbles) - 1
    
    # Case2 :
    # 다음 구슬의 목록 중 같은 위치에 구슬이 이미 있다면
    # 더 영향력 있는 구슬만 남기고
    # 현재 시간을 가장 최근 충돌 시간에 기록합니다.
    else:
        next_marbles[index] = collide(next_marbles[index], marble)
        last_collision_time = curr_time
    

# 모든 구슬들을 한 칸씩 움직이는 시뮬레이션을 진행합니다.
def simulate():
    global marbles, next_marbles
    
    for marble in marbles:
        # Step1 : 각 구슬에 대해 한 칸 움직인 이후의 위치를 받아옵니다.
        next_marble = move(marble)
        
        # Step2 : 그 다음 구슬의 목록에 반영합니다.
        push_next_marble(next_marble)
    
    marbles = next_marbles[:]
    
    # 그 다음 simulation 때 다시 사용해야 하므로
    # 충돌 여부를 빠르게 판단하기 위해 쓰였던 next_marble_index 배열과
    # 다음 구슬의 목록을 기록했던 next_marbles를 미리 초기화해줍니다.
    
    for x, y, _, _, _ in next_marbles:
        next_marble_index[x][y] = BLANK
    
    next_marbles = []


for _ in range(t):
    # 새로운 테스트 케이스가 시작될때마다 기존에 사용하던 값들을 초기화해줍니다.
    marbles = []
    last_collision_time = -1
    
    # 입력
    n = int(input())
    for i in range(1, n + 1):
        x, y, weight, d = tuple(input().split())
        x, y, weight = int(x), int(y), int(weight)
        
        # 구슬이 움직이는 도중에 충돌하는 문제를 깔끔하게 처리하기 위해
        # 좌표를 2배로 불려 1초에 한칸 씩 이동하는 문제로 바꿉니다.
        # 이렇게 문제가 바뀌면 따로 구슬이 움직이는 도중 충돌하는 경우를 생각하지
        # 않아도 됩니다.
        x, y = x * 2, y * 2
        
        # 좌표를 전부 양수로 만들어야 동일한 위치에서 충돌이 일어나는지를
        # 판단하는 데 사용할 next_marble_index 배열에
        # 각 구슬의 위치마다 자신의 index를 저장할 수 있으므로
        # 좌표를 전부 양수로 만듭니다.
        # 입력으로 들어올 수 있는 좌표값 중 가장 작은 값이 -2000 이므로
        # OFFSET을 2000으로 잡아 전부 더해줍니다.
        # 같은 OFFSET을 모든 구슬에 전부 더해주는 것은
        # 답에 전혀 영향을 미치지 않습니다.
        x += OFFSET; y += OFFSET;
        marbles.append((x, y, weight, mapper[d], i))
    
    # OFFSET이 더해진 구슬들의 처음 위치는 전부 
    # (0, 0)에서 (4000, 4000) 사이에 있기 때문에 
    # COORD SIZE + 1(4001)만큼 이동하면
    # 입력으로 주어진 구슬들이 모두 (0, 0) ~ (4000, 4000)
    # 영역 밖으로 벗어나게 되므로 더 이상 충돌이 일어나지 않게 됩니다.
    # 따라서 시뮬레이션을 총 COORD_SIZE번 진행합니다.
    for i in range(1, COORD_SIZE + 1):
        curr_time = i
        simulate()
    
    # 출력
    print(last_collision_time)


    # 변수 선언 및 입력
t = int(input())
n = 0
marbles = []
collisions = []
disappear = []

last_collision_time = -1

# 3 - move_dir 방향이 move_dir 방향과 정 반대가 되도록
# move_dir에 따른 dx, dy 값을 적절하게 정의합니다.
# 후에 두 구슬의 방향이 서로 정 반대인지 쉽게 판단하기 위함입니다. 
mapper = {
    'U': 0,
    'R': 1,
    'L': 2,
    'D': 3
}


# 구슬을 무게를 내림차순으로 정렬합니다.
# 무게가 동일할 경우 숫자를 내림차순으로 정렬하여
# 정렬 이후 더 앞선 구슬들이
# 충돌시에 항상 더 영향력을 가질 수 있도록 합니다.
def cmp(marble):
    (_, _, weight, _, num) = marble
    return (-weight, -num)


# 해당 구슬의 k초 후의 위치를 계산하여 반환합니다.
def move(marble, k):
    dxs, dys = [0, 1, -1, 0], [1, 0, 0, -1]
    
    (x, y, _, move_dir, _) = marble
    nx, ny = x + dxs[move_dir] * k, y + dys[move_dir] * k
    return (nx, ny)


# 두 구슬만 좌표 평면 위에 존재한다 했을 때
# 충돌이 일어난다면 언제 일어나는지 그 시간을 반환합니다.
# 만약 충돌이 일어나지 않는다면 -1을 반환합니다.
def collision_occur_time(marble1, marble2):
    x1, y1, _, dir1, _ = marble1
    x2, y2, _, dir2, _ = marble2

    # Case1 : 두 구슬의 방향이 같은 경우에는 절대 충돌하지 않습니다.
    if dir1 == dir2:
        return -1

    # Case2 : 두 구슬의 방향이 반대인 경우에는 
    #         x, y 값 중 하나가 일치해야 하고
    #         두 구슬의 거리를 반으로 나눈 값 만큼
    #         두 구슬을 각각의 방향으로 움직였을 때 
    #         서로 같은 위치로 도달해야 충돌한다고 볼 수 있습니다. 
    if dir1 == 3 - dir2:
        # x, y 둘 다 일치하지 않으면 불가합니다.
        if x1 != x2 and y1 != y2:
            return -1
        
        # x, y 둘 중에 하나가 일치한다면 
        # 처음에 모든 좌표를 다 2배씩 해줬기 때문에 
        # dist는 짝수임을 보장할 수 있습니다. 
        dist = abs(x1 - x2) if x1 != x2 else abs(y1 - y2)
        half = dist // 2
        
        if move(marble1, half) == move(marble2, half):
            return half
        else:
            return -1

    # Case3 : 두 방향이 서로 나란히 있지 않은 경우에는
    #         두 구슬의 x좌표, y좌표의 차이가 정확히 일치해야 하며
    #         두 구슬의 각각의 방향으로 그 거리의 차이 만큼씩 움직였을 때
    #         서로 같은 위치로 도달해야 충돌한다고 볼 수 있습니다. 

    x_dist, y_dist = abs(x1 - x2), abs(y1 - y2)
    
    if x_dist == y_dist and move(marble1, x_dist) == move(marble2, x_dist):
        return x_dist
    else:
        return -1


# 모든 구슬쌍에 대해 충돌이 일어나는지 확인하고
# 발생 가능한 충돌들에 대해 시간순으로 정렬해줍니다.
def arrange_collisions():
    marble_cnt = len(marbles)
    for i in range(n):
        for j in range(i + 1, n):
            time = collision_occur_time(marbles[i], marbles[j])
            if time != -1:
                collisions.append((time, i, j))
    
    # tuple은 기본적으로 앞의 원소부터 오름차순으로 정렬하므로
    # 다음과 같이 정렬시 시간순으로 오름차순으로 정렬됨을 보장할 수 있습니다.
    collisions.sort()


# 시간에 따라 충돌 단위로 시뮬레이션을 진행합니다.
def simulate():
    global last_collision_time
    
    for collision_time, index1, index2 in collisions:
        # 두 구슬 중 하나라도 이미 이전의 충돌로 인해 소멸되어 버렸다면
        # 두 구슬은 실제로 충돌이 일어날 수 없었다는 의미이므로
        # 패스합니다.
        if disappear[index1] or disappear[index2]:
            continue
        
        # 처음에 구슬의 목록을 (무게 순, 번호가 더 큰 순)으로
        # 정렬해놨기 때문에 index1 < index2인 경우 
        # 항상 index1이 더 영향력이 크기 때문에 살아남게 되고
        # index2는 소멸하게 됩니다.
        disappear[index2] = True
        last_collision_time = collision_time


for _ in range(t):
    # 입력
    n = int(input())
    
    # 새로운 테스트 케이스가 시작될때마다 기존에 사용하던 값들을 초기화해줍니다.
    marbles = []
    collisions = []
    last_collision_time = -1
    
    disappear = [
        0 for _ in range(n + 1)
    ]
    
    for i in range(1, n + 1):
        x, y, weight, d = tuple(input().split())
        x, y, weight = int(x), int(y), int(weight)
        
        # 구슬이 움직이는 도중에 충돌하는 문제를 깔끔하게 처리하기 위해
        # 좌표를 2배로 불려 1초에 한칸 씩 이동하는 문제로 바꿉니다.
        # 이렇게 문제가 바뀌면 따로 구슬이 움직이는 도중 충돌하는 경우를 생각하지
        # 않아도 됩니다.
        x, y = x * 2, y * 2
        
        marbles.append((x, y, weight, mapper[d], i))
    
    # 충돌시 영향력이 더 높은 구슬이 앞으로 오도록 정렬합니다.
    # 영향력이 더 높다 함은 무게가 더 크거나, 무게가 같더라도 번호가 더 커
    # 충돌시 살아남게 되는 구슬을 의미합니다.
    marbles.sort(key = cmp)
    
    # 모든 구슬쌍에 대해 충돌이 일어나는 경우를 구해
    # 시간순으로 정리해줍니다.
    arrange_collisions()
    
    # 시간에 따라 충돌 단위로 시뮬레이션을 진행합니다.
    simulate()
    
    # 출력
    print(last_collision_time)



    # 변수 선언 및 입력
k, n = tuple(map(int, input().split()))
selected_nums = []


# 선택된 원소들을 출력해줍니다.
def print_permutation():
    for num in selected_nums:
        print(num, end = " ")
    print()


def find_duplicated_permutations(cnt):
    # n개를 모두 뽑은 경우 답을 출력해줍니다.
    if cnt == n:
        print_permutation()
        return
    
    # 1부터 k까지의 각 숫자가 뽑혔을 때의 경우를 탐색합니다.
    for i in range(1, k + 1):
        selected_nums.append(i)
        find_duplicated_permutations(cnt + 1)
        selected_nums.pop()


find_duplicated_permutations(0)



# 변수 선언 및 입력:
n = int(input())
ans = 0
seq = list()


def is_beautiful():
    # 연달아 같은 숫자가 나오는 시작 위치를 잡습니다.
    i = 0
    while i < n:
        # 만약 연속하여 해당 숫자만큼 나올 수 없다면
        # 아름다운 수가 아닙니다.
        if i + seq[i] - 1 >= n:
            return False
        # 연속하여 해당 숫자만큼 같은 숫자가 있는지 확인합니다.
        # 하나라도 다른 숫자가 있다면
        # 아름다운 수가 아닙니다.
        for j in range(i, i + seq[i]):
            if seq[j] != seq[i]:
                return False
            
        i += seq[i]
        
    return True


def count_beautiful_seq(cnt):
    global ans
    
    if cnt == n:
        if is_beautiful():
            ans += 1
        return
	
    for i in range(1, 5):
        seq.append(i)
        count_beautiful_seq(cnt + 1)
        seq.pop()


count_beautiful_seq(0)
print(ans)



# 변수 선언 및 입력:
n = int(input())
bomb_type = [
    [0 for _ in range(n)]
    for _ in range(n)
]
bombed = [
    [False for _ in range(n)]
    for _ in range(n)
]

ans = 0

bomb_pos = list()


def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < n


def bomb(x, y, b_type):
    # 폭탄 종류마다 터질 위치를 미리 정의합니다.
    bomb_shapes = [
        [],
        [[-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0]],
        [[-1, 0], [1, 0], [0, 0], [0, -1], [0, 1]],
        [[-1, -1], [-1, 1], [0, 0], [1, -1], [1, 1]]
    ]
    
    # 격자 내 칸에 대해서만 영역을 표시합니다.
    for i in range(5):
        dx, dy = bomb_shapes[b_type][i];
        nx, ny = x + dx, y + dy
        if in_range(nx, ny):
            bombed[nx][ny] = True

            
def calc():
    # Step1. 폭탄이 터진 위치를 표시하는 배열을
    # 초기화합니다.
    for i in range(n):
        for j in range(n):
            bombed[i][j] = False
            
    # Step2. 각 폭탄의 타입에 따라 
    # 초토화 되는 영역을 표시합니다.
    for i in range(n):
        for j in range(n):
            if bomb_type[i][j]:
                bomb(i, j, bomb_type[i][j])
	
    # Step3. 초토화된 영역의 수를 구합니다.
    cnt = 0
    for i in range(n):
        for j in range(n):
            if bombed[i][j]:
                cnt += 1
    
    return cnt


def find_max_area(cnt):
    global ans
    
    if cnt == len(bomb_pos):
        ans = max(ans, calc())
        return

    for i in range(1, 4):
        x, y = bomb_pos[cnt]
        
        bomb_type[x][y] = i
        find_max_area(cnt + 1)
        bomb_type[x][y] = 0


for i in range(n):
    given_row = list(map(int, input().split()))
    for j, bomb_place in enumerate(given_row):
        if bomb_place:
            bomb_pos.append((i, j))

find_max_area(0)

print(ans)


# 변수 선언 및 입력:

n = int(input())
segments = [
    tuple(map(int, input().split()))
    for _ in range(n)
]

ans = 0
selected_segs = list()


def overlapped(seg1, seg2):
    (ax1, ax2), (bx1, bx2) = seg1, seg2

    # 두 선분이 겹치는지 여부는
    # 한 점이 다른 선분에 포함되는 경우로 판단 가능합니다. 
    return (ax1 <= bx1 and bx1 <= ax2) or (ax1 <= bx2 and bx2 <= ax2) or \
           (bx1 <= ax1 and ax1 <= bx2) or (bx1 <= ax2 and ax2 <= bx2)


def possible():
    # 단 한쌍이라도 선분끼리 겹치면 안됩니다
    for i, seg1 in enumerate(selected_segs):
        for j, seg2 in enumerate(selected_segs):
            if i < j and overlapped(seg1, seg2):
                return False

    return True


def find_max_segments(cnt):
    global ans
    
    if cnt == n:
        if possible():
            ans = max(ans, len(selected_segs))
        return
    
    selected_segs.append(segments[cnt])
    find_max_segments(cnt + 1)
    selected_segs.pop()
    
    find_max_segments(cnt + 1)


find_max_segments(0)
print(ans)


# 변수 선언 및 입력:
n, m = tuple(map(int, input().split()))

lines = list()
selected_lines = list()

ans = m

# 처음 상황과, 선택한 가로줄만 사용했을 때의
# 상황을 시뮬레이션하여
# 둘의 결과가 같은지 확인합니다.
def possible():
    # Step1. 시작 숫자를 셋팅합니다.
    num1, num2 = [i for i in range(n)], [i for i in range(n)]
	
    # Step2. 위에서부터 순서대로 적혀있는 
    # 가로줄에 대해 양쪽 번호에 해당하는 숫자를 바꿔줍니다. 
    for _, idx in lines:
        num1[idx], num1[idx + 1] = num1[idx + 1], num1[idx]
    for _, idx in selected_lines:
        num2[idx], num2[idx + 1] = num2[idx + 1], num2[idx]
	
    # Step3. 두 상황의 결과가 동일한지 확인합니다.
    for i in range(n):
        if num1[i] != num2[i]:
            return False

    return True


def find_min_lines(cnt):
    global ans
    
    if cnt == m:
        if possible():
            ans = min(ans, len(selected_lines))
        return
    
    selected_lines.append(lines[cnt])
    find_min_lines(cnt + 1)
    selected_lines.pop()
	
    find_min_lines(cnt + 1)


for _ in range(m):
    a, b = tuple(map(int, input().split()))
    lines.append((b, a - 1))

lines.sort()

find_min_lines(0);
print(ans)


# 변수 선언 및 입력
n, m, c = tuple(map(int, input().split()))
weight = [
    list(map(int, input().split()))
    for _ in range(n)
]
a = list()
max_val = 0


def find_max_sum(curr_idx, curr_weight, curr_val):
    global max_val
    
    if curr_idx == m:
        # 고른 무게들의 합이 c를 넘지 않는 경우에만 갱신합니다.
        if curr_weight <= c:
            max_val = max(max_val, curr_val)
        return
    
    # curr_idx index에 있는 숫자를 선택하지 않은 경우
    find_max_sum(curr_idx + 1, curr_weight, curr_val)
    
    # curr_idx index에 있는 숫자를 선택한 경우
    # 무게는 a[curr_idx] 만큼 늘지만
    # 문제 정의에 의해 가치는 a[curr_idx] * a[curr_idx] 만큼 늘어납니다.
    find_max_sum(curr_idx + 1, curr_weight + a[curr_idx],
                 curr_val + a[curr_idx] * a[curr_idx])


# (sx, sy) ~ (sx, sy + m - 1) 까지의 숫자들 중 적절하게 골라
# 무게의 합이 c를 넘지 않게 하면서 얻을 수 있는 최대 가치를 반환합니다.
def find_max(sx, sy):
    global a, max_val
    
    # 문제를 a[0] ~ a[m - 1]까지 m개의 숫자가 주어졌을 때
    # 적절하게 골라 무게의 합이 c를 넘지 않게 하면서 얻을 수 있는 최대 가치를
    # 구하는 문제로 바꾸기 위해
    # a 배열을 적절하게 채워넣습니다.
    a = weight[sx][sy:sy + m]
    
    # 2^m개의 조합에 대해 최적의 값을 구합니다.
    max_val = 0
    find_max_sum(0, 0, 0)
    return max_val
    

# [a, b], [c, d] 이 두 선분이 겹치는지 판단합니다.
def intersect(a, b, c, d):
    # 겹치지 않을 경우를 계산하여 그 결과를 반전시켜 반환합니다.
    return not (b < c or d < a)


# 두 도둑의 위치가 올바른지 판단합니다.
def possible(sx1, sy1, sx2, sy2):
    # 두 도둑이 훔치려는 물건의 범위가
    # 격자를 벗어나는 경우에는 불가능합니다.
    if sy1 + m - 1 >= n or sy2 + m - 1 >= n:
        return False
    
    # 두 도둑이 훔칠 위치의 행이 다르다면
    # 겹칠 수가 없으므로 무조건 가능합니다.
    if sx1 != sx2:
        return True
    
    # 두 구간끼리 겹친다면
    # 불가능합니다.
    if intersect(sy1, sy1 + m - 1, sy2, sy2 + m - 1):
        return False
    
    # 행이 같으면서 구간끼리 겹치지 않으면
    # 가능합니다.
    return True


# 첫 번째 도둑은 (sx1, sy1) ~ (sx1, sy1 + m - 1) 까지 물건을 훔치려 하고
# 두 번째 도둑은 (sx2, sy2) ~ (sx2, sy2 + m - 1) 까지의 물건을
# 훔치려 한다고 했을 때 가능한 모든 위치를 탐색해봅니다.
ans = max([
    find_max(sx1, sy1) + find_max(sx2, sy2)
    for sx1 in range(n)
    for sy1 in range(n)
    for sx2 in range(n)
    for sy2 in range(n)
    if possible(sx1, sy1, sx2, sy2)
])
print(ans)


# 변수 선언 및 입력
n, m, c = tuple(map(int, input().split()))
weight = [
    list(map(int, input().split()))
    for _ in range(n)
]

# best_val[sx][sy] : (sx, sy) ~ (sx, sy + m - 1)까지 물건을
#                    잘 골라 얻을 수 있는 최대 가치를 preprocessing
#                    때 저장해놓을 배열입니다.
best_val = [
    [0 for _ in range(n)]
    for _ in range(n)
]

a = list()
max_val = 0


def find_max_sum(curr_idx, curr_weight, curr_val):
    global max_val
    
    if curr_idx == m:
        # 고른 무게들의 합이 c를 넘지 않는 경우에만 갱신합니다.
        if curr_weight <= c:
            max_val = max(max_val, curr_val)
        return
    
    # curr_idx index에 있는 숫자를 선택하지 않은 경우
    find_max_sum(curr_idx + 1, curr_weight, curr_val)
    
    # curr_idx index에 있는 숫자를 선택한 경우
    # 무게는 a[curr_idx] 만큼 늘지만
    # 문제 정의에 의해 가치는 a[curr_idx] * a[curr_idx] 만큼 늘어납니다.
    find_max_sum(curr_idx + 1, curr_weight + a[curr_idx],
                 curr_val + a[curr_idx] * a[curr_idx])


# (sx, sy) ~ (sx, sy + m - 1) 까지의 숫자들 중 적절하게 골라
# 무게의 합이 c를 넘지 않게 하면서 얻을 수 있는 최대 가치를 반환합니다.
def find_max(sx, sy):
    global a, max_val
    
    # 문제를 a[0] ~ a[m - 1]까지 m개의 숫자가 주어졌을 때
    # 적절하게 골라 무게의 합이 c를 넘지 않게 하면서 얻을 수 있는 최대 가치를
    # 구하는 문제로 바꾸기 위해
    # a 배열을 적절하게 채워넣습니다.
    a = weight[sx][sy:sy + m]
    
    # 2^m개의 조합에 대해 최적의 값을 구합니다.
    max_val = 0
    find_max_sum(0, 0, 0)
    return max_val
    

# [a, b], [c, d] 이 두 선분이 겹치는지 판단합니다.
def intersect(a, b, c, d):
    # 겹치지 않을 경우를 계산하여 그 결과를 반전시켜 반환합니다.
    return not (b < c or d < a)


# 두 도둑의 위치가 올바른지 판단합니다.
def possible(sx1, sy1, sx2, sy2):
    # 두 도둑이 훔치려는 물건의 범위가
    # 격자를 벗어나는 경우에는 불가능합니다.
    if sy1 + m - 1 >= n or sy2 + m - 1 >= n:
        return False
    
    # 두 도둑이 훔칠 위치의 행이 다르다면
    # 겹칠 수가 없으므로 무조건 가능합니다.
    if sx1 != sx2:
        return True
    
    # 두 구간끼리 겹친다면
    # 불가능합니다.
    if intersect(sy1, sy1 + m - 1, sy2, sy2 + m - 1):
        return False
    
    # 행이 같으면서 구간끼리 겹치지 않으면
    # 가능합니다.
    return True


# preprocessing 과정입니다.
# 미리 각각의 위치에 대해 최적의 가치를 구해 best_val 배열에 저장해놓습니다.
for sx in range(n):
    for sy in range(n):
        if sy + m - 1 < n:
            best_val[sx][sy] = find_max(sx, sy)


# 첫 번째 도둑은 (sx1, sy1) ~ (sx1, sy1 + m - 1) 까지 물건을 훔치려 하고
# 두 번째 도둑은 (sx2, sy2) ~ (sx2, sy2 + m - 1) 까지의 물건을
# 훔치려 한다고 했을 때 가능한 모든 위치를 탐색해봅니다.
ans = max([
    best_val[sx1][sy1] + best_val[sx2][sy2]
    for sx1 in range(n)
    for sy1 in range(n)
    for sx2 in range(n)
    for sy2 in range(n)
    if possible(sx1, sy1, sx2, sy2)
])
print(ans)


# 변수 선언 및 입력
n, m, c = tuple(map(int, input().split()))
weight = [
    list(map(int, input().split()))
    for _ in range(n)
]

# best_val[sx][sy] : (sx, sy) ~ (sx, sy + m - 1)까지 물건을
#                    잘 골라 얻을 수 있는 최대 가치를
#                    이미 계산한 적이 있다면 그 값을 적어놓고
#                    아직 계산해본 적이 없다면 -1이 들어있습니다.
best_val = [
    [-1 for _ in range(n)]
    for _ in range(n)
]

a = list()
max_val = 0


def find_max_sum(curr_idx, curr_weight, curr_val):
    global max_val
    
    if curr_idx == m:
        # 고른 무게들의 합이 c를 넘지 않는 경우에만 갱신합니다.
        if curr_weight <= c:
            max_val = max(max_val, curr_val)
        return
    
    # curr_idx index에 있는 숫자를 선택하지 않은 경우
    find_max_sum(curr_idx + 1, curr_weight, curr_val)
    
    # curr_idx index에 있는 숫자를 선택한 경우
    # 무게는 a[curr_idx] 만큼 늘지만
    # 문제 정의에 의해 가치는 a[curr_idx] * a[curr_idx] 만큼 늘어납니다.
    find_max_sum(curr_idx + 1, curr_weight + a[curr_idx],
                 curr_val + a[curr_idx] * a[curr_idx])


# (sx, sy) ~ (sx, sy + m - 1) 까지의 숫자들 중 적절하게 골라
# 무게의 합이 c를 넘지 않게 하면서 얻을 수 있는 최대 가치를 반환합니다.
def find_max(sx, sy):
    global a, max_val
    
    # 이미 (sx, sy) ~ (sx, sy + m - 1) 사이의 최적 조합을
    # 계산해본 적이 있다는 뜻이므로, 그 값을 바로 반환합니다.
    if best_val[sx][sy] != -1:
        return best_val[sx][sy]
    
    # 문제를 a[0] ~ a[m - 1]까지 m개의 숫자가 주어졌을 때
    # 적절하게 골라 무게의 합이 c를 넘지 않게 하면서 얻을 수 있는 최대 가치를
    # 구하는 문제로 바꾸기 위해
    # a 배열을 적절하게 채워넣습니다.
    a = weight[sx][sy:sy + m]
    
    # 2^m개의 조합에 대해 최적의 값을 구합니다.
    max_val = 0
    find_max_sum(0, 0, 0)
    
    # 나중에 또 (sx, sy) ~ (sx, sy + m - 1) 사이의 조합을
    # 계산하려는 시도가 있을 수 있으므로 best_val 배열에 caching 해놓습니다.
    best_val[sx][sy] = max_val
    return max_val
    

# [a, b], [c, d] 이 두 선분이 겹치는지 판단합니다.
def intersect(a, b, c, d):
    # 겹치지 않을 경우를 계산하여 그 결과를 반전시켜 반환합니다.
    return not (b < c or d < a)


# 두 도둑의 위치가 올바른지 판단합니다.
def possible(sx1, sy1, sx2, sy2):
    # 두 도둑이 훔치려는 물건의 범위가
    # 격자를 벗어나는 경우에는 불가능합니다.
    if sy1 + m - 1 >= n or sy2 + m - 1 >= n:
        return False
    
    # 두 도둑이 훔칠 위치의 행이 다르다면
    # 겹칠 수가 없으므로 무조건 가능합니다.
    if sx1 != sx2:
        return True
    
    # 두 구간끼리 겹친다면
    # 불가능합니다.
    if intersect(sy1, sy1 + m - 1, sy2, sy2 + m - 1):
        return False
    
    # 행이 같으면서 구간끼리 겹치지 않으면
    # 가능합니다.
    return True


# 첫 번째 도둑은 (sx1, sy1) ~ (sx1, sy1 + m - 1) 까지 물건을 훔치려 하고
# 두 번째 도둑은 (sx2, sy2) ~ (sx2, sy2 + m - 1) 까지의 물건을
# 훔치려 한다고 했을 때 가능한 모든 위치를 탐색해봅니다.
ans = max([
    find_max(sx1, sy1) + find_max(sx2, sy2)
    for sx1 in range(n)
    for sy1 in range(n)
    for sx2 in range(n)
    for sy2 in range(n)
    if possible(sx1, sy1, sx2, sy2)
])
print(ans)


# 변수 선언 및 입력
k, n = tuple(map(int, input().split()))
selected_nums = []


# 선택된 원소들을 출력해줍니다.
def print_permutation():
    for num in selected_nums:
        print(num, end = " ")
    print()


def find_duplicated_permutations(cnt):
    # n개를 모두 뽑은 경우 답을 출력해줍니다.
    if cnt == n:
        print_permutation()
        return
    
    for i in range(1, k + 1):
        if cnt >= 2 and i == selected_nums[-1] and \
                        i == selected_nums[-2]:
            continue
        else:
            selected_nums.append(i)
            find_duplicated_permutations(cnt + 1)
            selected_nums.pop()


find_duplicated_permutations(0)


# 변수 선언 및 입력:
n, m, k = tuple(map(int, input().split()))
nums = list(map(int, input().split()))
pieces = [1 for _ in range(k)]

ans = 0


# 점수를 계산합니다.
def calc():
    score = 0
    for piece in pieces:
        score += (piece >= m)
    
    return score


def find_max(cnt):
    global ans
    
    # 말을 직접 n번 움직이지 않아도
    # 최대가 될 수 있으므로 항상 답을 갱신합니다.
    ans = max(ans, calc())
    
    # 더 이상 움직일 수 없으면 종료합니다.
    if cnt == n: 
        return
	
    for i in range(k):
        # 움직여서 더 이득이 되지 않는
        # 말은 더 이상 움직이지 않습니다.
        if pieces[i] >= m:
            continue
        
        pieces[i] += nums[cnt]
        find_max(cnt + 1)
        pieces[i] -= nums[cnt]


find_max(0)
print(ans)


import sys

# 변수 선언 및 입력
n = int(input())
numbers = [4, 5, 6]

series = []

# 가능한 수열인지 여부를 판별합니다.
def is_possible_series():
    # 수열의 가장 앞부터 각 인덱스가 시작점일 때
    # 인접한 연속 부분 수열이 동일한 경우가 있는지를 탐색합니다.
    for idx in range(len(series)):
        # 가능한 연속 부분 수열의 길이 범위를 탐색합니다.
        length = 1
        while True:
            # 연속 부분 수열의 시작과 끝 인덱스를 설정하여 줍니다.
            start1, end1 = idx, idx + length - 1
            start2, end2 = end1 + 1, (end1 + 1) + length - 1
            
            # 두번째 연속 부분 수열의 끝 인덱스가 범위를 넘어가면 탐색을 종료합니다.
            if end2 >= len(series):
                break
            
            # 인접한 연속 부분 수열이 같은지 여부를 확인합니다.
            if series[start1:end1 + 1] == series[start2:end2 + 1]:
                return False
            
            length += 1
    
    return True


def find_min_series(cnt):
    # n개의 숫자가 선택됐을 때 불가능한 수열인 경우 탐색을 종료합니다.
    # 가능한 수열인 경우 이를 출력하고 프로그램을 종료합니다.
    if cnt == n:
        if not is_possible_series():
            return
        
        for elem in series:
            print(elem, end = "")
        sys.exit(0)
    
    # 사용 가능한 각 숫자가 뽑혔을 때의 경우를 탐색합니다.
    for number in numbers:
        series.append(number)
        find_min_series(cnt + 1)
        series.pop()
        

find_min_series(0)

import sys

# 변수 선언 및 입력
n = int(input())
numbers = [4, 5, 6]

series = []

# 가능한 수열인지 여부를 판별합니다.
def is_possible_series():
    # 가능한 연속 부분 수열의 길이 범위를 탐색합니다.
    length = 1
    while True:
        # 연속 부분 수열의 시작과 끝 인덱스를 설정하여 줍니다.
        start1, end1 = len(series) - length, len(series) - 1
        start2, end2 = start1 - length, start1 - 1

        if start2 < 0:
            break

        # 인접한 연속 부분 수열이 같은지 여부를 확인합니다.
        if series[start1:end1 + 1] == series[start2:end2 + 1]:
            return False

        length += 1
    
    return True


def find_min_series(cnt):
    # n개의 숫자가 선택됐을 때 불가능한 수열인 경우 탐색을 종료합니다.
    # 가능한 수열인 경우 이를 출력하고 프로그램을 종료합니다.
    if cnt == n:
        for elem in series:
            print(elem, end = "")
        sys.exit(0)
    
    # 사용 가능한 각 숫자가 뽑혔을 때의 경우를 탐색합니다.
    for number in numbers:
        series.append(number)
        # 해당 시점까지 만들 수열이 조건을 만족하는 경우에만
        # 탐색을 진행합니다.
        if is_possible_series():
            find_min_series(cnt + 1)
        series.pop()
        

find_min_series(0)



# 변수 선언 및 입력:
n = int(input())
num = [
    list(map(int, input().split()))
    for _ in range(n)
]
move_dir = [
    list(map(int, input().split()))
    for _ in range(n)
]

ans = 0


def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < n


def can_go(x, y, prev_num):
    return in_range(x, y) and num[x][y] > prev_num


def find_max(x, y, cnt):
    global ans
    
    # 언제 끝날지 모르기 때문에
    # 항상 최댓값을 갱신해줍니다.
    ans = max(ans, cnt)
    
    dxs = [-1, -1, 0, 1, 1, 1, 0, -1]
    dys = [0, 1, 1, 1, 0, -1, -1, -1]
    
    d = move_dir[x][y] - 1
    
    for i in range(n):
        nx, ny = x + dxs[d] * i, y + dys[d] * i
        if can_go(nx, ny, num[x][y]):
            find_max(nx, ny, cnt + 1)

            
r, c = tuple(map(int, input().split()))

find_max(r - 1, c - 1, 0)
print(ans)



# 변수 선언 및 입력

n, m = tuple(map(int, input().split()))
visited = [
    False for _ in range(n + 1)
]


# 방문한 원소들을 출력해줍니다.
def print_combination():
    for i in range(1, n + 1):
        if visited[i]:
            print(i, end = " ")
    
    print()


# 지금까지 뽑은 갯수와 마지막으로 뽑힌 숫자를 추적하여
# 그 다음에 뽑힐 수 있는 원소의 후보를 정합니다.
def find_combination(cnt, last_num):
    # m개를 모두 뽑은 경우 답을 출력해줍니다.
    if cnt == m:
        print_combination()
        return
    
    # 뽑을 수 있는 원소의 후보들을 탐색합니다.
    for i in range(last_num + 1, n + 1): 
        visited[i] = True;
        find_combination(cnt + 1, i);
        visited[i] = False;


# 가능한 범위를 순회하며 해당 숫자가 
# 조합의 첫번째 숫자일 때를 탐색합니다.
for i in range(1, n + 1):
    visited[i] = True
    find_combination(1, i)
    visited[i] = False


    # 변수 선언 및 입력

n, m = tuple(map(int, input().split()))
combination = []


# 방문한 원소들을 출력해줍니다.
def print_combination():
    for elem in combination:
        print(elem, end = " ")
    
    print()


def find_combination(curr_num, cnt):
    # n개의 숫자를 모두 탐색했으면 더 이상 탐색하지 않습니다.
    if curr_num == n + 1:
        # 탐색하는 과정에서 m개의 숫자를 뽑은 경우 답을 출력해줍니다.
        if cnt == m:
            print_combination()
        return

    # curr_num에 해당하는 숫자를 사용했을 때의 경우를 탐색합니다.
    combination.append(curr_num);
    find_combination(curr_num + 1, cnt + 1);
    combination.pop();

    # curr_num에 해당하는 숫자를 사용하지 않았을 때의 경우를 탐색합니다.
    find_combination(curr_num + 1, cnt);


find_combination(1, 0)



import functools

# 변수 선언 및 입력
n, m = tuple(map(int, input().split()))
a = list(map(int, input().split()))
visited = [
    False for _ in range(n)
]

ans = 0


def calc():
    selected_numbers = [
        a[i]
        for i in range(n)
        if visited[i]
    ]
    
    # xor 연산의 항등원인 0을 초기값으로 설정합니다.
    return functools.reduce(
        lambda acc, cur: acc ^ cur,
        selected_numbers,
        0
    )


def find_max_xor(curr_idx, cnt):
    global ans
    
    if cnt == m:
        # 선택된 모든 조합에 대해 xor 연산을 적용해봅니다.
        ans = max(ans, calc())
        return
    
    if curr_idx == n:
        return
    
    # curr_idx index에 있는 숫자를 선택하지 않은 경우
    find_max_xor(curr_idx + 1, cnt)
    
    # curr_idx index에 있는 숫자를 선택한 경우
    visited[curr_idx] = True
    find_max_xor(curr_idx + 1, cnt + 1)
    visited[curr_idx] = False


find_max_xor(0, 0)

print(ans)


import functools

# 변수 선언 및 입력
n, m = tuple(map(int, input().split()))
a = list(map(int, input().split()))

ans = 0


def find_max_xor(curr_idx, cnt, curr_val):
    global ans
    
    if cnt == m:
        ans = max(ans, curr_val)
        return
    
    if curr_idx == n:
        return
    
    # curr_idx index에 있는 숫자를 선택하지 않은 경우
    find_max_xor(curr_idx + 1, cnt, curr_val)
    
    # curr_idx index에 있는 숫자를 선택한 경우
    find_max_xor(curr_idx + 1, cnt + 1, curr_val ^ a[curr_idx])


find_max_xor(0, 0, 0)

print(ans)

import sys

COIN_NUM = 9
INT_MAX = sys.maxsize

# 변수 선언 및 입력:
n = int(input())
m = 3

grid = [
    input()
    for _ in range(n)
]

coin_pos = list()
selected_pos = list()

start_pos = (-1, -1)
end_pos = (-1, -1)

ans = INT_MAX


def dist(a, b):
    (ax, ay), (bx, by) = a, b
    return abs(ax - bx) + abs(ay - by)


def calc():
    num_moves = dist(start_pos, selected_pos[0])
    for i in range(m - 1):
        num_moves += dist(selected_pos[i], selected_pos[i + 1])
    num_moves += dist(selected_pos[m - 1], end_pos)
    
    return num_moves


def find_min_moves(curr_idx, cnt):
    global ans
    
    if cnt == m:
        # 선택된 모든 조합에 대해 이동 횟수를 계산합니다.
        ans = min(ans, calc())
        return
    
    if curr_idx == len(coin_pos):
        return
    
    # curr_idx index 에 있는 동전을 선택하지 않은 경우
    find_min_moves(curr_idx + 1, cnt)
    
    # curr_idx index 에 있는 동전을 선택한 경우
    selected_pos.append(coin_pos[curr_idx])
    find_min_moves(curr_idx + 1, cnt + 1)
    selected_pos.pop()

    
for i in range(n):
    for j in range(n):
        if grid[i][j] == 'S':
            start_pos = (i, j)
        if grid[i][j] == 'E':
            end_pos = (i, j)

# 동전을 오름차순으로 각 위치를 집어넣습니다.
# 이후에 증가하는 순서대로 방문하기 위함입니다.
for num in range(1, COIN_NUM + 1):
    for i in range(n):
        for j in range(n):
            if grid[i][j] == str(num):
                coin_pos.append((i, j))
                
find_min_moves(0, 0)

if ans == INT_MAX:
    ans = -1

print(ans)


import sys

INT_MAX = sys.maxsize

# 변수 선언 및 입력:

n = int(input())
num = list(map(int, input().split()))
visited = [False for _ in range(2 * n)]

ans = INT_MAX


def calc():
    diff = 0
    for i in range(2 * n):
        diff = (diff + num[i]) if visited[i] else diff - num[i]
    
    return abs(diff)


def find_min(idx, cnt):
    global ans
    
    if cnt == n:
        ans = min(ans, calc())
        return
    
    if idx == 2 * n:
        return
    
    # 현재 숫자를 첫 번째 그룹에 사용한 경우입니다.
    visited[idx] = True
    find_min(idx + 1, cnt + 1)
    visited[idx] = False
    
    # 현재 숫자를 두 번째 그룹에 사용한 경우입니다.
    find_min(idx + 1, cnt)
    

find_min(0, 0)
print(ans)


import sys

INT_MAX = sys.maxsize

# 변수 선언 및 입력:

n = int(input())
num = list(map(int, input().split()))

ans = INT_MAX


def find_min(idx, cnt, diff):
    global ans
    
    if idx == 2 * n:
        if cnt == n:
            ans = min(ans, abs(diff))
        return
    
    # 현재 숫자를 첫 번째 그룹에 사용한 경우입니다.
    find_min(idx + 1, cnt + 1, diff + num[idx])
    # 현재 숫자를 두 번째 그룹에 사용한 경우입니다.
    find_min(idx + 1, cnt, diff - num[idx])
    

find_min(0, 0, 0)
print(ans)


# 변수 선언 및 입력
n, m = tuple(map(int, input().split()))
grid = [
    list(map(int, input().split()))
    for _ in range(n)
]

visited = [
    [0 for _ in range(m)]
    for _ in range(n)
]

# 주어진 위치가 격자를 벗어나는지 여부를 반환합니다.
def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < m


# 주어진 위치로 이동할 수 있는지 여부를 확인합니다.
def can_go(x, y):
    if not in_range(x, y):
        return False
    
    if visited[x][y] or grid[x][y] == 0:
        return False
    
    return True


def dfs(x, y):
    dxs, dys = [0, 1], [1, 0]
    
    for dx, dy in zip(dxs, dys):
        new_x, new_y = x + dx, y + dy
        
        if can_go(new_x, new_y):
            visited[new_x][new_y] = 1
            dfs(new_x, new_y)
            
            
visited[0][0] = 1
dfs(0, 0)

print(visited[n - 1][m - 1])



# 변수 선언 및 입력
n, m = tuple(map(int, input().split()))
grid = [
    list(map(int, input().split()))
    for _ in range(n)
]

visited = [
    [0 for _ in range(m)]
    for _ in range(n)
]

# 주어진 위치가 격자를 벗어나는지 여부를 반환합니다.
def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < m


# 주어진 위치로 이동할 수 있는지 여부를 확인합니다.
def can_go(x, y):
    if not in_range(x, y):
        return False
    
    if visited[x][y] or grid[x][y] == 0:
        return False
    
    return True


def dfs(x, y):
    dxs, dys = [0, 1], [1, 0]
    
    # 탐색을 시작하기 전에 해당 위치를 방문했음을 표시해줍니다.
    visited[x][y] = 1
    
    for dx, dy in zip(dxs, dys):
        new_x, new_y = x + dx, y + dy
        
        if can_go(new_x, new_y):
            dfs(new_x, new_y)
            
            
dfs(0, 0)

print(visited[n - 1][m - 1])



# 변수 선언 및 입력
n, m = tuple(map(int, input().split()))

#index를 1번 부터 사용하기 위해 m+1만큼 할당합니다.
graph = [
    [0 for _ in range(n + 1)]
    for _ in range(n + 1)
]

visited = [False for _ in range(n + 1)]
vertex_cnt = 0

def dfs(vertex):
    global vertex_cnt
    global visited
    
    # 해당 정점에서 이어져있는 모든 정점을 탐색해줍니다.
    for curr_v in range(1, n + 1):
        # 아직 간선이 존재하고 방문한 적이 없는 정점에 대해서만 탐색을 진행합니다.
        if graph[vertex][curr_v] and not visited[curr_v]:
            visited[curr_v] = True
            vertex_cnt += 1
            dfs(curr_v)
    
for i in range(m):
    v1, v2 = tuple(map(int, input().split()))

    # 각 정점이 서로 이동이 가능한 양방향 그래프이기 때문에
    # 각 정점에 대한 간선을 각각 저장해줍니다.
    graph[v1][v2] = 1
    graph[v2][v1] = 1

visited[1] = True
dfs(1)

print(vertex_cnt)



# 변수 선언 및 입력
n, m = tuple(map(int, input().split()))

#index를 1번 부터 사용하기 위해 m+1만큼 할당합니다.
graph = [[] for _ in range(n + 1)]

visited = [False for _ in range(n + 1)]
vertex_cnt = 0

def dfs(vertex):
    global vertex_cnt
    global visited
    
    # 해당 정점에서 이어져있는 모든 정점을 탐색해줍니다.
    for i in range(len(graph[vertex])):
        curr_v = graph[vertex][i]
        # 아직 간선이 존재하고 방문한 적이 없는 정점에 대해서만 탐색을 진행합니다.
        if not visited[curr_v]:
            visited[curr_v] = True
            vertex_cnt += 1
            dfs(curr_v)
    
for i in range(m):
    v1, v2 = tuple(map(int, input().split()))

    # 각 정점이 서로 이동이 가능한 양방향 그래프이기 때문에
    # 각 정점에 대한 간선을 각각 저장해줍니다.
    graph[v1].append(v2)
    graph[v2].append(v1)

visited[1] = True
dfs(1)

print(vertex_cnt)


# 변수 선언 및 입력
n = int(input())
grid = [
    list(map(int, input().split()))
    for _ in range(n)
]

visited = [
    [False for _ in range(n)]
    for _ in range(n)
]

people_num = 0
people_nums = list()

# 주어진 위치가 격자를 벗어나는지 여부를 반환합니다.
def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < n


# 주어진 위치로 이동할 수 있는지 여부를 확인합니다.
def can_go(x, y):
    if not in_range(x, y):
        return False
    
    if visited[x][y] or grid[x][y] == 0:
        return False
    
    return True

def dfs(x, y):
    global people_num
    
    # 0: 오른쪽, 1: 아래쪽, 2: 왼쪽, 3: 위쪽
    dxs, dys = [0, 1, 0, -1], [1, 0, -1, 0]
    
    # 네 방향에 각각에 대하여 DFS 탐색을 합니다.
    for dx, dy in zip(dxs, dys):
        new_x, new_y = x + dx, y + dy
        
        if can_go(new_x, new_y):
            visited[new_x][new_y] = True
            
            #  마을에 존재하는 사람을 한 명 추가해줍니다.
            people_num += 1
            dfs(new_x, new_y)

# 격자의 각 위치에서 탐색을 시작할 수 있는 경우
# 한 마을에 대한 DFS 탐색을 수행합니다.
for i in range(n):
    for j in range(n):
        if can_go(i, j):
            # 해당 위치를 방문할 수 있는 경우 visited 배열을 갱신하고
            # 새로운 마을을 탐색한다는 의미로 people_num을 1으로 갱신합니다.
            visited[i][j] = True
            people_num = 1
            
            dfs(i, j)
            
            # 한 마을에 대한 탐색이 끝난 경우 마을 내의 사람 수를 저장합니다.
            people_nums.append(people_num)

# 각 마을 내 사람의 수를 오름차순으로 정렬합니다.
people_nums.sort()

print(len(people_nums))

for i in range(len(people_nums)):
    print(people_nums[i])


    import sys
sys.setrecursionlimit(2500)

# 변수 선언 및 입력
n, m = tuple(map(int, input().split()))
grid = [
    list(map(int, input().split()))
    for _ in range(n)
]

visited = [
    [False for _ in range(m)]
    for _ in range(n)
]

zone_num = 0


# visited 배열을 초기화해줍니다.
def initialize_visited():
    for i in range(n):
        for j in range(m):
            visited[i][j] = False


# 주어진 위치가 격자를 벗어나는지 여부를 반환합니다.
def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < m


# 주어진 위치로 이동할 수 있는지 여부를 확인합니다.
def can_go(x, y, k):
    if not in_range(x, y):
        return False
    
    if visited[x][y] or grid[x][y] <= k:
        return False
    
    return True


def dfs(x, y, k):
    # 0: 오른쪽, 1: 아래쪽, 2: 왼쪽, 3: 위쪽
    dxs, dys = [0, 1, 0, -1], [1, 0, -1, 0]
    
    # 네 방향에 각각에 대하여 DFS 탐색을 합니다.
    for dx, dy in zip(dxs, dys):
        new_x, new_y = x + dx, y + dy
        
        if can_go(new_x, new_y, k):
            visited[new_x][new_y] = True
            dfs(new_x, new_y, k)


def get_zone_num(k):
    global zone_num
    
    # 새로운 탐색을 시작한다는 의미로 zone_num를 0으로 갱신하고 
    # visited 배열을 초기화해줍니다.
    zone_num = 0
    initialize_visited()
    
    # 격자의 각 위치에 대하여 탐색을 시작할 수 있는 경우
    # 해당 위치로부터 시작한 DFS 탐색을 수행합니다.
    for i in range(n):
        for j in range(m):
            if can_go(i, j, k):
                # 해당 위치를 탐색할 수 있는 경우 visited 배열을 갱신하고
                # 안전 영역을 하나 추가해줍니다.
                visited[i][j] = True
                zone_num += 1
                
                dfs(i, j, k)


# 가능한 안전 영역의 최솟값이 0이므로 다음과 같이 초기화 해줄 수 있습니다.
max_zone_num = -1
answer_k = 0
max_height = 100

# 각 가능한 비의 높이에 대하여 안전 영역의 수를 탐색합니다.
for k in range(1, max_height +1):
    get_zone_num(k)
    
    # 기존의 최대 영역의 수보다 클 경우 이를 갱신하고 인덱스를 저장합니다.
    if zone_num > max_zone_num:
        max_zone_num, answer_k = zone_num, k

print(answer_k, max_zone_num)


from collections import deque

# 변수 선언 및 입력
n, m = tuple(map(int, input().split()))

a = [
    list(map(int, input().split()))
    for _ in range(n)
]

visited = [
    [False for _ in range(m)]
    for _ in range(n)
]

q = deque()

# 주어진 위치가 격자를 벗어나는지 여부를 반환합니다.
def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < m


# 주어진 위치로 이동할 수 있는지 여부를 확인합니다.
def can_go(x, y):
    return in_range(x, y) and a[x][y] and not visited[x][y]


def bfs():
    # queue에 남은 것이 없을때까지 반복합니다.
    while q:
        # queue에서 가장 먼저 들어온 원소를 뺍니다.
        x, y = q.popleft()
        
        # queue에서 뺀 원소의 위치를 기준으로 4방향을 확인해봅니다.
        dxs, dys = [0, 1, 0, -1], [1, 0, -1, 0]
        for dx, dy in zip(dxs, dys):
            new_x, new_y = x + dx, y + dy
        
            # 아직 방문한 적이 없으면서 갈 수 있는 곳이라면
            # 새로 queue에 넣어주고 방문 여부를 표시해줍니다.
            if can_go(new_x, new_y):
                q.append((new_x, new_y))
                visited[new_x][new_y] = True

                
# bfs를 이용해 최소 이동 횟수를 구합니다.
# 시작점을 queue에 넣고 시작합니다.
q.append((0, 0))
visited[0][0] = True

bfs()

# 우측 하단을 방문한 적이 있는지 여부를 출력합니다.
answer = 1 if visited[n - 1][m - 1] else 0
print(answer)


from collections import deque

# 변수 선언 및 입력:
n, k = tuple(map(int, input().split()))
grid = [
    list(map(int, input().split()))
    for _ in range(n)
]

# bfs에 필요한 변수들 입니다.
bfs_q = deque()
visited = [
    [False for _ in range(n)]
    for _ in range(n)
]


def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < n


def can_go(x, y):
    return in_range(x, y) and not grid[x][y] and \
           not visited[x][y]


def bfs():
    # queue에 남은 것이 없을때까지 반복합니다.
    while bfs_q:
        # queue에서 가장 먼저 들어온 원소를 뺍니다.
        x, y = bfs_q.popleft()
        
        dxs, dys = [1, -1, 0, 0], [0, 0, 1, -1]

        # queue에서 뺀 원소의 위치를 기준으로 4방향을 확인해봅니다.
        for dx, dy in zip(dxs, dys):
            nx, ny = x + dx, y + dy

            # 아직 방문한 적이 없으면서 갈 수 있는 곳이라면
            # 새로 queue에 넣어주고 방문 여부를 표시해줍니다. 
            if can_go(nx, ny):
                bfs_q.append((nx, ny))
                visited[nx][ny] = True

                
# 시작점을 모두 bfs queue에 넣습니다.
for _ in range(k):
    x, y = tuple(map(int, input().split()))
    bfs_q.append((x - 1, y - 1))
    visited[x - 1][y - 1] = True

# bfs를 진행합니다.
bfs()

ans = sum([
    1
    for i in range(n)
    for j in range(n)
    if visited[i][j]
])

print(ans)


from collections import deque
import enum

class Element(enum.Enum):
    WATER = 0
    GLACIER = 1
    
# 변수 선언 및 입력
n, m = tuple(map(int, input().split()))

a = [
    list(map(int, input().split()))
    for _ in range(n)
]

# bfs에 필요한 변수들 입니다.
q = deque()
glaciers_to_melt = deque()
visited = [
    [False for _ in range(m)]
    for _ in range(n)
]
cnt = 0

# 0: 오른쪽, 1: 아래쪽, 2: 왼쪽, 3: 위쪽
dxs, dys = [0, 1, 0, -1], [1, 0, -1, 0]

# 소요 시간과 가장 마지막으로 녹은 빙하의 수를 저장합니다.
elapsed_time = 0
last_melt_cnt = 0

# 주어진 위치가 격자를 벗어나는지 여부를 반환합니다.
def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < m

# 범위를 벗어나지 않으면서 물이여야 하고 방문한적이 
# 없어야 갈 수 있습니다.
def can_go(x, y):
    return in_range(x, y) and a[x][y] == Element.WATER.value and not visited[x][y]


def is_glacier(x, y):
    return in_range(x, y) and a[x][y] == Element.GLACIER.value and not visited[x][y]


# visited 배열을 초기화합니다.
def initialize():
    for i in range(n):
        for j in range(m):
            visited[i][j] = False
            
            
# 빙하에 둘러쌓여 있지 않은 물들을 전부 구해주는 BFS입니다.
# 문제에서 가장자리는 전부 물로 주어진다 했기 때문에
# 항상 (0, 0)에서 시작하여 탐색을 진행하면
# 빙하에 둘러쌓여 있지 않은 물들은 전부 visited 처리가 됩니다.
def bfs():
    # BFS 함수가 여러 번 호출되므로
    # 사용하기 전에 visited 배열을 초기화 해줍니다.
    initialize()
    
    # 항상 (0, 0)에서 시작합니다.
    q.append((0, 0))
    visited[0][0] = True
    
    while q:
        # queue에서 가장 먼저 들어온 원소를 뺍니다.
        x, y = q.popleft()
        
        # queue에서 뺀 원소의 위치를 기준으로 네 방향을 확인합니다.
        for dx, dy in zip(dxs, dys):
            new_x, new_y = x + dx, y + dy

            # 더 갈 수 있는 곳이라면 Queue에 추가합니다.
            if can_go(new_x, new_y):
                q.append((new_x, new_y))
                visited[new_x][new_y] = True

                
# 현재 위치를 기준으로 인접한 영역에
# 빙하에 둘러쌓여 있지 않은 물이 있는지를 판단합니다.   
def outside_water_exist_in_neighbor(x, y):
    for dx, dy in zip(dxs, dys):
        new_x, new_y = x + dx, y + dy
        if in_range(new_x, new_y) and visited[new_x][new_y]:
            return True
        
    return False


# 인접한 영역에 빙하에 둘러쌓여 있지 않은 물이 있는 빙하를 찾아
# 녹여줍니다.
def melt():
    global last_melt_cnt
    
    for i in range(n):
        for j in range(m):
            if a[i][j] == Element.GLACIER.value and \
                    outside_water_exist_in_neighbor(i, j):
                a[i][j] = Element.WATER.value
                last_melt_cnt += 1
                
                
# 빙하를 한 번 녹입니다.
def simulate():
    global elapsed_time, last_melt_cnt
    
    elapsed_time += 1
    last_melt_cnt = 0
    
    # 빙하에 둘러쌓여 있지 않은 물의 위치를 전부
    # visited로 체크합니다.
    bfs()
    
    # 인접한 영역에 빙하에 둘러쌓여 있지 않은 물이 있는 빙하를 찾아
    # 녹여줍니다.
    melt()
    

# 빙하가 아직 남아있는지 확인합니다.
def glacier_exist():
    for i in range(n):
        for j in range(m):
            if a[i][j] == Element.GLACIER.value:
                return True
    return False


while True:
    simulate()
    
    # 빙하가 존재하는 한 계속 빙하를 녹입니다.
    if not glacier_exist():
        break
        
print(elapsed_time, last_melt_cnt)




from collections import deque
import enum

class Element(enum.Enum):
    WATER = 0
    GLACIER = 1
    
# 변수 선언 및 입력
n, m = tuple(map(int, input().split()))

a = [
    list(map(int, input().split()))
    for _ in range(n)
]

# bfs에 필요한 변수들 입니다.
q = deque()
glaciers_to_melt = deque()
visited = [
    [False for _ in range(m)]
    for _ in range(n)
]
cnt = 0

dxs, dys = [0, 1, 0, -1], [1, 0, -1, 0]

# 소요 시간과 가장 마지막으로 녹은 빙하의 수를 저장합니다.
elapsed_time = 0
last_melt_cnt = 0

# 주어진 위치가 격자를 벗어나는지 여부를 반환합니다.
def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < m

# 범위를 벗어나지 않으면서 물이여야 하고 방문한적이 
# 없어야 갈 수 있습니다.
def can_go(x, y):
    return in_range(x, y) and a[x][y] == Element.WATER.value and not visited[x][y]


# 범위를 벗어나지 않으면서 빙하여야 하고 이미 
# 선택된 적이 없어야 중복 없이 녹아야할 빙하 목록에 
# 해당 빙하를 문제 없이 추가할 수 있습니다.
def is_glacier(x, y):
    return in_range(x, y) and a[x][y] == Element.GLACIER.value and not visited[x][y]


# 아직 방문해보지 못한 빙하에 둘러쌓여 있지 않은 물 영역을 더 탐색해주는 BFS입니다.
def bfs():
    while q:
        # queue에서 가장 먼저 들어온 원소를 뺍니다.
        x, y = q.popleft()
        
        # queue에서 뺀 원소의 위치를 기준으로 네 방향을 확인합니다.
        for dx, dy in zip(dxs, dys):
            new_x, new_y = x + dx, y + dy

            # 더 갈 수 있는 곳이라면 Queue에 추가합니다.
            if can_go(new_x, new_y):
                q.append((new_x, new_y))
                visited[new_x][new_y] = True
            # 만약 아직 방문하지 않은 빙하가 있는 곳이라면
            elif is_glacier(new_x, new_y):
                # 빙하에 둘러쌓여 있지 않은 물에 인접한 빙하이므로 이번에 녹아야 할 빙하이므로 
                # 따로 저장해줍니다.
                # 중복되어 같은 빙하 정보가 기록되는 것을 막기위해
                # 이때에도 visited 값을 true로 설정해줍니다.
                glaciers_to_melt.append((new_x, new_y))
                visited[new_x][new_y] = True


# 녹여야 할 빙하들을 녹여줍니다.
def melt():
    while glaciers_to_melt:
        x, y = glaciers_to_melt.popleft()
        a[x][y] = Element.WATER.value
        
        
# 빙하를 한 번 녹입니다.
def simulate():
    global elapsed_time, last_melt_cnt, q

    # 빙하에 둘러쌓여 있지 않은 물의 영역을 넓혀보며
    # 더 녹일 수 있는 빙하가 있는지 봅니다. 
    bfs()
    
    # 더 녹일 수 있는 빙하가 없다면 시뮬레이션을 종료합니다.
    if not glaciers_to_melt:
        return False
    
    # 더 녹일 빙하가 있다면 답을 갱신해주고
    # 그 다음 시뮬레이션에서는 해당 빙하들의 위치를 시작으로
    # 빙하에 둘러쌓여 있지 않은 물의 영역을 더 탐색할 수 있도록 queue에 
    # 녹아야 할 빙하들의 위치를 넣어줍니다.
    elapsed_time += 1
    last_melt_cnt = len(glaciers_to_melt)

    q = glaciers_to_melt.copy()

    # 녹아야 할 빙하들을 녹여줍니다.
    melt()
    
    return True
    
    
# 처음에는 (0, 0) 에서 시작하여 초기 빙하에 둘러쌓여 있지 않은 물들을 찾을 수 있도록 합니다.
q.append((0, 0))
visited[0][0] = True

while True:
    is_glacier_exist = simulate()
    
    # 빙하에 둘러쌓여 있지 않은 물의 영역을 넓혀보며 더 녹일 수 있는 빙하가 있는지 봅니다.
    if not is_glacier_exist:
        break
        
print(elapsed_time, last_melt_cnt)



import sys
# recursion의 최대 깊이(N*M)인 10000으로 설정해줍니다.
sys.setrecursionlimit(10000)

INT_MAX = sys.maxsize

# 변수 선언 및 입력
n, m = tuple(map(int, input().split()))

a = [
    list(map(int, input().split()))
    for _ in range(n)
]

# backtracking에 필요한 변수들 입니다.
visited = [
    [False for _ in range(m)]
    for _ in range(n)
]

ans = INT_MAX

def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < m


# 격자를 벗어나지 않으면서, 뱀도 없고, 아직 방문한 적이 없는 곳이라면
# 이동이 가능합니다.
def can_go(x, y):
    return in_range(x, y) and a[x][y] and not visited[x][y]


# backtracking을 통해 최소 이동 횟수를 구합니다.
def find_min(x, y, cnt):
    global ans
    
    if x == n - 1 and y == m - 1:
        ans = min(ans, cnt)
        return
    
    dxs, dys = [0, 1, 0, -1], [1, 0, -1, 0]
    
    # 현재 위치를 기준으로 4방향을 확인해봅니다.
    for dx, dy in zip(dxs, dys):
        new_x, new_y = x + dx, y + dy
        
        # 아직 방문한 적이 없으면서 갈 수 있는 곳이라면
        # 더 진행해봅니다.
        if can_go(new_x, new_y):
            # 지금까지의 선택이 최단경로 로서 부적합했을 수 있으므로
            # 퇴각시 visited값을 다시 false로 바꿔 
            # 다른 방향으로 진행할때도 기회를 주어 모든 가능한 
            # 경로를 전부 탐색할 수 있도록 합니다. 
            visited[new_x][new_y] = True
            find_min(new_x, new_y, cnt + 1)
            visited[new_x][new_y] = False
            
# 현재까지 이동 횟수가 0일때, (0, 0)에서 시작하는
# 재귀를 호출합니다.
find_min(0, 0, 0)

# 불가능한 경우라면 -1을 답으로 넣어줍니다.
if ans == INT_MAX:
    ans = -1
    
print(ans)


import sys
from collections import deque

INT_MAX = sys.maxsize

# 변수 선언 및 입력
n, m = tuple(map(int, input().split()))

a = [
    list(map(int, input().split()))
    for _ in range(n)
]

# bfs에 필요한 변수들 입니다.
q = deque()
visited = [
    [False for _ in range(m)]
    for _ in range(n)
]
# step[i][j] : 시작점으로부터 (i, j) 지점에 도달하기 위한 
# 최단거리를 기록합니다.
step = [
    [0 for _ in range(m)]
    for _ in range(n)
]

ans = INT_MAX

def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < m


# 격자를 벗어나지 않으면서, 뱀도 없고, 아직 방문한 적이 없는 곳이라면
# 지금 이동하는 것이 최단거리임을 보장할 수 있으므로 가야만 합니다. 
def can_go(x, y):
    return in_range(x, y) and a[x][y] and not visited[x][y]


# queue에 새로운 위치를 추가하고
# 방문 여부를 표시해줍니다.
# 시작점으로 부터의 최단거리 값도 갱신해줍니다.
def push(new_x, new_y, new_step):
    q.append((new_x, new_y))
    visited[new_x][new_y] = True
    step[new_x][new_y] = new_step
    
    
# bfs를 통해 최소 이동 횟수를 구합니다.
def find_min():
    global ans
    
    dxs, dys = [0, 1, 0, -1], [1, 0, -1, 0]
    
    # queue에 남은 것이 없을때까지 반복합니다.
    while q:
        # queue에서 가장 먼저 들어온 원소를 뺍니다.
        x, y = q.popleft()    
    
        # queue에서 뺀 원소의 위치를 기준으로 4방향을 확인해봅니다.
        for dx, dy in zip(dxs, dys):
            new_x, new_y = x + dx, y + dy
        
            # 아직 방문한 적이 없으면서 갈 수 있는 곳이라면
            # 새로 queue에 넣어줍니다.
            if can_go(new_x, new_y):
                # 최단 거리는 이전 최단거리에 1이 증가하게 됩니다.
                push(new_x, new_y, step[x][y] + 1)
    
    # 우측 하단에 가는 것이 가능할때만 답을 갱신해줍니다.
    if visited[n - 1][m - 1]:
        ans = step[n - 1][m - 1]

# bfs를 이용해 최소 이동 횟수를 구합니다.
# 시작점을 queue에 넣고 시작합니다.
push(0, 0, 0)
find_min()

# 불가능한 경우라면 -1을 답으로 넣어줍니다.
if ans == INT_MAX:
    ans = -1
    
print(ans)


import sys
import enum

sys.setrecursionlimit(100000)
OPERATOR_NUM = 4
INT_MAX = sys.maxsize

class OPERATOR(enum.Enum):
    SUBTRACT = 0
    ADD = 1
    DIV2 = 2
    DIV3 = 3
    
n = int(input())
ans = INT_MAX

# num이라는 값에 해당 operator를 사용할 수 있는지를 판단합니다.
# 2로 나누거나 3으로 나누려는 경우 num이 해당 값으로 나누어 떨어질 때에만
# 해당 연산을 사용 가능합니다.
def possible(num, op):
    if op == OPERATOR.SUBTRACT.value or op == OPERATOR.ADD.value:
        return True
    elif op == OPERATOR.DIV2.value:
        return num % 2 == 0
    else:
        return num % 3 == 0

    
# num에 op 연산을 수행했을 때의 결과를 반환합니다.
def calculate(num, op):
    if op == OPERATOR.SUBTRACT.value:
        return num - 1
    elif op == OPERATOR.ADD.value:
        return num + 1
    elif op == OPERATOR.DIV2.value:
        return num // 2
    else:
        return num // 3
        
        
# 모든 가지수를 다 조사해보며 최소 연산 횟수를 계산합니다.
def find_min(num, cnt):
    global ans
    
    # 1이 되었을 경우 답이랑 비교하여 갱신합니다.
    if num == 1:
        ans = min(ans, cnt)
        return
    
    # 답은 최대 n - 1을 넘을수는 없으므로 
    # 더 이상 탐색을 진행하지 않습니다.
    if cnt >= n - 1:
        return

    # 4가지의 연산을 시도해 봅니다.
    # 해당 연산을 쓸 수 있는 경우에만
    # 더 탐색을 진행합니다.
    for i in range(OPERATOR_NUM):
        if possible(num, i):
            find_min(calculate(num, i), cnt + 1)


# 모든 가지수를 다 조사해보며 최소 연산 횟수를 계산합니다.
find_min(n, 0)            
print(ans)


from collections import deque
import sys
import enum

OPERATOR_NUM = 4
INT_MAX = sys.maxsize

class OPERATOR(enum.Enum):
    SUBTRACT = 0
    ADD = 1
    DIV2 = 2
    DIV3 = 3
    
n = int(input())
ans = INT_MAX

q = deque()
visited = [False for _ in range(2 * n)]

# step[i] : 정점 n에서 시작하여 정점 i 지점에 도달하기 위한 
# 최단거리를 기록합니다.
step = [0 for _ in range(2 * n)]

# num이라는 값에 해당 operator를 사용할 수 있는지를 판단합니다.
# 2로 나누거나 3으로 나누려는 경우 num이 해당 값으로 나누어 떨어질 때에만
# 해당 연산을 사용 가능합니다.
def possible(num, op):
    if op == OPERATOR.SUBTRACT.value or op == OPERATOR.ADD.value:
        return True
    elif op == OPERATOR.DIV2.value:
        return num % 2 == 0
    else:
        return num % 3 == 0

    
# num에 op 연산을 수행했을 때의 결과를 반환합니다.
def calculate(num, op):
    if op == OPERATOR.SUBTRACT.value:
        return num - 1
    elif op == OPERATOR.ADD.value:
        return num + 1
    elif op == OPERATOR.DIV2.value:
        return num // 2
    else:
        return num // 3
        
# 1에서 2n - 1 사이의 숫자만 이용해도 올바른 답을 구할 수 있으므로 
# 그 범위 안에 들어오는 숫자인지를 확인합니다.
def in_range(num):
    return 1 <= num and num <= 2 * n - 1


# 1에서 2n - 1 사이의 숫자이면서 아직 방문한 적이 없다면 가야만 합니다. 
def can_go(num):
    return in_range(num) and not visited[num]

# queue에 새로운 위치를 추가하고 방문 여부를 표시해줍니다.
# 시작점으로 부터의 최단거리 값도 갱신해줍니다.
def push(num, new_step):
    q.append(num)
    visited[num] = True
    step[num] = new_step
    
# BFS를 통해 최소 연산 횟수를 구합니다.
def find_min():
    global ans
    
    # queue에 남은 것이 없을때까지 반복합니다.
    while q:
        # queue에서 가장 먼저 들어온 원소를 뺍니다.
        curr_num = q.popleft()
        
        # queue에서 뺀 원소의 위치를 기준으로 4가지 연산들을 적용해봅니다.
        for i in range(OPERATOR_NUM):
            # 연산을 적용할 수 없는 경우라면 패스합니다.
            if not possible(curr_num, i):
                continue
            
            new_num = calculate(curr_num, i)
            # 아직 방문한 적이 없으면서 갈 수 있는 곳이라면 새로 queue에 넣어줍니다.
            if can_go(new_num):
                # 최단 거리는 이전 최단거리에 1이 증가하게 됩니다. 
                push(new_num, step[curr_num] + 1)
        
        # 1번 정점까지 가는 데 필요한 최소 연산 횟수를 답으로 기록합니다.
        ans = step[1]

# BFS를 통해 최소 연산 횟수를 구합니다.
push(n, 0)
find_min()            
print(ans)


n = int(input())

def fib(n):
    if n == 1 or n == 2:
        return 1
    
    return fib(n - 1) + fib(n - 2)

print(fib(n))


n = int(input())

UNUSED = -1

memo = [UNUSED for _ in range(n + 1)]

def fib(n):
    if memo[n] != UNUSED:
        return memo[n]
    
    if n == 1 or n == 2:
        return 1
    
    memo[n] = fib(n - 1) + fib(n - 2)
    return memo[n]

print(fib(n))


n = int(input())

MAX_NUM = 45

dp = [0 for _ in range(MAX_NUM + 1)]

dp[1] = dp[2] = 1

for i in range(3, n + 1):
    dp[i] = dp[i - 1] + dp[i - 2]
    
print(dp[n])

n = int(input())

grid = [
    list(map(int, input().split()))
    for _ in range(n)
]

max_sum = 0

def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < n


def find_max_sum(x, y, sum):
    global max_sum
    
    # 도착 지점에 도착하면 최대 합을 갱신해줍니다.
    if x == n - 1 and y == n - 1:
        max_sum = max(max_sum, sum)
        return
    
    dxs, dys = [1, 0], [0, 1]
    
    # 가능한 모든 방향에 대해 탐색해줍니다.
    for dx, dy in zip(dxs, dys):
        new_x, new_y = x + dx, y + dy
        
        if in_range(new_x, new_y):
            find_max_sum(new_x, new_y, sum + grid[new_x][new_y])
                         
                    
find_max_sum(0, 0, grid[0][0])
print(max_sum)


n = int(input())

grid = [
    list(map(int, input().split()))
    for _ in range(n)
]


def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < n


def find_max_sum(x, y):
    # 도착 지점에 도착하면 최대 합을 갱신해줍니다.
    if x == n - 1 and y == n - 1:
        return grid[n - 1][n - 1]
    
    dxs, dys = [1, 0], [0, 1]
    
    # 가능한 모든 방향에 대해 탐색해줍니다.
    max_sum = 0 # 주어진 숫자의 범위가 1보다 크기 때문에 항상 갱신됨이 보장됩니다.
    for dx, dy in zip(dxs, dys):
        new_x, new_y = x + dx, y + dy
        
        if in_range(new_x, new_y):
            max_sum = max(max_sum, find_max_sum(new_x, new_y) + grid[x][y])
            
    return max_sum
                    
print(find_max_sum(0, 0))


n = int(input())

UNUSED = -1

grid = [
    list(map(int, input().split()))
    for _ in range(n)
]

memo = [
    [UNUSED for _ in range(n)]
    for _ in range(n)
]

def in_range(x, y):
    return 0 <= x and x < n and 0 <= y and y < n


def find_max_sum(x, y):
    # 미리 계산된 적이 있는 경우 해당 값을 사용해줍니다.
    if memo[x][y] != UNUSED:
        return memo[x][y]
    
    # 도착 지점에 도착하면 최대 합을 갱신해줍니다.
    if x == n - 1 and y == n - 1:
        return grid[n - 1][n - 1]
    
    dxs, dys = [1, 0], [0, 1]
    
    # 가능한 모든 방향에 대해 탐색해줍니다.
    max_sum = 0 # 주어진 숫자의 범위가 1보다 크기 때문에 항상 갱신됨이 보장됩니다.
    for dx, dy in zip(dxs, dys):
        new_x, new_y = x + dx, y + dy
        
        if in_range(new_x, new_y):
            max_sum = max(max_sum, find_max_sum(new_x, new_y) + grid[x][y])
        
    # 게산된 값을 memo 배열에 저장해줍니다.
    memo[x][y] = max_sum
    return max_sum


print(find_max_sum(0, 0))

n = int(input())

num = [
    list(map(int, input().split()))
    for _ in range(n)
]

dp = [
    [0 for _ in range(n)]
    for _ in range(n)
]

def initialize():
    # 시작점의 경우 dp[0][0] = num[0][0]으로 초기값을 설정해줍니다
    dp[0][0] = num[0][0]
    
    # 최좌측 열의 초기값을 설정해줍니다.
    for i in range(1, n):
        dp[i][0] = dp[i-1][0] + num[i][0]
    
    # 최상단 행의 초기값을 설정해줍니다.
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + num[0][j]
        
        
# 초기값 설정
initialize()

# 탐색하는 위치의 위에 값과 좌측 값 중에 큰 값에
# 해당 위치의 숫자를 더해줍니다. 
for i in range(1, n):
    for j in range(1, n):
        dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + num[i][j]
        
print(dp[n-1][n-1])




str1 = input()
str2 = input()

str1_len, str2_len = len(str1), len(str2)

# string의 index가 0부터 시작하기 때문에
# 이를 1부터 시작하기 위해서 앞에 #을 추가해줍니다.
str1, str2 = '#' + str1, '#' + str2

dp = [
    [0 for _ in range(str2_len + 1)]
    for _ in range(str1_len + 1)
]

def initialize():
    # dp[1][1] 값은 첫 번째 문자열의 첫 번째 문자와
    # 두 번째 문자열의 첫 번째 문자가 같은지 여부를 저장합니다.
    dp[1][1] = int(str1[1] == str2[1])
    
    # 두 번째 문자열의 1번 인덱스의 문자까지만 사용했을 때 
    # 가능한 부분 수열의 최대 길이를 채워넣어줍니다.
    for i in range(2, str1_len + 1):
        if str1[i] == str2[1]:
            dp[i][1] = 1
        else:
            dp[i][1] = dp[i-1][1]
    
    # 첫 번째 문자열의 1번 인덱스의 문자까지만 사용했을 때 
    # 가능한 부분 수열의 최대 길이를 채워넣어줍니다.
    for j in range(2, str2_len + 1):
        if str1[1] == str2[j]:
            dp[1][j] = 1
        else:
            dp[1][j] = dp[1][j-1]
            
initialize()

for i in range(2, str1_len + 1):
    # 첫 번째 문자열의 i 번째까지 문자열을 고려했고
    # 두 번째 문자열의 j 번째까지 문자열을 고려했을 때
    # 가능한 부분 수열의 최대 길이를 구해줍니다.
    for j in range(2, str2_len + 1):
        # Case 1:
        # 첫 번째 문자열의 i번째 문자와,  두 번째 문자열 j번째 문자가 일치하는 경우
        # 첫 번째 문자열에서 i-1번째 문자까지 고려하고, 
        # 두 번째 문자열의 j-1번째 문자까지 고려했을 때 
        # 가능한 공통 부분 수열의 뒤에 문자 하나를 새로 추가할 수 있게 됩니다. 
        # 따라서 dp[i-1][j-1]에 1을 추가해주면 됩니다
        if str1[i] == str2[j]:
            dp[i][j] = dp[i-1][j-1] + 1
            
        # Case 2:
        # 첫 번째 문자열의 i 번째 문자를 공통 부분 수열을 만드는데 고려하지 않는 경우와
        # 두 번째 문자열의 j 번째 문자를 공통 부분 수열을 만드는데 고려하지 않는 경우 중
        # 더 큰 값을 선택하여 줍니다. 
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
print(dp[str1_len][str2_len])



UNUSED = -1

str1 = input()
str2 = input()

str1_len, str2_len = len(str1), len(str2)

# string의 index가 0부터 시작하기 때문에
# 이를 1부터 시작하기 위해서 앞에 #을 추가해줍니다.
str1, str2 = '#' + str1, '#' + str2

memo = [
    [UNUSED for _ in range(str2_len + 1)]
    for _ in range(str1_len + 1)
]

def find_max_len(i, j):
    # 만약 주어진 문자열의 범위가 가능한 범위를 넘어가는 경우
    # 더 이상 매칭을 진행할 수 없으므로, 
    # 해당 상황에서 최장 증가 부분 수열의 길이는 0이 됩니다
    if i > str1_len or j > str2_len:
        return 0
    
    # 이미 탐색한 적이 있다면 해당 값을 사용해줍니다.
    if memo[i][j] != UNUSED:
        return memo[i][j]
    
    # Case 1:
    if str1[i] == str2[j]:
        memo[i][j] = find_max_len(i + 1, j + 1) + 1
    # Case 2:
    else:
        memo[i][j] = max(find_max_len(i + 1, j), find_max_len(i, j + 1))
        
    return memo[i][j]


print(find_max_len(1, 1))





import sys
sys.setrecursionlimit(10000)

INT_MAX = sys.maxsize

n, m = tuple(map(int, input().split()))

coin = [0 for _ in range(n + 1)]
given_seq = list(map(int, input().split()))
coin[1:] = given_seq[:]

# 최소를 구하는 문제이므로
# 초기값을 INT_MAX로 설정합니다.
min_cnt = INT_MAX

# 지금까지 cnt개의 동전을 사용하여
# 합을 sum 만들었을 때, 최종적으로 합 m을
# 만들기 위해 탐색을 더 진행하는 재귀입니다.
def find_min_cnt(sum, cnt):
    global min_cnt
    
    # 합이 m이 되면 최소 동전의 수를 갱신해줍니다.
    if sum == m:
        min_cnt = min(min_cnt, cnt)
        return
    
    # 동전들을 추가적으로 한 번씩 더 사용해봅니다.
    for i in range(1, n + 1):
        if sum + coin[i] <= m:
            find_min_cnt(sum + coin[i], cnt + 1)
            
            
# 0개의 동전을 사용하여, 합 0을 만들었을 경우를
# 초기 상태로 설정하여 재귀를 호출합니다.
find_min_cnt(0, 0)

# 거슬러주는것이 불가능 할 시, -1을 출력합니다.
if min_cnt == INT_MAX:
    min_cnt = -1

print(min_cnt)



import sys
sys.setrecursionlimit(10000)

UNUSED = -1
MAX_ANS = 10001

n, m = tuple(map(int, input().split()))

coin = [0 for _ in range(n + 1)]
given_seq = list(map(int, input().split()))
coin[1:] = given_seq[:]

memo = [UNUSED for _ in range(m + 1)]

# sum에서부터 시작하여 최종적으로 합 m을 만드는 데
# 필요한 최소 동전의 수를 반환하는 재귀입니다.
def find_min_cnt(sum):
    # 미리 계산된 적이 있는 경우 해당 값을 사용해줍니다.
    if memo[sum] != UNUSED:
        return memo[sum]

    # 합이 m이 되면 동전이 추가적으로 필요 없으므로
    # 필요한 동전의 수 0을 반환 합니다.
    if sum == m:
        memo[sum] = 0
        return 0
    
    # 최소를 구하는 문제이므로
    # 초기값을 답이 될 수 있는 최대보다 조금 더 큰
    # MAX_ANS로 설정합니다.
    min_cnt = MAX_ANS
    
    # 동전들을 하나씩 사용해봅니다.
    for i in range(1, n + 1):
        if sum + coin[i] <= m:
            min_cnt = min(min_cnt, find_min_cnt(sum + coin[i]) + 1)
            
    memo[sum] = min_cnt
    return min_cnt
            
            
# 합 0에서부터 시작하여 합 m을 만들기 위해 필요한
# 최소 동전의 수를 계산합니다.
min_cnt = find_min_cnt(0)

# 거슬러주는것이 불가능 할 시, -1을 출력합니다.
if min_cnt == MAX_ANS:
    min_cnt = -1

print(min_cnt)



MAX_ANS = 10001

n, m = tuple(map(int, input().split()))

coin = [0 for _ in range(n + 1)]
given_seq = list(map(int, input().split()))
coin[1:] = given_seq[:]

# dp[i] : 지금까지 선택한 동전의 합이 i일 때, 가능한 최소 동전 횟수
# 최소를 구하는 문제이므로, 초기에는 전부 MAX_ANS을 넣어줍니다.
dp = [MAX_ANS for _ in range(m + 1)]

# 초기 조건으로 아직 아무런 동전도 고르지 않은 상태를 정의합니다.
# 따라서 지금까지 선택한 동전의 합이 0이며
# 지금까지 사용한 동전의 수는 0개이므로,
# dp[0] = 0을 초기 조건으로 설정합니다.
dp[0] = 0

# 지금까지 선택한 동전의 합이 i이기 위해 
# 필요한 최소 동전 횟수를 계산합니다.
for i in range(1, m + 1):
    # 합 i를 만들기 위해 마지막으로 사용한 동전이 j번째 동전이었을 경우를
    # 전부 고려해봅니다. 마지막으로 사용한 동전이 j번째 동전이었을 경우
    # 최종 합이 i가 되기 위해서는 이전 합이 i - coin[j] 였어야 하므로
    # 해당 상태를 만들기 위해 필요한 최소 동전의 수인 
    # dp[i - coin[j]]에 동전을 새로 1개 추가했으므로
    # 1을 더한 값들 중 최솟값을 선택하면 됩니다.
    # 단, 합 i가 coin[j]보다 작은 경우에는 j번째
    # 동전을 써서 합 i를 절대 만들 수 없으므로
    # i >= coin[j] 조건을 만족하는 경우에 대해서만 고려합니다.
    for j in range(1, n + 1):
        if i >= coin[j]:
            dp[i] = min(dp[i], dp[i - coin[j]] + 1)

# 합을 정확히 m을 만들었을 때
# 필요한 최소 동전의 수를 구해야 하므로
# dp[m]이 답이 됩니다.
min_cnt = dp[m]

# 거슬러주는것이 불가능 할 시, -1을 출력합니다.
if min_cnt == MAX_ANS:
    min_cnt = -1

print(min_cnt)


from collections import deque

n, m = tuple(map(int, input().split()))

coin = [0 for _ in range(n + 1)]
given_seq = list(map(int, input().split()))
coin[1:] = given_seq[:]

# bfs에 필요한 변수들 입니다.
q = deque()
visited = [False for _ in range(m + 1)]

# step[i][j] : 정점 0에서 시작하여 정점 i 지점에 도달하기 위한  
# 최단거리를 기록합니다.
step = [0 for _ in range(m + 1)]
ans = 0

# m 이내의 숫자만 이용해도 올바른 답을 구할 수 있으므로 
# 그 범위 안에 들어오는 숫자인지를 확인합니다.
def in_range(num):
    return num <= m


# m 이내의 숫자이면서 아직 방문한 적이 없다면 가야만 합니다. 
def can_go(num):
    return in_range(num) and not visited[num]


# queue에 새로운 위치를 추가하고
# 방문 여부를 표시해줍니다.
# 시작점으로 부터의 최단거리 값도 갱신해줍니다.
def push(num, new_step):
    q.append(num)
    visited[num] = True
    step[num] = new_step

    
# BFS를 통해 최소 연산 횟수를 구합니다.
def find_min():
    global ans
    
    # queue에 남은 것이 없을때까지 반복합니다.
    while q:
        # queue에서 가장 먼저 들어온 원소를 뺍니다.
        curr_num = q.popleft()
        
        # queue에서 뺀 원소의 위치를 기준으로 n개의 동전들을 사용해봅니다.
        for i in range(1, n + 1):
            # 아직 방문한 적이 없으면서 갈 수 있는 곳이라면 새로 queue에 넣어줍니다.
            if can_go(curr_num + coin[i]):
                # 최단 거리는 이전 최단거리에 1이 증가하게 됩니다. 
                push(curr_num + coin[i], step[curr_num] + 1)
                
    # m번 정점까지 가는 데 필요한 최소 연산 횟수를 답으로 기록합니다.
    # 만약 m번 정점으로 갈 수 없다면, -1을 기록합니다    
    if visited[m]:
        ans = step[m]
    else:
        ans = -1
        
        
# BFS를 통해 최소 연산 횟수를 구합니다.
push(0, 0)
find_min()
print(ans)




import sys

INT_MIN = -1 * sys.maxsize

n = int(input())

# dp[i] : 마지막으로 고른 원소의 위치가 i인
# 부분 수열 중 최장 부분 수열의 길이
# 최대를 구하는 문제이므로, 초기에는 전부 INT_MIN을 넣어줍니다.
dp = [INT_MIN for _ in range(n + 1)]
a = [0 for _ in range(n + 1)]

given_seq = list(map(int, input().split()))
a[1:] = given_seq[:]

# 0번째 index와 비교했을 때 항상 갱신될 수 있는 값을 넣어줍니다.
dp[0], a[0] = 0, 0

for i in range(1, n + 1):
    # i번째 보다 앞에 있는 원소들 중 
    # a[i]보다는 값이 작은 곳에 새로운 원소인 a[i]를 추가했을 때의 
    # 부분 수열 중 최대 부분 수열의 길이를 계산합니다.
    for j in range(0, i):
        if a[j] < a[i]:
            dp[i] = max(dp[i], dp[j] + 1)
            
# 마지막 원소의 위치가 i일 때의 부분 수열들 중
# 가장 길이가 긴 부분 수열을 고릅니다.
answer = 0
for i in  range(n + 1):
    answer = max(answer, dp[i])
    
print(answer)





import sys

INT_MIN = -1 * sys.maxsize
MAX_VALUE = 10000

n = int(input())

# dp[i][j] : i 번째 원소까지 고려했고
# 마지막으로 고른 원소의 값이 j일 때의 최장 부분 수열의 길이
# 최대를 구하는 문제이므로, 초기에는 전부 INT_MIN을 넣어줍니다.
dp = [
    [INT_MIN for _ in range(MAX_VALUE + 1)]
    for _ in range(n + 1)
]

a = [0 for _ in range(n + 1)]

given_seq = list(map(int, input().split()))
a[1:] = given_seq[:]

# 0번째 원소에 0이라는 숫자로 항상 부분 수열을 만들되
# 이때까지의 부분 수열의 길이는 0이었기 때문에, 
# 각각의 위치에 있는 원소를 시작으로 하는 
# 모든 부분 수열을 만들 수 있게 해줍니다.
dp[0][0] = 0

for i in range(1, n + 1):
    for j in range(0, MAX_VALUE):
        # j와 현재 위치에 있는 숫자 a[i]가 다르다면
        # i - 1번째 중에 최장 부분 수열이 있는 경우만 생각합니다.
        if j != a[i]:
            dp[i][j] = dp[i - 1][j]
        # j값이 현재 위치에 있는 숫자 a[i]와 일치한다면
        # 2가지 경우를 다 고려하여 그 중 가장 좋은 값을 택해야 합니다.
        else:
            # Case 1 : i - 1번째 중에 최장 부분 수열이 있는 경우
            dp[i][j] = dp[i - 1][j]
            
            # Case 2 : a[i]를 최장 부분 수열에 이용한 경우
            # 증가 부분 수열이 되어야 하므로 a[i]보다 작은 부분까지만 탐색하여
            # a[i]라는 원소를 하나 더 추가했을 때의 부분 수열 길이 중 최댓값을 갱신합니다.
            for l in range(0, a[i]):
                if dp[i - 1][l] != INT_MIN:
                    dp[i][j] = max(dp[i][j], dp[i - 1][l] + 1)
            
# n개의 원소를 다 고려했을 때, 마지막으로 끝나는 숫자가 j일때의 부분 수열들 중
# 가장 길이가 긴 부분 수열을 고릅니다.
answer = 0
for j in range(MAX_VALUE + 1):
    answer = max(answer, dp[n][j])
    
print(answer)


import sys

INT_MIN = -1 * sys.maxsize
MAX_VALUE = 10000

n = int(input())

# dp[j] : 지금까지 마지막으로 고른 원소의 값이 j일 때의
# 최장 부분 수열의 길이
# 최대를 구하는 문제이므로, 초기에는 전부 INT_MIN을 넣어줍니다.
dp = [INT_MIN for _ in range(MAX_VALUE + 1)]

a = [0 for _ in range(n + 1)]

given_seq = list(map(int, input().split()))
a[1:] = given_seq[:]

# 0번째 원소에 0이라는 숫자로 항상 부분 수열을 만들되
# 이때까지의 부분 수열의 길이는 0이었기 때문에, 
# 각각의 위치에 있는 원소를 시작으로 하는 
# 모든 부분 수열을 만들 수 있게 해줍니다.
dp[0] = 0

for i in range(1, n + 1):
    # j가 a[i]인 경우만 고민해서 누적합니다.
    j = a[i]
    for l in range(a[i]):
        if dp[l] != INT_MIN:
            dp[j] = max(dp[j], dp[l] + 1)

# 마지막으로 끝나는 숫자가 j일때의 부분 수열들 중
# 가장 길이가 긴 부분 수열을 고릅니다.
answer = 0
for j in range(MAX_VALUE + 1):
    answer = max(answer, dp[j])
    
print(answer)



import sys

INT_MIN = -sys.maxsize

# 변수 선언 및 입력
n = int(input())
red = [
    0
    for _ in range(2 * n + 1)
]
blue = [
    0
    for _ in range(2 * n + 1)
]

# dp[i][j][k] :
# i번째 카드 쌍까지 고려해봤을 때
# 지금까지 빨간색 카드는 정확히 j장 뽑았고
# 지금까지 파란색 카드는 정확히 k장 뽑았다 했을 때
# 얻을 수 있는 뽑힌 숫자들의 최대 합
dp = [
    [
        [0 for _ in range(2 * n + 1)]
        for _ in range(2 * n + 1)
    ]
    for _ in range(2 * n + 1)
]


def initialize():
    # 최대를 구하는 문제이므로, 
    # 초기에는 전부 INT_MIN을 넣어줍니다.
    for i in range(2 * n + 1):
        for j in range(2 * n + 1):
            for k in range(2 * n + 1):
                dp[i][j][k] = INT_MIN
    
    # 0번째 카드 쌍까지 고려해봤을 때에는
    # 아직 고른 카드가 없기 때문에
    # 빨간색, 파란색 카드 모두 0개를 뽑은 상황에
    # 뽑은 숫자들의 합은 0입니다.
    dp[0][0][0] = 0


for i in range(1, 2 * n + 1):
    red[i], blue[i] = tuple(map(int, input().split()))
    
initialize()

for i in range(1, 2 * n + 1):
    # i개의 카드 쌍에 대해 전부 카드 선택을 완료했을 때
    # 지금까지 뽑은 빨간색 카드 수가 j이고
    # 지금까지 뽑은 파란색 카드 수가 k였을 때
    # 가능한 선택한 카드 숫자의 최대합을 계산합니다.

    # 이러한 상황을 만들기 위한 선택지는 크게 2가지 입니다.
    for j in range(i + 1):
        for k in range(i + 1):
            # Case 1
            # i번째 카드 쌍에서 빨간색 카드를 선택하여
            # 최종적으로 빨간색이 j개, 파란색이 k개가 된 경우입니다.
            # 따라서 i - 1번째 카드 쌍 까지는 빨간색을 j - 1개, 파란색을 k개 뽑았어야 비로소
            # i번째에 빨간색 카드를 선택하게 되므로서 빨간색이 j개, 파란색이 k개가 될 수 있습니다.
            # 이 경우 dp[i - 1][j - 1][k] 에 i번째 카드 쌍 중 빨간색 카드에 적혀있는 숫자인
            # red[i]를 더한 것이 한 가지 경우가 됩니다.
            # 당연히 j가 0보다 커야지만이 만들어질 수 있는 경우입니다.
            if j > 0:
                dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j - 1][k] + red[i])

            # Case 2
            # i번째 카드 쌍에서 파란색 카드를 선택하여
            # 최종적으로 빨간색이 j개, 파란색이 k개가 된 경우입니다.
            # 따라서 i - 1번째 카드 쌍 까지는 빨간색을 j개, 파란색을 k - 1개 뽑았어야 비로소
            # i번째에 파란색 카드를 선택하게 되므로서 빨간색이 j개, 파란색이 k개가 될 수 있습니다.
            # 이 경우 dp[i - 1][j][k - 1] 에 i번째 카드 쌍 중 파란색 카드에 적혀있는 숫자인
            # blue[i]를 더한 것이 한 가지 경우가 됩니다.
            # 당연히 k가 0보다 커야지만이 만들어질 수 있는 경우입니다.
            if k > 0:
                dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j][k - 1] + blue[i])

# 총 2 * n개의 카드 쌍에 대해 전부 카드 선택을 완료했을 때
# 빨간색 카드, 파란색 카드 모두 각각 n개씩 뽑았다 했을 때
# 가능한 최대 합에 해당하는 dp 값을 출력합니다.
print(dp[2 * n][n][n])


import sys

INT_MIN = -sys.maxsize

# 변수 선언 및 입력
n = int(input())
red = [
    0
    for _ in range(2 * n + 1)
]
blue = [
    0
    for _ in range(2 * n + 1)
]

# dp[i][j] :
# i번째 카드 쌍까지 고려해봤을 때
# 지금까지 빨간색 카드를 정확히 j장 뽑았다 했을 때
# 얻을 수 있는 뽑힌 숫자들의 최대 합
dp = [
    [0 for _ in range(2 * n + 1)]
    for _ in range(2 * n + 1)
]


def initialize():
    # 최대를 구하는 문제이므로, 
    # 초기에는 전부 INT_MIN을 넣어줍니다.
    for i in range(2 * n + 1):
        for j in range(2 * n + 1):
            dp[i][j] = INT_MIN
    
    # 0번째 카드 쌍까지 고려해봤을 때에는
    # 아직 고른 카드가 없기 때문에
    # 빨간색 카드를 0개를 뽑은 상황에
    # 뽑은 숫자들의 합은 0입니다.
    dp[0][0] = 0


for i in range(1, 2 * n + 1):
    red[i], blue[i] = tuple(map(int, input().split()))
    
initialize()

for i in range(1, 2 * n + 1):
    # i개의 카드 쌍에 대해 전부 카드 선택을 완료했을 때
    # 지금까지 뽑은 빨간색 카드 수가 j일 때
    # 가능한 선택한 카드 숫자의 최대합을 계산합니다.

    # 이러한 상황을 만들기 위한 선택지는 크게 2가지 입니다.
    for j in range(i + 1):
        # Case 1
        # i번째 카드 쌍에서 빨간색 카드를 선택하여
        # 최종적으로 빨간색이 j개가 된 경우입니다.
        # 따라서 i - 1번째 카드 쌍 까지는 빨간색을 j - 1개 뽑았어야 비로소
        # i번째에 빨간색 카드를 선택하게 되므로서 빨간색이 j개가 될 수 있습니다.
        # 이 경우 dp[i - 1][j - 1] 에 i번째 카드 쌍 중 빨간색 카드에 적혀있는 숫자인
        # red[i]를 더한 것이 한 가지 경우가 됩니다.
        # 당연히 j가 0보다 커야지만이 만들어질 수 있는 경우입니다.
        if j > 0:
            dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + red[i])

        # Case 2
        # i번째 카드 쌍에서 파란색 카드를 선택하여
        # 최종적으로 빨간색이 j개가 된 경우입니다.
        # 따라서 i - 1번째 카드 쌍 까지는 빨간색을 j개 뽑았어야
        # i번째에 파란색 카드를 선택하게 되므로서 빨간색이 그대로 j개가 될 수 있습니다.
        # 이 경우 dp[i - 1][j] 에 i번째 카드 쌍 중 파란색 카드에 적혀있는 숫자인
        # blue[i]를 더한 것이 한 가지 경우가 됩니다.
        # 당연히 i - j가 0보다 커야지만이 만들어질 수 있는 경우입니다.
        if i - j > 0:
            dp[i][j] = max(dp[i][j], dp[i - 1][j] + blue[i])

# 총 2 * n개의 카드 쌍에 대해 전부 카드 선택을 완료했을 때
# 빨간색 카드를 n개씩 뽑았다 했을 때
# 가능한 최대 합에 해당하는 dp 값을 출력합니다.
print(dp[2 * n][n])





# 변수 선언 및 입력
n = int(input())
cards = [
    tuple(map(int, input().split()))
    for _ in range(2 * n)
]


# red - blue값을 내림차순으로 정렬합니다.
def cmp(card):
    red, blue = card
    return -(red - blue)


# red - blue 값을 내림차순으로 정렬합니다.
# 내림차순으로 정렬을 하게 되면
# 앞에 있는 n개의 카드에서는 빨간색 카드를,
# 뒤에 있는 n개의 카드에서는 파란색 카드를 선택하는 것이 항상 좋습니다.
cards.sort(key = cmp)

max_sum = 0

# 앞 n개에서는 빨간색 카드를 선택합니다.
max_sum += sum([
    red
    for red, _ in cards[:n]
])

# 뒤 n개에서는 파란색 카드를 선택합니다.
max_sum += sum([
    blue
    for _, blue in cards[n:]
])

print(max_sum)





import sys

INT_MIN = -sys.maxsize
    
# 변수 선언 및 입력
n, m = tuple(map(int, input().split()))
s = [
    0
    for _ in range(n + 1)
]
e = [
    0
    for _ in range(n + 1)
]
v = [
    0
    for _ in range(n + 1)
]

# dp[i][j] :
# i번째 날까지 입을 옷을 전부 결정했고
# 마지막 날에 입은 옷이 j번 옷이라 했을 때,
# 얻을 수 있는 최대 만족도
dp = [
    [0 for _ in range(n + 1)]
    for _ in range(m + 1)
]


def initialize():
    # 최댓값을 구하는 문제이므로, 
    # 초기에는 전부 INT_MIN을 넣어줍니다.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = INT_MIN
    
    # 첫 날에 옷을 입는 경우를 초기 조건으로 설정합니다.
    # 첫 번째 날에 입을 수 있는 옷들에 대해서만 가능하며,
    # j번째 옷을 첫 번째 날에 입는다면
    # 위치 1까지 고려했을 때, 마지막 날에 입은 옷이 j번 옷이 되고,
    # 만족도는 화려함의 차이로 결정되므로 초기 만족도 값은 0 이므로 
    # dp[1][j] = 0이 초기 조건이 됩니다. 
    
    for j in range(1, n + 1):
        if s[j] == 1:
            dp[1][j] = 0


for i in range(1, n + 1):
    s[i], e[i], v[i] = tuple(map(int, input().split()))

initialize()

for i in range(2, m + 1):
    # i번째 날까지 입을 옷을 전부 결정했고
    # 마지막 날에 입은 옷이 j번 옷이라 했을 때,
    # 얻을 수 있는 최대 만족도를 계산합니다.

    for j in range(1, n + 1):
        for k in range(1, n + 1):
            # i - 1번째 날에 k번 옷을 입은 경우를 고려해봅니다.
            # 단, k번 옷이 i - 1번째 날에 입을 수 있었어야 하고
            # j번 옷이 i번째 날에 입을 수 있는 경우에만 고려해볼 수 있습니다.
            # 이 상황에서의 최대 만족도를 의미하는 dp[i - 1][k] 값에
            # 새롭게 얻게 되는 만족도는 두 옷의 화려함의 차이이므로
            # |v[j] - v[k]|를 더한 값이 하나의 선택지가 될 수 있습니다.
            
            if s[k] <= i - 1 and i - 1 <= e[k] and s[j] <= i and i <= e[j]:
                dp[i][j] = max(dp[i][j], dp[i - 1][k] + abs(v[j] - v[k]))

# m번째 날짜까지 전부 입을 옷을 결정했을 때,
# 마지막 날에 입은 옷이 j번 옷인 경우 중
# 가장 높은 만족도를 얻을 수 있는 경우를 선택합니다.

ans = max(dp[m][1:n + 1])

print(ans)



import sys

INT_MIN = -sys.maxsize

# 변수 선언 및 입력
n = int(input())
a = [
    0
    for _ in range(n + 1)
]

# prefix_sum[i] : 1번째부터 i번째까지 
#                 a배열 원소의 합을 저장하고 있습니다. 
prefix_sum = [
    0
    for _ in range(n + 1)
]


# 누적합 배열에 적절한 값을 채워줍니다.
def preprocess():
    prefix_sum[1] = a[1]
    
    for i in range(2, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + a[i]


# 배열 a의 i번째 원소부터 j번째 원소까지의 합을 반환합니다.
def sum_in_range(i, j):
    return prefix_sum[j] - prefix_sum[i] + a[i]


given_seq = list(map(int, input().split()))
a[1:] = given_seq[:]

preprocess()

# 최댓값을 구해야 하는 문제이므로
# 초기값을 INT_MIN으로 설정합니다.
ans = INT_MIN

# 모든 연속 부분수열 쌍에 대해 그들의 합 중
# 최댓값을 계산합니다.
# 연속 부분수열이 i로 시작해서 j로 끝나는 
# (i, j)쌍을 전부 조사해야 합니다.
for i in range(1, n + 1):
    for j in range(i, n + 1):
        ans = max(ans, sum_in_range(i, j))

print(ans)



import sys

INT_MIN = -sys.maxsize

# 변수 선언 및 입력
n = int(input())
a = [
    0
    for _ in range(n + 1)
]

# prefix_sum[i] : 1번째부터 i번째까지 
#                 a배열 원소의 합을 저장하고 있습니다. 
prefix_sum = [
    0
    for _ in range(n + 1)
]


# 누적합 배열에 적절한 값을 채워줍니다.
def preprocess():
    prefix_sum[1] = a[1]
    
    for i in range(2, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + a[i]


given_seq = list(map(int, input().split()))
a[1:] = given_seq[:]

preprocess()

# 최댓값을 구해야 하는 문제이므로
# 초기값을 INT_MIN으로 설정합니다.
ans = INT_MIN

# 모든 연속 부분수열 쌍에 대해 그들의 합 중
# 최댓값을 계산합니다.
# 이를 0 <= index1 < index2 <= n 를 만족하는 두 위치
# index1, index2를 골라 누적합의 차가 최대가 되도록 하는
# 문제로 해결이 가능하므로, 
# index2를 먼저 고정하고 index1은 index2 앞의 원소들 중
# 가장 작은 원소를 골라야 차이를 최대화 할 수 있으므로
# index2가 바뀜에따라 계속 최솟값을 O(1) 시간에 갱신하면서
# 나아갈 수 있습니다.
index1 = 0
for index2 in range(1, n + 1):
    ans = max(ans, prefix_sum[index2] - prefix_sum[index1])
    if prefix_sum[index1] > prefix_sum[index2]:
        index1 = index2

print(ans)




import sys

INT_MIN =  -sys.maxsize

# 변수 선언 및 입력
n = int(input())
a = [
    0
    for _ in range(n + 1)
]

# dp[i] : 선택한 연속 부분 수열의 마지막 원소의 위치가 i라 했을 때,
#         얻을 수 있는 최대 합
dp = [
    0
    for _ in range(n + 1)
]


def initialize():
    # 최댓값을 구하는 문제이므로, 
    # 초기에는 전부 INT_MIN을 넣어줍니다.
    for i in range(1, n + 1):
        dp[i] = INT_MIN
    
    # 첫 번째 원소를 연속 부분 수열의 원소로 사용하는 경우를
    # 초기 조건으로 설정합니다.
    # 이때는, 이 원소만 연속 부분 수열에 속하게 되므로
    # dp[1] = a[1]이 됩니다.
    dp[1] = a[1];


given_seq = list(map(int, input().split()))
a[1:] = given_seq[:]

initialize()

# 선택한 연속 부분 수열의 마지막 원소의 위치가 i라 했을 때,
# 얻을 수 있는 최대 합을 계산합니다.
for i in range(2, n + 1):
    # 이렇 상황을 만들기 위한 선택지는 크게 2가지 입니다.
        
    # Case 1
    # 그 이전 연속 부분 수열에 i번째 원소를
    # 더 추가하는 경우입니다.
    # 추가를 위해서는 정확히 i - 1번째로 끝나는 연속 부분 수열
    # 중 최대 합이 필요하므로, dp[i - 1] + a[i]가
    # 하나의 선택지가 됩니다.

    # Case 2
    # i 번째 원소부터 연속 부분 수열을 만들기 시작하는 경우입니다.
    # 이 경우에는 원소가 a[i] 하나 뿐이므로, a[i]가 
    # 또 다른 선택지가 됩니다.

    dp[i] = max(dp[i - 1] + a[i], a[i])

ans = max(dp[1:n + 1])
print(ans)


import sys

INT_MIN = -sys.maxsize

# 변수 선언 및 입력
n = int(input())
a = [
    0
    for _ in range(n + 1)
]


given_seq = list(map(int, input().split()))
a[1:] = given_seq[:]

# 최댓값을 구해야 하는 문제이므로
# 초기값을 INT_MIN으로 설정합니다.
ans = INT_MIN

# 현재 연속 부분 수열 내 원소의 합을
# 저장합니다.
sum_of_nums = 0;

for i in range(1, n + 1):
    # 만약 현재 연속 부분 수열 내 원소의 합이
    # 0보다 작아진다면, 지금부터 새로운
    # 연속 부분 수열을 만드는 것이 더 유리합니다.
    if sum_of_nums < 0:
        sum_of_nums = a[i]
    
    # 그렇지 않다면 기존 연속 부분 수열에 
    # 현재 원소를 추가하는 것이 더 좋습니다.
    else:
        sum_of_nums += a[i]
    
    ans = max(ans, sum_of_nums)

print(ans)


import sys

INT_MIN = -sys.maxsize

# 변수 선언 및 입력
n = int(input())
a = [
    0
    for _ in range(n + 1)
]

# [start_idx, end_idx] 구간 내에서의
# 최대 연속 부분합을 계산하여 반환합니다.
def find_max(start_idx, end_idx):
    # 원소가 하나일 때에는 그 원소를 고르는 것 만이
    # 연속 부분 수열을 만들 수 있는 방법이므로
    # 해당 원소값을 반환합니다.
    if start_idx == end_idx:
        return a[start_idx]
    
    # 최댓값을 구해야 하는 문제이므로
    # 초기값을 INT_MIN으로 설정합니다.
    max_sum = INT_MIN

    # 가운데를 기준으로 divide & conquer를 진행합니다.
    mid = (start_idx + end_idx) // 2

    # Case 1 : 
    # [start_idx, mid] 사이에서 가능한 최대 연속 부분 합을 계산합니다.
    max_sum = max(max_sum, find_max(start_idx, mid))
    
    # Case 2 :
    # [mid + 1, end_idx] 사이에서 가능한 최대 연속 부분 합을 계산합니다.
    max_sum = max(max_sum, find_max(mid + 1, end_idx))

    # Case 3 :
    # mid, mid + 1번째 원소를 모두 연속 부분 수열에 포함시키는 경우입니다.
    # 이 경우의 최대 연속 부분 합은
    # mid원소를 끝으로 하는 최대 연속 부분 수열과
    # mid + 1번째 원소를 시작으로 하는 최대 연속 부분 수열을 합한 경우입니다.

    left_max_sum, sum_of_nums = -sys.maxsize, 0
    for i in range(mid, start_idx - 1, -1):
        sum_of_nums += a[i]
        left_max_sum = max(left_max_sum, sum_of_nums)
        
    right_max_sum, sum_of_nums = -sys.maxsize, 0
    for i in range(mid + 1, end_idx + 1):
        sum_of_nums += a[i]
        right_max_sum = max(right_max_sum, sum_of_nums)

    max_sum = max(max_sum, left_max_sum + right_max_sum)
    
    # 3가지 경우 중 최대를 반환합니다.
    return max_sum;


given_seq = list(map(int, input().split()))
a[1:] = given_seq[:]

print(find_max(1, n))



