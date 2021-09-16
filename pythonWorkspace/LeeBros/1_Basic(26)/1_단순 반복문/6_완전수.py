'''
완전수란 자기 자신을 제외한 모든 양의 약수들을 더했을 때 자기 자신이 되는 수 입니다.

예를 들어 6 = 1 + 2 + 3 이므로 6은 완전수 입니다.

주어지는 start에서 end 이내에 있는 숫자들 중 완전수가 몇 개인지 출력하는 코드를 작성해보세요.

입력 형식
첫 번째 줄에 start, end 값이 각각 공백을 사이에 두고 주어집니다.
1 ≤ start ≤ end ≤ 1,000

출력 형식
start와 end 사이에 있는 서로 다른 완전수의 개수를 출력하는 코드를 작성해보세요.

입출력 예제
예제1
입력:
3 30

출력: 
2

예제 설명
3과 30 사이에 있는 완전수는 6, 28 입니다.
'''

start, end = map(int, input().strip().split())
cnt = 0

for i in range(start,end+1):
    sum_ = 0
    for j in range(1,i):
        if i%j == 0:
            sum_ += j
    if sum_ == i:
        cnt += 1

print(cnt)