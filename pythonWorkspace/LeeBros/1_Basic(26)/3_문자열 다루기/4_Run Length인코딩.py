'''
문자열 A가 주어졌을 때 문자열 A에 Run-Length Encoding을 적용한 이후의 결과를 구해보려고 합니다.
Run-Length Encoding이란 간단한 비손실 압축 방식으로, 연속해서 나온 문자와 연속해서 나온 개수로 나타내는 방식입니다.

예를 들어, 문자열 A가 aaabbbbcbb인 경우 순서대로 a가 3번, b가 4번, c가 1번 그리고 b가 2번 나왔으므로 Run-Length Encoding을 적용하게 되면 a3b4c1b2가 됩니다.
문자열 A가 주어졌을 때, Run-Length Encoding을 적용한 이후의 결과를 출력하는 프로그램을 작성해보세요.

입력 형식
첫 번째 줄에 문자열 A가 주어집니다. 문자열 A는 소문자 알파벳으로만 이루어져 있습니다.
1 ≤ 문자열 A의 길이 ≤ 1,000

출력 형식
첫 번째 줄에는 문자열 A에 Run-Length Encoding을 적용한 이후의 길이를 출력합니다.
두 번째 줄에는 문자열 A에 Run-Length Encoding을 적용한 이후의 결과를 출력합니다.

입출력 예제
예제1
입력:
aaabbbbcbb

출력: 
8
a3b4c1b2

예제2
입력:
aaaaaaaaaabb

출력: 
5
a10b2
'''
arr = [i for i in input().strip()]
ans = []

if len(arr) == 1:
    ans.append(arr[0])
    ans.append(len(arr)) 

while True:

    compare = arr[0]
    del arr[0]

    if len(arr) == 0:
        break
    
    cnt = 1
    
    for i in arr:
        if compare != i:
            ans.append(compare)
            #print(1)
            ans.append(cnt)
            #print(2)
            compare = i
            cnt = 1
    
        else:
            cnt += 1
            compare = i
    
    if compare == arr[-1]:
        ans.append(compare)
        #print(3)
        ans.append(cnt) 
        #print(4)  
        break


length = len(ans)

for k in range(1,len(ans),2):
    if ans[k] >= 10 and ans[k] < 100:
        length += 1
    
    if ans[k] >= 100 and ans[k] < 1000:
        length += 2
    
    if ans[k] == 1000:
        length += 2

print(length, "" .join(map(str, ans)), sep="\n")
