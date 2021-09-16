'''
두 개의 단어가 입력으로 주어질 때 두 단어에 속하는 문자들의 순서를 바꾸어 
동일한 단어로 만들 수 있는지 여부를 출력하는 코드를 작성해보세요.

입력 형식
첫 번째 줄에는 첫 번째 단어가 주어지고, 두 번째 줄에는 두 번째 단어가 주어집니다.
1 ≤ 단어의 길이(n) ≤ 100,000

각 단어에 속하는 문자는 총 128개로 구성되어 있는 아스키 코드로 나타낼 수 있습니다.

출력 형식
두 단어에 속하는 문자들의 순서를 바꾸어 동일한 단어로 만들 수 있으면 “Yes”를 출력하고, 
만들 수 없으면 “No”를 출력합니다.

입출력 예제
예제1
입력:
aba
aab

출력: 
Yes

예제2
입력:
abab
abba

출력: 
Yes

예제3
입력:
aaa
aa

출력: 
No
'''

str1 = input()
str2 = input()

print("Yes" if sorted(str1) == sorted(str2) else "No")

##
##
##

import sys
input = sys.stdin.readline #공백 포함이기 때문에 strip()써주기

x = input().strip() 
y = input().strip()

alpha = "abcdefghijklmnopqrstuvwxyz"

x_arr = [0 for i in alpha]
y_arr = [0 for i in alpha]

for i in x:
    idx = alpha.find(i)
    x_arr[idx] += 1

for i in y:
    idx = alpha.find(i)
    y_arr[idx] += 1

if x_arr == y_arr:
    print("Yes")
else:
    print("No")





# a = "bcad"
# a = sorted(a, reverse= True) #리스트 뿐만 아니라 문자열도 정렬 가능, sorted는 원래 리스트 건드리지 않고 값만 반환
# print(a) 

# a = "bcad"
# print(list(reversed(a))) #거꾸로 출력 (내림차순 아님, list로 반환해야함)


# a = [4,1,3,2]
# a.sort(reverse=True)  # 원래 리스트를 바꿔버림, 내림차순 정렬
# print(a)

# a = [4,1,3,2]
# a.reverse() #거꾸로 출력 (내림차순 아님)
# print(a)
