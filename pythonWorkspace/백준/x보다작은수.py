# import sys
# input = sys.stdin.readline

### data = [int(d) if d.strip().isdigit() else d.strip() for d in input().split(",")]
### print(data)

# n, x = map(int, input().split(" "))
# nlist = [int(i) for i in input().strip().split()]

# for k in nlist:
#     if k < x:
#         print(k, end=" ")



# import sys
# input = sys.stdin.readline
# nlist = []
# n, x = map(int, input().split(" "))
# nlist = [int(i) for i in input().strip().split()]

# for k in nlist:
#     if k < x:
#         print(k, end=" ")

a,b = map(int,input().split())
score = [int(x) for x in input().split() if int(x)<b] #if 만 있을 땐 뒤에, if else가 있을 땐 앞에 쓴다.
print(" ".join(map(str, score))) #join은 list의 '문자열'을 특정 구분자로 출력 또는 문자열을 리스트로 반환

