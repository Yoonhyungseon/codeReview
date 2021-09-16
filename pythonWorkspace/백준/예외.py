import sys
input = sys.stdin.readline

while(True):
    try:
        a, b = map(int, input().split())
        print(a+ b)
    except:
        break
        

# try:
#     실행할 코드
# except (예외이름 as 변수 ):
#     예외 발생시 처리되는 코드
# else:
#     예외 발생하지 않을 때 실행할 코드
# finally:
#     예외 발생 여부와 상관없이 실행할 코드
    