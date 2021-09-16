import random

def compare_answer(x, req):
    if x < req:
        return 1
    elif x == req:
        return 0
    elif x > req:
        return -1

x = random.randint(1,100)

for i in range(10):
    req = int(input("두자리 정수 입력 : "))
    if compare_answer(x, req) > 0: print("아닙니다. 더 작은숫자입니다") 
    if compare_answer(x, req) == 0: print("정답입니다. {0}번만에 맞췄습니다.".format(i+1))
    if compare_answer(x, req) < 0: print("아닙니다. 더 큰숫자입니다") 

    if i==9: print("게임 끝!!!")





     