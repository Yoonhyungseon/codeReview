import sys
input = sys.stdin.readline

a = int(input())
origine = a
cnt = 1

while(True):
    x = a//10
    y = a%10

    X = y*10    
    Y = (x+y)%10

    if origine == X+Y:
        print(cnt)
        break
    else:
        cnt += 1
        a = X+Y
