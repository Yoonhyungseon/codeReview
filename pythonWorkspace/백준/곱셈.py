A = int(input())
B = int(input())

b1 = B//100
b2 = (B - b1*100)//10
b3 = B - b1*100 - b2*10

an1 = A*b3
an2 = A*b2
an3 = A*b1
an_ = an1 + an2*10 + an3*100

print(an1, an2, an3, an_, sep="\n")