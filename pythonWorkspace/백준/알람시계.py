h,m = map(int, input().split())
if m-45<0 :
    h-=1
    m = 60-abs(m-45)
else:
    m = m-45

if h < 0:
    h = 24 - abs(h)

print(h,m,sep=" ")