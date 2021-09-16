def tree(n):
    for i in range(1,n,2):
        print((" "*((n-1-i)//2)) + ("*"*i))

    print(" "*((n-2)//2) + "|")
    return 0

tree(10)
