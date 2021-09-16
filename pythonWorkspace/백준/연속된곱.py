def solution(n):

    na=[]
    for k in range(1,100):
        answer = 0 # 구하고자하는 값 answer을 0으로 초기화 
        
        for i in range(1, k + 1): # 어떤 자연수 부터 시작할지 결정하는 반복문
            sum = 1 #총합을 초기화, 초기화하는 위치가 중요하다.
            
            for j in range(i, k + 1): #i부터 시작해서 어디까지 더할지 결정하는 반복문
                sum = sum * j  #sum에 j값을 계속 더 해줘서 총합을 구해준다.

                if sum == k:  # 그 총합이 구하고자하는 n 값이 되면
                    answer = answer + 1 #answer(연속 되는 자연수들도 몇 번 표현할 수 있는지)를 1올려주고
                    break  #한 번 찾았으면 그 이 후에는 더 나올 수 없으므로 반복문을 깨준다.
 
                if sum > k:
                    break #같은 원리로 수가 더 커져버리면 찾을 가능성이 없으므로 반복문을 깨준다.

        if answer-1 != 0 :
            na.append(k)

    return na[n-1]

print(solution(2))