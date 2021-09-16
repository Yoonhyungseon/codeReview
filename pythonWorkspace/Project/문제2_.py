import itertools

list1=[5,6,7,8,9]
list2=[1,2,3,4]
list3=[2,4,6,8]

list_ = [list1, list2, list3]

def fnc(arr):
    cnt = 0

    arr = [i for i in range(len(list_))]

    def permute(arr):
        result = [arr[:]]
        temp = [0] * len(arr)
        i = 0
        while i < len(arr):
            if temp[i] < i:
                if i % 2 == 0:
                    arr[0], arr[i] = arr[i], arr[0]
                else:
                    arr[temp[i]], arr[i] = arr[i], arr[temp[i]]
                result.append(arr[:])
                temp[i] += 1
                i = 0
            else:
                temp[i] = 0
                i += 1
        return result

    for i in permute(arr):
        ans = list(itertools.product(*[list_[i[0]],list_[i[1]],list_[i[2]]]))
        cnt += len(ans)

    print("경우의 수 : {0}가지" .format(cnt))

fnc(list_)
