class MaxHeapClass:
    
    def __init__(self, data):
        self.heap_array = list()
        self.heap_array.append(data)
        self.buildHeap()
        
    def __str__(self):
        return str(self.heap_array)
    
    def buildHeap(self):
        size = len(self.heap_array)
        p = (size//2)-1
        while p>=0:
            maxHeapify(self.heap_array, p, size)
            p -= 1

    def maxHeapify(self, ar, p, size):
        left_child = 2*p+1
        right_child = 2*p+2
        largest = p

        if left_child <= size-1 and ar[left_child] > ar[largest]:
            largest = left_child
        if right_child <= size-1 and ar[right_child] > ar[largest]:
            largest = right_child
        if largest is not p:
            ar[largest], ar[right_child] = ar[right_child], ar[largest]
            maxHeapify(ar, largest, size)
    
    def heapSort(self, data):
        size = len(data)
        buildHeap()
        end = size-1
        while(end > 0):
            data[0], data[end] = data[end], data[0]
            maxHeapify(data, 0, end)
            end -= 1

    def insert(self, location):
        print("-")

arr = [1,3,2,4,9,7]
MaxHeapClass(arr)

print(arr)
