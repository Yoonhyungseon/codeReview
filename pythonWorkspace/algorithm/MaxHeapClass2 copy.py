class MaxHeapClass:

    def __init__(self, b):
        self.a = list()
        self.a.append(b)
        self.size = len(self.a)
        
    def __str__(self):
        return str(self.a)
    
    def heapsort(self):
        heapify(self.a, self.size)
        end = self.size-1
        while(end > 0):
            self.swap(a, 0, end)
            self.siftdown(a, 0, end)
            end -= 1

    def swap(self, a,i,j):
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp

    def siftdown(self, a, i, size):
        l = 2*i+1
        r = 2*i+2
        largest = i
        if l <= size-1 and a[l] > a[i]:
            largest = l
        if r <= size-1 and a[r] > a[largest]:
            largest = r
        if largest != i:
            self.swap(a, i, largest)
            self.siftdown(a, largest, size)

    def heapify(self, a, size):
        p = (size//2)-1
        while p>=0:
            self.siftdown(a, p, size)
            p -= 1

    def buildHeap(self):
        self.heapify(self.a, self.size)

   