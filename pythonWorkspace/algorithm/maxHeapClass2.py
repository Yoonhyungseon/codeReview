class MaxHeap:
    
    def __init__(self):
        self.data = [None]


    def insert(self, item):
        self.data.append(item)
        i = len(self.data) - 1
        while i > 1:
            if self.data[i] > self.data[(i // 2)]:
                self.data[i], self.data[(i // 2)] = self.data[(i // 2)], self.data[i]
                i = i // 2
            else: break
 
    def remove(self):
        if len(self.data) > 1:
            self.data[1], self.data[-1] = self.data[-1], self.data[1]
            data = self.data.pop(-1)
            self.maxHeapify(1)
        else:
            data = None
        return data

    def maxHeapify(self, i):
        left = 2 * i
        right = (2 * i) + 1
        smallest = i
        if left < len(self.data) and self.data[i] < self.data[left]:
            smallest = left
        if right < len(self.data) and self.data[i] > self.data[right]:            
            smallest = right
        if smallest != i:
            self.data[i], self.data[smallest] = self.data[smallest], self.data[i]
            self.maxHeapify(smallest)
