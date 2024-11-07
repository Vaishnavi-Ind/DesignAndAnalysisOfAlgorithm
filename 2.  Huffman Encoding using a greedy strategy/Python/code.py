import heapq

class HuffmanNode:
    def __init__(self, data, char):
        self.data = data
        self.char = char
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.data < other.data

def print_code(root, code):
    if root is None:
        return
    if root.left is None and root.right is None and root.char.isalpha():
        print(f"{root.char}: {code}")
        return
    print_code(root.left, code + "0")
    print_code(root.right, code + "1")

def huffman_coding(charArray, charfreq):
    heap = [HuffmanNode(charfreq[i], charArray[i]) for i in range(len(charArray))]
    heapq.heapify(heap)

    while len(heap) > 1:
        x = heapq.heappop(heap)
        y = heapq.heappop(heap)
   
        f = HuffmanNode(x.data + y.data, '-')
        f.left = x
        f.right = y
        
        heapq.heappush(heap, f)
    
    root = heap[0]
    
    print_code(root, "")

charArray = ['a', 'b', 'c', 'd', 'e', 'f']
charfreq = [5, 9, 12, 13, 16, 45]
huffman_coding(charArray, charfreq)
