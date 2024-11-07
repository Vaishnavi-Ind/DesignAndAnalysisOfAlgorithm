class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value
        self.ratio = value / weight

def get_max_value(items, capacity):
    items.sort(key=lambda item: item.ratio, reverse=True)
    
    total_value = 0

    for item in items:
        if capacity == 0:
            break

        if capacity >= item.weight:
            capacity -= item.weight
            total_value += item.value
        else:
            total_value += item.value * (capacity / item.weight)
            capacity = 0

    return total_value

if __name__ == "__main__":
    items = [
        Item(10, 60),
        Item(20, 100),
        Item(30, 120)
    ]
    
    capacity = 50
    max_value = get_max_value(items, capacity)
    print("Maximum value in knapsack:", max_value)
