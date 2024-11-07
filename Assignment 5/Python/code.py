import random
import time

def deterministic_quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        deterministic_quick_sort(arr, low, pi - 1)
        deterministic_quick_sort(arr, pi + 1, high)

def randomized_quick_sort(arr, low, high):
    if low < high:
        pi = randomized_partition(arr, low, high)
        randomized_quick_sort(arr, low, pi - 1)
        randomized_quick_sort(arr, pi + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            swap(arr, i, j)
    swap(arr, i + 1, high)
    return i + 1

def randomized_partition(arr, low, high):
    random_index = random.randint(low, high)
    swap(arr, random_index, high)
    return partition(arr, low, high)

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def generate_random_array(size, range_limit):
    return [random.randint(0, range_limit - 1) for _ in range(size)]

def copy_array(arr):
    return arr[:]

def main():
    size = 10000
    range_limit = 10000

    original_array = generate_random_array(size, range_limit)

    arr1 = copy_array(original_array)
    start_time = time.time_ns()
    deterministic_quick_sort(arr1, 0, len(arr1) - 1)
    end_time = time.time_ns()
    print("Deterministic QuickSort time:", end_time - start_time, "ns")

    arr2 = copy_array(original_array)
    start_time = time.time_ns()
    randomized_quick_sort(arr2, 0, len(arr2) - 1)
    end_time = time.time_ns()
    print("Randomized QuickSort time:", end_time - start_time, "ns")

if __name__ == "__main__":
    main()
