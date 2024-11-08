def fibonacci(n):
    num1, num2 = 0, 1
    for _ in range(n):
        print(num1, end=" ")
        num1, num2 = num2, num1 + num2

# Main Function
if __name__ == "__main__":
    N = 10
    fibonacci(N)
