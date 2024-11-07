def fib(n):
    # Function 
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

# Main function 
if __name__ == "__main__":
    N = 10
    for i in range(N):
        print(fib(i), end=" ")
