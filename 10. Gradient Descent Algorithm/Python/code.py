def function(x):
    return (x + 3) ** 2

def gradient(x):
    return 2 * (x + 3)

def gradient_descent(starting_x, learning_rate, iterations):
    x = starting_x
    for i in range(iterations):
        grad = gradient(x)
        x = x - learning_rate * grad
        print(f"Iteration {i+1}: x = {x}, f(x) = {function(x)}")
    return x

starting_x = 2
learning_rate = 0.1
iterations = 50

local_minimum = gradient_descent(starting_x, learning_rate, iterations)
print(f"\nLocal minimum occurs at x = {local_minimum}, f(x) = {function(local_minimum)}")
