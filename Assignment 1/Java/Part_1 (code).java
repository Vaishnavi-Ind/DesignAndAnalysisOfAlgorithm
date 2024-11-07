import java.io.*;
class FibonacciNonRecursive {
 // Function to print N Fibonacci Numbers
 static void Fibonacci(int N) {
 int num1 = 0, num2 = 1;
 for (int i = 0; i < N; i++) {
 // Print the number
 System.out.print(num1 + " ");
 // Swap
 int num3 = num2 + num1;
 num1 = num2;
 num2 = num3;
 }
 }
 // Main Function
 public static void main(String args[]) {
 int N = 10;
 Fibonacci(N);
 }
}
