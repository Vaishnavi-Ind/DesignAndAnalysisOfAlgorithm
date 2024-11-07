import java.io.*;
class FibonacciRecursive {
 // Function to print the Fibonacci series
 static int fib(int n) {
 if (n <= 1)
 return n;
 return fib(n - 1) + fib(n - 2);
 }
 // Main Function
 public static void main(String args[]) {
 int N = 10;
 for (int i = 0; i < N; i++) {
 System.out.print(fib(i) + " ");
 }
 }
}