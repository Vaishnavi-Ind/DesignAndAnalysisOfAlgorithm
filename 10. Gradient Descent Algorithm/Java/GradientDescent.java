public class GradientDescent {

    public static double function(double x) {
        return Math.pow(x + 3, 2);
    }

    public static double gradient(double x) {
        return 2 * (x + 3);
    }

    public static double gradientDescent(double startingX, double learningRate, int iterations) {
        double x = startingX;
        for (int i = 0; i < iterations; i++) {
            double grad = gradient(x);
            x = x - learningRate * grad;
            System.out.printf("Iteration %d: x = %.6f, f(x) = %.6f%n", i + 1, x, function(x));
        }
        return x;
    }

    public static void main(String[] args) {
        double startingX = 2;
        double learningRate = 0.1;
        int iterations = 50;

        double localMinimum = gradientDescent(startingX, learningRate, iterations);
        System.out.printf("%nLocal minimum occurs at x = %.6f, f(x) = %.6f%n", localMinimum, function(localMinimum));
    }
}
