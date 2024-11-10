public class NQueens {

    public static boolean isSafe(int[][] board, int row, int col, int n) {
        for (int i = 0; i < row; i++)
            if (board[i][col] == 1) return false;

        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--)
            if (board[i][j] == 1) return false;

        for (int i = row, j = col; i >= 0 && j < n; i--, j++)
            if (board[i][j] == 1) return false;

        return true;
    }

    public static boolean solveNQueens(int[][] board, int row, int n) {
        if (row == n) return true;

        for (int col = 0; col < n; col++) {
            if (isSafe(board, row, col, n)) {
                board[row][col] = 1;
                if (solveNQueens(board, row + 1, n)) return true;
                board[row][col] = 0;
            }
        }
        return false;
    }

    public static int[][] nQueensWithFirstQueen(int n, int firstRow, int firstCol) {
        int[][] board = new int[n][n];
        board[firstRow][firstCol] = 1;
        if (!solveNQueens(board, firstRow + 1, n)) {
            System.out.println("No solution exists");
            return null;
        }
        return board;
    }

    public static void printBoard(int[][] board) {
        for (int[] row : board) {
            for (int cell : row) {
                System.out.print(cell == 1 ? "Q " : ". ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        int n = 8;
        int firstRow = 0, firstCol = 0;
        int[][] board = nQueensWithFirstQueen(n, firstRow, firstCol);
        if (board != null) printBoard(board);
    }
}
