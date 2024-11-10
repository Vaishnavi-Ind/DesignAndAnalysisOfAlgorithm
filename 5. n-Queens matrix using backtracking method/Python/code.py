def is_safe(board, row, col, n):
    for i in range(row):
        if board[i][col] == 1:
            return False
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    for i, j in zip(range(row, -1, -1), range(col, n)):
        if board[i][j] == 1:
            return False
    return True

def solve_n_queens(board, row, n):
    if row == n:
        return True
    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1
            if solve_n_queens(board, row + 1, n):
                return True
            board[row][col] = 0
    return False

def n_queens_with_first_queen(n, first_row, first_col):
    board = [[0] * n for _ in range(n)]
    board[first_row][first_col] = 1
    if not solve_n_queens(board, first_row + 1, n):
        print("No solution exists")
        return None
    return board

def print_board(board):
    for row in board:
        print(" ".join("Q" if cell else "." for cell in row))

n = 8
first_row, first_col = 0, 0
board = n_queens_with_first_queen(n, first_row, first_col)
if board:
    print_board(board)
