import numpy as np

class TicTacToe:
    def __init__(self, board_size = 3) -> None:
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype = int)
        self.current_player = 1
        self.players = {
            1: 'X',
            -1: 'Y'
        }
        self.game_over = False

    def is_valid_move(self, row, column):
        return 0 <= row < self.board_size and 0 <= column < self.board_size and self.board[row][column] == 0
    
    def make_move(self, row, column):
        if self.is_valid_move(row, column):
            self.board[row][column] = self.current_player
            return True
        else:
            return False
        
    def switch_player(self):
        self.current_player *= -1

        
    def check_win(self):
        for i in range(self.board_size):
            if np.abs(self.board[i].sum()) == self.board_size or np.abs(self.board[:, i].sum()) == self.board_size:
                return True
        if np.abs(self.board.trace()) == self.board_size or np.abs(np.fliplr(self.board).trace()) == self.board_size:
            return True
        return False
    
    def check_draw(self):
        return len(np.where(self.board == 0)[0]) == 0
    
    def print_board(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        for row in self.board:
            print(" | ".join(symbols[cell] for cell in row))
            print("-" * (self.board_size * 4 - 1))

    def start_game(self):
        print(f"Let's play Tic-Tac-Toe on a {self.board_size}x{self.board_size} board!")
        self.print_board()

        while not self.game_over:
            while True:
                try:
                    row, column = map(int, input(f"Player {self.players[self.current_player]}'s turn. Enter row and column (0-{self.board_size - 1}): ").split())
                    if self.make_move(row, column):
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Please enter two integers.")

            self.print_board()

            if self.check_win():
                print(f"Player {self.players[self.current_player]} wins!")
                self.game_over = True
            elif self.check_draw():
                print("It's a draw!")
                self.game_over = True
            self.switch_player()


tic_tac_toe = TicTacToe()
tic_tac_toe.start_game()