import numpy as np
import random


class TicTacToe:
    def __init__(self, board_size = 3, computer_play = True) -> None:
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype = int)
        self.current_player = 1
        self.curr_player_is_computer = False
        self.players = {
            1: 'X',
            -1: 'O',
        }
        self.computer_play = computer_play
        self.game_over = False
        

    # return array of row,col for empty spots.
    def get_moves(self, state = None):
        if state is None:
            board = np.argwhere(self.board == 0)
        else:
            board = np.argwhere(state == 0)
        return board

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
        self.curr_player_is_computer  = not self.curr_player_is_computer

    def computer_move(self):
        while True:
            row = random.randint(0, self.board_size - 1)
            col = random.randint(0, self.board_size - 1)
            if self.board[row][col] == 0:
                return (row,col)
        
    def check_win(self, board_state = None):
        if board_state is None:
            for i in range(self.board_size):
                if np.abs(self.board[i].sum()) == self.board_size or np.abs(self.board[:, i].sum()) == self.board_size:
                    return True
            if np.abs(self.board.trace()) == self.board_size or np.abs(np.fliplr(self.board).trace()) == self.board_size:
                return True
        else:
            for i in range(self.board_size):
                if np.abs(board_state[i].sum()) == self.board_size or np.abs(board_state[:, i].sum()) == self.board_size:
                    return True
            if np.abs(board_state.trace()) == self.board_size or np.abs(np.fliplr(board_state).trace()) == self.board_size:
                return True
        return False
    
    def check_draw(self):
        return len(np.where(self.board == 0)[0]) == 0
    
    def print_board(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        for row in self.board:
            print(" | ".join(symbols[cell] for cell in row))
            print("-" * (self.board_size * 4 - 1))


    def get_input(self):
        if self.computer_play:
            if self.curr_player_is_computer is False:
                row, column = map(int, input(f"Player {self.players[self.current_player]}'s turn. Enter row and column (0-{self.board_size - 1}): ").split())
            else:
                row, column = self.computer_move()
        else:
            row, column = map(int, input(f"Player {self.players[self.current_player]}'s turn. Enter row and column (0-{self.board_size - 1}): ").split())
        return (row, column)
        

    def start_game(self):
        print(f"Let's play Tic-Tac-Toe on a {self.board_size}x{self.board_size} board!")
        self.print_board()

        while not self.game_over:
            while True:
                try:
                    row, column = self.get_input()
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


class TrainTicTacToe(TicTacToe):
    
    def __init__(self):
        # super().__init__()
        self.x_states = {}
        self.o_states = {}
        self.track_win_count = {
            'X': 0,
            'O': 0,
            'D': 0
        }
        self.learning_rate = 1

    def exploit(self, chance):
        threshold = 0.6
        exploit = None
        if chance >= threshold:
            exploit = False
        else: 
            exploit = True
        return exploit

    def ttt_ex_vs_oh(self):
        """
        Plays a game of Tic-Tac-Toe between 'X' and 'O' using the current board state 
        and Q-value tables (self.q_states and self.o_states). This function implements 
        an exploration-exploitation strategy to balance between the best move based 
        on Q-values (exploitation) and exploring random moves.

        The function iteratively plays moves until a winner is determined or the game 
        ends in a draw. At each turn, it uses a random chance value to decide between 
        exploitation (finding the highest reward move) and exploration (taking a random 
        move).

        Args:
            None

        Returns:
            tuple: A tuple containing:
                * winner (str): The winner of the game ('X', 'O', or 'D' for draw).
                * game_history (list): A list of 2D board states representing the complete 
                                        game history from start to end.
        """
        super().__init__()
        game = []
        play = True
        winner = None
        status = 0
        while play:
            game.append(self.board.copy())
            chance = round(random.random(), 2)
            if(self.exploit(chance) == True):
                player = self.players[self.current_player]
                row, col = self.find_highest_reward_move(player)
            else:
                row, col = self.random_move()
            
            if self.players[self.current_player] == 'X':
                self.board[row][col] = 1
            else:
                self.board[row][col] = -1
            win = self.check_win()
            draw = self.check_draw()
            if win:
                play = False
                status = 100
                if self.players[self.current_player] == 'X':
                    winner = 'X'
                else:
                    winner = 'O'
                self.update_track_win_count(status)
            elif draw:
                play = False
                winner = 'D'
                self.update_track_win_count(status)
            self.switch_player()
        game.append(self.board.copy())

        return (winner, game)
    
    def record_x_best_moves(self):
        x_states = self.x_states.keys()
        self.x_best_moves = {}

        for state in x_states:
            flat_state = np.array(state)
            reshaped_state = flat_state.reshape((3, 3))
            moves = self.get_moves(reshaped_state)
            if np.any(moves == 0) is False:
                print('no move detected')
            highest_reward = ()
            for row, col in moves:
                board = reshaped_state.copy()
                index = self.row_col_to_index((row,col))
                board[row][col] = 1
                board_tuple = tuple(board.flatten())
                if board_tuple in self.x_states:
                    if highest_reward:
                        if self.x_states[board_tuple] > highest_reward[0]:
                            highest_reward = (self.x_states[board_tuple], index)
                    else:
                        highest_reward = (self.x_states[board_tuple], index)
            if highest_reward:
                self.x_best_moves[state] = highest_reward[1]
        return self.x_best_moves
    
    def get_count_end_states(self):
        x_states = self.x_states.keys()
        count = 0
        for state in x_states:
            flat_state = np.array(state)
            board = flat_state.reshape((3, 3))
            has_empty_spots = np.any(board == 0)
            if self.check_win(board_state=board) or not has_empty_spots:
                count += 1
        print(f'total_occupied_states:{count}')

    

    def update_track_win_count(self, status):
        if status == 100:
            if self.players[self.current_player] == 'X':
                self.track_win_count['X'] += 1
            elif self.players[self.current_player] == 'O':
                self.track_win_count['O'] += 1
        else:
            self.track_win_count['D'] += 1
        


    def find_highest_reward_move(self, player):
        '''
        Finds the move with the highest expected reward for the given player (X or O) 
        using the current board state and the Q-value table (self.q_states).

        Iterates through valid moves, simulating them and checking the opponent's 
        reward in the Q-table. Chooses the move with the highest reward, exploring 
        randomly if no rewarding moves exist.

        Args:
            player (str): The player ('X' or 'O') for whom to find the best move.

        Returns:
            tuple: A tuple containing the row and column indices of the move with the 
                    highest expected reward (row, col).
        '''
        if player == 'X':
            highest_reward_state = None
            # reward = None   -- commented out to add the reward inside the loop
            reward_found = False
            moves = self.get_moves()
            for row, col in moves:
                reward = None
                board = self.board.copy()
                board[row][col] = 1
                board_tuple = tuple(board.flatten())
                if board_tuple in self.x_states:
                    reward = self.x_states[board_tuple]
                    reward_found = True
                
                if reward is None:
                    continue
                else:
                    if highest_reward_state:
                        if reward > highest_reward_state[0]:
                            highest_reward_state = (reward, (row, col))             
                    else:
                        highest_reward_state = (reward, (row, col))
            if reward_found is False:
                row, col = random.choice(moves)
                board = self.board.copy()
                board[row][col] = -1
                reward = 0 #doesn't matter
                highest_reward_state = (reward, (row, col))
        else:
            highest_reward_state = None
            reward_found = False
            # reward = None   -- commented out to add the reward inside the loop
            moves = self.get_moves()
            for row, col in moves:
                reward = None
                board = self.board.copy()
                board[row][col] = -1
                board_tuple = tuple(board.flatten())
                if board_tuple in self.o_states:
                    reward = self.o_states[board_tuple]
                    reward_found = True
                
                if reward is None:
                    continue
                else:
                    if highest_reward_state:
                        if reward > highest_reward_state[0]:
                            highest_reward_state = (reward, (row, col))             
                    else:
                        highest_reward_state = (reward, (row, col))
            if reward_found is False:
                row, col = random.choice(moves)
                board = self.board.copy()
                board[row][col] = -1
                reward = 0 #doesn't matter
                highest_reward_state = (reward, (row, col))
        return highest_reward_state[1]
    
    def random_move(self):
        moves = self.get_moves()
        row, col = random.choice(moves)
        return (row, col)
    
    def reversed(self, games):
        reversed_games = games[::-1]
        return reversed_games
    
    def export_x_states(self):
        return self.x_states
    
    def update_dict(self, states, game, learning_rate, discount, reward):
        n_games = self.reversed(game)
        cumulative_reward = reward
        for board in n_games:
            board_tuple = tuple(board.flatten())

            # if board is not in dictionary put it in, initial reward is 0
            if (board_tuple in states) is False:
                states[board_tuple] = 0

            # update reward
            states[board_tuple] = states[board_tuple] + learning_rate * ((discount * cumulative_reward) - states[board_tuple])
            cumulative_reward = states[board_tuple]

    def run_ttt_ex_vs_oh(self, n_games = 15000):
        """
        Plays Tic-Tac-Toe games between 'X' and 'O' for a specified number of iterations 
        (n_games) to train the AI player using Q-learning.

        This function iterates through n_games, playing a game between 'X' and 'O' at each 
        iteration. After each game, the Q-values in the state dictionaries (self.x_states 
        and self.o_states) are updated based on the game outcome (win, loss, or draw) using 
        rewards and a learning rate.

        Args:
            n_games: The number of games to play for training (int). Defaults to 15000.

        Returns:
            None. This function updates the internal Q-value dictionaries for training 
            but does not return any value.
        """
        for i in range(n_games):
            winner, game = self.ttt_ex_vs_oh()
            learning_rate = self.learning_rate
            if winner == 'X':
                self.update_dict(self.x_states, game, learning_rate, discount = 0.9, reward = 100)
                self.update_dict(self.o_states, game, learning_rate, discount = 0.9, reward = -100)
            elif winner == 'O':
                self.update_dict(self.o_states, game, learning_rate, discount = 0.9, reward = 100)
                self.update_dict(self.x_states, game, learning_rate, discount = 0.9, reward = -100)
            else:
                self.update_dict(self.x_states, game, learning_rate, discount = 0.9, reward = -20)
                self.update_dict(self.o_states, game, learning_rate, discount = 0.9, reward = -20)

            if i % 1000 == 0:
                print(f'Epoch:{i}, X-States: {len(self.x_states)}, O-States: {len(self.o_states)} || win X:{self.track_win_count["X"]} || win O:{self.track_win_count["O"]} || win D:{self.track_win_count["D"]}')

    def run(self):
        self.run_ttt_ex_vs_oh()
        self.print_info()
    
    def print_info(self):
        print(f'X-States: {len(self.x_states)} || O-States: {len(self.o_states)}')
        print(f'max_x_state_value: {max(self.x_states.values())} || max_o_state_value: {max(self.o_states.values())}')
        print(f'learning_rate: {self.learning_rate}')

        print(f'win X total:{self.track_win_count["X"]} || win O total:{self.track_win_count["O"]} || win D total:{self.track_win_count["D"]}')
    

    def row_col_to_index(self, row_col):
        row = row_col[0]
        col = row_col[1]
        
        index = row * 3 + col
        return index

    def get_input(self):
        """Overrides the method from the base class."""

        if self.computer_play:
            if self.curr_player_is_computer is False:
                row, column = map(int, input(f"Player {self.players[self.current_player]}'s turn. Enter row and column (0-{self.board_size - 1}): ").split())
            else:
                row, column = self.find_highest_reward_move({self.players[self.current_player]})
        else:
            row, column = map(int, input(f"Player {self.players[self.current_player]}'s turn. Enter row and column (0-{self.board_size - 1}): ").split())
        return (row, column)
        

    def start_game(self):
        """Overrides the method from the base class."""

        print(f"Let's play Tic-Tac-Toe on a {self.board_size}x{self.board_size} board!")
        super().__init__()
        self.print_board()
        while not self.game_over:
            while True:
                try:
                    row, column = self.get_input()
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


train = TrainTicTacToe()
train.learning_rate = 1
train.run()
x_states = train.x_states
x_best_moves = train.record_x_best_moves()
train.get_count_end_states()
print(f'length of x_best_moves:{len(x_best_moves)}')

# def match_values(dict1, dict2):
#     matching_dict = {}

#     for key in dict1:
#         if key in dict2:
#             value = dict2[key]
#             matching_dict[key] = value

#     return matching_dict

# # File path to save the dictionary
# file_path = 'state_action.pkl'

# import pickle

# data = match_values(x_states, x_best_moves)
# # Save the dictionary to a file using JSON
# with open(file_path, 'wb') as file:
#     pickle.dump(data, file)

while True:
    train.start_game()