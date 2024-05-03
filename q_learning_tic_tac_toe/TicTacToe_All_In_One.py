import numpy as np
import random
import types

class TicTacToe:
    """
    A class for playing a game of Tic-Tac-Toe.

    This class provides methods for managing a Tic-Tac-Toe game, including initializing the game board, making moves, checking the game status, and determining the winner.

    Additionally, the class includes two extra functions, `get_blocking_move` and `get_win_move`, which are not required for a standard game of Tic-Tac-Toe but can be used to adjust the move logic in inherited classes:
    
    - `get_blocking_move()`: Returns the move that would block the opponent from winning.
    - `get_win_move()`: Returns the move that would allow the current player to win.

    These extra functions are useful for customizing the game's logic in derived classes.
    """
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
    def get_moves(self):
        board = np.argwhere(self.board == 0)
        return board
    
    def random_move(self):
        moves = self.get_moves()
        row, col = random.choice(moves)
        return (row, col)

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
        print()


    def get_input(self):
        if self.computer_play:
            if self.curr_player_is_computer is False:
                row, column = map(int, input(f"Player {self.players[self.current_player]}'s turn. Enter row and column (0-{self.board_size - 1}): ").split())
            else:
                row, column = self.computer_move()
        else:
            row, column = map(int, input(f"Player {self.players[self.current_player]}'s turn. Enter row and column (0-{self.board_size - 1}): ").split())
        return (row, column)
    
    def get_blocking_move(self, player):
        """
        Finds the row and column where the specified player can place a move
        to block their opponent's win.

        Args:
            self: An instance of the Tic-Tac-Toe game class (or similar).
            player: Either 1 (X) or -1 (O) indicating the player whose blocking move
                    is being sought.

        Returns:
            A tuple (row, col) representing the blocking move for the player, or None
            if no such move exists.
        """
        board = self.board
        board_size = board.shape[0]
        opponent = -player  # Determine opponent's value (X for O and vice versa)

        # Check rows and columns for potential opponent wins that the player can block
        for i in range(board_size):
            row_sum =  board[i, :].sum()
            col_sum =  board[:, i].sum()

            if (row_sum == 2 * opponent) and any( board[i, :] == 0):
                # Find the empty space in the row/column
                empty_col = np.where( board[i, :] == 0)[0][0]
                return (i, empty_col)
            elif (col_sum == 2 * opponent) and any( board[:,i] == 0):
                empty_row = np.where( board[:,i] == 0)[0][0]
                return (empty_row, i)
      
        # Check diagonals for potential opponent wins that the player can block
        diag_sum1 = np.trace( board)
        diag_sum2 = np.trace(np.fliplr( board))

        if (diag_sum1 == 2 * opponent or diag_sum2 == 2 * opponent):
            # Iterate through diagonal elements (considering wrap-around for second diagonal)
            for i in range(board_size):
                if  board[i, i] == 0:
                    return (i, i)
                for i in range(board_size):
                    if board[board_size - 1 - i, i] == 0:
                        return (board_size - 1 - i, i)
        return None
    
    def get_win_move(self, player):
        """
        Finds the row and column where the specified player can place a move
        to win

        Args:
            self: An instance of the Tic-Tac-Toe game class (or similar).
            player: Either 1 (X) or -1 (O) indicating the player whose win move
                    is being sought.

        Returns:
            A tuple (row, col) representing the win move for the player, or None
            if no such move exists.
        """
        board = self.board
        board_size = board.shape[0]
        opponent = player  # Determine opponent's value (X for O and vice versa)


        for i in range(board_size):
            row_sum =  board[i, :].sum()
            col_sum =  board[:, i].sum()

            if (row_sum == 2 * opponent) and any( board[i, :] == 0):
            # Find the empty space in the row/column
                empty_col = np.where( board[i, :] == 0)[0][0]
                return (i, empty_col)
            elif (col_sum == 2 * opponent) and any( board[:,i] == 0):
                empty_row = np.where( board[:,i] == 0)[0][0]
                return (empty_row, i)

        diag_sum1 = np.trace( board)
        diag_sum2 = np.trace(np.fliplr( board))

        if (diag_sum1 == 2 * opponent or diag_sum2 == 2 * opponent):
            # Iterate through diagonal elements (considering wrap-around for second diagonal)
            for i in range(board_size):
                if  board[i, i] == 0:
                    return (i, i)
                for i in range(board_size):
                    if board[board_size - 1 - i, i] == 0:
                        return (board_size - 1 - i, i)

        return None
        

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
    """
    This class is specifically used to train an agent to play tic-tac-toe using the Q-learning algorithm.

    Inherits from the TicTacToe class and extends it to facilitate training an AI agent using reinforcement learning. The class is designed to manage the game state and provide necessary feedback to the Q-learning agent during training.

    Features:
    - Overrides methods from the base TicTacToe class as needed to support training.
    - Contains methods to update the Q-table and implement the Q-learning algorithm.
    - Provides a way to simulate games and allow the agent to learn from experience.
    
    Usage:
    Create an instance of this class and use its methods to control the training process. You can set parameters such as the learning rate and discount factor according to the Q-learning algorithm requirements.
    """
    
    def __init__(self):
        # super().__init__()
        self.x_states = {}
        self.o_states = {}
        self.training_win_count = {
            'X': 0,
            'O': 0,
            'D': 0
        }
        self.game_win_count = {
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
                self.update_training_win_count(status)
            elif draw:
                play = False
                winner = 'D'
                self.update_training_win_count(status)
            self.switch_player()
        game.append(self.board.copy())

        return (winner, game)
    

    def update_game_win_count(self, status):
        if status == 100:
            if self.players[self.current_player] == 'X':
                self.game_win_count['X'] += 1
            elif self.players[self.current_player] == 'O':
                self.game_win_count['O'] += 1
        else:
            self.game_win_count['D'] += 1

    def update_training_win_count(self, status):
        if status == 100:
            if self.players[self.current_player] == 'X':
                self.training_win_count['X'] += 1
            elif self.players[self.current_player] == 'O':
                self.training_win_count['O'] += 1
        else:
            self.training_win_count['D'] += 1
        


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
            reward = None
            moves = self.get_moves()
            for row, col in moves:
                board = self.board.copy()
                board[row][col] = 1
                board_tuple = tuple(board.flatten())
                if board_tuple in self.x_states:
                    reward = self.x_states[board_tuple]
                
                if reward is None:
                    continue
                else:
                    if highest_reward_state:
                        if reward > highest_reward_state[0]:
                            highest_reward_state = (reward, (row, col))             
                    else:
                        highest_reward_state = (reward, (row, col))
            if reward is None:
                row, col = random.choice(moves)
                board = self.board.copy()
                board[row][col] = -1
                reward = 0 #doesn't matter
                highest_reward_state = (reward, (row, col))
        else:
            highest_reward_state = None
            reward = None
            moves = self.get_moves()
            for row, col in moves:
                board = self.board.copy()
                board[row][col] = -1
                board_tuple = tuple(board.flatten())
                if board_tuple in self.o_states:
                    reward = self.o_states[board_tuple]
                
                if reward is None:
                    continue
                else:
                    if highest_reward_state:
                        if reward > highest_reward_state[0]:
                            highest_reward_state = (reward, (row, col))             
                    else:
                        highest_reward_state = (reward, (row, col))
            if reward is None:
                row, col = random.choice(moves)
                board = self.board.copy()
                board[row][col] = -1
                reward = 0 #doesn't matter
                highest_reward_state = (reward, (row, col))
        return highest_reward_state[1]
    
    def reversed(self, games):
        reversed_games = games[::-1]
        return reversed_games
    
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
                print(f'Epoch:{i}, X-States: {len(self.x_states)}, O-States: {len(self.o_states)} || win X:{self.game_win_count["X"]} || win O:{self.game_win_count["O"]} || win D:{self.game_win_count["D"]}')

    def update_game_win_count(self, status):
        if status == 100:
            if self.players[self.current_player] == 'X':
                self.game_win_count['X'] += 1
            elif self.players[self.current_player] == 'O':
                self.game_win_count['O'] += 1
        else:
            self.game_win_count['D'] += 1

    def run(self):
        self.run_ttt_ex_vs_oh()
        self.print_info()
    
    def print_info(self):
        print(f'X-States: {len(self.x_states)} || O-States: {len(self.o_states)}')
        print(f'max_x_state_value: {max(self.x_states.values())} || max_o_state_value: {max(self.o_states.values())}')
        print(f'learning_rate: {self.learning_rate}')

        print(f'win X total:{self.training_win_count["X"]} || win O total:{self.training_win_count["O"]} || win D total:{self.training_win_count["D"]}')
    

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
        status = 0
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
                status = 100
                self.game_over = True
                self.update_game_win_count(status)
            elif self.check_draw():
                print("It's a draw!")
                self.game_over = True
                self.update_game_win_count(status)
            self.switch_player()


class ModsTicTacToe(TrainTicTacToe):

    """
    This class contains modifications needed to approach the training in different ways. The constructor parameter mod is used to select the right method to add/overwrite method
    from the inherited class.

    The default mod_code is 0, i.e, use the default training approach and game play available in TrainTicTac.
    The mod_code 1 is used to pit the trained dictionary agent with the blocking move TicTacToe logic(This logic detects if the opposition has any winning moves, and places it's current player position there accordinly)
    The mod_code 2 is used to modify the training process of the dictionary.
    The mod_code 3 is used to modify the gameplay of the trained_dictionary.
    The mod_code 4 is used to train a neural network based on the data generated from the trained_dictionary and pit against the trained dictionary.
    The mod_code 5 is used to pit the trained neural network against the blocking move TicTacToe logic.
    """

    def __init__(self, mod_code=0):

        config_code = {
            0: 'q-dictionary vs You',
            1: 'trained_dict_vs_summer_ttt_2023',
            2: 'trained_dict_exploitation_mod_vs_summer_ttt_2023',
            3: 'trained_dict_random_first_move_vs_summer_ttt_2023',
            4: 'neural_network_x_vs_trained_dict',
            5: 'neural_nework_x_vs_summer_ttt_2023'
        }
        print(f'{"-"* 15} Executing {config_code[mod_code]} {"-"* 15}')
        super().__init__()
        self.mod_code = mod_code
        self.conditional_override()
        self.train_dict()        

    def train_dict(self):
        self.learning_rate = 1
        self.run()
        self.execute_game()

    def execute_game(self):
        if self.mod_code == 4:
            for i in range(0,1):
                self.start_game()
            print(self.game_win_count)
        else:
            for i in range(0,100):
                self.start_game()
            print(self.game_win_count)

    def conditional_override(self):
        if self.mod_code == 1:
            self.trained_ai_turn = True
            def get_input(self):
                """Overrides the method from the base class."""

                if self.computer_play:
                    if self.trained_ai_turn:
                        row, column = self.find_highest_reward_move(self.players[self.current_player])
                        self.trained_ai_turn = False
                    else:
                        '''
                        mimics tictactoe summer 2023 game logic.
                        '''
                        win = self.get_win_move(self.current_player)
                        block_move = self.get_blocking_move(self.current_player)
                        if win:
                            print('win move detected')
                            row, column = win
                        elif block_move:
                            print('block move detected')
                            row, column = block_move
                        else:
                            row, column = self.random_move()
                        self.trained_ai_turn = True
                else:
                    row, column = map(int, input(f"Player {self.players[self.current_player]}'s turn. Enter row and column (0-{self.board_size - 1}): ").split())
                return (row, column)
            bound_method = types.MethodType(get_input, self)
            setattr(self, 'get_input', bound_method)

        elif self.mod_code == 2:
            def run_ttt_ex_vs_oh(self, n_games = 15000):
                self.track_iterations['total_iterations'] = n_games - 1 
                self.track_iterations['exploit_from_iteration'] = (n_games - 1) // 2

                for i in range(n_games):
                    self.track_iterations['iteration_track'] = i
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
                        print(f'Epoch:{i}, X-States: {len(self.x_states)}, O-States: {len(self.o_states)} || win X:{self.game_win_count["X"]} || win O:{self.game_win_count["O"]} || win D:{self.game_win_count["D"]}')
                
            def exploit(self, chance):
                if self.track_iterations['iteration_track'] >= self.track_iterations['exploit_from_iteration']:
                    exploit = True
                else: 
                    exploit = False
                return exploit
            
            self.track_iterations = {
                'total_iterations': 0,
                'exploit_from_iteration': 0,
                'iteration_track': 0
            }

            # Bind the function as a method to the instance using types.MethodType
            bound_method = types.MethodType(run_ttt_ex_vs_oh, self)
            setattr(self, 'run_ttt_ex_vs_oh', bound_method)
            bound_method = types.MethodType(exploit, self)
            setattr(self, 'exploit', bound_method)

        elif self.mod_code == 3:
            self.x_random_move = True
            def get_input(self):
                """Overrides the method from the base class."""

                if self.computer_play:
                    if self.trained_ai_turn:
                        if self.x_random_move:
                            row, column = self.random_move()
                            self.x_random_move = False
                        else:
                            row, column = self.find_highest_reward_move(self.players[self.current_player])
                        self.trained_ai_turn = False
                    else:
                        win = self.get_win_move(self.current_player)
                        block_move = self.get_blocking_move(self.current_player)
                        if win:
                            print('win move detected')
                            row, column = win
                        elif block_move:
                            print('block move detected')
                            row, column = block_move
                        else:
                            row, column = self.random_move()
                        self.trained_ai_turn = True
                else:
                    row, column = map(int, input(f"Player {self.players[self.current_player]}'s turn. Enter row and column (0-{self.board_size - 1}): ").split())
                return (row, column)
            bound_method = types.MethodType(get_input, self)
            setattr(self, 'get_input', bound_method)

        elif self.mod_code == 4:
            def initialize_neural_net():
                from NeuralNetwork import NeuralNetTicTacToe
                self.neural_net_play = True
                neural_net = NeuralNetTicTacToe()
                neural_net.train()
                self.neural_net_obj = neural_net
                
                def get_input(self):
                    """Overrides the method from the base class."""

                    if self.computer_play:
                        if self.neural_net_play:

                            row, column = self.neural_net_obj.predict(self.board)
                            if self.board[row][column] == 0:
                                self.neural_net_play = False
                                self.neural_net_obj.valid(True)
                            else:
                                while(self.board[row][column] != 0):
                                    self.neural_net_obj.valid(False)
                                    row, column = self.neural_net_obj.predict(self.board)
                                self.neural_net_obj.valid(True)
                                self.neural_net_play = False
                        else:
                            row, column = self.find_highest_reward_move(self.players[self.current_player])
                            self.neural_net_play = True
                    else:
                        row, column = map(int, input(f"Player {self.players[self.current_player]}'s turn. Enter row and column (0-{self.board_size - 1}): ").split())
                    return (row, column)
        

                def start_game(self):
                    """Overrides the method from the base class."""

                    print(f"Let's play Tic-Tac-Toe on a {self.board_size}x{self.board_size} board!")
                    super().__init__()
                    self.print_board()
                    status = 0
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
                            status = 100
                            self.game_over = True
                            self.update_game_win_count(status)
                        elif self.check_draw():
                            print("It's a draw!")
                            self.game_over = True
                            self.update_game_win_count(status)
                            self.switch_player()

                bound_method = types.MethodType(get_input, self)
                setattr(self, 'get_input', bound_method)
                bound_method = types.MethodType(start_game, self)
                setattr(self, 'start_game', bound_method)
            initialize_neural_net()

        elif self.mod_code == 5:
            def initialize_neural_net():
                from NeuralNetwork import NeuralNetTicTacToe
                self.neural_net_play = True
                neural_net = NeuralNetTicTacToe()
                neural_net.train()
                self.neural_net_obj = neural_net
                self.neural_net_turn = True

                def get_input(self):
                    """Overrides the method from the base class."""

                    if self.computer_play:
                        if self.neural_net_turn:
                            row, column = self.neural_net_obj.predict(self.board)
                            if self.board[row][column] == 0:
                                self.neural_net_play = False
                                self.neural_net_obj.valid(True)
                            else:
                                while(self.board[row][column] != 0):
                                    self.neural_net_obj.valid(False)
                                    row, column = self.neural_net_obj.predict(self.board)
                                self.neural_net_obj.valid(True)
                            self.neural_net_turn = False
                        else:
                            '''
                            mimics tictactoe summer 2023 game logic.
                            '''
                            win = self.get_win_move(self.current_player)
                            block_move = self.get_blocking_move(self.current_player)
                            if win:
                                print('win move detected')
                                row, column = win
                            elif block_move:
                                print('block move detected')
                                row, column = block_move
                            else:
                                row, column = self.random_move()
                            self.neural_net_turn = True
                    else:
                        row, column = map(int, input(f"Player {self.players[self.current_player]}'s turn. Enter row and column (0-{self.board_size - 1}): ").split())
                    return (row, column)
                    

                def start_game(self):
                    """Overrides the method from the base class."""

                    print(f"Let's play Tic-Tac-Toe on a {self.board_size}x{self.board_size} board!")
                    TicTacToe.__init__(self)
                    self.print_board()
                    status = 0
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
                            status = 100
                            self.game_over = True
                            self.update_game_win_count(status)
                        elif self.check_draw():
                            print("It's a draw!")
                            self.game_over = True
                            self.update_game_win_count(status)
                        self.switch_player()


                bound_method = types.MethodType(get_input, self)
                setattr(self, 'get_input', bound_method)
                bound_method = types.MethodType(start_game, self)
                setattr(self, 'start_game', bound_method)

            initialize_neural_net()

execute_mod = ModsTicTacToe()      
# execute_mod = ModsTicTacToe(1)
# execute_mod = ModsTicTacToe(2)
# execute_mod = ModsTicTacToe(3)
# execute_mod = ModsTicTacToe(4)
# execute_mod = ModsTicTacToe(5)