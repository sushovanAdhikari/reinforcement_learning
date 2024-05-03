import numpy as np
from random import randrange, random

# init_game_board
def init_game_board(player_state, ghost_state, maze_walls, game_board_size):
    global game_board
    game_board = np.zeros(game_board_size)
    player_state_row_index, player_state_col_index = row_col_state_index(player_state,game_board_size)
    ghost_state_row_index, ghost_state_col_index = row_col_state_index(ghost_state, game_board_size)
    game_board[player_state_row_index][player_state_col_index] = 1
    game_board[ghost_state_row_index][ghost_state_col_index] = 2
    for wall in maze_walls:
        wall_row, wall_col = row_col_state_index(wall,game_board_size)
        game_board[wall_row][wall_col] = -1
    return game_board

def row_col_state_index(state, game_board_size):
    row = state // game_board_size[1]
    col = state % game_board_size[1]
    return (row, col)

def init_reward_values(total_states, player_state):
    reward_values = np.zeros(total_states, dtype=int)
    reward_values[player_state] = 100
    return reward_values

def print_game_board(current_state, previous_state, game_board):
    current_state_row, current_state_col = row_col_state_index(current_state, game_board_size)
    if previous_state:
        previous_state_row, previous_state_col = row_col_state_index(previous_state, game_board_size)
        game_board[previous_state_row][previous_state_col] = 0
    game_board[current_state_row][current_state_col] = 2

    cnt = 1
    if cnt == 1: 
        spaces = '_' * 10
        print(spaces, 'Printing the Board States', spaces)
        global _input_values
        _input_values = []
        cnt += 1

    
    _input_values.append(game_board)


    for row in range(len(game_board)):
        for col in range(len(game_board[0])):
            if game_board[row][col] == 0: 
                print('-', end=" ")
            elif game_board[row][col] == 1:
                print('P', end=" ")
            elif game_board[row][col] == 2:
                print('G', end=" ")
            else:
                print('#', end=" ")
        print()

# generate an adjacency matrix for a given grid size
# rows represent states starting from zero and columns represent Down Up Left and Right action
def generate_adjacency_matrix(grid_size):
    num_states = grid_size[0] * grid_size[1]
    adjacency_matrix = np.zeros((num_states, 4), dtype=int) - 1
    for i in range(num_states):
        row = i // grid_size[1]
        col = i % grid_size[1]
        if row > 0:
            adjacency_matrix[i, 0] = i - grid_size[1] # Down
        if row < grid_size[0] - 1:
            adjacency_matrix[i, 1] = i + grid_size[1] # Up
        if col > 0:
            adjacency_matrix[i, 2] = i - 1            # Left
        if col < grid_size[1] - 1:
            adjacency_matrix[i, 3] = i + 1            # Right
    return adjacency_matrix

# create another numpy matrix same as adj_matrix, copy negative value places, fill the rest with zeroes.
def init_q_table(adj_matrix):
    q_table = np.zeros_like(adj_matrix)

    # Boolean indexing to copy negative values from the original matrix to the new matrix
    q_table[adj_matrix < 0] = adj_matrix[adj_matrix < 0]
    return q_table

def select_random_action(available_actions):
    random_number = np.random.choice(available_actions)
    return random_number

game_board = None
game_board_size = (6,8)
maze_walls = [9,10,11,12,14,22,25,26,28,29,30,32,34,35]
total_game_states = game_board_size[0] * game_board_size[1]

# Create a list of available states by excluding the maze wall states
available_states = [state for state in range(total_game_states) if state not in maze_walls]
adjacency_matrix = generate_adjacency_matrix(game_board_size)

def q_learning_game():
    threshold = 0.5

    # saves player position as the key and q-table generated for that specifc position.
    global players_q_table
    players_q_table = {}

    for i in range(1):
        player_state = i
        q_table = init_q_table(adjacency_matrix)
        reward_values = init_reward_values(total_game_states, player_state)
        iterations = 0
        while (iterations < 300):
            
            # the maze walls should be discarded from the random selection pool.
            ghost_state = np.random.choice(available_states)
            
            while(ghost_state != player_state): 
                
                chance = round(random(), 2)
            
                if chance > threshold:
                    action = select_random_action(len(q_table[0]))
                else:
                    # Get the index(action) of the highest value in the specified row
                    # note: q_table column values represents the action
                    action = np.argmax(q_table[ghost_state])
        
                next_state = adjacency_matrix[ghost_state][action]
                if next_state in maze_walls:
                    continue
                
                temp_value = reward_values[next_state] + 0.9 * max(q_table[next_state])
                q_table[ghost_state][action] = reward_values[next_state] + 0.9 * max(q_table[next_state])
                ghost_state = next_state
                
            iterations += 1
        players_q_table[i] = q_table

q_learning_game()

for key, value in players_q_table.items():
    readibility = '-' * 5 + 'Q-Table for Player Position: ' + str(key) + '-' * 5
    print('\n',readibility)
    print(f'{value}')

def navigate_to_goal(q_table, adjacency_matrix, player_state, ghost_state):
    game_board = init_game_board(player_state, ghost_state, maze_walls, game_board_size)
    current_state = ghost_state
    previous_state = None
    while current_state != player_state:
        action = np.argmax(q_table[current_state])
        next_state = adjacency_matrix[current_state][action]
        print(f"Current State: {current_state}, Action: {action}, Next State: {next_state}")
        print_game_board(current_state, previous_state, game_board)
        previous_state = current_state
        current_state = next_state

    print(f"Current State: {current_state}, Action: {action}, Next State: {next_state}")
    print_game_board(current_state, previous_state, game_board)
    print('The ghost of Anabelle got you.')

# def start_game()
def start_game():
    player_state = int(input("Enter the starting state for poor player(currently supports 0-3): "))
    ghost_state = int(input("Enter the ghost state: "))
    if player_state > 3:
        print('0-3 states only supported for player')
    elif ghost_state > 47:
        print('Value exceeds the grid. Provide value <= 47 ')
    elif ghost_state in maze_walls:
        print('The ghost cannot be in a wall.')
    else:
        q_table = players_q_table[player_state]
        navigate_to_goal(q_table, adjacency_matrix, player_state, ghost_state)
    start_game()    

start_game()