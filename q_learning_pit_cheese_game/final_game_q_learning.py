import numpy as np
from random import randrange, random

'''
author: sushovan.adhikari
Goal: finding the optimum path towards the goal state(cheese state) using q-learning. Total 12 position(1D grid),
only 2 movements valid, left and right. From the left corner, left movement leads to the same state, and likewise for the right corner.
'''

# initialize q-table
q_table = np.empty((0, 2))

# immediate reward values
board_reward_values = np.array([-100,0,0,0,0,0,0,0,0,0,0,100])

game_grid = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']

# initialize q-movement-matrix
q_movement_matrix = np.array(
        [[0,1],
        [0,2],
        [1,3],
        [2,4],
        [3,5],
        [4,6],
        [5,7],
        [6,8],
        [7,9],
        [8,10],
        [9,11],
        [10,11]]
    )  
    
# initialize game board
def init_game(grid, goal_state, pit_state, start_state):
    grid[goal_state] = 'C'
    grid[pit_state] = 'X'
    grid[start_state] = 'P'

# print game states
def print_game_board(current_state, previous_state, grid):
    grid[current_state] = 'P'
    if previous_state:
        grid[previous_state] = 'o'
    horizontal_border = "+" + "-" * (len(grid) * 2 + 1) + "+"
    print(horizontal_border)
    print(' '.join(grid))
    print(horizontal_border)

for state, value in enumerate(board_reward_values):
        state_action = np.array([0,0])
        q_table = np.vstack([q_table,state_action])

def select_random_action(available_actions):
    random_number = np.random.choice(available_actions)
    return random_number

def q_learning_game(reward_state, pit_state):
    threshold = 0.5
    iterations = 0

    while (iterations < 300):
        print("iterations: ", iterations)
        
        num_of_states = len(board_reward_values) 
        player_state = np.random.choice(num_of_states)
        
        while(player_state != reward_state and player_state != pit_state): 
            
            chance = round(random(), 2)
        
            if chance > threshold:
                action = select_random_action(len(q_table[0]))
            else:
                # Get the index(action) of the highest value in the specified row
                # note: q_table column values represents the action
                action = np.argmax(q_table[player_state])
    
            next_state = q_movement_matrix[player_state][action]
            
            q_table[player_state][action] = board_reward_values[next_state] + 0.9 * max(q_table[next_state])
            player_state = next_state
        print(q_table)
            
        iterations += 1

def navigate_to_goal(q_table, q_movement_matrix, start_state, goal_state, pit_state):
    current_state = start_state
    init_game(game_grid, goal_state, pit_state, start_state)
    previous_state = None
    while current_state != goal_state and current_state != pit_state:
        action = np.argmax(q_table[current_state])
        next_state = q_movement_matrix[current_state][action]
        print(f"Current State: {current_state}, Action: {action}, Next State: {next_state}")
        print_game_board(current_state, previous_state, game_grid)
        previous_state = current_state
        current_state = next_state
    if current_state == pit_state:
        print('Pit got in your way. Move the pit and try again!')
        return
    print(f"Current State: {current_state}, Action: {action}, Next State: {next_state}")
    print_game_board(current_state, previous_state, game_grid)
    print('Win! You found the CHEESE')

q_learning_game(11,0)
    
# User inputs
start_state = int(input("Enter the starting state for navigation: "))
goal_state = int(input("Enter the goal state: "))
pit_state = int(input("Enter the pit state: "))

navigate_to_goal(q_table, q_movement_matrix, start_state, goal_state, pit_state)

