import numpy as np
from random import randrange

'''
The goal is to help agent(computer/index whatever) to find the optimum path toward the reward state in the 2*3 grid(total 6 positions.)
'''

# Function to generate a adjacency matrix for a given grid size
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

def select_random_state(states):
    pick_random_state = randrange(states)
    return pick_random_state

def select_random_action(state, q_table):
    available_action = q_table[state]
    # Find indices where elements are not equal to -1
    available_action_clean = np.where(available_action != -1)[0]
    random_number = np.random.choice(available_action_clean)
    return random_number

def display_q_table(q_table):
    print("Q-Table:")
    print(q_table)

# Function to generate a random immediate reward array
def generate_immediate_rewards(num_states, goal_state):
    immediate_rewards = np.zeros(num_states, dtype=int)
    immediate_rewards[goal_state] = 100
    return immediate_rewards

# Function to initialize Q-table with initial values
def initialize_q_table(q_adjacent_matrix):
    # Create a new matrix filled with zeros of the same shape as the original matrix
    q_table = np.zeros_like(q_adjacent_matrix)
    
    # Boolean indexing to copy negative values from the original matrix to the new matrix
    q_table[q_adjacent_matrix < 0] = q_adjacent_matrix[q_adjacent_matrix < 0]
    return q_table

# User inputs
grid_size = tuple(map(int, input("Enter grid size (rows columns): ").split()))
goal_state = int(input("Enter the goal state: "))
n_iterations = int(input("Enter the number of training iterations: "))
start_state = int(input("Enter the starting state for navigation: "))

# Generate environment parameters
num_states = grid_size[0] * grid_size[1]
num_actions = 4  # Up, Down, Left, Right
q_adjacent_matrix = generate_adjacency_matrix(grid_size)
q_immediate = generate_immediate_rewards(num_states, goal_state)
q_table = initialize_q_table(q_adjacent_matrix)


def q_learning(n_iterations, q_table, q_immediate, q_adjacent_matrix):
    for j in range(0, n_iterations):
        my_state = select_random_state(len(q_immediate)-1);
        while(my_state != goal_state):
            my_action = select_random_action(my_state, q_table)
            next_state = q_adjacent_matrix[my_state][my_action]
            immediate_reward = q_immediate[next_state]
            delayed_reward = max(q_table[next_state]) * 0.9
            total_reward = immediate_reward + delayed_reward
            q_table[my_state][my_action] = total_reward
            my_state = next_state

        if j >= n_iterations - 3:
            print(q_table,'\n')

    return q_table

# Train Q-learning
trained_q_table = q_learning(n_iterations, q_table, q_immediate, q_adjacent_matrix)


def navigate_to_goal(q_table, start_state, goal_state, q_adjacent_matrix):
    current_state = start_state
    while current_state != goal_state:
        action = np.argmax(q_table[current_state])
        next_state = q_adjacent_matrix[current_state][action]
        print(f"Current State: {current_state}, Action: {action}, Next State: {next_state}")
        current_state = next_state

# Navigate using trained Q-table
print("Navigating from start state to goal state using trained Q-table:")
navigate_to_goal(trained_q_table, start_state, goal_state, q_adjacent_matrix)