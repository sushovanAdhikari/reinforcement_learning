import gymnasium as gym
import numpy as np

# Create the environment
env = gym.make('MountainCar-v0', render_mode='human')

# Function to control the car
def control_car(state):
    # Extract position and velocity
    position, velocity = state
    
    # Simple heuristic to control the car
    # If car is moving to the right and has some speed, keep accelerating to the right
    if velocity > 0:
        return 2  # Action 2: accelerate to the right
    # If car is on the right side and not moving fast enough, accelerate left
    elif velocity <= 0 and position >= -0.5:
        return 0  # Action 0: accelerate to the left
    # If car is on the left side, accelerate to the right
    else:
        return 2  # Action 2: accelerate to the right

# Run the algorithm
episodes = 10  # Number of episodes to run
for episode in range(episodes):
    # Reset the environment and get the initial state
    state, info = env.reset()
    done = False
    
    # Track the number of steps in each episode
    step_count = 0
    
    while not done:
        # Choose an action using the control_car function
        action = control_car(state)
        
        # Take a step in the environment
        state, reward, done, truncated, info = env.step(action)
        # state, reward, done, truncated, info = env.step(env.action_space.sample())
        
        # Increment the step count
        step_count += 1
        
        # Render the environment
        env.render()
        
        # Break if the car reaches the goal or if the episode takes too many steps
        if done or step_count > 500:  # Adjust the step limit as needed
            break

env.close()



# # Create the environment
# env = gym.make('MountainCar-v0', render_mode='human')
# pos_low, pos_high = env.observation_space.low[0], env.observation_space.high[0]
# vel_low, vel_high = env.observation_space.low[1], env.observation_space.high[1]

# # Function to control the car
# def control_car(state):
#     position, velocity = state
#     command = np.zeros(shape=3, dtype="int8")

#     target_position = env.goal_position
    
#     if velocity > 0:
#         command[2] = 1
#     elif velocity <= 0 and position >= -0.5:
#         command[0] = 1
#     else:
#         command[2] = 1
#     return command 
    
# episodes = 10 
# for episode in range(episodes):
#     state, info = env.reset()
#     done = False
    
#     # Track the number of steps in each episode
#     step_count = 0
    
#     while True:

#         # Break if the car reaches the goal or if the episode takes too many steps
#         if done or step_count > 50000:  # Adjust the step limit as needed
#             command = np.zeros(shape=3, dtype="int8")
#             command[1] = 1
#             state, reward, done, truncated, info = env.step(env.action_space.sample(command))
#             env.render()
        
#         else:
#             # Choose an action using the control_car function
#             command = control_car(state)
            
#             # Take a step in the environment
#             state, reward, done, truncated, info = env.step(env.action_space.sample(command))
#             # Increment the step count
#             step_count += 1
            
#             # Render the environment
#             env.render()

# env.close()