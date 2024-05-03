import gymnasium as gym
import numpy as np
import random
import time

# Create the environment
env = gym.make('MountainCar-v0')
pos_low, pos_high = env.observation_space.low[0], env.observation_space.high[0]
vel_low, vel_high = env.observation_space.low[1], env.observation_space.high[1]
pos_num_bins, vel_num_bins = 20, 20
position_bins = np.linspace(pos_low, pos_high, pos_num_bins + 1)
velocity_bins = np.linspace(vel_low, vel_high, vel_num_bins + 1)
actions = 3
q_table = np.zeros((pos_num_bins, vel_num_bins, actions), dtype="float32")
ln = 0.9

episodes = 10000
mode = 0
epsilon_start = 0.8  # Initial epsilon value
epsilon_min = 0.05  # Minimum epsilon value to ensure some exploration remains
epsilon_decay = 0.999  # Epsilon decay factor
epsilon = epsilon_start
done_count = 0


def explore():
    command = np.zeros(shape=3, dtype="int8")
    index = np.random.choice(len(command))
    command[index] = 1
    return (command, index)

def exploit(p,v):
    command = np.zeros(shape=3, dtype="int8")
    values = q_table[p, v, :]
    index_of_highest = np.argmax(values)
    command[index_of_highest] = 1
    return (command, index_of_highest)

def pick_action(mode,p,v,epsilon=0):
    index = None
    if mode == 0:
        chance = round(random.random(), 2)
        if chance < epsilon:
            command, index = explore()
        else: 
            command, index = exploit(p,v)
    else:
        command, index = exploit(p,v)
    return (command, index)

def get_position(pos):
    # Use np.digitize to find the appropriate bin index
    bin_index = np.digitize(pos, position_bins) - 1  # Subtract 1 to get bin index from 0 to 19
    # Ensure the bin index is within range 0 to 19
    bin_index = max(0, min(bin_index, 19))
    return bin_index

def get_velocity(vel):
    # Use np.digitize to find the appropriate bin index
    bin_index = np.digitize(vel, velocity_bins) - 1  # Subtract 1 to get bin index from 0 to 19
    # Ensure the bin index is within range 0 to 19
    bin_index = max(0, min(bin_index, 19))
    return bin_index

def update_q_table(p,v,new_p,new_v,reward,action_index):
    a = action_index
    max_next_reward = np.max(q_table[new_p, new_v, :])
    q_table[p,v,a] = q_table[p,v,a] + ln * (reward + 0.9 * max_next_reward - q_table[p,v,a])



for episode in range(episodes):
    state, info = env.reset()
    done, truncated = False, False
    p, v = state[0], state[1]
    p = get_position(p)
    v = get_velocity(v)
    while True:
        a, action_index = pick_action(mode,p,v,epsilon)
        state, reward, done, truncated, info = env.step(env.action_space.sample(a))
        new_p = get_position(state[0])
        new_v = get_velocity(state[1])
        update_q_table(p,v,new_p,new_v,reward,action_index)
        p,v = new_p, new_v
        if done:
            done_count += 1
            break
        if done_count > 15:
            if truncated:
                break
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
env.close()


env = gym.make('MountainCar-v0', render_mode='human')
mode = 1
state, info = env.reset()
done = False
step_count = 0
p, v = state[0], state[1]
p = get_position(p)
v = get_velocity(v)
control_block = True
while control_block:
    a = pick_action(mode,p, v)[0]
    state, reward, done, truncated, info = env.step(env.action_space.sample(a))
    env.render()
    new_p = get_position(state[0])
    new_v = get_velocity(state[1])
    p, v = new_p, new_v
    while done:
        delay_seconds = 10
        control_block = False
        time.sleep(delay_seconds)
        break
env.close()