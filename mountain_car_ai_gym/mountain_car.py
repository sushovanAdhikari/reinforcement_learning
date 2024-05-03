import gymnasium as gym
import numpy as np

env = gym.make('MountainCarContinuous-v0', render_mode="human")

observation, info = env.reset()

done = False
while not done:
    
    # Change the action. The range is from -1 to 1
    action = [1]
    
    env.render()

    observation, reward, done, truncated, info = env.step(action)

env.close()