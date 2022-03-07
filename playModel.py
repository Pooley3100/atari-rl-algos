from tkinter import E
import models
import envWrapper
import torch
import numpy as np
import gym
from gym.wrappers import Monitor

#Setup env and device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

env = envWrapper.makeEnv('PongNoFrameskip-v4')
env = Monitor(env, 'Project/Videos', force=True)

#Load Model
#TODO: Select which model the network is built of.
model = models.NeuralNetworkAdvanced(env.observation_space.shape, env.action_space.n).to(device)
model.load_state_dict(torch.load('Project/Models/dqnWeights'))

current_state = env.reset()
count = 0
while count < 3:
    env.render()
    # Depending on epsilon get action from target network or random action

    current_state_a = np.array([current_state], copy=False)
    current_state_t = torch.tensor(current_state_a).to(device)
    _, action = torch.max(model(current_state_t), dim = 1)
    action = int(action.item())
    
    new_state, reward, done, _ = env.step(action)

    current_state = new_state
    if done:
        count += 1
        current_state = env.reset()

env.close()