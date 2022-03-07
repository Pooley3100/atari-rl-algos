import random as rnd
import numpy as np
import torch

class Agent():
    def __init__(self, env):
        self.current_state = env.reset()

    def step(self, env, approximate, model, epsilon, device, replay_mem, REPLAY_MIN):       
        if approximate:
            # No epsilon used
            current_state_a = np.array([self.current_state], copy=False)
            current_state_t = torch.tensor(current_state_a).to(device)
            # Action probabilities
            action_probs = model(current_state_t).cpu().detach().squeeze().numpy()
            action = np.random.choice(np.arange(env.action_space.n), p=action_probs)

        else:
            # Epsilon used and replay memory must be filled else random action
            random = rnd.uniform(0,1)
            if random <= epsilon or len(replay_mem) < REPLAY_MIN:
                action = env.action_space.sample()
            else:
                current_state_a = np.array([self.current_state], copy=False)
                current_state_t = torch.tensor(current_state_a).to(device)
                _, action = torch.max(model(current_state_t), dim = 1)
                action = int(action.item())

        new_state, reward, done, _ = env.step(action)
        transition = (self.current_state, action, reward, new_state, done)
        replay_mem.append(transition)

        self.current_state = new_state

        if done:
            self.current_state = env.reset()

        return replay_mem, reward, done, env


        




