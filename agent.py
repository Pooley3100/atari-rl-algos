import random as rnd
import numpy as np
import torch
from torchviz import make_dot


class Agent:
    def __init__(self, env):
        self.current_state = env.reset()

    def step(self, env, approximate, model, epsilon, device, replay_mem, REPLAY_MIN):       
        if approximate:
            # No epsilon used
            current_state_a = np.array([self.current_state], copy=False)
            current_state_t = torch.tensor(current_state_a).to(device)
            # Action probabilities
            action_probs, value= model(current_state_t) # TODO: REINFORCE will not return value (critic)
            action_probs_n = action_probs.cpu().detach().numpy()
            action = np.random.choice(env.action_space.n, p=action_probs_n.squeeze(0))
            log_prob = torch.log(action_probs.squeeze(0)[action]).cpu()
        else:
            # Epsilon used and replay memory must be filled else random action
            random = rnd.uniform(0, 1)
            value = None
            action_probs = None
            if random <= epsilon or len(replay_mem) < REPLAY_MIN:
                action = env.action_space.sample()
            else:
                current_state_a = np.array([self.current_state], copy=False)
                current_state_t = torch.tensor(current_state_a).to(device)
                _, action = torch.max(model(current_state_t), dim = 1)
                action = int(action.item())

        new_state, reward, done, _ = env.step(action)
        if not approximate:
            transition = (self.current_state, action, reward, new_state, done)
        else:
            transition = (self.current_state, action, reward, new_state, done, log_prob)
        replay_mem.append(transition)

        self.current_state = new_state

        if done:
            # Draw neural network
            # make_dot(action_probs, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")

            self.current_state = env.reset()

        return replay_mem, reward, done, env, value, action_probs, action


        




