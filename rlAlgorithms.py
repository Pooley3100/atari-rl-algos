import collections
import random
import torch
import numpy as np
import torch.nn as nn
import gym

import agent
from statistics import mean

from torch.utils.tensorboard import SummaryWriter

import models

writer = SummaryWriter()


class DQN():
    def __init__(self, model, env, targetModel, optimizer, settings, device, ddqn) -> None:
        self.env = env
        self.ddqn = ddqn
        self.settings = settings
        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.targetModel = targetModel
        # Replay memory
        self.replay_mem = collections.deque(maxlen=settings['MEMORY_SIZE'])
        self.agent = agent.Agent(self.env)
        self.eps = 0
        self.frames = 0

    '''
    every x episodes update target model

    replay memory uses collection dqeue to store transitions. Then send batch size of random selection to neural networks.

    so the q algorithm takes current qs on current state, then future qs on next state (this is with target tho?)
    now it iterates through each one index and uses reward * discount * future q to find next q unless terminal state, current qs[index(action)]
    signifies which gets the new q. Append that to x and y and go to next loop of batch
    now train with this. 
    '''

    def train_DQN(self):
        if len(self.replay_mem) < self.settings['REPLAY_MIN']:
            return False
        # Train works with batch size
        batch = random.sample(self.replay_mem, self.settings['BATCH_SIZE'])

        # Get individual transition from batched replay memory
        current_states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        future_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])

        # Convert to tensor from numpy (supposedly this is faster?)
        # unsqueeze -1 converts 1d arrays to 2d array, (i.e. [3] to [3,1])
        state_t = torch.as_tensor(current_states, dtype=torch.float32).to(self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        fstate_t = torch.as_tensor(future_states, dtype=torch.float32).to(self.device)

        # Now get predicted state action values for all states, 
        # send state to network to get values for each action
        # gather along columns to get predicted q values for each action in the batchd
        # then squeeze to make tensor 1d again, giving list of q values for each action
        state_action_qs = self.model(state_t).gather(1, actions_t).squeeze(-1)

        if self.ddqn:
            future_state_actions = self.model(fstate_t).max(1)[1]
            max_future_qs = self.targetModel(fstate_t).gather(1, future_state_actions.unsqueeze(-1)).squeeze(-1)
        else:
            future_qs = self.targetModel(fstate_t)
            max_future_qs = future_qs.max(1)[0]

        # One trick for dones
        # targets = max_future_qs*DISCOUNT*rewards_t*(1 - dones_t)
        # Another
        max_future_qs[dones_t] = 0.0
        max_future_qs = max_future_qs.detach()
        targets = max_future_qs * self.settings['DISCOUNT'] + rewards_t

        # different loss functions
        # loss = F.smooth_l1_loss(state_action_qs, targets)
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(state_action_qs, targets)
        writer.add_scalar("Loss/train", loss, self.frames)
        loss.backward()
        self.optimizer.step()

        return True

    def play(self):
        epsilon = self.settings['EPSILON_START']
        frames = 0
        reward_buf = collections.deque(maxlen=10)
        for eps in range(200000):
            done = False
            reward_sum = 0
            self.eps = eps

            while not done:
                frames += 1
                self.frames = frames

                if frames <= self.settings['FINAL_EXPLORATION']:
                    epsilon = epsilon - ((self.settings['EPSILON_START'] - self.settings['EPSILON_END'])/self.settings['FINAL_EXPLORATION'])

                if eps % 100 == 0:
                    self.env.render()

                approximate = False

                self.replay_mem, reward, done, self.env, _, _, _ = self.agent.step(self.env, approximate, self.model, epsilon,
                                                                          self.device, self.replay_mem,
                                                                          self.settings['REPLAY_MIN'])

                # Train method
                self.train_DQN()

                reward_sum += reward
                writer.add_scalar('Epsilon', epsilon, frames)

                if frames % self.settings['TARGET_UPDATE'] == 0:
                    self.targetModel.load_state_dict(self.model.state_dict())
                    torch.save(self.model.state_dict(), 'Models/dqnWeights')

            reward_buf.append(reward_sum)
            writer.add_scalar('Total/Reward', np.mean(reward_buf), eps)
            writer.add_scalar('Total/Epsilon', epsilon, eps)

            # Current logging
            if eps % 5 == 0:
                print('Episode Mean', np.mean(reward_buf))
                print('Epsilon', epsilon)
                print('Episode Number', eps)
                writer.flush()


class E_SARSA():
    def __init__(self, model, env, targetModel, optimizer, settings, device) -> None:
        self.env = env
        self.settings = settings
        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.targetModel = targetModel
        # Replay memory
        self.replay_mem = collections.deque(maxlen=settings['MEMORY_SIZE'])
        self.agent = agent.Agent(self.env)
        self.epsilon = self.settings['EPSILON_START']

    def train_E_SARSA(self):
        if len(self.replay_mem) < self.settings['REPLAY_MIN']:
            return False

        # Train works with batch size
        batch = random.sample(self.replay_mem, self.settings['BATCH_SIZE'])

        # Get individual transition from batched replay memory
        current_states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        future_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])

        # Convert to tensor from numpy
        state_t = torch.as_tensor(current_states, dtype=torch.float32).to(self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        fstate_t = torch.as_tensor(future_states, dtype=torch.float32).to(self.device)

        # This is predict
        state_action_qs = self.model(state_t).gather(1, actions_t).squeeze(-1)

        future_qs = self.targetModel(fstate_t)

        expected_q = np.zeros(future_qs.shape[0])
        max_future_qs = torch.sort(future_qs, dim=1, descending=True)[0].cpu().detach()
        num_greedy_actions = np.ones(future_qs.shape[0])

        future_max_qs_num = np.array(max_future_qs)
        future_qs_num = future_qs.cpu().detach().numpy()
        for i in range(future_qs_num.shape[0]):
            col = 0
            while future_max_qs_num[i][col] == future_max_qs_num[i][col + 1]:
                num_greedy_actions[i] += 1
                col += 1
                if col >= self.env.action_space.n - 1:
                    break

        # Probabilities from geek to geek article
        # Action probabilites
        non_greedy_probability = self.epsilon / self.env.action_space.n
        greedy_probability = ((1 - self.epsilon) / num_greedy_actions) + non_greedy_probability

        for i in range(future_qs_num.shape[0]):
            for j in range(self.env.action_space.n):
                if future_qs_num[i][j] == max_future_qs[i][0]:
                    expected_q[i] += future_qs_num[i][j] * greedy_probability[i]
                else:
                    expected_q[i] += future_qs_num[i][j] * non_greedy_probability

        expected_q = torch.tensor(expected_q, dtype=torch.float32).to(self.device)

        expected_q[dones_t] = 0.0
        targets = rewards_t + self.settings['DISCOUNT'] * expected_q

        self.optimizer.zero_grad()
        loss = nn.MSELoss()(state_action_qs, targets)
        loss.backward()
        self.optimizer.step()

        return True

    def play(self):
        frames = 0
        for eps in range(100000):
            done = False
            reward_buf = 0

            while not done:
                frames += 1
                if frames <= self.settings['FINAL_EXPLORATION']:
                    self.epsilon = self.epsilon - ((self.settings['EPSILON_START'] - self.settings['EPSILON_END']) / self.settings['FINAL_EXPLORATION'])

                if eps % 100 == 0:
                    self.env.render()

                approximate = False

                self.replay_mem, reward, done, self.env, _, _, _ = self.agent.step(self.env, approximate, self.model,
                                                                                   self.epsilon, self.device,
                                                                                   self.replay_mem,
                                                                                   self.settings['REPLAY_MIN'])

                # Train method

                self.train_E_SARSA()

                reward_buf += reward

                if frames % self.settings['TARGET_UPDATE'] == 0:
                    self.targetModel.load_state_dict(self.model.state_dict())
                    torch.save(self.model.state_dict(), 'Models/eSarsaWeights-Pong')

            writer.add_scalar('Total/Reward', reward_buf, eps)
            writer.add_scalar('Total/Epsilon', self.epsilon, eps)

            # Current logging
            if eps % 5 == 0:
                print('Episode Reward', np.mean(reward_buf))
                print('Episode', eps)
                print('Epsilon', self.epsilon)
                writer.flush()


class REINFORCE:
    def __init__(self, model, env, optimizer, settings, device) -> None:
        self.model = model
        self.env = env
        self.optimizer = optimizer
        self.batched_mem = []
        self.settings = settings
        self.device = device
        self.agent = agent.Agent(self.env)
        self.replay_mem = []


    def reinforce_train(self, model, discount_rewards):
        # Get individual transition from batched replay memory
        current_states = np.array([transition[0] for transition in self.batched_mem])
        current_states_t = torch.tensor(current_states, dtype=torch.float32).to(self.device)

        action_probs, _ = model(current_states_t)
        log_probs = torch.stack([transition[5] for transition in self.batched_mem]).to(self.device)

        loss = []
        policy_gradient = -log_probs * discount_rewards

        self.optimizer.zero_grad()
        policy_gradient.sum().backward()
        self.optimizer.step()

        return True


    def calc_discount(self, replay_mem):
        rewards = np.array([transition[2] for transition in replay_mem])
        discount_rewards = []
        for i in range(len(rewards)):
            discount = 0
            gamma = 0
            for j in rewards[i:]:
                discount = discount + self.settings['DISCOUNT']**gamma * j
                gamma += 1
            discount_rewards.append(discount)

        discount_rewards = np.array(discount_rewards).astype(np.float32)
        discount_rewards = (discount_rewards - discount_rewards.mean()) / (discount_rewards.std() + 1e-6)
        # Baseline removing mean is added here along with dividing by discount rewards standard deviation to provide normalization, std deviation reference from chris yoon article.

        discount_rewards = torch.tensor(discount_rewards).to(self.device)
        return discount_rewards

    # discont rewards need to be calculated, only once episode done, this is batched rewards
    # single policy estimator used in the nework
    # times discount rewards with log of policy estimators, gather with actions as indicies 
    # loss is negative of that averaged
    def play(self):
        discount_rewards = []
        reward_buf = 0

        for eps in range(3000):
            done = False
            reward_buf = 0

            while not done:
                if eps % 50 == 0:
                    self.env.render()

                self.replay_mem, reward, done, self.env, _, _, action = self.agent.step(self.env, True, self.model, None, self.device, self.replay_mem,
                                                           self.settings['REPLAY_MIN'])

                if done:
                    discount_rewards = (self.calc_discount(self.replay_mem))
                    self.batched_mem.extend(self.replay_mem)
                    self.replay_mem.clear()

                    self.reinforce_train(self.model, discount_rewards)
                    # Clear Memory once update done
                    discount_rewards = []
                    self.batched_mem = []

                reward_buf += reward

            writer.add_scalar('Total/Reward', reward_buf, eps)

            # Current logging
            if eps % 5 == 0:
                print('Reward' , reward_buf)
                print('Episode', eps)
                writer.flush()


class ActorCritic:
    def __init__(self, env, settings, device, in_channels, out_channels):
        if settings['Model'] == 'Basic':
            self.Actor = models.ActorBasic(in_channels, out_channels).to(device)
            self.Critic = models.CriticBasic(in_channels).to(device)
        else:
            self.Actor = models.ActorAdvanced(in_channels, out_channels).to(device)
            self.Critic = models.CriticAdvanced(in_channels, out_channels).to(device)
        self.env = env
        self.settings = settings
        self.agent = agent.Agent(self.env)
        self.ActorOptim = torch.optim.Adam(self.Actor.parameters(), lr=settings['LEARNING_RATE'])
        self.CriticOptim = torch.optim.Adam(self.Critic.parameters(), lr=settings['LEARNING_RATE'])
        self.device = device

    def play(self):
        # Loop through epsiodes, every batch train
        total_rewards = 0

        replay_mem = []

        ep_count = 0
        state = self.env.reset()
        for eps in range(100000000):
            #Find action from state using proability dist

            state_n = np.array([state], copy=False)
            probs = self.Actor(torch.as_tensor(state_n, dtype=torch.float32).to(self.device))
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            fstate, reward, done, _ = self.env.step(action[0].cpu().detach().data.numpy())

            # Get Value
            value = self.Critic(torch.as_tensor(state_n, dtype=torch.float32).to(self.device))

            # Get log probability on action
            log_prob = dist.log_prob(action)

            replay_mem.append((state, action, reward, fstate, done, log_prob, value))

            state = fstate

            total_rewards += reward

            if done or (eps+1 % self.settings['BATCH_SIZE'] == 0):
                self.train(replay_mem, ep_count)

                replay_mem = []

                # Logging (temporary I guess)
                if done:
                    state = self.env.reset()
                    ep_count += 1
                    print(np.mean(total_rewards))
                    print(ep_count)
                    writer.add_scalar('Total/Reward', np.mean(total_rewards), ep_count)
                    writer.flush()
                    total_rewards = 0

    def train(self, replay_mem, ep_count):
        rewards = np.array([transition[2] for transition in replay_mem])
        dones = np.array([transition[4] for transition in replay_mem])
        fstates = np.array([transition[3] for transition in replay_mem])
        values = torch.stack([transition[6] for transition in replay_mem]).to(self.device)
        # fstates = torch.tensor(fstates).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        log_probs = torch.stack([transition[5] for transition in replay_mem]).to(self.device)
        dones = torch.tensor(dones, dtype=torch.long).to(self.device)

        # Find Q_Values with Discounted Rewards
        q_vals = np.zeros((len(rewards), 1))
        fstates_n = np.array([fstates[-1]], copy=False)
        q_val = self.Critic.forward(torch.as_tensor(fstates_n).to(self.device))
        writer.add_scalar('Critic/Values', q_val, ep_count)
        for i in reversed(range(len(rewards))):
            q_val = rewards[i] + self.settings['DISCOUNT']*q_val*(1-dones[i])
            q_vals[i] = q_val.cpu().detach().numpy()

        q_vals = torch.as_tensor(q_vals, dtype=torch.float32).to(self.device)

        advantage = q_vals - values
        value_loss = 0.5 * advantage.pow(2).mean() # MSE loss, multiplying by 0.5 as better option found on chris yoon.
        self.CriticOptim.zero_grad()
        value_loss.backward()
        self.CriticOptim.step()

        action_loss = (-log_probs * advantage.detach()).mean()
        self.ActorOptim.zero_grad()
        action_loss.backward()
        self.ActorOptim.step()


class WorkerA2C:
    def __init__(self, model, settings, device, env_name):
        self.ActorCritic = model
        self.settings = settings

        self.device = device

        self.env = gym.make(env_name)
        self.state = self.env.reset()

    def discount_rewards(self, replay_mem):
        q_vals = []
        rewards = np.array([transition[2] for transition in replay_mem])
        dones = np.array([transition[4] for transition in replay_mem])
        future_states = np.array([transition[3] for transition in replay_mem])

        fstates = torch.tensor(future_states).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # Find Q_Values with Discounted Rewards
        q_vals = np.zeros((len(rewards), 1))
        q_val = self.ActorCritic.get_values(fstates[-1])
        for i in reversed(range(len(rewards))):
            q_val = rewards[i] + self.settings['DISCOUNT'] * q_val * (1 - dones[i])
            q_vals[i] = q_val.cpu().detach().numpy()

        return q_vals

    def play_sample(self, render):
        # Loop through epsiodes, every batch train

        total_rewards = 0

        replay_mem = []
        values = []
        end_reward = None
        ep_count = 0
        for eps in range(self.settings['BATCH_SIZE']):
            # Find action from state using proability dist
            probs, value = self.ActorCritic(torch.as_tensor(self.state, dtype=torch.float32).to(self.device))
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()

            fstate, reward, done, _ = self.env.step(action.cpu().detach().data.numpy())

            # Get log probability on action
            log_prob = dist.log_prob(action)

            replay_mem.append((self.state, action, reward, fstate, done, log_prob, value))

            self.state = fstate

            values.append(value)

            # use this bit more for tensorboard
            total_rewards += reward

            if done:
                self.state = self.env.reset()
                ep_count += 1
                end_reward = total_rewards

        values = self.discount_rewards(replay_mem)

        return values, replay_mem, end_reward


class TrainA2C:
    def __init__(self, model, settings, optim, device, env_name):
        self.ActorCritic = model
        self.env_name = env_name
        self.settings = settings
        self.optimizer = optim
        self.device = device

    def play(self):
        workers_num = 8  # This could be a hyperparameter
        workers = []
        for i in range(workers_num):
            workers.append(WorkerA2C(self.ActorCritic, self.settings, self.device, self.env_name))

        ep_count = 0
        while True:
            values = []
            log_probs = []
            end_total = []
            values_network = []
            for worker in workers:
                if ep_count % 50 == 0:
                    render = True
                else:
                    render = False

                worker_values, worker_mem, end_reward = worker.play_sample(render)

                log_probs.extend(torch.stack([transition[5] for transition in worker_mem]))
                values.extend(worker_values)
                values_network.extend(([transition[6] for transition in worker_mem]))

                if end_reward is not None:
                    end_total.append(end_reward)

            if len(end_total) > 0:
                # Not quite sure how to get all the rewards as workers can finish at different times
                ep_count += 1
                print(ep_count, np.mean(end_total))
                writer.add_scalar('Total/Reward', np.mean(end_total), ep_count)
                writer.flush()
                end_total = []

            self.train(values, log_probs, values_network)

    def train(self, values, log_probs, values_network):
        # TODO Change this loss algo, this currently does not work very well

        values_n = np.array(values)
        values_t = torch.as_tensor(values_n, dtype=torch.float32).to(self.device)
        log_probs_t = torch.as_tensor(log_probs, dtype=torch.float32).to(self.device)
        values_network_t = torch.stack(values_network).to(self.device)

        advantages = values_t.squeeze(1) - values_network_t
        critic_loss = 0.5 * advantages.pow(2).mean()

        actor_loss = (-log_probs_t * advantages.detach()).mean()

        # Now the loss equation = Total Loss = Action Loss + Value Loss - entropy. No entropy here though currently
        total_loss = critic_loss + actor_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

