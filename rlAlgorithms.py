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

    # TODO: Seperate training file for each algorithm
    # TODO: DQN algorithm see why previous method did not work
    # TODO: alternate ways to calculate dqn
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
        # rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        # dones_t = torch.as_tensor(dones, dtype=torch.long).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_t = torch.ByteTensor(dones).to(self.device)
        fstate_t = torch.as_tensor(future_states, dtype=torch.float32).to(self.device)

        # Now get predicted state action values for all states, 
        # send state to network to get values for each action
        # gather along columns to get predicted q values for each action in the batch
        # then squeeze to make tensor 1d again, giving list of q values for each action
        state_action_qs = self.model(state_t).gather(1, actions_t).squeeze(-1)

        # Now send next states to target model, get action values, find max value and just take that
        # Dim 0 is batch
        future_qs = self.targetModel(fstate_t)
        max_future_qs = future_qs.max(1)[0]

        # One trick for dones
        # targets = max_future_qs*DISCOUNT*rewards_t*(1 - dones_t)
        # Another
        max_future_qs[dones_t] = 0.0
        max_future_qs = max_future_qs.detach()
        targets = max_future_qs * self.settings['DISCOUNT'] + rewards_t

        # action_q_values = torch.gather(input=current_qs, dim=1, index=actions_t)

        # TODO different loss functions
        # loss = F.smooth_l1_loss(state_action_qs, targets)
        loss = nn.MSELoss()(state_action_qs, targets)
        writer.add_scalar("Loss/train", loss, self.frames)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return True

    def play(self):
        epsilon = self.settings['EPSILON_START']
        reward_buf = collections.deque(maxlen=10000)
        frames = 0
        for eps in range(1000):
            done = False
            reward_buf.clear()
            self.eps = eps

            while not done:
                frames += 1
                self.frames = frames
                # TODO: Different way of doing this 
                # TODO: alternate epsilon methods for non approximation
                if (len(self.replay_mem) > self.settings['REPLAY_MIN']):
                    epsilon = max(epsilon * self.settings['EPSILON_DECAY'], self.settings['EPSILON_END'])

                if eps % 100 == 0:
                    self.env.render()

                # TODO: Calculate whether using approximate methods or not
                approximate = False

                self.replay_mem, reward, done, self.env, _, _, _ = self.agent.step(self.env, approximate, self.model, epsilon,
                                                                          self.device, self.replay_mem,
                                                                          self.settings['REPLAY_MIN'])

                # Train method

                self.train_DQN()

                reward_buf.append(reward)
                writer.add_scalar('Epsilon', epsilon, frames)

                if frames % self.settings['TARGET_UPDATE'] == 0:
                    self.targetModel.load_state_dict(self.model.state_dict())
                    torch.save(self.model.state_dict(), 'Models/dqnWeights')

            epReward = mean(reward_buf)
            writer.add_scalar('Total/Reward', epReward, eps)
            writer.add_scalar('Total/Epsilon', epsilon, eps)

            # Current logging
            if eps % 5 == 0:
                print(np.mean(reward_buf))
                print(eps)
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

        # Convert to tensor from numpy (supposedly this is faster?)
        # unsqueeze -1 converts 1d arrays to 2d array, (i.e. [3] to [3,1])
        state_t = torch.as_tensor(current_states, dtype=torch.float32).to(self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).to(self.device)
        # rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        # dones_t = torch.as_tensor(dones, dtype=torch.long).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_t = torch.LongTensor(dones).to(self.device)
        fstate_t = torch.as_tensor(future_states, dtype=torch.float32).to(self.device)

        # One method
        # This is predict
        state_action_qs = self.model(state_t).gather(1, actions_t).squeeze(-1)
        # state_action_qs = self.model(state_t).cpu().detach().numpy()
        # state_action_qs = self.model(state_t)
        # Calculate number of greedy actions

        future_qs = self.targetModel(fstate_t)

        expected_q = np.zeros(future_qs.shape[0])
        max_future_qs = torch.sort(future_qs, dim=1, descending=True)[0].cpu().detach()
        greedy_actions = np.ones(future_qs.shape[0])

        future_max_qs_num = np.array(max_future_qs)
        future_qs_num = future_qs.cpu().detach().numpy()
        for i in range(future_qs_num.shape[0]):
            col = 0
            while future_max_qs_num[i][col] == future_max_qs_num[i][col + 1]:
                greedy_actions[i] += 1
                col += 1

        # Action probabilites
        non_greedy_action_probability = self.epsilon / self.env.action_space.n
        greedy_action_probability = ((1 - self.epsilon) /greedy_actions) + non_greedy_action_probability
        # greedy_action_probability = (1 - self.epsilon)
        # non_greedy_action_probability = self.epsilon

        for i in range(future_qs_num.shape[0]):
            for j in range(self.env.action_space.n):
                if future_qs_num[i][j] == max_future_qs[i][0]:
                    # test = future_qs_num[i][j] * greedy_action_probability
                    # test1 = future_qs_num[i][j]
                    expected_q[i] += future_qs_num[i][j] * greedy_action_probability[0]
                else:
                    expected_q[i] += future_qs_num[i][j] * non_greedy_action_probability

        expected_q = torch.tensor(expected_q, dtype=torch.float32).to(self.device)

        expected_q[dones_t] = 0.0
        targets = rewards_t + self.settings['DISCOUNT'] * expected_q

        # loss = F.smooth_l1_loss(state_action_qs, targets)

        # One method <<<<<<<<<<<<<,

        # Now get predicted state action values for all states,
        # send state to network to get values for each action
        # gather along columns to get predicted q values for each action in the batch
        # then squeeze to make tensor 1d again, giving list of q values for each action
        # state_action_qs = self.model(state_t).gather(1, actions_t).squeeze(-1)

        # Now send next states to target model, get action values, find max value and just take that
        # Dim 0 is batch
        # future_qs = self.targetModel(fstate_t)
        # mean_qs = future_qs.mean(1)

        # One trick for dones
        # targets = max_future_qs*DISCOUNT*rewards_t*(1 - dones_t)
        # Another
        # mean_qs[dones_t] = 0.0
        # mean_qs = mean_qs.detach()
        # targets = mean_qs * self.settings['DISCOUNT'] + rewards_t

        # action_q_values = torch.gather(input=current_qs, dim=1, index=actions_t)

        # TODO different loss functions
        # loss = F.smooth_l1_loss(state_action_qs, targets)

        # calculate if action is greddy or if action is non greedy, then act upon that information

        # Calculate the sum of next q values * their probabilities

        # If using non-greedy action, then choose with probability distribution from softmax?
        # targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1).to(self.device)

        loss = nn.MSELoss()(state_action_qs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return True

    def play(self):
        self.epsilon = self.settings['EPSILON_START']
        reward_buf = collections.deque(maxlen=10000)
        frames = 0
        for eps in range(1000):
            done = False
            reward_buf.clear()

            while not done:
                frames += 1
                # TODO: Different way of doing this 
                # TODO: alternate epsilon methods for non approximation
                if (len(self.replay_mem) > self.settings['REPLAY_MIN']):
                    self.epsilon = max(self.epsilon * self.settings['EPSILON_DECAY'], self.settings['EPSILON_END'])

                if eps % 100 == 0:
                    self.env.render()

                # TODO: Calculate whether using approximate methods or not
                approximate = False

                self.replay_mem, reward, done, self.env, _, _, _ = self.agent.step(self.env, approximate, self.model,
                                                                                   self.epsilon, self.device,
                                                                                   self.replay_mem,
                                                                                   self.settings['REPLAY_MIN'])

                # Train method

                self.train_E_SARSA()

                reward_buf.append(reward)

                if frames % self.settings['TARGET_UPDATE'] == 0:
                    self.targetModel.load_state_dict(self.model.state_dict())
                    torch.save(self.model.state_dict(), 'Models/eSarsaWeights')

            writer.add_scalar('Total/Reward', np.mean(reward_buf), eps)
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

    # TODO: Alternate methods for calculating reinforce method.
    # Now for training if using REINFORCE METHOD
    # This is called end of each episode
    def reinforce_train(self, model, discount_rewards):
        # Get individual transition from batched replay memory
        current_states = np.array([transition[0] for transition in self.batched_mem])
        actions = np.array([transition[1] for transition in self.batched_mem])

        current_states_t = torch.tensor(current_states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(-1).to(self.device)

        # TODO Softmax not currently used, or option in some networks
        action_probs, _ = model(current_states_t)
        log_probs = torch.stack([transition[5] for transition in self.batched_mem]).to(self.device)
        # log_probs = torch.tensor(log_probs, dtype=torch.float32).to(self.device)
        # for i in range(action_probs.shape[0]):
        #     action_prob_max = np.random.choice(self.env.action_space.n, p=np.squeeze(action_probs[i].cpu().detach().numpy()))
        #     log_prob = torch.log(action_probs[i][action_prob_max])
        #     log_probs.append(log_prob)

        # log_probs = torch.stack(log_probs).to(self.device)

        loss = []
        # for prob, discount in zip(log_probs, discount_rewards):
        #     loss.append(-prob * discount)
        policy_gradient = -log_probs * discount_rewards

        #policy_gradient = torch.stack(policy_gradient).sum()
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

        # TODO some baseliene can be added here
        discount_rewards = np.array(discount_rewards).astype(np.float32)
        discount_rewards = (discount_rewards - discount_rewards.mean()) / (discount_rewards.std() + 1e-9)
        ## Trial <<< TODO Found on github chris yoon as a way to stabelise and normalise

        discount_rewards = torch.tensor(discount_rewards).to(self.device)
        return discount_rewards

    # discont rewards need to be calculated, only once episode done, this is batched rewards
    # single policy estimator used in the nework
    # times discount rewards with log of policy estimators, gather with actions as indicies 
    # loss is negative of that averaged
    # then empty batch ?
    # End
    def play(self):
        discount_rewards = []
        reward_buf = collections.deque(maxlen=10000)

        for eps in range(3000):
            done = False
            reward_buf.clear()

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

                reward_buf.append(reward)

            writer.add_scalar('Total/Reward', np.mean(reward_buf), eps)

            # Current logging
            if eps % 5 == 0:
                print(mean(reward_buf))
                print(eps)
                writer.flush()


class ActorCritic:
    def __init__(self, env, settings, device, in_channels, out_channels):
        self.Actor = models.ActorBasic(in_channels, out_channels).to(device)
        self.Critic = models.CriticBasic(in_channels).to(device)
        self.env = env
        self.settings = settings
        self.agent = agent.Agent(self.env)
        self.ActorOptim = torch.optim.Adam(self.Actor.parameters(), lr=settings['LEARNING_RATE'])
        self.CriticOptim = torch.optim.Adam(self.Critic.parameters(), lr=settings['LEARNING_RATE'])
        self.device = device

    def play(self):
        # Loop through epsiodes, every batch train
        # TODO Add multiple workers for asynchronicity
        # TODO change up baselines for this and REINFORCE
        total_rewards = 0

        replay_mem = []

        ep_count = 0
        state = self.env.reset()
        for eps in range(100000000):
            #Find action from state using proability dist
            probs = self.Actor(torch.as_tensor(state, dtype=torch.float32).to(self.device))
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()

            fstate, reward, done, _ = self.env.step(action.cpu().detach().data.numpy())

            # Get Value
            value = self.Critic(torch.as_tensor(state, dtype=torch.float32).to(self.device))

            # Get log probability on action
            log_prob = dist.log_prob(action)

            replay_mem.append((state, action, reward, fstate, done, log_prob, value))

            state = fstate

            total_rewards += reward

            if done or (eps+1 % self.settings['BATCH_SIZE'] == 0):
                self.train(replay_mem)

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

    def train(self, replay_mem):
        rewards = np.array([transition[2] for transition in replay_mem])
        dones = np.array([transition[4] for transition in replay_mem])
        fstates = np.array([transition[3] for transition in replay_mem])
        values = torch.stack([transition[6] for transition in replay_mem]).to(self.device)
        fstates = torch.tensor(fstates).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        log_probs = torch.stack([transition[5] for transition in replay_mem]).to(self.device)
        dones = torch.tensor(dones, dtype=torch.long).to(self.device)

        # Find Q_Values with Discounted Rewards
        q_vals = np.zeros((len(rewards), 1))
        q_val = self.Critic.forward(fstates[-1])
        for i in reversed(range(len(rewards))):
            q_val = rewards[i] + self.settings['DISCOUNT']*q_val*(1-dones[i])
            q_vals[i] = q_val.cpu().detach().numpy()


        # Convert values, q values and log to tensor to allow loss calculation
        # q_vals = torch.tensor(q_vals, dtype=torch.float32).to(self.device)
        # q_vals = torch.FloatTensor(q_vals).to(self.device)
        # values = torch.tensor(values, dtype=torch.float32).to(self.device)
        q_vals = torch.as_tensor(q_vals, dtype=torch.float32).to(self.device)

        # values = values.max(1)[0]
        # log_probs = torch.stack(log_probs).to(self.device)

        # Now the loss equation = Total Loss = Action Loss + Value Loss - Entropy
        advantage = q_vals - values
        value_loss = 0.5 * advantage.pow(2).mean()
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

        self.entropy_sum = 0
        self.device = device

        self.env = gym.make(env_name)

    def true_rewards(self, replay_mem):
        q_vals = []
        rewards = np.array([transition[2] for transition in replay_mem])
        dones = np.array([transition[4] for transition in replay_mem])
        future_states = np.array([transition[3] for transition in replay_mem])

        fstates = torch.tensor(future_states).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # Find Q_Values with Discounted Rewards
        q_vals = np.zeros((len(rewards), 1))
        q_val = self.ActorCritic.get_critic(fstates[-1])
        for i in reversed(range(len(rewards))):
            q_val = rewards[i] + self.settings['DISCOUNT'] * q_val * (1 - dones[i])
            q_vals[i] = q_val.cpu().detach().numpy()

        return q_vals

    def play_sample(self, model, render):
        self.ActorCritic = model
        # Loop through epsiodes, every batch train
        # TODO change up baselines for this and REINFORCE
        total_rewards = 0

        replay_mem = []
        values = []
        end_reward = None
        ep_count = 0
        state = self.env.reset()
        for eps in range(self.settings['BATCH_SIZE']):
            # Find action from state using proability dist
            probs, value = self.ActorCritic(torch.as_tensor(state, dtype=torch.float32).to(self.device))
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()

            fstate, reward, done, _ = self.env.step(action.cpu().detach().data.numpy())

            # Get log probability on action
            log_prob = dist.log_prob(action)

            replay_mem.append((state, action, reward, fstate, done, log_prob, value))

            state = fstate

            # values.append(value.cpu().detach().numpy()[0, 0])
            values.append(value)
            # log_probs.append(torch.log(action_probs.squeeze(0))[action])

            # use this bit more for tensorboard
            total_rewards += reward

            # action_prob = action_probs.cpu().detach().numpy()
            # entropy = -np.sum(np.mean(action_prob) * np.log(action_prob))
            # self.entropy_sum += entropy

            if done:
                state = self.env.reset()
                ep_count += 1
                end_reward = total_rewards

        values = self.true_rewards(replay_mem)
        actions = ([transition[1] for transition in replay_mem])
        states = np.array([transition[0] for transition in replay_mem])
        return values, actions, states, end_reward


class TrainA2C:
    def __init__(self, model, settings, optim, device, env_name):
        self.ActorCritic = model
        self.env_name = env_name
        self.settings = settings
        self.entropy_sum = 0
        self.optimizer = optim
        self.device = device
        self.entropy_coef = 0.01 # TODO Also should be hyperparameters
        self.value_coef = 0.5

    def play(self):
        workers_num = 8  # TODO make this a hyperparameter
        workers = []
        for i in range(workers_num):
            workers.append(WorkerA2C(self.ActorCritic, self.settings, self.device, self.env_name))

        ep_count = 0
        while True:
            values_t = []
            actions_t = []
            states_t = []
            end_total = []
            for worker in workers:
                if ep_count % 50 == 0:
                    render = True
                else:
                    render = False
                values, actions, states, end_reward = worker.play_sample(self.ActorCritic, render)
                for i in range(len(values)):
                    values_t.append(values[i])
                    actions_t.append(actions[i])
                    states_t.append(states[i])
                    if end_reward is not None:
                        end_total.append(end_reward)

            if len(end_total) > 1:
                ep_count += 1
                print(ep_count, mean(end_total))
                writer.add_scalar('Total/Reward', np.mean(end_total), ep_count)
                writer.flush()
                end_total = []

            self.train(values_t, actions_t, states_t)

    def train(self, values_t, actions_t, states_t):
        # TODO Change this to try other loss algos

        values_t = np.array(values_t)
        values_t = torch.as_tensor(values_t, dtype=torch.float32).to(self.device)
        actions_t = torch.as_tensor(actions_t, dtype=torch.float32).to(self.device)
        states_t = np.array(states_t)
        states_t = torch.as_tensor(states_t, dtype=torch.float32).to(self.device)
        values, log_probs, entropy = self.ActorCritic.eval_action(states_t, actions_t)

        advantages = values_t - values
        critic_loss = advantages.pow(2).mean()

        actor_loss = -(log_probs * advantages.detach()).mean()
        # TODO sort of get this, have found mutiple total_loss calculations
        total_loss = (self.value_coef * critic_loss) + actor_loss - (self.entropy_coef * entropy)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ActorCritic.parameters(), 0.5)
        self.optimizer.step()

