import collections
import random
import torch
import numpy as np
import torch.nn as nn
import agent
from statistics import mean

from torch.utils.tensorboard import SummaryWriter
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

    #TODO: Seperate training file for each algorithm
    #TODO: DQN algorithm see why previous method did not work
    #TODO: alternate ways to calculate dqn
    def train_DQN(self):
        if len(self.replay_mem) < self.settings['REPLAY_MIN']:
            return False
        # Train works with batch size
        batch = random.sample(self.replay_mem, self.settings['BATCH_SIZE'])

        #Get individual transition from batched replay memory
        current_states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        future_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])

        # Convert to tensor from numpy (supposedly this is faster?)
        # unsqueeze -1 converts 1d arrays to 2d array, (i.e. [3] to [3,1])
        state_t = torch.as_tensor(current_states, dtype=torch.float32).to(self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).to(self.device)    
        #rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)        
        #dones_t = torch.as_tensor(dones, dtype=torch.long).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_t = torch.ByteTensor(dones).to(self.device)
        fstate_t = torch.as_tensor(future_states, dtype=torch.float32).to(self.device)

        # Now get predicted state action values for all states, 
        # send state to network to get values for each action
        # gather along columns to get predicted q values for each action in the batch
        # then squeeze to make tensor 1d again, giving list of q values for each action
        state_action_qs = self.model(state_t).gather(1, actions_t).squeeze(-1)

        #Now send next states to target model, get action values, find max value and just take that
        #Dim 0 is batch
        future_qs = self.targetModel(fstate_t)
        max_future_qs = future_qs.max(1)[0]

        # One trick for dones
        #targets = max_future_qs*DISCOUNT*rewards_t*(1 - dones_t)
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
                frames+=1
                self.frames = frames
                # TODO: Different way of doing this 
                # TODO: alternate epsilon methods for non approximation
                if(len(self.replay_mem) > self.settings['REPLAY_MIN']):
                    epsilon = max(epsilon*self.settings['EPSILON_DECAY'], self.settings['EPSILON_END'])

                if eps % 100 == 0:
                    self.env.render()

                # TODO: Calculate whether using approximate methods or not
                approximate = False
                
                self.replay_mem, reward, done, self.env = self.agent.step(self.env, approximate, self.model, epsilon, self.device, self.replay_mem, self.settings['REPLAY_MIN'])
                
                #Train method
                
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

class PPO():
    pass

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

        #Get individual transition from batched replay memory
        current_states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        future_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])

        # Convert to tensor from numpy (supposedly this is faster?)
        # unsqueeze -1 converts 1d arrays to 2d array, (i.e. [3] to [3,1])
        state_t = torch.as_tensor(current_states, dtype=torch.float32).to(self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).to(self.device)    
        #rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)        
        #dones_t = torch.as_tensor(dones, dtype=torch.long).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_t = torch.LongTensor(dones).to(self.device)
        fstate_t = torch.as_tensor(future_states, dtype=torch.float32).to(self.device)

        # One method
        # # This is predict
        # state_action_qs = self.model(state_t).gather(1, actions_t).squeeze(-1)
        # #state_action_qs = self.model(state_t).cpu().detach().numpy()
        # #state_action_qs = self.model(state_t)
        # # Calculate number of greedy actions
        
        # future_qs = self.targetModel(fstate_t)

        # expected_q = np.ones(future_qs.shape[0])
        # max_future_qs = torch.sort(future_qs,dim=1,descending=True)[0].cpu().detach()
        # greedy_actions = np.ones(future_qs.shape[0])

        # future_max_qs_num = np.array(max_future_qs)
        # future_qs_num = future_qs.cpu().detach().numpy()
        # for i in range(future_qs_num.shape[0]):
        #     col = 0
        #     while future_max_qs_num[i][col] == future_max_qs_num[i][col+1]:
        #         greedy_actions[i]+=1
        #         col+=1
        #     continue


        # # Action probabilites
        # non_greedy_action_probability = self.epsilon / self.env.action_space.n
        # greedy_action_probability = ((1 - self.epsilon) /greedy_actions) + non_greedy_action_probability

        # for i, x in enumerate(future_qs_num):
        #     for j in range(self.env.action_space.n):
        #         if future_qs_num[i][j] == max_future_qs[i][0]:
        #             #test = future_qs_num[i][j] * greedy_action_probability
        #             #test1 = future_qs_num[i][j]
        #             expected_q[i] += future_qs_num[i][j] * greedy_action_probability[i]
        #         else:
        #             expected_q[i] += future_qs_num[i][j] * non_greedy_action_probability

        # expected_q = torch.tensor(expected_q, dtype=torch.float32).to(self.device)
        
        # expected_q[dones_t] = 0.0
        # targets = rewards_t + self.settings['DISCOUNT'] * expected_q
        
        # loss = F.smooth_l1_loss(state_action_qs, targets)

        # One method <<<<<<<<<<<<<,

        # Now get predicted state action values for all states, 
        # send state to network to get values for each action
        # gather along columns to get predicted q values for each action in the batch
        # then squeeze to make tensor 1d again, giving list of q values for each action
        state_action_qs = self.model(state_t).gather(1, actions_t).squeeze(-1)

        #Now send next states to target model, get action values, find max value and just take that
        #Dim 0 is batch
        future_qs = self.targetModel(fstate_t)
        mean_qs = future_qs.mean(1)

        # One trick for dones
        #targets = max_future_qs*DISCOUNT*rewards_t*(1 - dones_t)
        # Another
        mean_qs[dones_t] = 0.0
        mean_qs = mean_qs.detach()
        targets = mean_qs * self.settings['DISCOUNT'] + rewards_t

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
                frames+=1
                # TODO: Different way of doing this 
                # TODO: alternate epsilon methods for non approximation
                if(len(self.replay_mem) > self.settings['REPLAY_MIN']):
                    self.epsilon = max(self.epsilon*self.settings['EPSILON_DECAY'], self.settings['EPSILON_END'])

                if eps % 100 == 0:
                    self.env.render()

                # TODO: Calculate whether using approximate methods or not
                approximate = False
                
                self.replay_mem, reward, done, self.env = self.agent.step(self.env, approximate, self.model, self.epsilon, self.device, self.replay_mem, self.settings['REPLAY_MIN'])
                
                #Train method
                
                self.train_E_SARSA()

                reward_buf.append(reward)

                if frames % self.settings['TARGET_UPDATE'] == 0:
                    self.targetModel.load_state_dict(self.model.state_dict())
                    torch.save(self.model.state_dict(), 'Models/eSarsaWeights')

            writer.add_scalar('Total/Reward', np.mean(reward_buf) , eps)
            writer.add_scalar('Total/Epsilon', self.epsilon, eps)


            # Current logging
            if eps % 5 == 0:
                print('Episode Reward', np.mean(reward_buf))
                print('Episode', eps)
                print('Epsilon', self.epsilon)
                writer.flush()

class REINFORCE():
    def __init__(self, model, env, optimizer, settings) -> None:
        self.model = model
        self.env = env
        self.optimizer = optimizer
        self.batched_mem = []
        self.settings = settings

    # TODO: Alternate methods for calculating reinforce method.
    #Now for training if using REINFORCE METHOD
    #This is called end of each episode
    def reinforce_train(self, model, discount_rewards):
        # Discounted rewards:
            
        #Get individual transition from batched replay memory
        current_states = np.array([transition[0] for transition in batched_mem])
        actions = np.array([transition[1] for transition in batched_mem])
        # dones = np.array([transition[4] for transition in replay_mem])

        current_states_t = torch.tensor(current_states, dtype=torch.float32).to(device)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(-1).to(device)
        discount_rewards_t = torch.tensor(discount_rewards, device=device)

        # TODO Softmax not currently used, or option in some networks
        log_loss = torch.log(model(current_states_t))
        test = torch.gather(log_loss, 1, actions_t).squeeze()
        loss = discount_rewards_t * torch.gather(log_loss, 1, actions_t).squeeze()
        loss = -loss.mean()
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        return True

    def calc_discount(self, replay_mem):
        rewards = np.array([transition[2] for transition in replay_mem])
        # TODO: Change how this is done
        
        # for i in range(len(rewards)):
        
        #     discount_rewards.append(DISCOUNT**i * rewards[i])    

        # return discount_rewards
        r = np.array([DISCOUNT**i * rewards[i] 
            for i in range(len(rewards))])
        # Reverse the array direction for cumsum and then
        # revert back to the original order
        r = r[::-1].cumsum()[::-1]
        result = r - r.mean()
        return (result)

    # discont rewards need to be calculated, only once episode done, this is batched rewards
    # single policy estimator used in the nework
    # times discount rewards with log of policy estimators, gather with actions as indicies 
    # loss is negative of that averaged
    # then empty batch ?
    # End
    def play(self):
        discount_rewards = []
        reward_buf = collections.deque(maxlen=10000)
        batch = 0
        agent = agent.Agent(env)
        for eps in range(1000):
            done = False
            reward_buf.clear()

            while not done:

                if eps % 100 == 0:
                    env.render()
                
                approximate = True
                replay_mem, reward, done, env = agent.step(env, approximate, model, epsilon, device, replay_mem, REPLAY_MIN)

                if done:
                    batch += 1
                    discount_rewards.extend(self.calc_discount(replay_mem))
                    batched_mem.extend(replay_mem)
                    replay_mem.clear()
                    if batch == BATCH_SIZE:
                        self.reinforce_train(model, discount_rewards)
                        # Clear Memory once update done
                        discount_rewards = []
                        batched_mem = []
                

                reward_buf.append(reward)


            # Current logging
            if eps % 5 == 0:
                print(mean(reward_buf))
                print(eps)
