import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TODO: Edit advanced network
# TODO: Edit basic to work better
class NeuralNetworkBasic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NeuralNetworkBasic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, out_channels)

        # now flatten
        # pass cnn to linear via flatten to final layer of 512, then pass to action space size.

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        # print(x.shape)
        # x = F.relu(self.conv3(x))
        # print(x.shape)
        # x = x.view(-1, 384)
        x = x.flatten(start_dim=1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # print(x.shape)
        return x


class NeuralNetworkAdvanced(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NeuralNetworkAdvanced, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            # was 512
            nn.Linear(conv_out_size, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions),
            # nn.Softmax(dim=-1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    # def get_qs(self, state):
    #     state_t = torch.as_tensor(current_state, dtype=torch.float32)
    #     # state_t = torch.permute(state_t, (2, 1, 0))
    #     q_values = model.forward(state_t[None, ...])  # None removes batch for time being
    #     return q_values


class PolicyNeuralNetworkAdvanced(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyNeuralNetworkAdvanced, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            # was 512
            nn.Linear(conv_out_size, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions),
            nn.Softmax(dim=-1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out), None

# TODO Change this
class PolicyNetwork(nn.Module):
    def __init__(self, env):
        super(PolicyNetwork, self).__init__()

        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        
        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16), 
            nn.ReLU(), 
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))
    
    def predict(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.critic = nn.Sequential(torch.nn.Linear(conv_out_size, 512),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(512, 1))

        self.actor = nn.Sequential(torch.nn.Linear(conv_out_size, 512),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(512, n_actions),
                                   torch.nn.Softmax(dim=-1))

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, state):
        x = self.conv(state).view(state.size()[0], -1)
        value = self.critic(x)

        prob_dist = self.actor(x)

        return prob_dist, value



