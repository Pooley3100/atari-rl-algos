import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TODO: Conv out
# TODO: network sizes and forward pass
class NeuralNetworkBasic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NeuralNetworkBasic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels[0], 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels),
            nn.ReLU()
        )

    def forward(self, state):
        return self.fc(state.view(state.size()[0], -1))


class PolicyNeuralNetworkBasic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PolicyNeuralNetworkBasic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels[0], 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        action_probs = self.fc(state.view(state.size()[0], -1))
        return action_probs, None


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
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
            # nn.Softmax(dim=-1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


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


class ActorCriticNetworkBasic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ActorCriticNetworkBasic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels[0], 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.critic = nn.Sequential(torch.nn.Linear(64, 1))

        self.actor = nn.Sequential(torch.nn.Linear(64, out_channels),
                                   torch.nn.Softmax(dim=-1))

    def forward(self, state):
        x = self.fc(state)
        value = self.critic(x)

        prob_dist = self.actor(x)

        return prob_dist, value

    def get_critic(self, state):
        x = self.fc(state)
        return self.critic(x)

    def eval_action(self, state, action):
        prob_dist, value = self.forward(state)
        dist = torch.distributions.Categorical(prob_dist)
        log_probs = dist.log_prob(action).view(-1, 1)
        entropy = dist.entropy().mean()

        return value, log_probs, entropy


class ActorBasic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ActorBasic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels[0], 128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64, out_channels),
            nn.Softmax()
        )

    def forward(self, state):
        return self.model(state)


class CriticBasic(nn.Module):
    def __init__(self, in_channels):
        super(CriticBasic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        return self.model(state)


class CriticAdvanced(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(CriticAdvanced, self).__init__()

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
            nn.Linear(100, 1),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class ActorAdvanced(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorAdvanced, self).__init__()

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
        return self.fc(conv_out)



