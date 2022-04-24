import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def flatten(x):
    batches = x.shape[0]
    return x.view(batches, -1)


class NeuralNetworkBasic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NeuralNetworkBasic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels[0], 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels),
        )

    def forward(self, state):
        return self.fc(flatten(state))


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
        action_probs = self.fc(flatten(state))
        return action_probs, None


class NeuralNetworkAdvanced(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NeuralNetworkAdvanced, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        cnn_out_size = self.cnn_out_size(in_channels)

        self.fc = nn.Sequential(
            # was 256
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels),
        )

    def cnn_out_size(self, in_channels):
        # Send dummy input through cnn then find product of this output for output size
        return int(np.prod(self.cnn(torch.ones(in_channels).unsqueeze(0)).size()))

    def forward(self, x):
        cnn = self.cnn(x)
        return self.fc(flatten(cnn))


class PolicyNeuralNetworkAdvanced(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PolicyNeuralNetworkAdvanced, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        cnn_out_size = self.cnn_out_size(in_channels)

        self.fc = nn.Sequential(
            # was 100
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels),
            nn.Softmax(dim=-1)
        )

    def cnn_out_size(self, in_channels):
        # Send dummy input through cnn then find product of this output for output size
        return int(np.prod(self.cnn(torch.ones(in_channels).unsqueeze(0)).size()))

    def forward(self, x):
        return self.fc(flatten(self.cnn(x))), None


class ActorCriticNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ActorCriticNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        cnn_out_size = self.cnn_out_size(in_channels)

        self.critic = nn.Sequential(torch.nn.Linear(cnn_out_size, 512),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(512, 1))

        self.actor = nn.Sequential(torch.nn.Linear(cnn_out_size, 512),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(512, out_channels),
                                   torch.nn.Softmax(dim=-1))

    def cnn_out_size(self, in_channels):
        # Send dummy input through cnn then find product of this output for output size
        return int(np.prod(self.cnn(torch.ones(in_channels).unsqueeze(0)).size()))

    def get_values(self, state):
        x = flatten(self.cnn(state))
        value = self.critc(x)

        return value

    def forward(self, state):
        x = flatten(self.cnn(state))

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

    def get_values(self, state):
        x = self.fc((state))
        value = self.critic(x)

        return value


class ActorBasic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ActorBasic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels[0], 100),
            nn.ReLU,
            nn.Linear(100, 32),
            nn.ReLU,
            nn.Linear(32, out_channels),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.model(state)


class CriticBasic(nn.Module):
    def __init__(self, in_channels):
        super(CriticBasic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels[0], 100),
            nn.ReLU(),
            nn.Linear(100, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state):
        return self.model(state)


class CriticAdvanced(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CriticAdvanced, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        cnn_out_size = self.cnn_out_size(in_channels)

        self.fc = nn.Sequential(
            # was 100
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def cnn_out_size(self, in_channels):
        # Send dummy input through cnn then find product of this output for output size
        return int(np.prod(self.cnn(torch.ones(in_channels).unsqueeze(0)).size()))

    def forward(self, x):
        return self.fc(flatten(self.cnn(x)))


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
            # was 100
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
            nn.Softmax(dim=-1)
        )

    def cnn_out_size(self, in_channels):
        # Send dummy input through cnn then find product of this output for output size
        return int(np.prod(self.cnn(torch.ones(in_channels).unsqueeze(0)).size()))

    def forward(self, x):
        return self.fc(flatten(self.cnn(x)))



