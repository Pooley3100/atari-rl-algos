import json
import gym
import torch
import envWrapper
import models
import rlAlgorithms

with open("settings.json") as read_file:
    settings = json.load(read_file)


# Tale advantage of gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


'''
Can all be selected by user to test, then tensorboard to graph
all the pre processing that happens before, no op, stacked frames etc. 
First make the 256 x 2 conv neural network
ensure that the input image is normalized
create model and target model
model to train and target to predict
'''


# SETUP ENVIRONMENT, SELECT GAME TO TRAIN ON HERE

# Non atari environment, BASIC models and gym.make
lunar_lander = 'LunarLander-v2'
cartpole = 'CartPole-v0'

# Atari environments with advanced models and envWrapper.makeEnv
pong = 'PongNoFrameskip-v4'
breakout = 'BreakoutNoFrameskip-v4'
space_invaders = 'SpaceInvadersNoFrameskip-v4'

env_name = pong
# env = gym.make(env_name)
env = envWrapper.makeEnv(env_name)

in_channels = env.observation_space.shape
out_channels = env.action_space.n

print("out channels", out_channels)
print("in channels", in_channels)

# Model
model = models.NeuralNetworkAdvanced(in_channels, out_channels).to(device)
# model.load_state_dict(torch.load('Models/dqnWeights')) # TEMPORARY LOAD IN TO RESUMER <<<<<<

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=settings['LEARNING_RATE'])

rlOption = 1
if rlOption == 1:
    # DQN
    print('Training DQN in {0}'.format(env_name))
    # Target Model required
    targetModel = models.NeuralNetworkAdvanced(in_channels, out_channels).to(device)
    targetModel.load_state_dict(model.state_dict())
    # Remember to either set ddqn to true or not based on testing preference.
    ddqn = False
    DQN = rlAlgorithms.DQN(model, env, targetModel=targetModel, optimizer=optimizer, settings=settings, device=device, ddqn = ddqn)
    DQN.play()

elif rlOption == 2:
    # Expexted SARSA
    print('Training E_SARSA in {0}'.format(env_name))
    # Target Model required
    targetModel = models.NeuralNetworkBasic(in_channels, out_channels).to(device)
    targetModel.load_state_dict(model.state_dict())
    e_sarsa = rlAlgorithms.E_SARSA(model, env, targetModel=targetModel, optimizer=optimizer, settings=settings, device=device)
    e_sarsa.play()

elif rlOption == 3:
    # REINFORCE
    print('Training REINFORCE in {0}'.format(env_name))
    # model = models.PolicyNeuralNetworkAdvanced(in_channels, out_channels).to(device)
    model = models.PolicyNeuralNetworkAdvanced(in_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['LEARNING_RATE'])
    reinforce = rlAlgorithms.REINFORCE(model, env, optimizer, settings, device)
    reinforce.play()

elif rlOption == 4:
    # Vanilla Actor Critic, no multiple workers.
    print('Training ActorCritic in {0}'.format(env_name))
    actorCritic = rlAlgorithms.ActorCritic(env=env, device=device, in_channels=in_channels, out_channels=out_channels, settings=settings)

    actorCritic.play()

elif rlOption == 5:
    # A2C
    print('Training A2C in {0}'.format(env_name))
    # model = models.ActorCriticNetwork(in_channels, out_channels).to(device)
    model = models.ActorCriticNetworkBasic(in_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['LEARNING_RATE'])
    a2c = rlAlgorithms.TrainA2C(model, settings, optimizer, device, lunar_lander)

    a2c.play()

torch.save(model.state_dict(), 'Models/Weights{}'.format(rlOption))



