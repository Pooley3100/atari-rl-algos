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


# SETUP ENVIRONMENT from settings.json

# Non atari environment, BASIC models and gym.make
lunar_lander = 'LunarLander-v2'
cartpole = 'CartPole-v0'

# Atari environments with advanced models and envWrapper.makeEnv
pong = 'PongNoFrameskip-v4'
breakout = 'BreakoutNoFrameskip-v4'
space_invaders = 'SpaceInvadersNoFrameskip-v4'

env = None
env_name = ''
if settings['Model'] == 'Basic':
    env_name = lunar_lander
    env = gym.make(env_name)
else:
    if settings['Game'] == 1:
        env_name = pong
    elif settings['Game'] == 2:
        env_name = breakout
    else:
        env_name = space_invaders
    env = envWrapper.makeEnv(env_name)

in_channels = env.observation_space.shape
out_channels = env.action_space.n

print("out channels", out_channels)
print("in channels", in_channels)

# rlOptions, 1 = DQN (ddqn variable as well), 2 = E_SARSA, 3 = REINFORCE, 4 = Actor Critic, 5 = A2C
rlOption = settings['rlOption']

if rlOption == 1 or rlOption == 2:
    # Model
    if settings['Model'] == 'Basic':
        print('Model Type Basic')
        model = models.NeuralNetworkBasic(in_channels, out_channels).to(device)
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=settings['LEARNING_RATE'])
    elif settings['Model'] == 'Advanced':
        print('Model Type Advanced')
        model = models.NeuralNetworkAdvanced(in_channels, out_channels).to(device)
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=settings['LEARNING_RATE'])

# model.load_state_dict(torch.load('Models/dqnWeights')) # TEMPORARY LOAD IN TO RESUME<<<<<<

if rlOption == 1:
    # DQN
    ddqn = settings['ddqn']
    if ddqn:
        print('Training DDQN in {0}'.format(env_name))
    else:
        print('Training DQN in {0}'.format(env_name))

    # Target Model required
    if settings['Model'] == 'Basic':
        targetModel = models.NeuralNetworkBasic(in_channels, out_channels).to(device)
    else:
        targetModel = models.NeuralNetworkAdvanced(in_channels, out_channels).to(device)
    targetModel.load_state_dict(model.state_dict())
    # Remember to either set ddqn to true or not based on testing preference.
    DQN = rlAlgorithms.DQN(model, env, targetModel=targetModel, optimizer=optimizer, settings=settings, device=device, ddqn = ddqn)
    DQN.play()

elif rlOption == 2:
    # Expexted SARSA
    print('Training E_SARSA in {0}'.format(env_name))
    # Target Model required
    if settings['Model'] == 'Basic':
        targetModel = models.NeuralNetworkBasic(in_channels, out_channels).to(device)
    else:
        targetModel = models.NeuralNetworkAdvanced(in_channels, out_channels).to(device)
    targetModel.load_state_dict(model.state_dict())
    e_sarsa = rlAlgorithms.E_SARSA(model, env, targetModel=targetModel, optimizer=optimizer, settings=settings, device=device)
    e_sarsa.play()

elif rlOption == 3:
    # REINFORCE
    print('Training REINFORCE in {0}'.format(env_name))
    if settings['Model'] == 'Basic':
        model = models.PolicyNeuralNetworkBasic(in_channels, out_channels).to(device)
    else:
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
    print('Note: Does not require optimal results')
    if settings['Model'] == 'Basic':
        model = models.ActorCriticNetworkBasic(in_channels, out_channels).to(device)
    else:
        model = models.ActorCriticNetwork(in_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['LEARNING_RATE'])
    a2c = rlAlgorithms.TrainA2C(model, settings, optimizer, device, lunar_lander)

    a2c.play()

torch.save(model.state_dict(), 'Models/Weights{}'.format(rlOption))



