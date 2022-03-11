import json
import torch
import envWrapper
import models
import rlAlgorithms



# TODO: Redo the venv installs, there are too many currently that are not needed
# TODO: Change up settings, and how epsilon decay is done

with open("settings.json") as read_file:
    settings = json.load(read_file)


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

# MsPacman-v5, Breakout-v5
# render_mode='human' for watching.
# env = gym.make('ALE/Breakout-v5', render_mode='human')
env = envWrapper.makeEnv('PongNoFrameskip-v4')

in_channels = env.observation_space.shape
out_channels = env.action_space.n

print("out channels", out_channels)
print("in channels", in_channels)


# Model
# TODO Softmax currently implemented
model = models.NeuralNetworkAdvanced(in_channels, out_channels).to(device)

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=settings['LEARNING_RATE'])

rlOption = 2
if rlOption == 1:
    # DQN
    print('Training DQN')
    # Target Model required
    targetModel = models.NeuralNetworkAdvanced(in_channels, out_channels).to(device)
    targetModel.load_state_dict(model.state_dict())
    DQN = rlAlgorithms.DQN(model, env, targetModel=targetModel, optimizer=optimizer, settings=settings, device=device)
    DQN.play()

elif rlOption == 2:
    # Expexted SARSA
    print('Training E_SARSA')
    # Target Model required
    targetModel = models.NeuralNetworkAdvanced(in_channels, out_channels).to(device)
    targetModel.load_state_dict(model.state_dict())
    e_sarsa = rlAlgorithms.E_SARSA(model, env, targetModel=targetModel, optimizer=optimizer, settings=settings, device=device)
    e_sarsa.play()

elif rlOption == 3:
    # REINFORCE
    print('Training REINFORCE')
    model = models.PolicyNeuralNetworkAdvanced(in_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['LEARNING_RATE'])
    reinforce = rlAlgorithms.REINFORCE(model, env, optimizer, settings, device)
    reinforce.play()

elif rlOption == 4:
    # A2C
    print('Training A2C')
    # TODO Potentially multiple workers with different envs, provides
    # Redo these declerations as requires different network
    model = models.ActorCriticNetwork(in_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['LEARNING_RATE'])
    a2c = rlAlgorithms.A2C(model, env, settings, optimizer, device)

    a2c.play()

torch.save(model.state_dict(), 'Models/Weights{}'.format(rlOption))



