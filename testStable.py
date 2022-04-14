# Note if to use this use seperate venv with requirementsStable.txt

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback
import gym


#env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
#env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=4, seed=0)
env = make_atari_env('SpaceInvadersNoFrameskip-v4', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)
# env = gym.make('LunarLander-v2')


# model = DQN('CnnPolicy', env, verbose=1, tensorboard_log='./runs/', buffer_size=40000)
model = A2C('CnnPolicy', env, verbose=1, tensorboard_log='./runs/',device='cuda:0')


while True:
    # model.load('A2CPongStableWeights')
    model.learn(total_timesteps=50000000)
    model.save('A2CPongStableWeights')