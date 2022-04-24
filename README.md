# atari-rl-algos
Testing Deep RL Algorithms with Atari Games

## To Install:
1. Create Virtual environment
2. pip install -r requirements.txt <p>
(if using testStable.py file, install requirementsStable.txt) </p>

## To run:
Run main.py

## Options:
All options for testing are changed within settings.json file

### To change algorithm change 'rlOption' from 1-5
  - 1 is DQN (set 'ddqn' variable to either true or false for Double DQN).
  - 2 is Expected SARSA
  - 3 is REINFORCE
  - 4 is Actor Critic
  - 5 is A2C (note: this one does not perform correctly)
  
### To change game, change 'Game' variable:
  - 1 is Pong
  - 2 is Breakout
  - 3 is Space Invaders
  
### To change Model, use either 'Advanced' or 'Basic':
  - 'Advanced' Model is Convolutional Layer Pytorch network.
  - 'Basic' Model, when this is selected the game will automatically be the Lunar Lander Environment.

## Results:
- Run plot.py to then visualize test run. 
- playModel.py can also be used to use trained model weights, this must be selected however in the file.
