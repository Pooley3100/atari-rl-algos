# atari-rl-algos
Testing Deep RL Algorithms with Atari Games

## To Install:
Create Venv and pip install requirements.tx. <p>
(if using testStable.py file, install requirementsStable.txt)

## To run:
Run main.py

## Options:
All options for testing are changed within settings.json file

### To change algorithm change rlOption from 1-5
  - 1 is DQN
  - 2 is Expected SARSA
  - 3 is REINFORCE
  - 4 is Actor Critic
  - 5 is A2C
  
### To change game, change Game variable:
  - 1 is Pong
  - 2 is Breakout
  - 3 is Space Invaders
  
### To change Model, use either 'Advanced' or 'Basic':
  - 'Advanced' Model is Convolutional Layer Pytorch network.
  - 'Basic' Model, when this is selected the game will automatically be the Lunar Lander Environment.

## Results:
- Run plot.py to then visualize test run. 
- playModel.py can also be used to use trained model weights, this must be selected however in the file.
