# Deep Reinforcement Learning Algorithms
Testing Deep RL Algorithms with Lunar Lander and Atari Games.

<div align="center">

Pong             |  Breakout
:-------------------------:|:-------------------------:
![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/17604361/164998169-2df7328c-2907-43fe-a62a-d54670f88276.gif) | ![ezgif com-gif-maker](https://user-images.githubusercontent.com/17604361/164995473-dfea1d8b-59ca-4a84-af8f-b05d13697dc8.gif)
  
</div>

## To Install:
1. git clone repo
2. Create virtual environment
3. pip install -r requirements.txt <p>
(if using testStable.py file, install requirementsStable.txt) </p>

## To run:
Run main.py

## Options:
All options for testing are changed within <b> settings.json </b> file

### To change algorithm change 'rlOption' from 1-5
  - 1 is DQN (set 'ddqn' variable to either true or false for Double DQN).
  - 2 is Expected SARSA
  - 3 is REINFORCE
  - 4 is Vanilla Actor Critic
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
- playModel.py can also be used to play trained model weights, this must be selected however in the playModel file.
