[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

In this project, I trained an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, my agent is trained using DDQN and experience replay to archieve an average score of +15 over 100 consecutive episodes.

### Code

Please find the complete running experiments in `Navigation.ipynb` which imports the following reinforcement learning code:
* In `dqn_agent.py` you can find the code defining the DQN learning agent `Agent` and its companion memory `ReplayBuffer` code.
* In `model.py` you can find the neural network architecture to approximating the action-value function.

You can find the saved model weights of the successful agent in `model.pt`.
