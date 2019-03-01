[//]: # (Image References)

[image1]: rewards.png "Rewards Plot"

# Project 3 Report: Collaboration and Competition

## Learning Algorithm
In this tennis task, a shared DDPG learning agent is shared by two players.
In other words, the implementation would be very like the multi-agent parallel training in project 2.

In `agent.py` you can find the DDPG learning agent.
Since the env contains 2 players, so the `act()` and `step()` methods of the DDPG agent takes batch input and gives batch output (batch size = 2).
After every 4 calls of `step()` (in other words, 4 time steps), the agent will learn from replay buffer (size 1e5) for 4 times with batch size of 128.
The DDPG hyperparameters also include:
* Discount factor gamma = 0.99
* Soft update target tau = 1e-3
* Learning rate of actor = 1e-4 and of critic = 3e-4

In `model.py` the NNs for actor and critic can be found.
They are both vanilla multi-layer full connected networks (3 linear layers, with hidden sizes around 300 ~ 400, tanh at output if requires [-1, 1] output).
Three lessons I learnt during training are:
* Large network is not always better without proper data and training -- Larger networks are hard and slow to train and converge.
* Batch normalization is very useful to regularize the input, thus vastly improve the learning of a network.
* Use leaky ReLU instead of ReLU makes NNs to learn faster in my case.

## Plot of Rewards
![Rewards Plot][image1]

At episode 1222, the 100-ep average reward already passed 0.5.
I set a higher bar for the learning algorithm at 0.6 (higher is still possible, but costs more GPU time).
So after 1240 episodes, finally I get policies that could play with average score of 0.6 for 100 continuous episodes.

## Ideas for Future work
There are lots of techniques we can apply to improve the learning algorithm in the future, for example:
* While we should only feed local observations to actor network(s) of each agent, the critic network(s) could take the complete observation of all states and chosen actions.
This could lead to better reward estimation.
* Adopt multi-node multi-GPU multi-CPU parallel training (E.g. DistributedDataParallel in PyTorch) to get much faster training speed.
