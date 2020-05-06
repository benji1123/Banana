## Instructions
1. `python==3.6` is required due to the `unityagents` dependency
2. Run `Navigation.ipynb` to train model

## Repo Anatomy
* __Banana_Windows_x86_64/__: directory that stores Unity environment
* __Navigation.ipynb__: notebook used to train agent
* __model.py__: neural net that outputs action Q-values from given state-vector
* __agent.py__: agent which interacts with environment and implements Q-Learning
* __checkpoint.pth__: stores computed weights for neural net from training

## Context
[//]: # (Image References)
[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Results
The agent solves the environment in under 400 episodes. 