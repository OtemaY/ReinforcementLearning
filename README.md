# ReinforcementLearning
Using Gymnasium API in Python to develop the Reinforcement Learning Algorithm in CartPole and Pong.


Deep Q-Learning (DQN) is a fundamental algorithm in the field of reinforcement learning (RL) that has garnered significant attention due to its success in solving complex decision-making tasks. RL
trains agents in an environment to make decisions in maximizing the rewards. DQN is grouped as a model-free RL family, which means it does not require any prior knowledge of the environment’s dynamics.
It uses the Q-network, which is a neural network, to approximate the Q-function. The Q-function calculates all the future rewards for an action in a given state. The training procedure includes updating the Q-network parameters iteratively in order to minimize the differences between the predicted and target Q-values. One advantage of DQN is that it uses experience replay, a technique
that is used to store past experiences in a replay buffer and sample batches in order to train the Q-network. This is useful in a way that it stabilizes training by breaking correlations between consecutive samples and makes use of past experiences effectively. The target network, which is a
separate copy of the Q-network in DQN, is used to generate target Q-values during training.
It will be updated periodically along with the Q-network in order to prevent training instability.[1]
This project examines the research paper ”Playing Atari with Deep Reinforcement Learning” by Mnih et al. (2013) and applies the Deep Q-learning algorithm to the CartPole-v1 and ALE Pong-v5 environments provided by the Gymnasium framework.


# CartPole-v1.

The CartPole environment, provided by OpenAI’s Gymnasium, simulates a cart and pole system, where the goal is to balance a pole on top of a moving cart by applying appropriate left or right forces. The state space consists of four variables: Cart position, Cart velocity, The angle of the pole measured from the vertical position, and The rate of change of the pole angle. 

The agent can take two actions: move the cart to the left or move it to the right. The main objective is to prevent the pole from falling over by applying the appropriate force to the cart. The episode terminates if the pole angle exceeds a certain threshold or the cart position moves outside a predefined range. The task we have undertaken in this project is to develop an agent that learns to balance the pole effectively by intelligently selecting actions based on the observed state.


# ALE/Pong-v5
The ALE Pong-v5 environment, also provided by OpenAI's Gymnasium, emulates the game of Pong, where two players control paddles on opposite sides of the screen, attempting to hit a ball back and forth. The objective in this environment is to control one of the paddles to successfully hit the ball and prevent it from passing the other paddle.
The state space in ALE Pong-v5 typically consists of the raw pixel values of the game screen, representing the visual information perceived by the agent, represented as Box(0, 255, (210, 160,3), uint8), where the pixel values are [0-255] and 3 channels of 210 by 160 frames.  The actions available to the agent usually include moving the paddle up or down to control its position.
The action space in ALE Pong-v5 offers 6 discrete actions, including NOOP (No operation), FIRE (unused), RIGHT, LEFT, RIGHTFIRE (equivalent to action 2), and LEFTFIRE (equivalent to action 3). Notably, only actions 2/4 (RIGHT) and 3/5 (LEFT) induce movement in the agent.

You can read more in our complete pdf document attached.
