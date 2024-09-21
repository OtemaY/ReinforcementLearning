# ReinforcementLearning
Using Gymnasium API in Python to develop the Reinforcement Learning Algorithm in CartPole and Pong.


Deep Q-Learning (DQN) is a fundamental algorithm in the field of reinforcement learning (RL) that has garnered significant attention due to its success in solving complex decision-making tasks. RL
trains agents in an environment to make decisions in maximizing the rewards. DQN is grouped as a model-free RL family, which means it does not require any prior knowledge of the environment’s dynamics.
It uses the Q-network, which is a neural network, to approximate the Q-function. The Q-function calculates all the future rewards for an action in a given state. The training procedure includes updating the Q-network parameters iteratively in order to minimize the differences between the predicted and target Q-values. One advantage of DQN is that it uses experience replay, a technique
that is used to store past experiences in a replay buffer and sample batches in order to train the Q-network. This is useful in a way that it stabilizes training by breaking correlations between consecutive samples and makes use of past experiences effectively. The target network, which is a
separate copy of the Q-network in DQN, is used to generate target Q-values during training.
It will be updated periodically along with the Q-network in order to prevent training instability.[1]
This project examines the research paper ”Playing Atari with Deep Reinforcement Learning” by Mnih et al. (2013) and applies the Deep Q-learning algorithm to the CartPole-v1 and ALE Pong-v5 environments provided by the Gymnasium framework.
