"""
In this file, you may edit the hyperparameters used for different environments.

memory_size: Maximum size of the replay memory.
n_episodes: Number of episodes to train for.
batch_size: Batch size used for training DQN.
target_update_frequency: How often to update the target network.
train_frequency: How often to train the DQN.
gamma: Discount factor.
lr: Learning rate used for optimizer.
eps_start: Starting value for epsilon (linear annealing).
Linear annealing is linear decay. gradually reducing the lr 

eps_end: Final value for epsilon (linear annealing).
anneal_length: How many steps to anneal epsilon for.
n_actions: The number of actions can easily be accessed with env.action_space.n, but we do
    some manual engineering to account for the fact that Pong has duplicate actions.
"""

# Hyperparameters for CartPole-v1
CartPole = {
    'memory_size': 50000,
    'n_episodes': 1000,
    'batch_size': 32,
    'target_update_frequency': 100,
    'train_frequency': 1,
    'gamma': 0.95,
    'lr': 1e-4,
    'eps_start': 1.0,
    'eps_end': 0.05,
    'anneal_length': 10**4,
    'n_actions': 2,
}

ALE_Pong = {
    'memory_size': 50000,
    'n_episodes': 100,
    'batch_size': 32,
    'target_update_frequency': 100,
    'train_frequency': 1,
    'gamma': 0.95,
    'lr': 1e-4,
    'eps_start': 1.0,
    'eps_end': 0.05,
    'anneal_length': 10**4,
    'n_actions': 6,
    'obs_stack': 4 #stack frames to convey velocity.
#     4 consecutive frames form a single state
}

