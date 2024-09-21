import random
import torch
import torch.nn as nn
import copy
import math
import numpy as np
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)

        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """

        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))
#         return sample


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]
        self.anneal_step = 0
       
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def act(self, observation, env, exploit=False): #exploration is default
        """Selects an action with an epsilon-greedy exploration strategy."""
         # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.

        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1 * self.anneal_step / self.anneal_length)

        self.anneal_step += 1
        sample = random.random()

        if sample > eps_threshold:
            with torch.no_grad():
                 return self(observation).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

class DQNP(nn.Module):
    def __init__(self, env_config):
        super(DQNP, self).__init__()

        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]
        self.anneal_step = 0
       
    
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.fc1(self.flatten(x)))
        x = self.fc2(x)

        return x

    def act(self, observation, env, valid_acts =[2,3], exploit=False): #exploration is default
        """Selects an action with an epsilon-greedy exploration strategy."""
         # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.

        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1 * self.anneal_step / self.anneal_length)

        self.anneal_step += 1
        sample = random.random()

        if sample > eps_threshold or exploit == True:
            with torch.no_grad():
                # greedy mode
                return self(observation).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.choice(valid_acts)]], device=device, dtype=torch.long)


def optimize(dqn, target_dqn, memory, optimizer,device): #this memory actual holds a batch
    """This function samples a batch from the replay buffer and optimizes the Q-network. We do our training here"""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

        
    # TODO: Sample a batch from the replay memory and concatenate so that there are
      #Remember to move them to GPU if it is available, e.g., by using Tensor.to(device)
    #       Note that special care is needed for terminal transitions!
        #       four tensors in total: observations, actions, next observations and rewards.

    batch = memory.sample(dqn.batch_size)

    batch_states, batch_actions, batch_next_states, batch_rewards= batch

    # Convert integers to tensors before concatenation
    batch_actions = [torch.tensor(action) for action in batch_actions]


    batch_states = torch.cat(batch_states)

    batch_rewards = torch.cat(batch_rewards)



# a mask to filter final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_states)),
                                  device=device, dtype=bool)
    
#   Infer the state dimension from the first non-None state
    state_shape = next(state.shape for state in batch_next_states if state is not None)

    # Replace None with zeros of the inferred shape
    batch_next_states = [torch.zeros(state_shape) if state is None else state for state in batch_next_states]
#     print(len(batch_next_states))
    
        # Handle None values in batch_next_states
    batch_next_states = torch.cat([state for state in batch_next_states if state is not None])
#     print(batch_next_states.size())

    
# Infer the state dimension from the first non-None state
    state_shape = next(state.shape for state in batch_states if state is not None)
    # Replace None with zeros
    batch_next_states = [torch.zeros(state_shape) if state is None else state for state in batch_next_states]

# Convert the list to a single PyTorch tensor
    batch_next_states = torch.stack(batch_next_states)

    

    # Compute the current Q values from the DQN    
    batch_actions = torch.tensor([action.item() for action in batch_actions], device=device, dtype=torch.long)

    q_values = dqn(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)



    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
 # Compute next state values only for non-terminal states
#     if non_final_next_states.size(0) > 0:
#         next_state_values[non_final_mask] = target_dqn(non_final_next_states).max(1)[0].detach()
    
    with torch.no_grad():

        next_q_values = target_dqn(batch_next_states)
        next_q_max, _ = torch.max(next_q_values, dim=1)

    q_value_targets = batch_rewards + (dqn.gamma * next_q_max * non_final_mask)

    
    # Compute loss.
    loss = F.mse_loss(q_values, q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()
