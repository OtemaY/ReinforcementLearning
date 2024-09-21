import argparse
import gymnasium as gym
import torch

import config
from utils import preprocess
from plots import pong_episodic_returns, cartpole_episodic_returns
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize
import os
import numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argpase - for usr friendly cmd line interfaces
parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v1'], default='CartPole-v1')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# python train.py --env CartPole-v1 --evaluate_freq 25 --evaluation_episodes 5


# Define the directory path
directory = "models"

# Create the directory if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole
}


# Replace value for each epoch training for
epoch = 0


# eppochs = torch.load(f'models/cart-returns-epoch-{epoch}.pt')


                  
if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and its config settings
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    
#  creates an instance of a DQN and moves it to specified device
    dqn = DQN(env_config=env_config).to(device)
    
        # TODO: Create and initialize target Q-network.

    target_dqn = DQN(env_config=env_config).to(device)
#     copying intial weights to target network. this weight model is a tensor default
    target_dqn.load_state_dict(dqn.state_dict())


 
    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp (Root mean square propagation).
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
#     by first initializing the biggest -inf no.
    best_mean_return = -float("Inf")

#     TRAINING_THRESHOLD = 50000
    
    steps = 0
    returns = []
#     start loop by iterating over n episodes defined in env_config
    for episode in range(env_config['n_episodes']):
        obs, info = env.reset() # starting from random initial condition, return initial state

        
             

        obs = preprocess(obs, env=args.env).unsqueeze(0) #reshaped and converted into a tensor

        terminated = False
        
#         Interaction with environment
        while not terminated:
            steps+=1

            # Get action from DQN.
            action = dqn.act(obs, env, exploit=False)
            action = action.item()

            # Act in the true environment.
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward = torch.tensor([reward], device=device)

      
            # Preprocess incoming observation.
            if not terminated:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
            else:
                next_obs = None
                
            
            # TODO: Add the transition to the replay memory. Remember to convert    everything to PyTorch tensors!
           

            memory.push(obs, action, next_obs, reward)
            obs = next_obs

              
      # TODO Run DQN.optimize() every env_config["train_frequency"] steps.
# env_config["memory_size"]


            if steps % env_config["train_frequency"] == 0: #add len mem
                optimize(dqn, target_dqn, memory, optimizer,device)

        # TODO: update the target dqn network every "target_update_frequency" steps
            if steps % env_config["target_update_frequency"] == 0:
#                target_dqn = copy.deepcopy(dqn).to(device)
                target_dqn.load_state_dict(dqn.state_dict())
                print(f'Updated target at {steps} steps')


        print("steps at end:", steps)

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')
            returns.append((episode,mean_return))

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                print('best_mean_return:', best_mean_return)

            else:
                print('Curr mean_return < best_mean_return')
                print('curr mean return is:', mean_return)
                torch.save(dqn.state_dict(), f'models/{args.env}_best.pt')
                
            print('last best_mean_return:', best_mean_return)
            
            
    eppochs = {
      'returns': returns,
      'args': env_config
    }
    
    torch.save(eppochs, f'models/cart-returns-epoch-{epoch}.pt')

    env.close()
    
    cartpole_episodic_returns()
    