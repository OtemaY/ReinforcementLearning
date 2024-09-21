# Training Pong Model

# Import requisite packages
import argparse
import gymnasium as gym
import torch

import config
from utils import preprocess
from plots import pong_episodic_returns, cartpole_episodic_returns
from evaluate import pongeval_policy
from dqn import DQN, DQNP, ReplayMemory, optimize
import os
import numpy as np
from pathlib import Path
from gymnasium.wrappers import atari_preprocessing


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argpase - for usr friendly cmd line interfaces
parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['ALE/Pong-v5'], default='ALE/Pong-v5')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# python trainpong.py --env ALE/Pong-v5 --evaluate_freq 25 --evaluation_episodes 5


# Define the directory path
directory = "modelspong"

# Create the directory if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'ALE/Pong-v5': config.ALE_Pong
}

# Replace value for each epoch training for
epoch = 0


eppochs = torch.load(f'modelspong/pong-returns-epoch-{epoch}.pt')

                  
if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and its config settings
    env = gym.make(args.env)
    env = atari_preprocessing.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30, scale_obs=True) 
    #scale_obs=True normalizes pixels 4rm to 255 to 0-1

    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    
    dqn = DQNP(env_config=env_config).to(device)
    
        # TODO: Create and initialize target Q-network.

    target_dqn = DQNP(env_config=env_config).to(device)
#     copying intial weights to target network.
    target_dqn.load_state_dict(dqn.state_dict())


 
    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp (Root mean square propagation).
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
#     by first initializing the biggest -inf no.
    best_mean_return = -float("Inf")

    
    steps = 0
    returns = []
    OBS_STACK =  env_config['obs_stack']
#     start loop by iterating over n episodes defined in env_config
    for episode in range(env_config['n_episodes']):
        obs, info = env.reset() # starting from random initial condition, return initial state

        
  

        obs = preprocess(obs, env=args.env).unsqueeze(0) #reshaped and converted into a tensor
        obs_stack = torch.cat(OBS_STACK *[obs]).unsqueeze(0).to(device)
        terminated = False
        
#         Interaction with environment
        while not terminated:
            steps+=1

            # Get action from DQN.
            action = dqn.act(obs_stack, env)
            action = action.item()

            # Act in the true environment.
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward = torch.tensor([reward], device=device)

      
            # Preprocess incoming observation.
            if not terminated:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
                next_obs_stack = torch.cat((obs_stack[:, 1:, ...], next_obs.unsqueeze(1)), dim=1).to(device)
            else:
                next_obs = None
                next_obs_stack = None
                
            
            # TODO: Add the transition to the replay memory. Remember to convert    everything to PyTorch tensors!
            

            memory.push(obs_stack, action, next_obs_stack, reward)
            obs_stack = next_obs_stack
#             print(obs_stack.size())

              
      # TODO Run DQN.optimize() every env_config["train_frequency"] steps.
# env_config["memory_size"]


            if steps % env_config["train_frequency"] == 0: 
                optimize(dqn, target_dqn, memory, optimizer,device)



        # TODO: update the target dqn network every "target_update_frequency" steps
            if steps % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())
                print(f'Updated target at {steps} steps')


        print("steps at end:", steps)

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = pongeval_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)

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
                torch.save(dqn, f'modelspong/{args.env}_best.pt')
                
            print('last best_mean_return:', best_mean_return)

    # Close environment after training is completed.
    
    eppochs = {
      'returns': returns,
      'args': env_config
    }
    
    torch.save(eppochs, f'modelspong/pong-returns-epoch-{epoch}.pt')

    env.close()
    
    pong_episodic_returns