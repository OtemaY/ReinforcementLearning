import torch
import os
import matplotlib.pyplot as plt

# Define the directory path
directory = "plots"

# Create the directory if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v1', 'ALE/Pong-v5']:
        return torch.tensor(obs, device=device).float()
    else:
        raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')
