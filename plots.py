import torch
import os
import matplotlib.pyplot as plt

# Define the directory path
directory = "plots"

# Create the directory if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Plotting for X episodes only

def cartpole_episodic_returns():
    for i in range(1000):
        file_path = f'models/cart-returns-epoch-0.pt'  
        
        data = torch.load(file_path, map_location=torch.device('cpu'))

        print(data)
        returns = data['returns']
        if not returns:
            print(f"No returns data found for epoch {i+1}.")
            continue

        x = [x for (x, y) in returns]
        y = [y for (x, y) in returns]

        fig = plt.figure()
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.9)

        fig.suptitle(f'Episodic Returns for 1000 Episodes', fontsize=10, fontweight='bold')
        ax.set_title('Average returns at episodes')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average return')

        plt.plot(x, y)
        plt.show()

        fig.savefig(f'plots/cartpole/returns-epoch-0-episode-{i+1}.png', format='png')

def pong_episodic_returns():
    for i in range(100):
        file_path = f'modelspong/pong-returns-epoch-0.pt'  
        
        data = torch.load(file_path, map_location=torch.device('cpu'))

        print(data)
        returns = data['returns']
        if not returns:
            print(f"No returns data found for epoch {i+1}.")
            continue

        x = [x for (x, y) in returns]
        y = [y for (x, y) in returns]

        fig = plt.figure()
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.9)

        fig.suptitle(f'Episodic Returns for 100 Episodes', fontsize=10, fontweight='bold')
        ax.set_title('Average returns at episodes')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average return')

        plt.plot(x, y)
        plt.show()

        fig.savefig(f'plots/returns-epoch-0-episode-{i+1}.png', format='png')

if __name__ == '__main__':
    pong_episodic_returns()
#   cartpole_episodic_returns()