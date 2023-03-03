import numpy as np
import torch
import torch.nn as nn
import gym
from network_utils import np2torch


class RewardNetwork(nn.Module):
    """
    Class for implementing Reward network
    """

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.lr = 3e-2
        observation_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        input_dim = observation_dim + action_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1)
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = self.lr)


    def forward(self, observations):
        # squeeze the result so it's either [batch size] * [traj_length] or just for [1] i.e. reward for a single (obs, action) pair
        output = self.network(observations).squeeze() 
        print("reward network output shape: ", output.shape)
        return output

    def calculate_advantage(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]

        TODO:
        Evaluate the baseline and use the result to compute the advantages.
        Put the advantages in a variable called "advantages" (which will be
        returned).

        Note:
        The arguments and return value are numpy arrays. The np2torch function
        converts numpy arrays to torch tensors. You will have to convert the
        network output back to numpy, which can be done via the numpy() method.
        """
        observations = np2torch(observations)
        #######################################################
        #########   YOUR CODE HERE - 1-4 lines.   ############
        baseline = self.forward(observations).detach().numpy()
        advantages = returns - baseline
        #######################################################
        #########          END YOUR CODE.          ############
        return advantages

    def update_reward(self, traj1, traj2, labels):
        """
        Args:
            traj1, traj2: a dictionary with two items, a batch of observations and actions, each an np.array of shape
             [batch size, traj_length, obs_dim or act_dim]
            labels: np.array of shape [batch size]
            label = 1 if traj1 is more preferable, = 2 if traj2 is more preferable, = 3 if equally preferable

        training data looks like:
        traj_1 & traj_2 (dictionary that has key "observation" and "action"),, labels (np array of shape [batch size])

        e.g.

        # batch size = 3, traj_length = 2, observation_dim = 3, action_dim = 1
        test_traj1 = {"observations": np.array([[[1, 2, 3], [4, 5, 6]],
                                                [[1, 2, 3], [4, 5, 6]],
                                                [[1, 2, 3], [4, 5, 6]]],
                                                dtype=float), "actions": np.array([[[1],[1]],[[1],[1]],[[1],[1]]], dtype=float)}
        test_traj2 = {"observations": np.array([[[0, 2, 3], [4, 5, 6]],
                                                [[0, 2, 3], [4, 5, 6]],
                                                [[0, 2, 3], [4, 5, 6]]],
                                                dtype=float), "actions": np.array([[[1],[1]],[[1],[1]],[[1],[1]]], dtype=float)}
        test_labels = np.array([3, 1, 2])
        """
        
        traj1_reward = self.traj_to_reward(traj1)
        traj2_reward = self.traj_to_reward(traj2)
        p1 = torch.log((traj1_reward)/(traj1_reward + traj2_reward))
        p2 = torch.log((traj2_reward)/(traj1_reward + traj2_reward))
        mu1, mu2 = self.mu(labels)

        loss = - torch.sum(p1 * mu1 + p2 * mu2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def mu(self, labels): # labels input dim = [batch size]
        print("labels.dtype", labels.dtype)
        labels = labels.astype(float)
        mu1 = np.piecewise(labels, [labels == 1, labels == 2, labels == 3], [1, 0, 1/2])
        mu2 = np.piecewise(labels, [labels == 1, labels == 2, labels == 3], [0, 1, 1/2])
        return np2torch(mu1), np2torch(mu2)
    
    def traj_to_reward(self, traj, exponential = True):
        observations = traj["observations"]
        actions = traj["actions"]
        temp = np.concatenate([observations, actions], axis = -1)
        reward_input = np2torch(np.concatenate((observations, actions), axis = -1)) # now dim = batch size x traj length x (obs dim + act dim)
        traj_reward = self.forward(reward_input) # shape = [batch size, traj length]
        traj_reward = torch.sum(traj_reward, dim = 1) # sum over the trajectory, for each t. Now dim = [batch size]
        if exponential:
            traj_reward = torch.exp(traj_reward)
        return traj_reward
    

# test
# test_traj1 = {"observations": np.array([[[1, 2, 3], [4, 5, 6]],
#                                         [[1, 2, 3], [4, 5, 6]],
#                                         [[1, 2, 3], [4, 5, 6]]],
#                                         dtype=float), "actions": np.array([[[1],[1]],[[1],[1]],[[1],[1]]], dtype=float)}
# test_traj2 = {"observations": np.array([[[0, 2, 3], [4, 5, 6]],
#                                         [[0, 2, 3], [4, 5, 6]],
#                                         [[0, 2, 3], [4, 5, 6]]],
#                                         dtype=float), "actions": np.array([[[1],[1]],[[1],[1]],[[1],[1]]], dtype=float)}
# test_labels = np.array([3, 1, 2])
# test_reward_network = RewardNetwork(gym.make('Pendulum-v1'))
# test_reward_network.update_reward(test_traj1, test_traj2, test_labels)
