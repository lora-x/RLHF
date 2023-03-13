import numpy as np
import torch
import torch.nn as nn
import gym
from network_utils import np2torch

class RewardNetwork(nn.Module):
    """
    Class for implementing Reward network, which is used by the CustomReward wrapper to replace env reward with reward generated by learned network.
    """

    def __init__(self, env, synthetic_feedback = "false"):
        super().__init__()
        self.env = env
        self.__initialize_network()
        self.synthetic_feedback = synthetic_feedback

    def __initialize_network(self):
        self.lr = 3e-2
        observation_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0] # TODO this can only handle continous action space for now
        input_dim = observation_dim + action_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1)
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = self.lr)


    def predict_reward(self, reward_input):
        # squeeze the result so it's either [batch size] * [traj_length] or just for torch.Size([]) i.e. a scalar reward for a single (obs, action) pair
        output = self.network(reward_input).squeeze() 
        return output
        

    def update_network(self, traj1, traj2, preferences):
        """
        actually, traj should be stored as list and then convert to np.array
        """

        """
        Args:
            traj1, traj2: a dictionary with two items, a batch of observations and actions, each an np.array of shape
             [batch size, traj_length, obs_dim or act_dim]
            preferences: np.array of shape [batch size]
            preference = 1 if traj1 is more preferable, = 2 if traj2 is more preferable, = 3 if equally preferable

        training data looks like:
        traj_1 & traj_2 (dictionary that has key "observation" and "action"),, preferences (np array of shape [batch size])

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
        test_preferences = np.array([3, 1, 2])
        """
        traj1_reward = self.__traj_to_reward(traj1)
        traj2_reward = self.__traj_to_reward(traj2)
        p1 = torch.log((traj1_reward)/(traj1_reward + traj2_reward))
        p2 = torch.log((traj2_reward)/(traj1_reward + traj2_reward))
        mu1, mu2 = self.__mu(preferences)

        loss = - torch.sum(p1 * mu1 + p2 * mu2)
        print("loss: ", loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def __mu(self, preferences): # preferences input dim = [batch size]
        preferences = np.asarray(preferences, dtype=np.float32)
        mu1 = np.piecewise(preferences, [preferences == 1, preferences == 2, preferences == 3], [1, 0, 1/2])
        mu2 = np.piecewise(preferences, [preferences == 1, preferences == 2, preferences == 3], [0, 1, 1/2])
        return np2torch(mu1), np2torch(mu2)
    
    def __traj_to_reward(self, traj, exponential = True):
        observations = np.asarray(traj["observations"], dtype=np.float32)
        actions = np.asarray(traj["actions"], dtype=np.float32)
        reward_input = np2torch(np.concatenate((observations, actions), axis = -1)) # now dim = batch size x traj length x (obs dim + act dim)
        traj_reward = self.predict_reward(reward_input) # shape = [batch size, traj length]
        traj_reward = torch.sum(traj_reward, dim = -1) # sum over the trajectory, for each t. Now dim = [batch size]
        if exponential:
            traj_reward = torch.exp(traj_reward)
        return traj_reward
    

# test
# batch size = 3, traj_length = 2, observation_dim = 3, action_dim = 1
# test_traj1 = {"observations": [[[1, 2, 3], [4, 5, 6]],
#                                         [[1, 2, 3], [4, 5, 6]],
#                                         [[1, 2, 3], [4, 5, 6]]], "actions": [[[1],[1]],[[1],[1]],[[1],[1]]]}
# test_traj2 = {"observations": [[[0, 2, 3], [4, 5, 6]],
#                                         [[0, 2, 3], [4, 5, 6]],
#                                         [[0, 2, 3], [4, 5, 6]]], "actions": [[[1],[1]],[[1],[1]],[[1],[1]]]}
# test_preferences = [3, 1, 2]
# test_reward_network = RewardNetwork(gym.make('Pendulum-v1'))
# test_reward_network.update_network(test_traj1, test_traj2, test_preferences)
