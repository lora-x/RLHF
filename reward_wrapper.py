import gym
import numpy as np
from network_utils import np2torch 

class CustomReward(gym.Wrapper):
    def __init__(self, env, reward_network, initial_observation):
        super().__init__(env)
        self.reward_network = reward_network
        self.env = env
        self.prev_observation = initial_observation

    def step(self, action):
        observation, _, done, info = self.env.step(action)
        print("in wrapper, self.prev_observation: ", self.prev_observation)
        print("in wrapper, observation: ", observation)
        reward_input = np.concatenate([self.prev_observation, action], axis=-1)
        # reward_input = np.concatenate([self.prev_observation, action], axis=-1)
        reward_input = np2torch(reward_input)
        reward = self.reward_network(reward_input)
        self.prev_observation = observation
        return observation, reward, done, info