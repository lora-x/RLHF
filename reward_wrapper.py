import gym
import numpy as np
from network_utils import np2torch 

class CustomReward(gym.Wrapper):
    def __init__(self, env, reward_network, initial_observation):
        super().__init__(env)
        self.reward_network = reward_network
        self.env = env
        self.prev_observation = initial_observation

    def ask_human(self):
        """
        Ask human to provide preference between two trajectories
        1 = traj1 is more preferable, 2 = traj2 is more preferable, 3 = both are equally preferable
        if undecidable, do not append the pair of trajs to database. 
        """
        preference = -1 # by default, do not append to database

        if len(self.trijectories) < 2:
            return
        
        # temporarily using random acion as placeholder -- might need to merge the object??  but this is supposed to be an env wrapper?
        # how can i not load the whole thing database? 

    def step(self, action):
        observation, _, done, info = self.env.step(action)
        reward_input = np.concatenate([self.prev_observation, action], axis=-1)
        reward_input = np2torch(reward_input)
        reward = self.reward_network(reward_input)
        reward = reward.detach().numpy() # tensor to numpy
        self.prev_observation = observation
        return observation, reward, done, info
    

# class Preference(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.traj1s = {"observations": [], "actions": []}
#         self.traj2s = {"observations": [], "actions": []}
#         self.preferences = []
