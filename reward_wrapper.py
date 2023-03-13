import gym
import numpy as np
from network_utils import np2torch 
from reward_network import RewardNetwork
from preference_db import PreferenceDb
import random

class CustomReward(gym.Wrapper):
    def __init__(self, env, synthetic_feedback = "false"): # by default uses human feedback
        super().__init__(env)
        self.env = env
        self.synthetic_feedback = synthetic_feedback
        self.reward_network = RewardNetwork(env)
        self.prev_observation = env.reset() # initial observation
        self.update_frequency = 1 # how often to update the reward network
        self.step_id = 0

        self.pref_db = PreferenceDb.get_instance()

    # temporarily using random acion as placeholder -- might need to merge the object??  but this is supposed to be an env wrapper?
    # how can i not load the whole thing database? 

    def step(self, action):
        observation, env_reward, done, info = self.env.step(action) # env reward is the algorithmically generated real reward by the original environment

        if self.synthetic_feedback == "true":
            # add original reward info so it can be used in the feedback wrapper
            info["env_reward"] = env_reward
        
        print("info['env_reward'] = ", info["env_reward"])

        print ("Sanity check: in reward wrapper step_id = ", self.step_id)

        # calculate reward 
        reward_input = np.concatenate([self.prev_observation, action], axis=-1)
        reward_input = np2torch(reward_input)
        reward = self.reward_network.predict_reward(reward_input)
        reward = reward.detach().numpy() # tensor to numpy
        self.prev_observation = observation

        # update the reward network at given frequency
        if self.step_id % self.update_frequency == 0:
            print("Updating reward network...")
            sampled_traj1s, sampled_traj2s, sampled_preferences = self.sample_preferences()
            self.reward_network.update_network(sampled_traj1s, sampled_traj2s, sampled_preferences)

        return observation, reward, done, info
    
    def sample_preferences(self, num_samples):
        """
        Sample k pairs with preferences from all recorded preferences
        """
        pref_db = self.pref_db
        indices = random.sample(range(1, pref_db.db_size), min(num_samples, pref_db.db_size))
        sampled_traj1s = [pref_db.traj1s[i] for i in indices]
        sampled_traj2s = [pref_db.traj2s[i] for i in indices]
        sampled_preferences = [pref_db.preferences[i] for i in indices]
        return sampled_traj1s, sampled_traj2s, sampled_preferences
