import gym
import numpy as np
from network_utils import np2torch 
from reward_network import RewardNetwork
from preference_db import PreferenceDb
import random
import json
from scipy import stats

class FeedbackReward(gym.Wrapper):
    def __init__(self, env): # by default uses human feedback
        super().__init__(env)
        self.env = env
        self.reward_network = RewardNetwork(env)
        self.prev_observation = env.reset() # initial observation
        self.update_frequency = 20 # how often to update the reward network
        self.batch_size = 64 # how many samples to use to update the reward network
        self.step_id = 0
        self.new_behavior = False
        self.correlation = {
            "coefficients": [],
            "pvalues": [],
            "step_ids": [],
        }
        
        print("Updating reward network every {} steps".format(self.update_frequency))
        self.pref_db = PreferenceDb.get_instance()

    def step(self, action):
        observation, env_reward, done, info = self.env.step(action) # env reward is the algorithmically generated real reward by the original environment
        # print ("in reward wrapper,input observation = ", observation)
        # add original reward info so it can be used in the feedback wrapper for synthetic feedback
        info["env_reward"] = env_reward

        # print ("Sanity check: in reward wrapper step_id = ", self.step_id, ". In sync?)

        # calculate reward 
        reward_input = np.concatenate([self.prev_observation, action], axis=-1)
        reward_input = np2torch(reward_input)
        # reward_input = np2torch(np.expand_dims(reward_input, axis = 0)) # now dim = 1 x (obs dim + act dim), bc network expects a non-empty batch size 
        reward = self.reward_network.predict_reward(reward_input, inference=True) # turn on eval mode so reward network doesn't do batch norm, avoid dimension mismatch
        reward = reward.detach().numpy() # tensor to numpy
        self.prev_observation = observation

        # update the reward network at given frequency
        if self.pref_db.db_size > 0 and self.step_id % self.update_frequency == 0:
            # print(f"At step {self.step_id}, updating reward network...")
            sampled_traj1s, sampled_traj2s, sampled_preferences = self.sample_preferences(self.batch_size)
            self.reward_network.update_network(sampled_traj1s, sampled_traj2s, sampled_preferences)

            # if env reward is true reward, i.e. not trying to train a new behavior, then evaluate the reward network with correlation
            if not self.new_behavior:
                # a lot of overlap between the update network function in reward_network
                sampled_traj1s, sampled_traj2s, sampled_preferences = self.sample_preferences(self.batch_size) # resample eval set
                predicted_rewards = self.reward_network.traj_to_reward(sampled_traj1s, mode = "eval").detach().numpy().flatten() # flatten from (batch_size, clip_length) to (batch_size * clip_length, )
                true_rewards = np.array(sampled_traj1s["env_rewards"]).flatten()
                # flatten both rewards to 1d array
                # print("predicted reward shape = ", predicted_rewards.shape)
                # print("true reward shape = ", true_rewards.shape)
                
                # calculate correlation
                result = stats.spearmanr(predicted_rewards, true_rewards)
                self.correlation["coefficients"].append(result.statistic)
                self.correlation["pvalues"].append(result.pvalue)
                self.correlation["step_ids"].append(self.step_id)

        self.step_id += 1

        # print("in reward wrapper, output observation = ", observation)
        return observation, reward, done, info
    
    def sample_preferences(self, num_samples):
        """
        Sample k pairs with preferences from all recorded preferences
        """
        pref_db = self.pref_db
        indices = random.sample(range(pref_db.db_size), min(num_samples, pref_db.db_size))
        # TODO: can probably do a better data structure back in feedback wrapper to avoid so much iteration
        sampled_traj1s = {"observations": [pref_db.traj1s["observations"][i] for i in indices], "actions": [pref_db.traj1s["actions"][i] for i in indices], "env_rewards": [pref_db.traj1s["env_rewards"][i] for i in indices]}
        sampled_traj2s = {"observations": [pref_db.traj2s["observations"][i] for i in indices], "actions": [pref_db.traj2s["actions"][i] for i in indices], "env_rewards": [pref_db.traj2s["env_rewards"][i] for i in indices]}
        sampled_preferences = [pref_db.preferences[i] for i in indices]
        return sampled_traj1s, sampled_traj2s, sampled_preferences
    
    def init_test_db(self):
        with open("test_db.json") as f:
            self.pref_db = json.load(f)
        return

# TEST

# env = FeedbackReward(gym.make('Pendulum-v1'))
# env.reset()
# env.init_test_db()

# for i in range(400):
#     observation, reward, done, info = env.step(env.action_space.sample())
#     if done:
#         env.reset()