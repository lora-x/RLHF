# -*- coding: UTF-8 -*-

import argparse
import numpy as np
import torch
import gym
import pybullet_envs
from policy_gradient import PolicyGradient
from ppo import PPO
from config import get_config
import random

# networks
from policy_gradient import PolicyGradient
from reward_network import RewardNetwork

# wrappers
from gym.wrappers import RecordVideo, Monitor
from human_feedback_wrapper import HumanFeedback
from rlhf_record_video import CustomRecordVideo
from reward_wrapper import CustomReward

import pdb

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-name", required=True, type=str, choices=["cartpole", "pendulum", "cheetah"]
)
parser.add_argument("--baseline", dest="use_baseline", action="store_true")
parser.add_argument("--no-baseline", dest="use_baseline", action="store_false")
parser.add_argument("--ppo", dest="ppo", action="store_true")
parser.add_argument("--seed", type=int, default=1)

parser.set_defaults(use_baseline=True)

def step_trigger(step):
    if step % FREQUENCY == 0:
        return True
    return False

FREQUENCY = 50
VIDEO_LENGTH = 30
NUM_BATCHES = 10
REWARD_UPDATE_FREQUENCY = 5
REWARD_BATCH_SIZE = 10

if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
    env = HumanFeedback(env)
    env = CustomReward(env)
    policy = PolicyGradient(env)
    for batch in NUM_BATCHES:

        # update policy network
        paths = policy.sample_paths()
        policy.update_network(paths)
        
        # update reward network
        if batch % REWARD_UPDATE_FREQUENCY == 0:
            sampled_traj1s, sampled_traj2s, sampled_preferences = env.sample_preferences(REWARD_BATCH_SIZE)
            reward_network.update_network(sampled_traj1s, sampled_traj2s, sampled_preferences)

        
        

# original hw3 code
# if __name__ == "__main__":
#     args = parser.parse_args()

#     torch.random.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     random.seed(args.seed)

#     config = get_config(args.env_name, args.use_baseline, args.ppo, args.seed)

#     env = gym.make(config.env_name)
#     observation = env.reset() # initial observation
#     reward_network = RewardNetwork(env)
#     env = CustomReward(env, reward_network, observation)
#     env = CustomRecordVideo(env, video_folder='./video', step_trigger=step_trigger, video_length=VIDEO_LENGTH) # NOT using custom frequency yet
    
#     # train model
#     model = PolicyGradient(env, config, args.seed) if not args.ppo else PPO(env, config, args.seed)
#     model.run()

# pseudo code from here

"""
for batch in num_batches:
    paths = policy.sample_paths()
    policy.update_network(paths)
    if batch % update_reward_network_frequency == 0:
        sampled_preference = env.traj_preference_pairs.sample()
        env.reward_network.update_network(sampled_preference)



then in policy.sample_paths(): // change to step based, not episode based
    for episode in num_episodes:
        for step in max_ep_len:
            action = policy(observation)
            frame = env.render(mode = 'rgb_array')
            observation, reward, done, info = env.step(action)
            append step to traj
        append traj to trajs
        if episode % ask_human_FREQUENCY == 0:
            env.ask_human(trajs[-1], traj[-2])

then in env.step(action):
    if should_record_step:
        frame = env.render(mode = 'rgb_array')
        frames.append(frame)
    if should_ask_human:

in env.ask_human(traj1_rbgarray_list, traj2_rbgarray_list):
    choose a segment of traj1 and traj2 to ask human
    env.traj_preference_pairs.append(traj_pair_with_preference)
        
"""