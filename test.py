import argparse
import numpy as np
import torch
import gym
import pybullet_envs
from policy_gradient import PolicyGradient
from config import get_config
import random
import wandb
import time


# networks
from policy_gradient import PolicyGradient
from ppo import PPO
from reward_network import RewardNetwork

# wrappers
from human_feedback_wrapper import HumanFeedback, SyntheticFeedback
from reward_wrapper import FeedbackReward


print("\n" * 5, "=================================\n", "\n" * 5)

# bullet_env = "InvertedPendulumBulletEnv-v0"
pendulum = "Pendulum-v1"
env = SyntheticFeedback(FeedbackReward(gym.make(pendulum), synthetic_feedback = True))
env.reset()

eval_env = gym.make(pendulum)
eval_env.reset()
observation = eval_env.reset()

config = get_config("pendulum")
ppo = PPO(env, eval_env, config, seed = 1)


ppo.train()

print("training done")

# # evaluation

# # wandb.init(
# #     project="RLHF",
# # )

total_reward = 0
done = False 
eval_env.reset()

while not done:
    eval_env.render(mode = "human")
    observation, reward, done, info = eval_env.step(ppo.policy.act(observation))
    total_reward += reward
    # wandb.log({"eval env reward" : reward})
    # if i > 0:
    #     wandb.log({"avg eval env reward" : total_reward/i})

eval_env.close()
print("total reward = ", total_reward)
