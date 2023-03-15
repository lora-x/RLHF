import argparse
import numpy as np
import torch
import gym
import pybullet_envs
from policy_gradient import PolicyGradient
from config import get_config
import random
import wandb


# networks
from policy_gradient import PolicyGradient
from ppo import PPO
from reward_network import RewardNetwork

# wrappers
from human_feedback_wrapper import HumanFeedback, SyntheticFeedback
from reward_wrapper import FeedbackReward


print("\n" * 5, "=================================\n", "\n" * 5)


env = SyntheticFeedback(FeedbackReward(gym.make("InvertedPendulumBulletEnv-v0"), synthetic_feedback = "true"))
env.reset()

eval_env = gym.make("InvertedPendulumBulletEnv-v0")
observation = eval_env.reset()
config = get_config("pendulum", True, True)
ppo = PPO(env, eval_env, config, seed = 1)

ppo.train()

# evaluation

# wandb.init(
#     project="RLHF",
# )

total_reward = 0


for i in range(200):
    observation, reward, done, info = eval_env.step(ppo.policy.act(observation))
    eval_env.render(mode = "human")
    total_reward += reward
    # wandb.log({"eval env reward" : reward})
    # if i > 0:
    #     wandb.log({"avg eval env reward" : total_reward/i})
    if done:
        eval_env.reset()


eval_env.close()



# for i in range(1000):
#     observation, reward, done, info = env.step(env.action_space.sample())
#     wandb.log({"predicted reward" : reward})
#     wandb.log({"env reward" : info["env_reward"]})
#     if done:
#         env.reset()

