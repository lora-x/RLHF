import argparse
import numpy as np
import torch
import pybullet_envs
import gym
from a3_ppo import PPO
from config import get_config
import random
import wandb

total_reward = 0 

env = gym.make("InvertedPendulumBulletEnv-v0")
env.reset()
done = False
while not done:
    action = env.action_space.sample()
    print("action = ", action)
    observation, reward, done, info = env.step(action)
    env.render(mode = "human")
    total_reward += reward
    # wandb.log({"eval env reward" : reward})
    # if i > 0:
    #     wandb.log({"avg eval env reward" : total_reward/i})
print("total reward = ", total_reward)
 # ==================== ====================


# """
# Without any special environment. Just PPO from assignment 3 on vanilla environment.
# """

# print("\n" * 5, "=================================\n", "\n" * 5)


# env = gym.make("InvertedPendulumBulletEnv-v0")
# env.reset()
# config = get_config("pendulum", True, True)
# ppo = PPO(env, config, seed = 1)

# ppo.train()


# # wandb.init(
# #     project="RLHF",
# # )

# total_reward = 0


# for i in range(1000):
#     observation, reward, done, info = env.step(ppo.policy.act(observation))
#     env.render(mode = "human")
#     total_reward += reward
#     # wandb.log({"eval env reward" : reward})
#     # if i > 0:
#         # wandb.log({"avg eval env reward" : total_reward/i})
#     if done:
#         env.reset()


# env.close()

# ==================== ====================

# for i in range(1000):
#     observation, reward, done, info = env.step(env.action_space.sample())
#     wandb.log({"predicted reward" : reward})
#     wandb.log({"env reward" : info["env_reward"]})
#     if done:
#         env.reset()

