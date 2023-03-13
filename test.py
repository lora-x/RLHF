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
from human_feedback_wrapper import HumanFeedback, SyntheticFeedback
from reward_wrapper import FeedbackReward

env = SyntheticFeedback(FeedbackReward(gym.make('Pendulum-v1'), synthetic_feedback = "true"))
env.reset()

for i in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()

