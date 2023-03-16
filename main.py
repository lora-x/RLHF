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
import pdb

# wrappers
from human_feedback_wrapper import HumanFeedback, SyntheticFeedback
from reward_wrapper import FeedbackReward


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-name", required=True, type=str, choices=["cartpole", "pendulum", "cheetah"]
)
parser.add_argument("--synthetic", dest="synthetic", action="store_true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--entropy", type=float, choices=[0.0, 0.01, 0.05, 0.1], default=0.05)

parser.set_defaults(use_baseline=True)


if __name__ == "__main__":

    args = parser.parse_args()

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("\n" * 5, "=================================\n", "\n" * 5)

    config = get_config(args.env_name, args.seed, args.entropy)
    
    if args.synthetic:
        env = SyntheticFeedback(FeedbackReward(gym.make(config.env_name), synthetic_feedback = True))
    else:
        env = HumanFeedback(FeedbackReward(gym.make(config.env_name), synthetic_feedback = False))
    eval_env = gym.make(config.env_name)
    
    # train model
    env.reset()
    eval_env.reset()
    model = PPO(env, eval_env, config, args.seed)
    model.train()
