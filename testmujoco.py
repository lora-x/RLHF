import mujoco
import gym

env = gym.make('Humanoid-v2')
obs = env.reset()

print(obs)
