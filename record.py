import gym, copy, time
import numpy as np

# networks
from policy_gradient import placeholder_policy
from reward_network import RewardNetwork

# wrappers
from gym.wrappers import RecordVideo, Monitor
from rlhf_record_video import CustomRecordVideo
from reward_wrapper import CustomReward


FREQUENCY = 50
VIDEO_LENGTH = 30

def step_trigger(step):
    if step % FREQUENCY == 0:
        return True
    return False

trajectories = []
trajectory = {
    "observations": [],
    "actions": [],
    "rewards": [],
}

# observations, actions = [], []

# when sampling trajectories, need to truncate to the shorter length of the two, using assert

# init
env = gym.make('Pendulum-v1')
observation = env.reset() # initial observation
reward_network = RewardNetwork(env)
env = CustomReward(env, reward_network, observation)
env = CustomRecordVideo(env, video_folder='./video', step_trigger=step_trigger, video_length=VIDEO_LENGTH) # NOT using custom frequency yet

done = False

frames = 10
i = 0
while i < 300:
    print("i: ", i)
    # sample action
    action = placeholder_policy(env, observation)
    # step
    next_observation, reward, done, info = env.step(action)
    print("observation: ", observation)
    print("next_observation: ", next_observation)
    trajectory["observations"].append(observation.tolist()) 
    trajectory["actions"].append(action.tolist())
    trajectory["rewards"].append(reward.tolist())
    # update observation for next step
    observation = next_observation 
    i += 1
    if i % frames == 0:
        trajectories.append(copy.deepcopy(trajectory))
        trajectory["observations"] = [] # clear the trajectory
        trajectory["actions"] = []
    # if done:
    #     observation = env.reset() # problem: this creates async for 1 round between observation here and prev_observation in reward_wrapper.py for next round
env.close()



# import gym
# import cv2
# import numpy as np
# from gym.wrappers import Monitor

# env = Monitor(gym.make('CartPole-v0'), './video', force=True)
# state = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()
#     state_next, reward, done, info = env.step(action)
# env.close()



# //////

# env = gym.make('Humanoid-v2')
# env.reset()

# # Set up the video recorder
# video_width = 640
# video_height = 480
# fps = 60
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video = cv2.VideoWriter('output.mp4', fourcc, fps, (video_width, video_height))

# for i in range(1000):
#     # Render the environment
#     img = env.render(mode='rgb_array', width=video_width, height=video_height)

#     # Write the frame to the video
#     video.write(np.uint8(img))

#     # Take a step in the environment
#     obs, _, done, _ = env.step(env.action_space.sample())
#     if done:
#         env.reset()

# # Release the video recorder
# video.release()