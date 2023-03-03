import gym
import copy
import time
from gym.wrappers import RecordVideo, Monitor
from rlhf_record_video import CustomRecordVideo

FREQUENCY = 50
VIDEO_LENGTH = 30

def step_trigger(step):
    if step % FREQUENCY == 0:
        return True
    return False

trajectories = []
trajectory = {
    "observations": [],
    "actions": []
}

env = CustomRecordVideo(gym.make('Pendulum-v1'), video_folder='./video', step_trigger=step_trigger, video_length==VIDEO_LENGTH) # NOT using custom frequency yet
observation = env.reset()
done = False

frames = 10
i = 0
while not done:
    trajectory["observations"].append(observation.tolist())
    action = env.action_space.sample()
    trajectory["actions"].append(action.tolist())
    observation, reward, done, info = env.step(action) # observation here is the next observation
    # print(env.step(action)) 
    i += 1
    # if done:
    #     env.reset()
    if i % frames == 0:
        trajectories.append(copy.deepcopy(trajectory))
        trajectory["observations"] = [] # clear the trajectory
        trajectory["actions"] = []

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