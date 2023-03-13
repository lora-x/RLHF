import gym
import numpy as np
from network_utils import np2torch 
import cv2
import random

class HumanFeedback(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.step_id = 0
        self.ask_human_frequency = 5
        self.record = False
        self.clip_length = 10 # how long each clip should be

        # a simple database for preferences
        # traj1s["observation"][i], traj1s["action"][i], traj2s["observation"][i], traj2s["action"][i], preferences[i] 
        #   form one pair of trajectories and the corresponding preference
        self.traj1s = {
            "observations": [],
            "actions": [],
        }
        self.traj2s = {
            "observations": [],
            "actions": [],
        }
        self.preferences = []

        # buffer to store current pair of trajectory being recorded
        # this is stored seperately from the sampled paths used to train the policy for modularity (so the policy can be changed to other policies)
        self._reset_traj_buffer()

        self.which_traj = 0 # index to record whether the first or second trajectory is being recorded now

    """
    TODO: make sure clip_length is not too long compared to the labeling frequency
    """
    def _check_label_schedule(self, step_id):
        return step_id % int(self.ask_human_frequency/2) == 0
    
    def _reset_traj_buffer(self):
        self.traj_buffer = [{
            "observations": [],
            "actions": [],
            "frames": [],
            "num_frames": 0,
        }] * 2

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if not self.record and self._check_label_schedule(self.step_id):
            self.record = True # start recording
        
        if self.record:

            # add step to current trajectory
            i = self.which_traj
            frame = self.env.render(mode='rgb_array')
            self.traj_buffer[i]["observations"].append(observation)
            self.traj_buffer[i]["actions"].append(action)
            self.traj_buffer[i]["frames"].append(frame)
            self.traj_buffer[i]["num_frames"] += 1

            if self.traj_buffer["num_frames"] == self.clip_length or done:
                self.record = False # stop recording
                if self.which_traj == 1: # if this is already the second traj, i.e. have two full trajs, ask human for preference
                    if self.traj_buffer[0]["num_frames"] == self.traj_buffer[1]["num_frames"]: # ask only if the two trajs have the same length
                        self._ask_human()
                    self._reset_traj_buffer()
                self.which_traj = 1 - self.which_traj # next time record the other trajectory

        self.step_id += 1

        return observation, reward, done, info
    
    def _ask_human(self):
        
        # render two videos
        cv2.imshow("Trajectory 1", self.traj_buffer[0]["frames"])
        cv2.imshow("Trajectory 2", self.traj_buffer[1]["frames"])

        # ask human for preference
        print ("Preference (1,2 for preference | Space for equal | Enter for incomparable ): ")
        key = cv2.waitKey(0)
        if key == 10: # ASCII code for enter
            return # skip this pair because they are incomparable
        if key == ord(1): 
            preference = 1
        elif key == ord(2):
            preference = 2
        elif key == 32: # ASCII code for space
            preference = 3

        # add to database
        traj1 = {k: v for k, v in self.traj_buffer[0].items() if k in ["observations", "actions"]}
        traj2 = {k: v for k, v in self.traj_buffer[1].items() if k in ["observations", "actions"]}
        print("in ask human, traj1 : ", traj1, "type = ", type(traj1))
        self.add_preference(traj1, traj2, preference)
        return
    
    def sample_preferences(self, k):
        """
        Sample k pairs with preferences from all recorded preferences
        """
        db_size = len(self.preferences)
        indices = random.sample(range(1, db_size), min(k, db_size))
        sampled_traj1s = [self.traj1s[i] for i in indices]
        sampled_traj2s = [self.traj2s[i] for i in indices]
        sampled_preferences = [self.preferences[i] for i in indices]
        return sampled_traj1s, sampled_traj2s, sampled_preferences
    
    def add_preference(self, traj1, traj2, preference):
        """
        TODO: is traj1 ndarray or list? should be list. maybe need to convert to list
        Add a pair of trajectories and the corresponding preference to the database
        """
        self.traj1s.append(traj1)
        self.traj2s.append(traj2)
        self.preferences.append(preference)
