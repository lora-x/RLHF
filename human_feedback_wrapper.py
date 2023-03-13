import gym
import numpy as np
from network_utils import np2torch 
import cv2
from preference_db import PreferenceDb

class HumanFeedback(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.step_id = 0
        self.ask_human_frequency = 100
        self.record = False
        self.clip_length = 20 # how long each clip should be


        self.pref_db = PreferenceDb.get_instance()

        # buffer to store current pair of trajectory being recorded
        # this is stored seperately from the sampled paths used to train the policy for modularity (so the policy can be changed to other policies)
        self._reset_traj_buffer()
        self.which_traj = 0 # index to record whether the first or second trajectory is being recorded now

    """
    TODO: make sure clip_length is not too long compared to the labeling frequency
    TODO: implement actual labeling schedule
    """
    def _check_label_schedule(self, step_id):
        return step_id % int(self.ask_human_frequency/2) == 0
    
    def _reset_traj_buffer(self):
        self.traj_buffer = [{
            "observations": [],
            "actions": [],
            "frames": [],
            "num_frames": 0,
        },
        {
            "observations": [],
            "actions": [],
            "frames": [],
            "num_frames": 0,
        }] # can't do [{}]*2 because it will create two references to the same list

    def step(self, action):

        # print ("Sanity check: in HL wrapper step_id = ", self.step_id, ". In sync?")

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

            if self.traj_buffer[i]["num_frames"] == self.clip_length or done:
                self.record = False # stop recording
                if self.which_traj == 1: # if this is already the second traj, i.e. have two full trajs, ask human for preference
                    if self.traj_buffer[0]["num_frames"] == self.traj_buffer[1]["num_frames"]: # ask only if the two trajs have the same length
                        self._ask_human()
                    self._reset_traj_buffer()
                self.which_traj = 1 - self.which_traj # next time record the other trajectory

        self.step_id += 1

        return observation, reward, done, info
    
    def _render_video(self, frames1, frames2):
        print("clip length = ", len(frames1))
        print("step_id = ", self.step_id)
        for i in range(len(frames1)):
            side_by_side_frame = np.concatenate((frames1[i], frames2[i]), axis=1)
            cv2.imshow("Trajectory 1 on Left, Trajectory 2 on Right", side_by_side_frame)
            cv2.waitKey(1)
    
    def _ask_human(self):
        
        import pdb
        # pdb.set_trace()

        # render two videos
        self._render_video(self.traj_buffer[0]["frames"], self.traj_buffer[1]["frames"])

        # ask human for preference
        print ("Preference (1,2 for preference | Space for equal | Delete/Backspace for incomparable ): ")
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == 8 or key == 127: # ASCII code for backspace or delete
            print("Skipping this incomparable pair.")
            return # skip this pair because they are incomparable
        if key == ord("1"): 
            preference = 1
            print("1")
        elif key == ord("2"):
            preference = 2
            print("2")
        elif key == 32: # ASCII code for space
            preference = 3
            print("=")
        else:
            print ("Invalid input. Skipping this pair. Use 1 or 2 for preference | Space for equal | Enter/Return for incomparable.")
            return

        # add to database
        self.add_preference(self.traj_buffer[0]["observations"],
                            self.traj_buffer[0]["actions"],
                            self.traj_buffer[1]["observations"],
                            self.traj_buffer[1]["actions"],
                            preference)
        return
        
    def add_preference(self, obs1, acts1, obs2, acts2, preference):
        """
        obs1 etc is a list (length clip_length) of ndarrays (size obs_dim) 
        Add a pair of trajectories and the corresponding preference to the database
        """
        self.pref_db.traj1s["observations"].append(obs1)
        self.pref_db.traj1s["actions"].append(acts1)
        self.pref_db.traj2s["observations"].append(obs2)
        self.pref_db.traj2s["actions"].append(acts2)
        self.pref_db.preferences.append(preference)
        self.pref_db.db_size += 1
        
# TEST
import gym
env = HumanFeedback(gym.make('Pendulum-v1'))
env.reset()

for i in range(400):
    env.step(env.action_space.sample())