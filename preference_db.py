"""
a simple database for preferences
format: traj1s["observation"][i], traj1s["action"][i], traj2s["observation"][i], traj2s["action"][i], preferences[i] 
    form one pair of trajectories and the corresponding preference

Uses a Singleton pattern to share the database between HumanFeedback wrapper (add pref) and CustomReward wrapper (sample pref)
"""
class PreferenceDb(object):
    _instance = None
    def __init__(self):
        if self._instance is not None:
            raise Exception("Only one instance of PreferenceDatabase is allowed.")
        else:
            self.traj1s = {
                "observations": [],
                "actions": [],
            }
            self.traj2s = {
                "observations": [],
                "actions": [],
            }
            self.preferences = []
            self.db_size = 0
            PreferenceDb._instance = self

    @staticmethod
    def get_instance():
        if PreferenceDb._instance is None:
            PreferenceDb()
        return PreferenceDb._instance