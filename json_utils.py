import json

sample_pair = {
    "traj1": {
        "observations": [1.2, 3.2, 3, -1.1],
        "actions": [0.1, 0.2, -0.3, 0.4]
    },
    "traj2": {
        "observations": [0.2, 0.2, 3, -1.1],
        "actions": [1, 2, -0.3, 0.4]
    },
    "preference": 1
}

sample_list = [sample_pair, sample_pair]

with open ("human_preference.json", "w") as jsonfile:
    json.dump(sample_list, jsonfile)

with open ("human_preference.json") as jsonfile:
    data = json.load(jsonfile)
    print(data)

