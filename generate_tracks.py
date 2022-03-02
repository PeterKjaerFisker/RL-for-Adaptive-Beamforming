# %% Imports
import json
import sys

import helpers


cmd_input = sys.argv
if len(cmd_input) > 1:
    CHANNEL_SETTINGS = sys.argv[1]
else:
    CHANNEL_SETTINGS = "pedestrian_LOS_8_users_20000_steps"

# Load Settings for simulation
with open(f'Settings/{CHANNEL_SETTINGS}.json', 'r') as fs:
    setting = json.load(fs)

# Load global parameters
FILENAME = setting["FILENAME"]  # Name of the data file to be loaded or saved
CASE = setting["CASE"]  # "car_highway", "pedestrian" or "car"

# ----------- Channel Simulation Parameters -----------
scenarios = setting["scenarios"]  # Quadriga scenarios, page 101 of Quadriga documentation
N = setting["N"]  # Number of steps in an episode
sample_period = setting["sample_period"]  # The sample period in [s]
M = setting["M"]  # Number of episodes
r_lim = setting["rlim"]  # Radius of the cell


para = [N, M, r_lim, sample_period, scenarios]
# Load Scenario configuration
with open(f'Cases/{CASE}.json', 'r') as fp:
    case = json.load(fp)

# Generate track
helpers.create_pos_log(case, para, f"data_pos_{FILENAME}.mat")
