# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""
# %% Imports
import json
import sys

import helpers

cmd_input = sys.argv
if len(cmd_input) > 1:
    SETTING = sys.argv[1]
else:
    SETTING = "Fisker_test_01"

# %% main
if __name__ == "__main__":
    # Load Settings for simulation
    with open(f'Settings/{SETTING}.json', 'r') as fs:
        setting = json.load(fs)

    # Load global parameters
    FILENAME = setting["FILENAME"]  # Name of the data file to be loaded or saved
    CASE = setting["CASE"]  # "car_highway", "pedestrian" or "car"
    ENGINE = setting["ENGINE"]
    SCENARIOS = setting["scenarios"]
    fc = setting["fc"]  # Center frequency

    parameters = [fc, SCENARIOS]

    simulation_data = helpers.quadriga_simulation(ENGINE, f"data_pos_{FILENAME}.mat", f"data_{FILENAME}",
                                                  parameters, multi_user=True)