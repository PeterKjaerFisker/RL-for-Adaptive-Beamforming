# -*- coding: utf-8 -*-
"""
@author: Dennis Sand & Peter Fisker
"""
# %% Imports
import json
import sys

import train_agent

cmd_input = sys.argv
if len(cmd_input) > 1:
    CHANNEL_SETTINGS = sys.argv[1]
    AGENT_SETTINGS = sys.argv[2]
else:
    CHANNEL_SETTINGS = "car_urban_LOS_16_users_10000_steps"
    AGENT_SETTINGS = "sarsa_TFFF_1-0-0-0-0-0_1000_1"

# %% main
if __name__ == "__main__":
    train_agent.main() CHANNEL_SETTINGS AGENT_SETTINGS