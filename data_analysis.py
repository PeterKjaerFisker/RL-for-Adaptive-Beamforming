# -*- coding: utf-8 -*-
"""
@author: Dennis Sand & Peter Fisker
"""
# %% Imports
import numpy as np
import sys

import helpers
import plots
import h5py

cmd_input = sys.argv
if len(cmd_input) > 1:
    DATA_NAME = sys.argv[1]
else:
    DATA_NAME = "car_urban_LOS_sarsa_TFFF_2-2-0-0-0-0_5000_300"

# %% Load results
data_reward = h5py.File(f'Results/{DATA_NAME}_results.hdf5', 'r+')
R_log_db = data_reward['R_log']
R_max_log_db = data_reward['R_max']
R_min_log_db = data_reward['R_min']
R_mean_log_db = data_reward['R_mean']

Save = False

# %% PLOT
print("Starts plotting")

# Calculate differences
R_log_db = 10*np.log10(R_log_db)
R_max_log_db = 10*np.log10(R_max_log_db)
R_min_log_db = 10*np.log10(R_min_log_db)
R_mean_log_db = 10*np.log10(R_mean_log_db)


Misalignment_log_dB = R_log_db - R_max_log_db
Meanalignment_log_dB = R_mean_log_db - R_max_log_db
Minalignment_log_dB = R_min_log_db - R_max_log_db

plots.ECDF(Save, Misalignment_log_dB, 1)

plots.Relative_reward(Save,
                      np.mean(Misalignment_log_dB, axis=0),
                      np.mean(Meanalignment_log_dB, axis=0),
                      np.mean(Minalignment_log_dB, axis=0))

plots.stability(Save, R_log_db, 50)

# Code for plotting specific episode
# start = 102
# stop = 103

# plots.mean_reward(Save, R_max_log_db[start:stop], R_mean_log_db[start:stop], R_min_log_db[start:stop], R_log_db[start:stop],
#                   ["R_max", "R_mean", "R_min", "R"], "Mean Rewards db",
#                   db=True)

plots.mean_reward(Save, R_max_log_db, R_mean_log_db, R_min_log_db, R_log_db,
                  ["R_max", "R_mean", "R_min", "R"], "Mean Rewards db",
                  db=True)

# X-db misalignment probability
x_db = 3
ACC_xdb = helpers.misalignment_prob(np.mean(R_log_db, axis=0), np.mean(R_max_log_db, axis=0), x_db)
print(F"{x_db}-db Mis-alignment probability: {ACC_xdb:0.3F} for full length")

NN = 1000
ACC_xdb_NL = helpers.misalignment_prob(np.mean(R_log_db[:, -NN:], axis=0), np.mean(R_max_log_db[:, -NN:], axis=0), x_db)
print(F"{x_db}-db Mis-alignment probability: {ACC_xdb_NL:0.3F} for the last {NN}")

ACC_xdb_NF = helpers.misalignment_prob(np.mean(R_log_db[:, 0:NN], axis=0), np.mean(R_max_log_db[:, 0:NN], axis=0), x_db)
print(F"{x_db}-db Mis-alignment probability: {ACC_xdb_NF:0.3F} for the first {NN}")

print("Done")
