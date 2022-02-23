# -*- coding: utf-8 -*-
"""
@author: Dennis Sand & Peter Fisker
"""
# %% Imports
import numpy as np

import helpers
import plots
# %% Load pickle

data = helpers.load_pickle('' ,'Car_highway_2500ep.pickle')
Agent = data['Agent']
R_log = data['R_log']
R_max_log = data['R_max']
R_min_log = data['R_min']
R_mean_log = data['R_mean']
setting = data['settings']
Save = False

# %% PLOT
print("Starts plotting")

# Get the Logs in power decibel
R_log_db = 10 * np.log10(R_log)
R_max_log_db = 10 * np.log10(R_max_log)
R_min_log_db = 10 * np.log10(R_min_log)
R_mean_log_db = 10 * np.log10(R_mean_log)
Misalignment_log_dB = R_log_db - R_max_log_db
Meanalignment_log_dB = R_mean_log_db - R_max_log_db
Minalignment_log_dB = R_min_log_db - R_max_log_db

plots.ECDF(Save, np.mean(Misalignment_log_dB[-3:-1 ,:], axis=0))
plots.Relative_reward(Save, np.mean(Misalignment_log_dB, axis=0), np.mean(Meanalignment_log_dB, axis=0), np.mean(Minalignment_log_dB, axis=0))
plots.stability(Save, R_log_db, 50)

plots.mean_reward(Save, R_max_log_db, R_mean_log_db, R_min_log_db, R_log_db,
                  ["R_max", "R_mean", "R_min", "R"], "Mean Rewards db",
                  db=True)

# plots.positions(pos_log, r_lim)

# X-db misalignment probability
x_db = 3
ACC_xdb = helpers.misalignment_prob(np.mean(R_log_db, axis=0),
                                    np.mean(R_max_log_db, axis=0), x_db)
print(F"{x_db}-db Mis-alignment probability: {ACC_xdb:0.3F} for full length")

NN = 1000
ACC_xdb_NL = helpers.misalignment_prob(np.mean(R_log_db[:, -NN:], axis=0),
                                       np.mean(R_max_log_db[:, -NN:], axis=0), x_db)
print(F"{x_db}-db Mis-alignment probability: {ACC_xdb_NL:0.3F} for the last {NN}")

ACC_xdb_NF = helpers.misalignment_prob(np.mean(R_log_db[:, 0:NN], axis=0),
                                       np.mean(R_max_log_db[:, 0:NN], axis=0), x_db)
print(F"{x_db}-db Mis-alignment probability: {ACC_xdb_NF:0.3F} for the first {NN}")

print("Done")
