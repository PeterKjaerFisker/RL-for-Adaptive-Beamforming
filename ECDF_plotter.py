# -*- coding: utf-8 -*-
"""
@author: Dennis Sand & Peter Fisker
"""
# %% Imports
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import numpy as np
import sys
import seaborn as sns

import plots
import h5py

cmd_input = sys.argv
if len(cmd_input) > 1:
    DATA_NAME = sys.argv[1]
else:
    # DATA_NAME = "pedestrian_LOS_SARSA_TTFT_2-2-1-8-2-32_7000_10000_validated_0.005_0.01_0.7"
    # DATA_NAME = "pedestrian_LOS_Q-LEARNING_TTFT_2-0-1-8-2-32_7000_2000_Q-LEARNING_TFFT_0-2-0-0-2-32_7000_2000"
    # DATA_NAME = "pedestrian_LOS_SARSA_TTFT_2-0-1-8-2-32_7000_10000_SARSA_TFFT_0-2-0-0-2-32_7000_10000_validated_0.01_0.01_0.6_0.0"
    # DATA_NAME = "pedestrian_LOS_SARSA_TTFT_2-0-1-8-4-64_7000_10000_SARSA_TFFT_0-2-0-0-4-64_7000_10000_validated_0.01_0.01_0.6_0.0"
    DATA_NAME = "car_urban_LOS_SARSA_TTFT_2-0-1-8-2-32_7000_10000_SARSA_TFFT_0-2-0-0-2-32_7000_10000_0.0_0.01_0.0_10000.0_results"
    DATA_NAME_2 = "car_urban_NLOS_SARSA_TTFT_2-0-1-8-2-32_7000_10000_SARSA_TFFT_0-2-0-0-2-32_7000_10000_0.0_0.01_0.0_10000.0_results"
    # DATA_NAME_3 = "pedestrian_LOS_SARSA_SarsaNoiseFactorS_2-2-1-8-2-32_7000_10000_0.0_0.01_0.7_15000.0_results"


# %% Load results
data_reward = h5py.File(f'Results/ECDFs/plot/{DATA_NAME}.hdf5', 'r+')
# data_reward = data_reward['Training']
# data_reward = data_reward['Validation']
R_log_db = data_reward['R_log']
R_max_log_db = data_reward['R_max']

data_reward_2 = h5py.File(f'Results/ECDFs/plot/{DATA_NAME_2}.hdf5', 'r+')
# data_reward_2 = data_reward_2['Training']
# data_reward_2 = data_reward_2['Validation']
R_log_db_2 = data_reward_2['R_log']
R_max_log_db_2 = data_reward_2['R_max']

# data_reward_3 = h5py.File(f'Results/ECDFs/plot/{DATA_NAME_3}.hdf5', 'r+')
# # data_reward_3 = data_reward_3['Training']
# # data_reward_3 = data_reward_3['Validation']
# R_log_db_3 = data_reward_3['R_log']
# R_max_log_db_3 = data_reward_3['R_max']



Save = False

# %% PLOT
print("Starts calculating")

# Calculate differences
R_log_db = 10*np.log10(R_log_db)
R_max_log_db = 10*np.log10(R_max_log_db)
Misalignment_log_dB = R_log_db - R_max_log_db

# Calculate differences
R_log_db_2 = 10*np.log10(R_log_db_2)
R_max_log_db_2 = 10*np.log10(R_max_log_db_2)
Misalignment_log_dB_2 = R_log_db_2 - R_max_log_db_2

# # Calculate differences
# R_log_db_3 = 10*np.log10(R_log_db_3)
# R_max_log_db_3 = 10*np.log10(R_max_log_db_3)
# Misalignment_log_dB_3 = R_log_db_3 - R_max_log_db_3

print("Starts plotting")
fig, ax = plt.subplots()
sns.ecdfplot(Misalignment_log_dB[9000:10000].flatten(), label='LOS Car')
sns.ecdfplot(Misalignment_log_dB_2[9000:10000].flatten(), label='NLOS Car')
# sns.ecdfplot(Misalignment_log_dB_3[9000:10000].flatten(), label='Sarsa: Noise factor')
# sns.ecdfplot(Misalignment_log_dB_3[9000:10000].flatten(), label='Sarsa')
# sns.ecdfplot(Misalignment_log_dB[0:1000].flatten(), label='0-999')
plt.axvline(-6, linestyle='--', color='black', label='-6 dB')
plt.axvline(-3, linestyle='-.', color='black', label='-3 dB')
# plt.title('E-CDF, Heuristic - Pedestrian LOS')
loc = plticker.MultipleLocator(base=0.05)  # this locator puts ticks at regular intervals
ax.yaxis.set_major_locator(loc)
ax.yaxis.tick_right()
plt.xlabel('Misalignment in dB')
plt.legend()
plt.show()

# %%
# plots.Relative_reward(Save,
#                       np.mean(Misalignment_log_dB, axis=0),
#                       np.mean(Meanalignment_log_dB, axis=0),
#                       np.mean(Minalignment_log_dB, axis=0))

# plots.stability(Save, R_log_db, 50)

# Code for plotting specific episode
# start = 9950
# stop = 9951
# start2 = 0
#
# plots.mean_reward(Save, R_max_log_db[start:stop][:, start2:], R_mean_log_db[start:stop][:, start2:], R_min_log_db[start:stop][:, start2:], R_log_db[start:stop][:, start2:],
#                   ["R_max", "R_mean", "R_min", "R"], "Mean Rewards db",
#                   db=True)

# plots.mean_reward(Save, R_max_log_db, R_mean_log_db, R_min_log_db, R_log_db,
#                   ["R_max", "R_mean", "R_min", "R"], "Mean Rewards db",
#                   db=True)

# X-db misalignment probability
# x_db = 3
# ACC_xdb = helpers.misalignment_prob(np.mean(R_log_db, axis=0), np.mean(R_max_log_db, axis=0), x_db)
# print(F"{x_db}-db Mis-alignment probability: {ACC_xdb:0.3F} for full length")
#
# NN = 1000
# ACC_xdb_NL = helpers.misalignment_prob(np.mean(R_log_db[:, -NN:], axis=0), np.mean(R_max_log_db[:, -NN:], axis=0), x_db)
# print(F"{x_db}-db Mis-alignment probability: {ACC_xdb_NL:0.3F} for the last {NN}")
#
# ACC_xdb_NF = helpers.misalignment_prob(np.mean(R_log_db[:, 0:NN], axis=0), np.mean(R_max_log_db[:, 0:NN], axis=0), x_db)
# print(F"{x_db}-db Mis-alignment probability: {ACC_xdb_NF:0.3F} for the first {NN}")

print("Done")
