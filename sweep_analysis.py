# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:36:31 2022

@author: dksan
"""
# %% Imports
import numpy as np
import sys
import matplotlib.pyplot as plt

import helpers
import plots
import h5py

# %% Load results
# pickle_data = helpers.load_pickle('Results/','before_preprocessing_car_urban_LOS_sarsa_TTFF_2-2-2-4-0-0_5000_300_results.pickle')
# bulk_data = helpers.bulk_loader('Results/Centralized_Agent_Sweeps/HDF5/Decimated/')
bulk_data = helpers.bulk_loader('Results/Centralized_Agent_Sweeps/')

# %% Data analysis

# span = [(0,100),(100,200),(200,300),(300,400),(400,500),(500,600),(600,700),(700,800),(800,900),(900,1000)]
span = [(0, 1000),(1000, 2000),(2000, 3000),(3000, 4000),(4000, 5000),(5000, 6000),(6000, 7000),(7000, 8000), (8000, 9000),(9000, 10000)]
# span = [(0, 250),(250, 500),(500, 750),(750, 1000),(1000, 1250),(1250, 1500),(1500, 1750),(1750, 2000), (2000, 2250),(2250, 2500)]

dim1 = len(bulk_data)
dim2 = len(span)

plot_array = np.zeros((dim1, dim2))

for test_idx, test in enumerate(bulk_data.keys()):
    # averaged_episodes = np.average(bulk_data[f'{test}/R_log'], axis = 1)
    R_log_db = 10 * np.log10(bulk_data[f'{test}/R_log'])
    R_max_log_db = 10 * np.log10(bulk_data[f'{test}/R_max'])
    # R_log_db = 10*np.log10(pickle_data['R_log'])
    # R_max_log_db = 10*np.log10(pickle_data['R_max'])

    # R_mean_log_db = 10*np.log10(bulk_data[f'{test}/R_mean'])

    Misalignment = R_log_db - R_max_log_db
    # Meanalignment = R_mean_log_db - R_max_log_db
    averaged_episodes = np.abs(np.average(Misalignment, axis=1))

    # averaged_episodes = np.abs(np.average(Meanalignment, axis = 1))

    for idx, (start, stop) in enumerate(span):
        plot_array[test_idx, idx] = np.average(averaged_episodes[start:stop])

# plots.barplot(False, plot_array, span)
plt.plot(plot_array.T, marker='x')
plt.show()

bulk_data.close()

# R_log_db = 10*np.log10(R_log_db)
# R_max_log_db = 10*np.log10(R_max_log_db)
# R_min_log_db = 10*np.log10(R_min_log_db)
# R_mean_log_db = 10*np.log10(R_mean_log_db)


# Misalignment_log_dB = R_log_db - R_max_log_db
