# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:36:31 2022

@author: dksan
"""
# %% Imports
import numpy as np
import sys
import matplotlib.pyplot as plt
import natsort as nt

import helpers
import plots
import h5py

# %% Load results
# pickle_data = helpers.load_pickle('Results/','before_preprocessing_car_urban_LOS_sarsa_TTFF_2-2-2-4-0-0_5000_300_results.pickle')
# bulk_data = helpers.bulk_loader('Results/Centralized_Agent_Sweeps/HDF5/Decimated/')
bulk_data = helpers.bulk_loader('Results/Single_hyper_epsilon_a2/')
# bulk_data = h5py.File('Results/Multi_test/pedestrian_LOS_SARSA_TFFT_2-2-0-0-2-32_7000_10000_validated_results.hdf5','r+')

# %% Data analysis

# span = [(0,100),(100,200),(200,300),(300,400),(400,500),(500,600),(600,700),(700,800),(800,900),(900,1000)]
# span = [(0, 1000),(1000, 2000),(2000, 3000),(3000, 4000),(4000, 5000),(5000, 6000),(6000, 7000),(7000, 8000), (8000, 9000),(9000, 10000)]
# span = [(0, 250),(250, 500),(500, 750),(750, 1000),(1000, 1250),(1250, 1500),(1500, 1750),(1750, 2000), (2000, 2250),(2250, 2500)]
# span = [(0,50),(50,100),(100,150),(150,200),(200,250),(250,300)]
# span = [(0,1000),(4500,5500),(9000,10000)]
# span = [(0,1000),(2250,3250),(4500,5500),(6750,7750),(9000,10000)]
span = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000), (5000, 6000), (6000, 7000), (7000, 8000),
        (8000, 9000), (9000, 10000)]
# span = [(x*100,(x+1)*100) for x in range(100)]
method = False
Save = False

dim1 = len(bulk_data)
dim2 = len(span)

plot_array = np.zeros((dim1, dim2))

plt.figure()
for test_idx, test in enumerate(nt.natsorted(bulk_data.keys())):

    Val_stat = len(bulk_data[list(bulk_data.keys())[test_idx]])

    if Val_stat != 2:
        R_log_db = 10 * np.log10(bulk_data[f'{test}/R_log'])
        R_max_log_db = 10 * np.log10(bulk_data[f'{test}/R_max'])

    else:
        # CHANGE IF TRAINING DATA SHOULD BE PLOTTED

        # R_log_db = 10 * np.log10(bulk_data[f'{test}/Training/R_log'])
        # R_max_log_db = 10 * np.log10(bulk_data[f'{test}/Training/R_max'])

        R_log_db = 10 * np.log10(bulk_data[f'{test}/Validation/R_log'])
        R_max_log_db = 10 * np.log10(bulk_data[f'{test}/Validation/R_max'])

        R_mean_log_db = 10 * np.log10(bulk_data[f'{test}/Validation/R_mean'])

    Misalignment = R_log_db - R_max_log_db
    averaged_episodes = np.abs(np.average(Misalignment, axis=1))

    for idx, (start, stop) in enumerate(span):
        plot_array[test_idx, idx] = np.average(averaged_episodes[start:stop])

        # Label construction

        # OLD LABELS
        # stripped_label = test.lstrip('car_urbanpedestrian_NLOS_sarsaSIMPLEQ-LEARNING_TF_') # Removes leading information
        # suffix_remove = stripped_label.lstrip('0123456789-')
        # stripped_label = stripped_label.removesuffix(suffix_remove)

        # NEW AND IMPROVED LABELS, still cant handle validated and hyper param stuff
        stripped_label = test.lstrip('car_urbanpedestrian_NLOS_sarsaSIMPLEQ-LEARNING_TF_')
        method_label = ''
        if method == True:  # Isolates and saves the update rule
            stripped_label = test.lstrip('car_urbanpedestrian_')
            stripped_label = stripped_label.lstrip('NLOS')
            stripped_label = stripped_label.lstrip('_')
            method_remove = stripped_label.lstrip('sarsaSIMPLEQ-LEARNING')
            method_label = helpers.remove_suffix(stripped_label, method_remove) + ' '
        stripped_label = test.lstrip('car_urbanpedestrian_NLOS_sarsaSIMPLEQ-LEARNING_TF_')

        # Isolates the first state
        suffix_remove = stripped_label.lstrip('0123456789-')
        leftovers = suffix_remove.lstrip('0123456789_validated.')
        label_first = helpers.remove_suffix(stripped_label, suffix_remove)
        stripped_label = method_label + label_first

        if leftovers != '':  # Isolates the second state
            stripped_label_second = leftovers.lstrip('car_urbanpedestrian_NLOS_sarsaSIMPLEQ-LEARNING_TF_')
            suffix_remove_second = stripped_label_second.lstrip('0123456789-')
            label_second = helpers.remove_suffix(stripped_label_second, suffix_remove_second) + ' '
            stripped_label = stripped_label + ' & ' + label_second

        validate_label = test.lstrip('car_urbanpedestrian_NLOS_sarsaSIMPLEQ-LEARNING_TF_0123456789-')
        if validate_label != '':
            validate_label = validate_label.lstrip('validated_')
            epsilon_remove = validate_label.lstrip('.0123456789')
            epsilon_label = helpers.remove_suffix(validate_label, epsilon_remove)
            stepsize_remove = epsilon_remove.lstrip('_')
            stepsize_remove = stepsize_remove.lstrip('.0123456789')
            stepsize_label = ' ' + helpers.remove_suffix(epsilon_remove, stepsize_remove).lstrip('_')
            discount_remove = stepsize_remove.lstrip('_')
            discount_remove = discount_remove.lstrip('.0123456789')
            discount_label = ' ' + helpers.remove_suffix(stepsize_remove, discount_remove).lstrip('_')
            weight_remove = discount_remove.lstrip('_')
            weight_remove = weight_remove.lstrip('.0123456789')
            weight_label = ' ' + helpers.remove_suffix(discount_remove, weight_remove).lstrip('_')
            stripped_label = stripped_label + epsilon_label + stepsize_label + discount_label + weight_label

    if stripped_label == '2-0-1-8-2-32 & 0-2-0-0-2-32 0.05 0.05 0.7 ':
        plt.plot(plot_array[test_idx, :], color='k', label=stripped_label, marker='d', linestyle='dashed')
    else:
        plt.plot(plot_array[test_idx, :], label=stripped_label, marker='d')

# Constructs the title for the plot (removes validated and hyper parameters)

if method == True:
    title_remove = test.lstrip('car_urbanpedestrian_')
    title_remove = title_remove.lstrip('NLOS')
    title_remove = title_remove.lstrip('_')
else:
    title_remove = test.lstrip('car_urbanpedestrian_NLOS_sarsaSIMPLEQ-LEARNING_')

# plot_title_first = test.removesuffix(title_remove)
plot_title_first = helpers.remove_suffix(test, title_remove)

id_length = title_remove.lstrip('SARSASIMPLEQ-LEARNING_')
id_length = id_length.lstrip('TF_')
id_length = id_length.lstrip('-0123456789')
id_length = id_length.lstrip('_')
id_remove = id_length.lstrip('0123456789_')
# id_length = id_length.removesuffix(id_remove)
# id_length = id_length.removesuffix('_')
id_length = helpers.remove_suffix(id_length, '_' + id_remove)

plot_title = plot_title_first + id_length

# plot_title = test.removesuffix('_'+title_remove)+suffix_remove
# title_remove = plot_title.lstrip('car_urbanpedestrian_NLOS_sarsaSIMPLEQ-LEARNING_0123456789')
# plot_title = plot_title.removesuffix('_'+title_remove)

lgd = plt.legend(bbox_to_anchor=(0.5, -0.25), loc="upper center")

plt.xticks(range(len(span)), span, rotation=20)
plt.xlabel("Episode range [-,-]")
plt.ylabel("Average absolute misalignment [dB]")
plt.grid(True, axis='x')
plt.title(plot_title)

if Save:
    plt.savefig("Figures/Performance_MPLS.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight') # Saves the figure

plt.show()

bulk_data.close()

# start = 9996
# stop = 9997
# start2 = 0
#
# plots.mean_reward(False, R_max_log_db[start:stop][:, start2:], R_mean_log_db[start:stop][:, start2:], R_mean_log_db[start:stop][:, start2:], R_log_db[start:stop][:, start2:],
#                   ["R_max", "R_mean", "R_min", "R"], "Mean Rewards db",
#                   db=True)
#
#
# plots.ECDF(False, Misalignment[start:stop], 1)
