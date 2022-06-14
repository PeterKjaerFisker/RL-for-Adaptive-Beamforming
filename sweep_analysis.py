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
bulk_data = helpers.bulk_loader('Results/New_folder/')
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
Save = True
method = False
Scenario = False
State = True
Hyper = False

generate_label = True

# Legend_list=['Adaptive: w = 15','Constant: ε = 0.005','Reference','Decaying: w = 300']
# Legend_list=['Reference','Adaptive: w = 10','Constant: ε = 0.01','Decaying: w = 300']
# Legend_list=['Reference','Constant: ε = 0.001','Constant: ε = 0.01','Constant: ε = 0.005']
# Legend_list=['Reference','Decaying: w = 150','Decaying: w = 300','Decaying: w = 600','Decaying: w = 900']
# Legend_list=['Reference','Adaptive: w = 0.3','Adaptive: w = 1','Adaptive: w = 10','Adaptive: w = 15']
# Legend_list=['Reference','α = 0.01','α = 0.005','α = 0.05']
# Legend_list=['Reference','γ = 0.5','γ = 0.6','γ = 0.7','γ = 0.8']
# Legend_list=['Heuristic','Q-learning','Sarsa','Simple']
# Legend_list=['Q-learning','Sarsa','Simple']
# Legend_list=['Q-learning','Sarsa','Wolf']
Legend_list = ['2-0-1-8-2-32 & 0-2-0-0-2-32']

# Legend_list=['Reference',
#              '2-2-0-0-2-32',
#              '2-2-0-0-32-32',
#              '2-2-1-8-0-0',
#              '2-2-1-8-2-32']

# Legend_list = ['Heuristic: No noise','Heuristic: Noise factor','Heuri
#
# stic: Thermal noise','Sarsa: No noise','Sarsa: Noise factor','Sarsa: thermal noise']
# Legend_list = ['Heuristic noise factor','Heuristic Thermal noise','Sarsa noise factor multi','Sarsa noise factor single','Sarsa thermal noise multi','Sarsa thermal noise single']
# Legend_list = ['Tuned multi agent','Tuned single agent']
# ['ε = ','α = ','γ = ','w = ']

dim1 = len(bulk_data)
dim2 = len(span)

plot_array = np.zeros((dim1, dim2))

plt.figure()
for test_idx, test in enumerate(nt.natsorted(bulk_data.keys())):

    Val_stat = len(bulk_data[list(bulk_data.keys())[test_idx]])

    if Val_stat == 2:
        R_log_db = 10 * np.log10(bulk_data[f'{test}/Validation/R_log'])
        R_max_log_db = 10 * np.log10(bulk_data[f'{test}/Validation/R_max'])
        # R_log_db = 10 * np.log10(bulk_data[f'{test}/Training/R_log'])
        # R_max_log_db = 10 * np.log10(bulk_data[f'{test}/Training/R_max'])

        R_mean_log_db = 10 * np.log10(bulk_data[f'{test}/Validation/R_mean'])

    elif Val_stat > 2:
        # CHANGE IF TRAINING DATA SHOULD BE PLOTTED

        R_log_db = 10 * np.log10(bulk_data[f'{test}/R_log'])
        R_max_log_db = 10 * np.log10(bulk_data[f'{test}/R_max'])

    elif Val_stat == 1:
        R_log_db = 10 * np.log10(bulk_data[f'{test}/Training/R_log'])
        R_max_log_db = 10 * np.log10(bulk_data[f'{test}/Training/R_max'])

    Misalignment = R_log_db - R_max_log_db
    averaged_episodes = np.abs(np.average(Misalignment, axis=1))

    for idx, (start, stop) in enumerate(span):
        plot_array[test_idx, idx] = np.average(averaged_episodes[start:stop])

        # Label construction
        # EVEN NEWER AND MORE IMPROVED LABELS
    strip_list = ['NLOS', 'SARSAQ-LEARNINGWEIGHTEDAVERAGEWOLFHUMANn',
                  'TFqwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNMSarsaThermanlNoiseMS', '-0123456789',
                  '0123456789', '0123456789', '.0123456789']
    labels = []

    multi = False
    # Isolate case
    case_remove = test.lstrip('carpedestrian')
    case_remove = case_remove.lstrip('_')
    case_remove = case_remove.lstrip('urbanhighway')
    case_remove = case_remove.lstrip('_')
    case_label = helpers.remove_suffix(test, '_' + case_remove)
    labels.append(case_label)
    old_remove = case_remove
    label_remove = ''

    stripped_label = ''
    if generate_label:
        # Everything else
        for strip_key in strip_list:
            # scenario_remove = case_remove.lstrip('NLOS')
            # scenario_remove = case_remove.lstrip('_')
            # scenario = helpers.remove_suffix(case_remove,'_'+scenario_remove)

            if strip_key == '.0123456789':
                old_remove = old_remove.lstrip('validated_')

                if label_remove != label_remove.lstrip('SARSAQ-LEARNINGWEIGHTEDAVERAGEWOLFHUMANn'):
                    multi = True
                    old_remove = old_remove.lstrip(strip_list[1])
                    old_remove = old_remove.lstrip('_')
                    old_remove = old_remove.lstrip(strip_list[2])
                    old_remove = old_remove.lstrip('_')
                    second_remove = old_remove.lstrip(strip_list[3])
                    second_remove = second_remove.lstrip('_')
                    state_label2 = helpers.remove_suffix(old_remove, '_' + second_remove)
                    labels.append(state_label2)
                    old_remove = second_remove
                    old_remove = old_remove.lstrip(strip_list[4])
                    old_remove = old_remove.lstrip('_')
                    old_remove = old_remove.lstrip(strip_list[5])
                    old_remove = old_remove.lstrip('_')

                old_remove = old_remove.lstrip('validated_')
                while old_remove != 'results.hdf5':
                    label_remove = old_remove.lstrip(strip_key)
                    label_remove = label_remove.lstrip('_')
                    isolated_label = helpers.remove_suffix(old_remove, '_' + label_remove)
                    labels.append(isolated_label)
                    old_remove = label_remove
            else:
                label_remove = old_remove.lstrip(strip_key)
                label_remove = label_remove.lstrip('_')
                isolated_label = helpers.remove_suffix(old_remove, '_' + label_remove)
                labels.append(isolated_label)
                old_remove = label_remove

        plot_title = ''
        stripped_label = ''
        if Scenario:
            stripped_label += labels[0] + ' ' + labels[1] + ' '
        # else:
        # plot_title += labels[0]+'_'+labels[1]+'_'

        if method:
            stripped_label += labels[2] + ' '
        # else:
        # plot_title += labels[2]+'_'

        if State:
            stripped_label += labels[4] + ' '
            if multi:
                stripped_label += '& ' + labels[7]
        # else:
        # plot_title += labels[4]+'_'
        # if multi:
        # plot_title += labels[7]+'_'

        if multi:
            hyper_list = labels[8:]
        else:
            hyper_list = labels[7:]

        # plot_title += labels[5] + '_' + labels[6] + '_'

        hyper_param_list = ['ε = ', 'α = ', 'γ = ', 'w = ']

        if hyper_list != []:
            for idx, param in enumerate(hyper_list):
                if Hyper:
                    stripped_label += hyper_param_list[idx] + param + ' '
            # else:
            # plot_title += param+'_'

    # plot_title = helpers.remove_suffix(plot_title,'_')

    # if stripped_label == 'pedestrian LOS SARSA ':
    #     plt.plot(plot_array[test_idx, :], color='k', label=stripped_label, marker='d', linestyle='dashed')
    # else:
    #     plt.plot(plot_array[test_idx, :], label=stripped_label, marker='d')

    if Legend_list[test_idx] == 'Reference':
        plt.plot(plot_array[test_idx, :], color='k', label=Legend_list[test_idx], marker='d', linestyle='dashed')
    else:
        plt.plot(plot_array[test_idx, :], label=Legend_list[test_idx], marker='d')

    print(test)
    print(stripped_label + f': mean {np.round(np.mean(Misalignment[9000:10000]), 1)} '
                           f'median {np.round(np.median(Misalignment[9000:10000]), 1)} '
                           f'1st quad {np.round(np.percentile(Misalignment[9000:10000], 25), 1)} '
                           f'3rd quad {np.round(np.percentile(Misalignment[9000:10000], 75), 1)}')
    print('')

# lgd = plt.legend(bbox_to_anchor=(0.5, -0.25), loc="upper center")
lgd = plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")

plt.xticks(range(len(span)), span, rotation=20)
plt.xlabel("Episode range [-,-]")
plt.ylabel("Average absolute misalignment [dB]")
plt.grid(True, axis='x')
# plt.title(plot_title)

if Save:
    plt.savefig("Figures/Performance_.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')  # Saves the figure

plt.show()

bulk_data.close()
