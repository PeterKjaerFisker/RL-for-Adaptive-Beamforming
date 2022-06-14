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
bulk_data = helpers.bulk_loader('Results/Simple_scenario/')
action_list_r=[]
action_list_t=[]
beam_list_r=[]
beam_list_t=[]


for test_idx, test in enumerate(nt.natsorted(bulk_data.keys())):
    print(test)
    Val_stat = len(bulk_data[list(bulk_data.keys())[test_idx]])

    if Val_stat == 2:
        action_r = bulk_data[f'{test}/Training/action_log_r'][9000:10000]
        action_t = bulk_data[f'{test}/Training/action_log_t'][9000:10000]
        # R_log_db = 10 * np.log10(bulk_data[f'{test}/Training/R_log'])
        # R_max_log_db = 10 * np.log10(bulk_data[f'{test}/Training/R_max'])
        beam_r = bulk_data[f'{test}/Training/beam_log_r'][9000:10000]
        beam_t = bulk_data[f'{test}/Training/beam_log_t'][9000:10000]

    elif Val_stat > 2:
        # CHANGE IF TRAINING DATA SHOULD BE PLOTTED

        
        action_r = bulk_data[f'{test}/action_log_r'][9000:10000]
        action_t = bulk_data[f'{test}/action_log_t'][9000:10000]
        
        beam_r = bulk_data[f'{test}/beam_log_r'][9000:10000]
        beam_t = bulk_data[f'{test}/beam_log_t'][9000:10000]
        
    elif Val_stat == 1:
        action_r = bulk_data[f'{test}/Training/action_log_r'][9000:10000]
        action_t = bulk_data[f'{test}/Training/action_log_t'][9000:10000]
        
        beam_r = bulk_data[f'{test}/Training/beam_log_r'][9000:10000]
        beam_t = bulk_data[f'{test}/Training/beam_log_t'][9000:10000]
        
    
    action_r_data = plt.hist(action_r.flatten(),6,label=f'{test_idx}')
    action_list_r.append(action_r_data[0]/np.sum(action_r_data[0]))
    action_t_data = plt.hist(action_t.flatten(),6,label=f'{test_idx}')
    action_list_t.append(action_t_data[0]/np.sum(action_t_data[0]))
    
    beam_r_data = plt.hist(beam_r.flatten(),14,label=f'{test_idx}')
    beam_list_r.append(beam_r_data[0]/np.sum(beam_r_data[0]))
    beam_t_data = plt.hist(beam_t.flatten(),30,label=f'{test_idx}')
    beam_list_t.append(beam_t_data[0]/np.sum(beam_t_data[0]))
    # action_list_t.append(action_t.flatten())
    
plt.show()
# label_input = ['Heuristic','Q-learning','Sarsa','Simple']
# label_input = ['Q-learning','Sarsa','Wolf']
label_input = ['LOS Heuristic','LOS Sarsa','NLOS Heuristic','NLOS Sarsa']

plots.barplot(False, action_list_r, ['Stay','Right','Left','Down','Up Right','Up Left'], labels=label_input)
plots.barplot(False, action_list_t, ['Stay','Right','Left','Down','Up Right','Up Left'], labels=label_input)
x_labels = ['' if x not in [0,2,6,14] else x for x in range(14)]
plots.barplot(False, beam_list_r, x_labels, labels=label_input)
x_labels = ['' if x not in [0,2,6,14] else x for x in range(30)]
plots.barplot(False, beam_list_t, x_labels, labels=label_input)
# plots.barplot(False, action_list_r, ['1','2','3','4'])



bulk_data.close()