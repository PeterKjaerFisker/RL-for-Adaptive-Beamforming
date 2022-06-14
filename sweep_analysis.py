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
bulk_data = helpers.bulk_loader('Results/Flat_codebook/')
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
# span = [(0,100), (100,200), (200,300), (300,400), (400,500),
        # (500,600), (600,700), (700,800), (800,900), (900,1000),
        # (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000),
        # (5000, 6000), (6000, 7000), (7000, 8000), (8000, 9000), (9000, 10000)]
# xvals = np.array([0,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000])
# span = [(x*200,(x+1)*200) for x in range(10)]
Save = True

generate_label = False
method = True
Scenario = False
State = False
Hyper = False

average_variance = False

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
# Legend_list=['LOS Heuristic','LOS Sarsa','NLOS Heuristic','NLOS Sarsa']
# Legend_list = ['Heuristic: No noise','Heuristic: Noise factor','Heuristic: Thermal noise','Sarsa: No noise','Sarsa: Noise factor','Sarsa: thermal noise']
# Legend_list = ['Heuristic noise factor','Heuristic Thermal noise','Sarsa noise factor multi','Sarsa noise factor single','Sarsa thermal noise multi','Sarsa thermal noise single']
# Legend_list = ['Tuned multi agent','Tuned single agent']
# Legend_list = ['α = 0.01','α = 0.005','α = 0.05']
# Legend_list = ['γ = 0.6','γ = 0.7','γ = 0.8']
Legend_list = ['LOS Car','NLOS Car','LOS Pedestrian','NLOS Pedestrian']
# Legend_list = ['1','2']
# Legend_list = ['Multi: Training period','Multi: Validation period','Single: Training period','Single: Validation period']
# Legend_list = ['w = 0.3 - Training','w = 0.3 - Validation','w = 1 - Training','w = 1 - Validation','w = 10 - Training','w = 10 - Validation','w = 15 - Training','w = 15 - Validation']
# Legend_list = ['Adaptive: w = 0.3','Adaptive: w = 1','Adaptive: w = 10','Adaptive: w = 15']
# ['ε = ','α = ','γ = ','w = ']
color_list = ['tab:blue','tab:orange','tab:green','tab:red']

dim1 = len(bulk_data)
dim2 = len(span)

plot_array = np.zeros((dim1, dim2))
plot_array_val = np.zeros((dim1, dim2)) # Kommenter ud igen

plt.figure()
for test_idx, test in enumerate(nt.natsorted(bulk_data.keys())):

    Val_stat = len(bulk_data[list(bulk_data.keys())[test_idx]])

    if Val_stat == 2:
        # R_log_db_val = 10 * np.log10(bulk_data[f'{test}/Validation/R_log']) # Kommenter ud igen
        # R_max_log_db_val = 10 * np.log10(bulk_data[f'{test}/Validation/R_max']) # Kommenter ud igen
        R_log_db = 10 * np.log10(bulk_data[f'{test}/Training/R_log'])
        R_max_log_db = 10 * np.log10(bulk_data[f'{test}/Training/R_max'])

        # R_mean_log_db = 10 * np.log10(bulk_data[f'{test}/Validation/R_mean'])

    elif Val_stat > 2:
        # CHANGE IF TRAINING DATA SHOULD BE PLOTTED

        
        R_log_db = 10 * np.log10(bulk_data[f'{test}/R_log'])
        R_max_log_db = 10 * np.log10(bulk_data[f'{test}/R_max'])
        
    elif Val_stat == 1:
        R_log_db = 10 * np.log10(bulk_data[f'{test}/Training/R_log'])
        R_max_log_db = 10 * np.log10(bulk_data[f'{test}/Training/R_max'])

    Misalignment = R_log_db - R_max_log_db
    # Misalignment_val = R_log_db_val - R_max_log_db_val # Kommeneter ud igen
    
    averaged_episodes = np.abs(np.average(Misalignment, axis=1))
    # avaraged_episodes_val = np.abs(np.average(Misalignment_val, axis=1)) # Kommenter ud igen

    for idx, (start, stop) in enumerate(span):
        plot_array[test_idx, idx] = np.average(averaged_episodes[start:stop])
        # plot_array_val[test_idx,idx] = np.average(avaraged_episodes_val[start:stop])# Kommenter ud igen

        # Label construction
        # EVEN NEWER AND MORE IMPROVED LABELS
    strip_list=['NLOS','SARSAQ-LEARNINGWEIGHTEDAVERAGEWOLFHUMANn','TFqwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNMSarsaThermanlNoiseMS','-0123456789','0123456789','0123456789','.0123456789']
    labels = []
    
    multi = False
    # Isolate case
    case_remove = test.lstrip('carpedestrian')
    case_remove = case_remove.lstrip('_')
    case_remove = case_remove.lstrip('urbanhighway')
    case_remove = case_remove.lstrip('_')
    case_label = helpers.remove_suffix(test,'_'+case_remove)
    labels.append(case_label)
    old_remove = case_remove
    label_remove = ''
    
    stripped_label = ''
    if generate_label:
        # Everything else
        for strip_key in strip_list:

            
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
                    state_label2 = helpers.remove_suffix(old_remove,'_'+second_remove)
                    labels.append(state_label2)
                    old_remove = second_remove
                    old_remove = old_remove.lstrip(strip_list[4])
                    old_remove = old_remove.lstrip('_')
                    old_remove = old_remove.lstrip(strip_list[5])
                    old_remove = old_remove.lstrip('_')
                    
                
                old_remove = old_remove.lstrip('validated_')
                while old_remove != '':
                    label_remove = old_remove.lstrip(strip_key)
                    label_remove = label_remove.lstrip('_')
                    isolated_label = helpers.remove_suffix(old_remove,'_'+label_remove)
                    labels.append(isolated_label)
                    old_remove = label_remove
            else:
                label_remove = old_remove.lstrip(strip_key)
                label_remove = label_remove.lstrip('_')
                isolated_label = helpers.remove_suffix(old_remove,'_'+label_remove)
                labels.append(isolated_label)
                old_remove = label_remove
            
        plot_title = ''
        stripped_label = ''
        if Scenario:
            stripped_label += labels[0]+' '+labels[1]+' '
 
        if method:
            stripped_label += labels[2]+' '

        if State:
            stripped_label += labels[4]+' '
            if multi:
                stripped_label += '& '+labels[7]

        if multi:
            hyper_list = labels[8:]
        else:
            hyper_list = labels[7:]
        

        hyper_param_list = ['ε = ','α = ','γ = ','w = ']
        
        if hyper_list != []:
            for idx, param in enumerate(hyper_list):
                if Hyper:
                    stripped_label += hyper_param_list[idx]+ param + ' '

    if not average_variance:
        if stripped_label == 'SARSA ':
            plt.plot(plot_array[test_idx, :], color='k', label=Legend_list[test_idx], marker='d', linestyle='dashed')
        else:
            plt.plot(plot_array[test_idx, :], label=Legend_list[test_idx], marker='d')
            # plt.plot(plot_array[test_idx, :], label=Legend_list[test_idx*2], marker='d', color = color_list[test_idx]) # Pølse plots kode
            # plt.plot(plot_array_val[test_idx, :], label=Legend_list[test_idx*2+1], marker='d', color = color_list[test_idx], linestyle='dashed') # Spaghetti plots kode

    print(test)
    print(stripped_label+f': mean {np.mean(Misalignment[9000:10000])} median {np.median(Misalignment[9000:10000])} 1st quad {np.percentile(Misalignment[9000:10000],25)} 3rd quad {np.percentile(Misalignment[9000:10000],75)}')
    print('')
    # print(stripped_label+f': mean {np.mean(Misalignment_val[9000:10000])} median {np.median(Misalignment_val[9000:10000])} 1st quad {np.percentile(Misalignment_val[9000:10000],25)} 3rd quad {np.percentile(Misalignment_val[9000:10000],75)}')
    # print('')
if average_variance:
    average_plot = np.mean(plot_array, axis = 0)
    # plt.plot(average_plot, marker='d')
    variances = np.var(plot_array, axis = 0)
    
    plt.errorbar(np.arange(len(average_plot)) ,average_plot, variances, marker = 'o',
                 color = 'tab:blue',capsize = 3, ecolor = 'tab:red',elinewidth = 1, capthick = 1)
    

# lgd = plt.legend(bbox_to_anchor=(0.5, -0.25), loc="upper center")
if not average_variance:
    lgd = plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
# lgd = plt.legend()

plt.xticks(range(len(span)), span, rotation=20)
plt.xlabel("Episode range [-,-]")
plt.ylabel("Average absolute misalignment [dB]")
plt.grid(True, axis='x')
# plt.title(plot_title)

if Save:
    plt.savefig("Figures/Performance_.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight') # Saves the figure

plt.show()

bulk_data.close()
