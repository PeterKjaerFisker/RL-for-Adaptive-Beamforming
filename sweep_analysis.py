# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:36:31 2022

@author: dksan
"""
# %% Imports
import numpy as np
import sys

import helpers
import plots
import h5py

# %% Load results

bulk_data = helpers.bulk_loader('Results/Sweep/')

# %% Data analysis

span = [(0,100),(2450,2550),(4950,5050),(7450,7550),(9900,10000)]

dim1 = len(bulk_data)
dim2 = len(span)

plot_array = np.zeros((dim1,dim2))

for test in range(dim1):
    averaged_episodes = np.average(bulk_data[f'{test}/R_log'], axis = 1)
    for idx, (start,stop) in enumerate(span):
        plot_array[test,idx] = np.average(averaged_episodes[start:stop])
        
plots.barplot(False, plot_array, span)

bulk_data.close()