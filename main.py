# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""
# %% Imports
import json
import sys
from time import time
import dill as pickle

import numpy as np
from tqdm import tqdm

import classes
import helpers
import plots

cmd_input = sys.argv
if len(cmd_input) > 1:
    SETTING = sys.argv[1]
else:
    SETTING = "Thing_01"

# %% main
if __name__ == "__main__":

    # Load Settings for simulation    
    with open(f'Settings/{SETTING}.json', 'r') as fs:
        setting = json.load(fs)

    # Load global parameters
    RUN = setting["RUN"]  # Whether new data should be generated regardless of if it already exist
    ENGINE = setting["ENGINE"]  # Engine for QUADRIGA simulation, "MATLAB" or "octave"
    FILENAME = setting["FILENAME"]  # Name of the data file to be loaded or saved
    CASE = setting["CASE"]  # "car_highway", "pedestrian" or "car"

    # ----------- Channel Simulation Parameters -----------
    scenarios = setting["scenarios"]  # Quadriga scenarios, page 101 of Quadriga documentation
    N = setting["N"]  # Number of steps in an episode
    sample_period = setting["sample_period"]  # The sample period in [s]
    M = setting["M"]  # Number of episodes
    r_lim = setting["rlim"]  # Radius of the cell
    fc = setting["fc"]  # Center frequency
    P_t = setting["P_t"]  # Transmission power
    lambda_ = 3e8 / fc  # Wave length

    # ----------- Reinforcement Learning Parameters -----------
    METHOD = setting["METHOD"]  # RL table update, "simple", "SARSA" or "Q-LEARNING"
    ADJ = setting["ADJ"]  # Whether action space should be all beams ("False") or adjacent ("True")
    ORI = setting["ORI"]  # Include the orientation in the state
    DIST = setting["DIST"]  # Include the dist in the state
    LOCATION = setting["LOCATION"]  # Include location in polar coordinates in the state
    n_actions = setting["n_actions"]  # Number of previous actions
    n_ori = setting["n_ori"]  # Number of previous orientations
    dist_res = setting["dist_res"]  # Resolution of the distance
    angle_res = setting["angle_res"]  # Resolution of the angle
    chunksize = setting["chunksize"]  # Number of samples taken out
    Episodes = setting["Episodes"]  # Episodes per chunk
    Nt = setting["transmitter"]["antennea"]  # Transmitter
    Nr = setting["receiver"]["antennea"]  # Receiver
    Nbt = setting["transmitter"]["beams"]  # Transmitter
    Nbr = setting["receiver"]["beams"]  # Receiver

    # Load Scenario configuration
    with open(f'Cases/{CASE}.json', 'r') as fp:
        case = json.load(fp)

    # ----------- Create the data -----------
    t_start = time()
    # Load or create the data
    channel_par, pos_log = helpers.get_data(RUN, ENGINE, case,
                                            f"data_pos_{FILENAME}.mat", f"data_{FILENAME}",
                                            [fc, N, M, r_lim, sample_period, scenarios])
    print(f"Took: {time() - t_start}", flush=True)

    # Re-affirm that "M" matches data
    M = len(pos_log)

    # ----------- Extract data from Quadriga simulation -----------
    print("Extracting data", flush=True)
    AoA_Global = channel_par[0][0]  # Angle of Arrival in Global coord. system
    AoD_Global = channel_par[1][0]  # Angle of Departure in Global coord. system
    coeff = channel_par[2][0]  # Channel Coefficients
    Orientation = channel_par[3][0]  # Orientation in Global coord. system

    if CASE == 'pedestrian':
        # Add some random noise to the orientation to simulate a moving person
        Orientation = helpers.noisy_ori(Orientation)

    # ----------- Prepare the simulation - Channel -----------
    print("Starts calculating", flush=True)
    # Make ULA antenna positions - Transmitter
    r_r = np.zeros((2, Nr))
    r_r[0, :] = np.linspace(0, (Nr - 1) * lambda_ / 2, Nr)

    # Make ULA antenna positions - Receiver
    r_t = np.zeros((2, Nt))
    r_t[0, :] = np.linspace(0, (Nt - 1) * lambda_ / 2, Nt)

    # Preallocate empty arrays
    beam_t = np.zeros((M, N))
    beam_r = np.zeros((M, N))
    AoA_Local = []

    # Calculate DFT-codebook - Transmitter
    precoder_codebook = helpers.codebook_old(Nbt, Nt)

    # Calculate DFT-codebook - Receiver
    combiner_codebook = helpers.codebook_old(Nbr, Nr)

    # Calculate the AoA in the local coordinate system
    for m in range(M):
        AoA_Local.append(helpers.get_local_angle(AoA_Global[m][0], Orientation[m][0][2, :]))

    # ----------- Prepare the simulation - RL -----------
    # Create the Environment
    Env = classes.Environment(combiner_codebook, precoder_codebook, Nt, Nr,
                              r_r, r_t, fc, P_t)

    # Create action space
    action_space = np.arange(Nbr)

    # Create the discrete orientation if ORI is true
    if ORI:
        ori_discrete = np.zeros([M, N])
        for m in range(M):
            ori_discrete[m, :] = helpers.discrete_ori(Orientation[m][0][2, :], Nbr)
    else:
        ori_discrete = None

    if DIST or LOCATION:
        dist_discrete = np.zeros([M, N])
        for m in range(M):
            dist_discrete[m, :] = helpers.discrete_dist(pos_log[m], dist_res, r_lim)
    else:
        dist_discrete = None

    if LOCATION:
        angle_discrete = np.zeros([M, N])
        for m in range(M):
            angle_discrete[m, :] = helpers.discrete_angle(pos_log[m], angle_res)
    else:
        angle_discrete = None

    # ----------- Starts the simulation -----------
    action_log = np.zeros([Episodes, chunksize])
    R_log = np.zeros([Episodes, chunksize])
    R_max_log = np.zeros([Episodes, chunksize])
    R_min_log = np.zeros([Episodes, chunksize])
    R_mean_log = np.zeros([Episodes, chunksize])

    for episode in tqdm(range(Episodes), desc="Episodes"):
        # Create the Agent
        Agent = classes.Agent(action_space, eps=0.1, alpha=["constant", 0.7])

        # Initiate the State at a random beam sequence
        State_tmp = [list(np.random.randint(0, Nbr, n_actions))]

        if DIST or LOCATION:
            State_tmp.append(list([dist_discrete[0]]))
        else:
            State_tmp.append(["N/A"])

        if ORI:
            State_tmp.append(list(np.random.randint(0, Nbr, n_ori)))
        else:
            State_tmp.append(["N/A"])

        if LOCATION:
            State_tmp.append(list([angle_discrete[0]]))
        else:
            State_tmp.append(["N/A"])

        State = classes.State(State_tmp)

        # Choose data
        data_idx = np.random.randint(0, N - chunksize) if (N - chunksize) else 0
        path_idx = np.random.randint(0, M)

        # Update the environment data
        Env.update_data(AoA_Local[path_idx][data_idx:data_idx + chunksize],
                        AoD_Global[path_idx][0][data_idx:data_idx + chunksize],
                        coeff[path_idx][0][data_idx:data_idx + chunksize])

        # Initiate the action
        action = np.random.choice(action_space)
        retning = np.random.randint(0,3)-1

        end = False
        # Run the episode
        for n in range(chunksize):
            if ORI:
                ori = int(ori_discrete[path_idx, data_idx + n])
                if n < chunksize - 1:
                    next_ori = int(ori_discrete[path_idx, data_idx + n + 1])
            else:
                ori = None
                next_ori = None

            if DIST or LOCATION:
                dist = dist_discrete[path_idx, data_idx + n]
                if n < chunksize - 1:
                    next_dist = dist_discrete[path_idx, data_idx + n + 1]
            else:
                dist = None
                next_dist = None

            if LOCATION:
                angle = angle_discrete[path_idx, data_idx + n]
                if n < chunksize - 1:
                    next_angle = angle_discrete[path_idx, data_idx + n + 1]
            else:
                angle = None
                next_angle = None

            if n == chunksize - 1:
                end = True

            para = [dist, ori, angle]
            para_action = [next_dist, ori, angle]
            para_next = [next_dist, next_ori, next_angle]

            if ADJ:
                State.update_state(action, para=para, retning=retning)
                action, retning = Agent.e_greedy_adj(State.get_state(para=para), action)
            else:
                State.update_state(action, para=para)
                action = Agent.e_greedy(State.get_state(para=para))

            R, R_max, R_min, R_mean = Env.take_action(n, action)

            if METHOD == "simple":
                Agent.update(State, action, R, para=para)
            elif METHOD == "SARSA":
                if ADJ:
                    next_action = Agent.e_greedy_adj(State.get_nextstate(action,
                                                                         para_next=para_action), action)
                else:
                    next_action = Agent.e_greedy(State.get_nextstate(action,
                                                                     para_next=para_action))
                Agent.update_sarsa(R, State, action,
                                   next_action,
                                   para_next=para_next, end=end)
            else:
                Agent.update_Q_learning(R, State, action,
                                        para_next=para_next,
                                        adj=ADJ, end=end)
                METHOD = "Q-LEARNING"

            action_log[episode, n] = action
            R_log[episode, n] = R
            R_max_log[episode, n] = R_max
            R_min_log[episode, n] = R_min
            R_mean_log[episode, n] = R_mean

    # %% PICKLE
    data = {
        'Agent': Agent,
        'R_log': R_log,
        'R_max': R_max_log,
        'R_min': R_min_log,
        'R_mean': R_mean_log,
        'settings': setting
    }

    with open('results.pickle', 'wb+') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    # %% PLOT
    print("Starts plotting")

    # Get the Logs in power decibel
    R_log_db = 10 * np.log10(R_log)
    R_max_log_db = 10 * np.log10(R_max_log)
    R_min_log_db = 10 * np.log10(R_min_log)
    R_mean_log_db = 10 * np.log10(R_mean_log)

    # plots.mean_reward(R_max_log, R_mean_log, R_min_log, R_log,
    #                   ["R_max", "R_mean", "R_min", "R"], "Mean Rewards")

    plots.mean_reward(R_max_log_db, R_mean_log_db, R_min_log_db, R_log_db,
                      ["R_max", "R_mean", "R_min", "R"], "Mean Rewards db",
                      db=True)

    plots.positions(pos_log, r_lim)

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
