# -*- coding: utf-8 -*-
"""
@authors: Dennis Sand & Peter Fisker
"""
# %% Imports
import json
import sys
from time import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import classes
import helpers
import agent_classes

cmd_input = sys.argv
if len(cmd_input) > 1:
    CHANNEL_SETTINGS = sys.argv[1]
    AGENT_SETTINGS = sys.argv[2]
else:
    CHANNEL_SETTINGS = "pedestrian_LOS_2_users_20000_steps_01"
    AGENT_SETTINGS = "SARSA_FFFF_0-0-0-0-0-0_7000_2000"

# %% main
if __name__ == "__main__":
    VALIDATION_SETTINGS = "pedestrian_LOS_10_users_20000_steps_validation"
    # Load Channel Settings for simulation
    with open(f'Settings/Channel_settings/{CHANNEL_SETTINGS}.json', 'r') as fs:
        channel_settings = json.load(fs)

    # Load Agent Settings for simulation
    with open(f'Settings/{AGENT_SETTINGS}.json', 'r') as fs:
        agent_settings = json.load(fs)

    # Load global parameters
    RESULT_NAME = agent_settings["RESULT_NAME"]
    FILENAME = channel_settings["FILENAME"]  # Name of the data file to be loaded or saved
    CASE = channel_settings["CASE"]  # "car_highway", "pedestrian" or "car"

    # ----------- Channel Simulation Parameters -----------
    N = channel_settings["N"]  # Number of steps in an episode
    M = channel_settings["M"]  # Number of episodes
    r_lim = channel_settings["rlim"]  # Radius of the cell
    fc = channel_settings["fc"]  # Center frequency
    P_t = channel_settings["P_t"]  # Transmission power
    lambda_ = 3e8 / fc  # Wave length

    # ----------- Reinforcement Learning Parameters -----------

    METHOD = agent_settings["METHOD"]  # RL table update, "simple", "SARSA" or "Q-LEARNING"
    ADJ = agent_settings["ADJ"]  # Whether action space should be all beams ("False") or adjacent ("True")
    chunksize = agent_settings["chunksize"]  # Number of samples taken out
    Episodes = agent_settings["Episodes"]  # Episodes per chunk
    Nt = agent_settings["transmitter"]["antennea"]  # Transmitter
    Nr = agent_settings["receiver"]["antennea"]  # Receiver
    Nlt = agent_settings["transmitter"]["layers"]  # Transmitter
    Nlr = agent_settings["receiver"]["layers"]  # Receiver
    Nbeam_tot_r = (2 ** (Nlr + 1)) - 2  # Total number of beams for the receiver
    Nbeam_tot_t = (2 ** (Nlt + 1)) - 2  # Total number of beams for the transmitter

    # ----------- Load the data -----------
    t_start = time()
    # Load the training data
    channel_par, pos_log = helpers.load_data(f"data_pos_{FILENAME}.mat",
                                             f"data_{FILENAME}")

    print(f"Took: {time() - t_start}", flush=True)

    # Re-affirm that "M" matches data
    M = len(pos_log)

    # ----------- Extract data from Quadriga simulation -----------
    print("Extracting data", flush=True)

    # ----- Training data -------
    AoA_Global = channel_par[0][0]  # Angle of Arrival in Global coord. system
    for i in range(len(AoA_Global)):
        AoA_Global[i][0] = np.squeeze(AoA_Global[i][0])

    AoD_Global = channel_par[1][0]  # Angle of Departure in Global coord. system
    for i in range(len(AoD_Global)):
        AoD_Global[i][0] = np.squeeze(AoD_Global[i][0])

    coeff = channel_par[2][0]  # Channel Coefficients
    for i in range(len(coeff)):
        coeff[i][0] = np.squeeze(coeff[i][0])

    Orientation = channel_par[3][0]  # Orientation in Global coord. system

    if CASE == 'pedestrian':
        # Add some random noise to the orientation to simulate a moving person
        Orientation = helpers.noisy_ori(Orientation)

    # ----------- Prepare the simulation - Channel -----------
    print("Starts calculating", flush=True)

    # Preallocate empty arrays
    AoA_Local = []

    # Calculate hierarchical-codebook - Transmitter
    precoder_codebook = helpers.codebook(Nt, Nlt, lambda_)

    # Calculate hierarchical-codebook - Receiver
    combiner_codebook = helpers.codebook(Nr, Nlr, lambda_)

    # Calculate the AoA in the local coordinate system of the user terminal
    for m in range(M):
        AoA_Local.append(helpers.get_local_angle(AoA_Global[m][0], Orientation[m][0][2, :]))

    # ----------- Prepare the simulation - RL -----------
    # Create the Environment
    Env = classes.Environment(combiner_codebook, precoder_codebook, Nt, Nr,
                              fc, P_t)

    # Initialize all possible actions for the Agent
    action_space_r = np.arange(Nbeam_tot_r)
    action_space_t = np.arange(Nbeam_tot_t)

    # ----------- Starts the simulation -----------

    # Initializing arrays for logs.
    action_log_r = np.zeros([Episodes, chunksize])
    action_log_t = np.zeros([Episodes, chunksize])
    beam_log_r = np.zeros([Episodes, chunksize])
    beam_log_t = np.zeros([Episodes, chunksize])
    R_log = np.zeros([Episodes, chunksize])
    R_max_log = np.zeros([Episodes, chunksize])
    R_min_log = np.zeros([Episodes, chunksize])
    R_mean_log = np.zeros([Episodes, chunksize])
    TD_log = np.zeros([Episodes, chunksize])

    EPSILON_METHOD = "constant"
    Agent = agent_classes.Agent(action_space_r, action_space_t, eps=[f'{EPSILON_METHOD}', 1], alpha=0.05)

    print('Rewards for training are now calculated')
    reward_start = time()

    Env.update_data(AoA_Local, AoD_Global, coeff)
    Env.create_reward_matrix()

    print(f'Rewards tog {time() - reward_start} sekunder at regne')

    data = defaultdict(lambda: 0)

    # Threshold = 0.0006  # This works very well
    Threshold = 0.0005  # This works worse than 0.0006
    Thresholds = np.arange(15, 25) * 10 ** -5
    thrs_idx = 0
    for episode in tqdm(range(Episodes), desc="Episodes"):

        if episode == 0:
            pass
        else:
            if episode % 200 == 0:
                data[Thresholds[thrs_idx]] = 10 * np.log10(
                    R_log[episode - 200:episode] / R_max_log[episode - 200:episode])
                if thrs_idx < len(Thresholds) - 1:
                    thrs_idx += 1
                else:
                    pass

        # Choose data
        path_idx = np.random.randint(0, M)
        data_idx = np.random.randint(0, N - chunksize) if (N - chunksize) else 0

        # Initiate the action
        previous_beam_nr_r = np.random.choice(action_space_r)
        previous_beam_nr_t = np.random.choice(action_space_t)
        previous_beam_nr = tuple([previous_beam_nr_r, previous_beam_nr_t])

        end = False
        STAY = False
        # Run the episode
        for n in range(chunksize):
            if n == chunksize - 1:
                end = True

            # Calculate the action
            if not STAY:
                beam_nr, action = Agent.e_greedy_adj('NO', previous_beam_nr, Nlr, Nlt)
                # beam_nr_r = np.random.choice(action_space_r)
                # beam_nr_t = np.random.choice(action_space_t)
                # beam_nr = tuple([previous_beam_nr_r, previous_beam_nr_t])
                # action = tuple([0, 0])
            else:
                beam_nr = previous_beam_nr
                action = tuple([0, 0])

            # Get reward from performing action
            R, R_noiseless, R_max, R_min, R_mean = Env.take_action(path_idx, n + data_idx, beam_nr[0], beam_nr[1])

            if R < Thresholds[thrs_idx]:
                STAY = False
            else:
                STAY = True

            action_log_r[episode, n] = action[0]
            action_log_t[episode, n] = action[1]
            beam_log_r[episode, n] = beam_nr[0]
            beam_log_t[episode, n] = beam_nr[1]
            R_log[episode, n] = R_noiseless
            R_max_log[episode, n] = R_max
            R_min_log[episode, n] = R_min
            R_mean_log[episode, n] = R_mean

            previous_beam_nr = beam_nr

    data[Thresholds[thrs_idx]] = 10 * np.log10(R_log[episode - 200:episode] / R_max_log[episode - 200:episode])

    # %%
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as plticker

    fig, ax = plt.subplots()

    for key, value in data.items():
        sns.ecdfplot(value.flatten(), label=f'Threshold = {round(key, 7)}, avg = {round(np.mean(value), 3)}')

    plt.axvline(-6, linestyle='--', color='black', label='-6 dB')
    plt.axvline(-3, linestyle='-.', color='black', label='-3 dB')
    plt.title('E-CDF, Heuristic')
    loc = plticker.MultipleLocator(base=0.05)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.yaxis.tick_right()
    plt.xlabel('Misalignment in dB')
    plt.legend()
    plt.show()

    # %% Save pickle and hdf5
    data_reward = {
        'Training': {
            'R_log': R_log,
            'R_max': R_max_log,
            'R_min': R_min_log,
            'R_mean': R_mean_log,
            'action_log_r': action_log_r,
            'action_log_t': action_log_t,
            'beam_log_r': beam_log_r,
            'beam_log_t': beam_log_t
        }
    }

    try:
        if "NLOS" in channel_settings["scenarios"][0]:
            helpers.dump_hdf5_validate(data_reward, 'Results/',
                                       f'{CASE}_NLOS_{RESULT_NAME}_validated_Human_results.hdf5')
            # helpers.dump_pickle(data_agent, 'Results/', f'{CASE}_NLOS_{RESULT_NAME}_results.pickle')
        else:
            helpers.dump_hdf5_validate(data_reward, 'Results/',
                                       f'{CASE}_LOS_{RESULT_NAME}_validated_Human_results.hdf5')
            # helpers.dump_pickle(data_agent, 'Results/', f'{CASE}_LOS_{RESULT_NAME}_results.pickle')
    except OSError as e:
        print(e)
        print("Saving to root folder instead")
        if "NLOS" in channel_settings["scenarios"][0]:
            helpers.dump_hdf5_validate(data_reward, '',
                                       f'{CASE}_NLOS_{RESULT_NAME}_validated_Human_results.hdf5')
            # helpers.dump_pickle(data_agent, '', f'{CASE}_NLOS_{RESULT_NAME}_results.pickle')
        else:
            helpers.dump_hdf5_validate(data_reward, '',
                                       f'{CASE}_LOS_{RESULT_NAME}_validated_Human_results.hdf5')
            # helpers.dump_pickle(data_agent, '', f'{CASE}_LOS_{RESULT_NAME}_results.pickle')
