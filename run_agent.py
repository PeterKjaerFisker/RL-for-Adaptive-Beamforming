# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""
# %% Imports
import json
import sys
from time import time

import numpy as np
from tqdm import tqdm

import classes
import helpers

cmd_input = sys.argv
if len(cmd_input) > 1:
    CHANNEL_SETTINGS = sys.argv[1]
    AGENT_SETTINGS = sys.argv[2]
    DATA_NAME = sys.argv[1]
else:
    CHANNEL_SETTINGS = "car_urban_LOS_16_users_10000_steps"
    AGENT_SETTINGS = "sarsa_TFFF_2-3-8-6-8_5000_5_Kopi"
    DATA_NAME = "car_urban_LOS_sarsa_TFFF_2-3-8-6-8_5000_5"

# %% Load pickle
data = helpers.load_pickle('Results/', f'{DATA_NAME}_results.pickle')
Agent = data['Agent']

# %% main
if __name__ == "__main__":

    # Load Channel Settings for simulation
    with open(f'Settings/{CHANNEL_SETTINGS}.json', 'r') as fs:
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
    ORI = agent_settings["ORI"]  # Include the User Terminal orientation in the state
    DIST = agent_settings["DIST"]  # Include the distance between User Terminal and Base Station in the state
    LOCATION = agent_settings["LOCATION"]  # Include location of User Terminal in polar coordinates in the state
    n_actions = agent_settings["n_actions"]  # Number of previous actions
    n_ori = agent_settings["n_ori"]  # Number of previous orientations
    ori_res = agent_settings["ori_res"]  # Resolution of the orientation
    dist_res = agent_settings["dist_res"]  # Resolution of the distance
    angle_res = agent_settings["angle_res"]  # Resolution of the angle
    chunksize = agent_settings["chunksize"]  # Number of samples taken out
    Episodes = agent_settings["Episodes"]  # Episodes per chunk
    Nt = agent_settings["transmitter"]["antennea"]  # Transmitter
    Nr = agent_settings["receiver"]["antennea"]  # Receiver
    Nlt = agent_settings["transmitter"]["layers"]  # Transmitter
    Nlr = agent_settings["receiver"]["layers"]  # Receiver
    Nbeam_tot = (2 ** (Nlr + 1)) - 2  # Total number of beams for the receiver

    # TODO virker useless
    # Load Scenario configuration
    with open(f'Cases/{CASE}.json', 'r') as fp:
        case = json.load(fp)

    # ----------- Load the data -----------
    t_start = time()
    # Load the data
    channel_par, pos_log = helpers.load_data(f"data_pos_{FILENAME}.mat", f"data_{FILENAME}")
    print(f"Took: {time() - t_start}", flush=True)

    # Re-affirm that "M" matches data
    M = len(pos_log)

    # ----------- Extract data from Quadriga simulation -----------
    print("Extracting data", flush=True)
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
    action_space = np.arange(Nbeam_tot)

    """
    Based on the settings for the simulation, different components used in the State 
    are calculated as needed. 
    - If the user terminal (UT) orientation is used, the orientation data from Quadriga is discretized and saved.
    - If either the distance between UT and BS or the location of the UT is used
      the distance between the UT and the BS is calculated from UT position data and discretized.
    - If the location of the UT is used, the angle to the UT in relation to the BS is calculated and discretized.  
    """
    if ORI:
        ori_discrete = np.zeros([M, N])
        for m in range(M):
            ori_discrete[m, :] = helpers.discrete_ori(Orientation[m][0][2, :], ori_res)
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

    # Initializing arrays for logs.
    action_log = np.zeros([Episodes, chunksize])
    R_log = np.zeros([Episodes, chunksize])
    R_max_log = np.zeros([Episodes, chunksize])
    R_min_log = np.zeros([Episodes, chunksize])
    R_mean_log = np.zeros([Episodes, chunksize])

    Agent.eps = 0

    for episode in tqdm(range(Episodes), desc="Episodes"):
        """
        For each episode we first initialize a random State to begin in. 
        Depending on the settings for the simulation, different elements are added to the State. 
        - If the distance or the location is used, the first elements from the discretized data are 
          added to the State, as random initial values. 
        - If the orientation is used, a list of random orientations are drawn and added to the State. 
          The amount of orientations drawn depends on the settings.  

        We then choose a track from the data set at random and similarly choose a random starting
        point in the chosen track. 
        The Environment is then updated with the relevant AoA, AoD and channel parameter data.  
        We then go through each position in the track, takes an action, gets a reward and updates the Q-table.
        """
        # Create the Agent
        # Agent = classes.Agent(action_space, eps=0.1, alpha=["constant", 0.7])

        # Choose data
        path_idx = np.random.randint(0, M)
        data_idx = np.random.randint(0, N - chunksize) if (N - chunksize) else 0

        # Update the environment data
        Env.update_data(AoA_Local[path_idx][data_idx:data_idx + chunksize],
                        AoD_Global[path_idx][0][data_idx:data_idx + chunksize],
                        coeff[path_idx][0][data_idx:data_idx + chunksize])

        # Initiate the State at a random beam sequence
        # TODO dette skal ikke blot være beams men én beam og et antal tidligere "retninger"
        State_tmp = [list(np.random.randint(0, Nbeam_tot, n_actions))]

        if DIST or LOCATION:
            State_tmp.append(list([dist_discrete[0][0]]))
        else:
            State_tmp.append(["N/A"])

        if ORI:
            State_tmp.append(list(np.random.randint(0, ori_res, n_ori)))
        else:
            State_tmp.append(["N/A"])

        if LOCATION:
            State_tmp.append(list([angle_discrete[0][0]]))
        else:
            State_tmp.append(["N/A"])

        State = classes.State(State_tmp, ORI, DIST, LOCATION)

        # Initiate the action
        action = np.random.choice(action_space) # TODO ændre action, i ADJ til at være retningen man går og ikke beam nr.
        retning = np.random.randint(0, 3) - 1   # TODO måske tilføj en seperat værdi der er beam nr. der ikke nødvendigivis er den del af state

        # TODO første action skal afhænge af initial state

        end = False
        # Run the episode
        for n in range(chunksize):

            # Update the current state
            # Check if the user terminal orientation is part of the state.
            if ORI:
                # Get the current discrete orientation of the user terminal
                ori = int(ori_discrete[path_idx, data_idx + n])
                # Check if current step is the last step in episode.
                # If not the last step, the next orientation of the user terminal is assigned to a variable
                if n < chunksize - 1:
                    next_ori = int(ori_discrete[path_idx, data_idx + n + 1])
            else:
                ori = "N/A"
                next_ori = "N/A"

            if DIST or LOCATION:
                dist = dist_discrete[path_idx, data_idx + n]
                if n < chunksize - 1:
                    next_dist = dist_discrete[path_idx, data_idx + n + 1]
            else:
                dist = "N/A"
                next_dist = "N/A"

            if LOCATION:
                angle = angle_discrete[path_idx, data_idx + n]
                if n < chunksize - 1:
                    next_angle = angle_discrete[path_idx, data_idx + n + 1]
            else:
                angle = "N/A"
                next_angle = "N/A"

            if n == chunksize - 1:
                end = True

            current_state_parameters = [dist, ori, angle]
            next_state_parameters = [next_dist, next_ori, next_angle]

            # Calculate the action
            if ADJ:
                State.state = State.build_state(action, current_state_parameters, retning)
                action, retning = Agent.e_greedy_adj(helpers.state_to_index(State.state), action, Nlr) # TODO måske ændre sidste output til "limiting factors"
                action_index = retning
            else:
                State.state = State.build_state(action, current_state_parameters)
                action = Agent.e_greedy(helpers.state_to_index(State.state))
                action_index = action

            # Get reward from performing action
            R, R_max, R_min, R_mean = Env.take_action(n, action)

            action_log[episode, n] = action_index
            R_log[episode, n] = R
            R_max_log[episode, n] = R_max
            R_min_log[episode, n] = R_min
            R_mean_log[episode, n] = R_mean

    # %% Save pickle
    data = {
        'Agent': Agent,
        'R_log': R_log,
        'R_max': R_max_log,
        'R_min': R_min_log,
        'R_mean': R_mean_log,
        'agent_settings': agent_settings,
        'channel_settings': channel_settings,
    }

    try:
        if "NLOS" in channel_settings["scenarios"][0]:
            helpers.dump_pickle(data, 'Results/', f'{CASE}_NLOS_{RESULT_NAME}_policy_evaluation_results.pickle')
        else:
            helpers.dump_pickle(data, 'Results/', f'{CASE}_LOS_{RESULT_NAME}_policy_evaluation_results.pickle')
    except OSError as e:
        print(e)
        print("Saving to root folder instead")
        if "NLOS" in channel_settings["scenarios"][0]:
            helpers.dump_pickle(data, '', f'{CASE}_NLOS_{RESULT_NAME}_policy_evaluation_results.pickle')
        else:
            helpers.dump_pickle(data, '', f'{CASE}_LOS_{RESULT_NAME}_policy_evaluation_results.pickle')

