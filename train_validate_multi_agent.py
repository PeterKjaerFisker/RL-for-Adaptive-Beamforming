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
import agent_classes

cmd_input = sys.argv
if len(cmd_input) > 1:
    CHANNEL_SETTINGS = sys.argv[1]
    AGENT_SETTINGS_R = sys.argv[2]
    AGENT_SETTINGS_T = sys.argv[3]
    validate_eps = float(sys.argv[4])
    validate_alpha = float(sys.argv[5])
    validate_gamma = float(sys.argv[6])
    validate_weight = float(sys.argv[7])
else:
    CHANNEL_SETTINGS = "pedestrian_LOS_2_users_20000_steps_01"
    AGENT_SETTINGS_R = "SARSA_TTFT_2-0-1-8-2-32_7000_10000"
    AGENT_SETTINGS_T = "SARSA_TFFT_0-2-0-0-2-32_7000_10000"
    validate_eps = 0.01
    validate_alpha = 0.01
    validate_gamma = 0.6
    validate_weight = 30

# %% main
if __name__ == "__main__":
    VALIDATION_SETTINGS = "pedestrian_LOS_10_users_20000_steps_validation"
    # Load Channel Settings for simulation
    with open(f'Settings/Channel_settings/{CHANNEL_SETTINGS}.json', 'r') as fs:
        channel_settings = json.load(fs)

    # Load Validation Settings for simulation
    with open(f'Settings/Validation_settings/{VALIDATION_SETTINGS}.json', 'r') as fs:
        validation_settings = json.load(fs)

    # Load Agent Settings for simulation
    with open(f'Settings/{AGENT_SETTINGS_R}.json', 'r') as fs:
        agent_settings_r = json.load(fs)

    with open(f'Settings/{AGENT_SETTINGS_T}.json', 'r') as fs:
        agent_settings_t = json.load(fs)

    # Load global parameters
    if agent_settings_r['chunksize'] != agent_settings_t['chunksize']:
        raise Exception('Chunksizes mismatch')
    else:
        chunksize = agent_settings_r['chunksize']

    if agent_settings_r['Episodes'] != agent_settings_t['Episodes']:
        raise Exception('Episodes mismatch')
    else:
        Episodes = agent_settings_r['Episodes']

    RESULT_NAME = agent_settings_r["RESULT_NAME"].lstrip(f'_{chunksize}_{Episodes}') + '_' + agent_settings_t[
        "RESULT_NAME"]
    FILENAME = channel_settings["FILENAME"]  # Name of the data file to be loaded or saved
    VALIDATION_FILE = validation_settings['FILENAME']
    CASE = channel_settings["CASE"]  # "car_highway", "pedestrian" or "car"

    # ----------- Channel Simulation Parameters -----------
    N = channel_settings["N"]  # Number of steps in an episode
    M = channel_settings["M"]  # Number of episodes
    r_lim = channel_settings["rlim"]  # Radius of the cell
    fc = channel_settings["fc"]  # Center frequency
    P_t = channel_settings["P_t"]  # Transmission power
    lambda_ = 3e8 / fc  # Wave length
    P_n = 0  # Power of the noise

    # Boltzmann constant
    # k = 1.380649 * 10 ** (-23)
    # # Power of the noise from [SOURCE]. extra 10 added to match P_t
    # P_n_db = 10 * np.log10(k * 290 * 400 * 10 ** 6 / 0.001) + 10
    # P_n_db += 10*np.log10(9/0.001)  # Add additional noise from noise factor (9dB) to see how well it performs.
    # P_n = 10**(P_n_db/10)

    # ----------- Validation Simulation Parameters ------------
    N_validation = validation_settings['N']  # Number of steps in an episode
    M_validation = validation_settings["M"]  # Number of episodes
    r_lim_validation = validation_settings["rlim"]  # Radius of the cell
    fc_validation = validation_settings["fc"]  # Center frequency
    P_t_validation = validation_settings["P_t"]  # Transmission power
    lambda_validation = 3e8 / fc  # Wave length
    Episodes_validation = 10000

    # ----------- Reinforcement Learning Parameters -----------
    METHOD_r = agent_settings_r["METHOD"]  # RL table update, "simple", "SARSA" or "Q-LEARNING"
    ORI_r = agent_settings_r["ORI"]  # Include the User Terminal orientation in the state
    DIST_r = agent_settings_r["DIST"]  # Include the distance between User Terminal and Base Station in the state
    LOCATION_r = agent_settings_r["LOCATION"]  # Include location of User Terminal in polar coordinates in the state
    n_actions_r_r = agent_settings_r["n_actions_r"]  # Number of previous actions
    n_actions_t_r = agent_settings_r["n_actions_t"]  # Number of previous actions
    n_ori_r = agent_settings_r["n_ori"]  # Number of previous orientations
    ori_res_r = agent_settings_r["ori_res"]  # Resolution of the orientation
    dist_res_r = agent_settings_r["dist_res"]  # Resolution of the distance
    angle_res_r = agent_settings_r["angle_res"]  # Resolution of the angle
    Nr = agent_settings_r["receiver"]["antennea"]  # Receiver
    Nlr = agent_settings_r["receiver"]["layers"]  # Receiver

    METHOD_t = agent_settings_t["METHOD"]  # RL table update, "simple", "SARSA" or "Q-LEARNING"
    ORI_t = agent_settings_t["ORI"]  # Include the User Terminal orientation in the state
    DIST_t = agent_settings_t["DIST"]  # Include the distance between User Terminal and Base Station in the state
    LOCATION_t = agent_settings_t["LOCATION"]  # Include location of User Terminal in polar coordinates in the state
    n_actions_r_t = agent_settings_t["n_actions_r"]  # Number of previous actions
    n_actions_t_t = agent_settings_t["n_actions_t"]  # Number of previous actions
    n_ori_t = agent_settings_t["n_ori"]  # Number of previous orientations
    ori_res_t = agent_settings_t["ori_res"]  # Resolution of the orientation
    dist_res_t = agent_settings_t["dist_res"]  # Resolution of the distance
    angle_res_t = agent_settings_t["angle_res"]  # Resolution of the angle
    Nt = agent_settings_t["transmitter"]["antennea"]  # Transmitter
    Nlt = agent_settings_t["transmitter"]["layers"]  # Transmitter

    Nbeam_tot_r = (2 ** (Nlr + 1)) - 2  # Total number of beams for the receiver
    Nbeam_tot_t = (2 ** (Nlt + 1)) - 2  # Total number of beams for the transmitter

    # ----------- Load the data -----------
    t_start = time()
    # Load the data
    channel_par, pos_log = helpers.load_data(f"data_pos_{FILENAME}.mat",
                                             f"data_{FILENAME}")
    # Load the validation data
    channel_par_validation, pos_log_validation = helpers.load_data(f"data_pos_{VALIDATION_FILE}.mat",
                                                                   f"data_{VALIDATION_FILE}")

    print(f"Took: {time() - t_start}", flush=True)

    # Re-affirm that "M" matches data
    M = len(pos_log)
    M_validation = len(pos_log_validation)

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

    # ----- Validation data -------
    AoA_Global_validation = channel_par_validation[0][0]  # Angle of Arrival in Global coord. system
    for i in range(len(AoA_Global_validation)):
        AoA_Global_validation[i][0] = np.squeeze(AoA_Global_validation[i][0])

    AoD_Global_validation = channel_par_validation[1][0]  # Angle of Departure in Global coord. system
    for i in range(len(AoD_Global_validation)):
        AoD_Global_validation[i][0] = np.squeeze(AoD_Global_validation[i][0])

    coeff_validation = channel_par_validation[2][0]  # Channel Coefficients
    for i in range(len(coeff_validation)):
        coeff_validation[i][0] = np.squeeze(coeff_validation[i][0])

    Orientation_validation = channel_par_validation[3][0]  # Orientation in Global coord. system

    if CASE == 'pedestrian':
        # Add some random noise to the orientation to simulate a moving person
        Orientation_validation = helpers.noisy_ori(Orientation_validation)

    # ----------- Prepare the simulation - Channel -----------
    print("Starts calculating", flush=True)

    # Preallocate empty arrays
    AoA_Local = []
    AoA_Local_validation = []

    # Calculate hierarchical-codebook - Transmitter
    precoder_codebook = helpers.codebook(Nt, Nlt, lambda_)

    # Calculate hierarchical-codebook - Receiver
    combiner_codebook = helpers.codebook(Nr, Nlr, lambda_)

    # Calculate the AoA in the local coordinate system of the user terminal
    for m in range(M):
        AoA_Local.append(helpers.get_local_angle(AoA_Global[m][0], Orientation[m][0][2, :]))

    for m in range(M_validation):
        AoA_Local_validation.append(
            helpers.get_local_angle(AoA_Global_validation[m][0], Orientation_validation[m][0][2, :]))

    # ----------- Prepare the simulation - RL -----------
    # Create the Environment
    Env = classes.Environment(combiner_codebook, precoder_codebook, Nt, Nr,
                              fc, P_t)

    # Initialize all possible actions for the Agent
    action_space_r = np.arange(Nbeam_tot_r)
    action_space_t = np.arange(Nbeam_tot_t)

    """
    Based on the settings for the simulation, different components used in the State 
    are calculated as needed. 
    - If the user terminal (UT) orientation is used, the orientation data from Quadriga is discretized and saved.
    - If either the distance between UT and BS or the location of the UT is used
      the distance between the UT and the BS is calculated from UT position data and discretized.
    - If the location of the UT is used, the angle to the UT in relation to the BS is calculated and discretized.  
    """
    if ORI_r:
        ori_discrete_r = np.zeros([M, N])
        for m in range(M):
            ori_discrete_r[m, :] = helpers.discrete_ori(Orientation[m][0][2, :], ori_res_r)

        ori_discrete_validation_r = np.zeros([M_validation, N_validation])
        for m in range(M_validation):
            ori_discrete_validation_r[m, :] = helpers.discrete_ori(Orientation_validation[m][0][2, :], ori_res_r)

    else:
        ori_discrete_r = None
        ori_discrete_validation_r = None

    if ORI_t:
        ori_discrete_t = np.zeros([M, N])
        for m in range(M):
            ori_discrete_t[m, :] = helpers.discrete_ori(Orientation[m][0][2, :], ori_res_t)

        ori_discrete_validation_t = np.zeros([M_validation, N_validation])
        for m in range(M_validation):
            ori_discrete_validation_t[m, :] = helpers.discrete_ori(Orientation_validation[m][0][2, :], ori_res_t)
    else:
        ori_discrete_t = None
        ori_discrete_validation_t = None

    if DIST_r or LOCATION_r:
        dist_discrete_r = np.zeros([M, N])
        for m in range(M):
            dist_discrete_r[m, :] = helpers.discrete_dist(pos_log[m], dist_res_r, r_lim)

        dist_discrete_validation_r = np.zeros([M_validation, N_validation])
        for m in range(M_validation):
            dist_discrete_validation_r[m, :] = helpers.discrete_dist(pos_log_validation[m], dist_res_r, r_lim)
    else:
        dist_discrete_r = None
        dist_discrete_validation_r = None

    if DIST_t or LOCATION_t:
        dist_discrete_t = np.zeros([M, N])
        for m in range(M):
            dist_discrete_t[m, :] = helpers.discrete_dist(pos_log[m], dist_res_t, r_lim)

        dist_discrete_validation_t = np.zeros([M_validation, N_validation])
        for m in range(M_validation):
            dist_discrete_validation_t[m, :] = helpers.discrete_dist(pos_log_validation[m], dist_res_t, r_lim)

    else:
        dist_discrete_t = None
        dist_discrete_validation_t = None

    if LOCATION_r:
        angle_discrete_r = np.zeros([M, N])
        for m in range(M):
            angle_discrete_r[m, :] = helpers.discrete_angle(pos_log[m], angle_res_r)

        angle_discrete_validation_r = np.zeros([M_validation, N_validation])
        for m in range(M_validation):
            angle_discrete_validation_r[m, :] = helpers.discrete_angle(pos_log_validation[m], angle_res_r)

    else:
        angle_discrete_r = None
        angle_discrete_validation_r = None

    if LOCATION_t:
        angle_discrete_t = np.zeros([M, N])
        for m in range(M):
            angle_discrete_t[m, :] = helpers.discrete_angle(pos_log[m], angle_res_t)

        angle_discrete_validation_t = np.zeros([M_validation, N_validation])
        for m in range(M_validation):
            angle_discrete_validation_t[m, :] = helpers.discrete_angle(pos_log_validation[m], angle_res_t)
    else:
        angle_discrete_t = None
        angle_discrete_validation_t = None

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
    TD_log_r = np.zeros([Episodes, chunksize])
    TD_log_t = np.zeros([Episodes, chunksize])

    # Initializing arrays for logs.
    action_log_r_validation = np.zeros([Episodes_validation, chunksize])
    action_log_t_validation = np.zeros([Episodes_validation, chunksize])
    beam_log_r_validation = np.zeros([Episodes_validation, chunksize])
    beam_log_t_validation = np.zeros([Episodes_validation, chunksize])
    R_log_validation = np.zeros([Episodes_validation, chunksize])
    R_max_log_validation = np.zeros([Episodes_validation, chunksize])
    R_min_log_validation = np.zeros([Episodes_validation, chunksize])
    R_mean_log_validation = np.zeros([Episodes_validation, chunksize])

    EPSILON_METHOD = "adaptive"
    # Agent_r = agent_classes.MultiAgent(action_space_r, agent_type='naive', eps=[f'{EPSILON_METHOD}', 0.05], alpha=0.05, gamma=0.7)

    # Agent_t = agent_classes.MultiAgent(action_space_t, agent_type='naive', eps=[f'{EPSILON_METHOD}', 0.05], alpha=0.05, gamma=0.7)
    
    Agent_r = agent_classes.MultiAgent(action_space_r, agent_type='naive', eps=[f'{EPSILON_METHOD}', 0.05], alpha=0.05, gamma=0.7)

    Agent_t = agent_classes.MultiAgent(action_space_t, agent_type='naive', eps=[f'{EPSILON_METHOD}', 0.05], alpha=0.05, gamma=0.7)
    
    Agent_r.eps = validate_eps
    Agent_r.alpha = validate_alpha
    Agent_r.gamma = validate_gamma
    Agent_r.eps_method = EPSILON_METHOD
    Agent_r.reset_eps_table()

    Agent_t.eps = validate_eps
    Agent_t.alpha = validate_alpha
    Agent_t.gamma = validate_gamma
    Agent_t.eps_method = EPSILON_METHOD
    Agent_t.reset_eps_table()

    print('Rewards are now calculated')
    reward_start = time()

    Env.update_data(AoA_Local, AoD_Global, coeff)
    Env.create_reward_matrix()

    print(f'Rewards tog {time() - reward_start} sekunder at regne')

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
        # Choose data
        path_idx = np.random.randint(0, M)
        data_idx = np.random.randint(0, N - chunksize) if (N - chunksize) else 0

        # TODO dette skal ikke blot være beams men én beam og et antal tidligere "actions"

        if n_actions_r_r > 0:
            State_tmp_r = [[tuple([x]) for x in np.random.randint(0, Nbeam_tot_r, n_actions_r_r)]]
        else:
            State_tmp_r = [list("N/A")]

        if n_actions_t_r > 0:
            State_tmp_r.append([tuple([x]) for x in np.random.randint(0, Nbeam_tot_t, n_actions_t_r)])
        else:
            State_tmp_r.append(["N/A"])

        if DIST_r or LOCATION_r:
            State_tmp_r.append(list([dist_discrete_r[0][0]]))
        else:
            State_tmp_r.append(["N/A"])

        if ORI_r:
            State_tmp_r.append(list(np.random.randint(0, ori_res_r, n_ori_r)))
        else:
            State_tmp_r.append(["N/A"])

        if LOCATION_r:
            State_tmp_r.append(list([angle_discrete_r[0][0]]))
        else:
            State_tmp_r.append(["N/A"])

        # ------------ initiate state for transmitter -----------

        if n_actions_r_t > 0:
            State_tmp_t = [[tuple([x]) for x in np.random.randint(0, Nbeam_tot_r, n_actions_r_t)]]
        else:
            State_tmp_t = [list("N/A")]

        if n_actions_t_t > 0:
            State_tmp_t.append([tuple([x]) for x in np.random.randint(0, Nbeam_tot_t, n_actions_t_t)])
        else:
            State_tmp_t.append(["N/A"])

        if DIST_t or LOCATION_t:
            State_tmp_t.append(list([dist_discrete_t[0][0]]))
        else:
            State_tmp_t.append(["N/A"])

        if ORI_t:
            State_tmp_t.append(list(np.random.randint(0, ori_res_t, n_ori_t)))
        else:
            State_tmp_t.append(["N/A"])

        if LOCATION_t:
            State_tmp_t.append(list([angle_discrete_t[0][0]]))
        else:
            State_tmp_t.append(["N/A"])

        State_r = classes.State(State_tmp_r, ORI_r, DIST_r, LOCATION_r, n_actions_r_r, n_actions_t_r)
        State_t = classes.State(State_tmp_t, ORI_t, DIST_t, LOCATION_t, n_actions_r_t, n_actions_t_t)

        # Initiate the action
        previous_beam_nr_r, previous_action_r = Agent_r.e_soft_adj(helpers.state_to_index(State_r.state),
                                                                   State_r.state[0][-1],
                                                                   Nlr)

        previous_beam_nr_t, previous_action_t = Agent_t.e_soft_adj(helpers.state_to_index(State_t.state),
                                                                   State_t.state[1][-1],
                                                                   Nlt)

        #  FOR DEBUG PURPOSES, DO NOT REMOVE YET
        # beam_nr_list_r, action_list_r = Agent_r.get_action_list_adj(State_r.state[0][-1][0], Nlr, Agent_r.action_space)
        # beam_nr_list_t, action_list_t = Agent_t.get_action_list_adj(State_t.state[1][-1][0], Nlt, Agent_t.action_space)
        # for action_r, action_t in zip(action_list_r, action_list_t):
        #     Agent_r.Q[helpers.state_to_index(State_r.state), tuple([action_r])][2] = 0
        #     Agent_t.Q[helpers.state_to_index(State_t.state), tuple([action_t])][2] = 0

        previous_state_r = State_r.state
        previous_state_t = State_t.state

        end = False
        Agent_r.reset_epsilon()
        Agent_t.reset_epsilon()
        # Run the episode
        for n in range(chunksize):

            if n == chunksize - 1:
                end = True
            # Update the current state

            current_state_parameters_r = State_r.get_state_parameters(path_idx, data_idx + n, ori_discrete_r,
                                                                      dist_discrete_r, angle_discrete_r)
            current_state_parameters_t = State_t.get_state_parameters(path_idx, data_idx + n, ori_discrete_t,
                                                                      dist_discrete_t, angle_discrete_t)
            State_r.state = State_r.build_state(tuple([previous_beam_nr_r, previous_beam_nr_t]),
                                                current_state_parameters_r,
                                                tuple([previous_action_r, previous_action_t]))

            # Calculate the action
            beam_nr_r, action_r = Agent_r.e_soft_adj(helpers.state_to_index(State_r.state),
                                                     previous_beam_nr_r,
                                                     Nlr)  # TODO måske ændre sidste output til "limiting factors"

            State_t.state = State_t.build_state(tuple([previous_beam_nr_r, previous_beam_nr_t]),
                                                current_state_parameters_t,
                                                tuple([previous_action_r, previous_action_t]))

            beam_nr_t, action_t = Agent_t.e_soft_adj(helpers.state_to_index(State_t.state),
                                                     previous_beam_nr_t,
                                                     Nlt)  # TODO måske ændre sidste output til "limiting factors"

            # Get reward from performing action
            R, R_noiseless, R_max, R_min, R_mean = Env.take_action(path_idx, n + data_idx, beam_nr_r, beam_nr_t, P_n)

            # Update Q-table
            if METHOD_r == "SARSA":
                TD_error_r = Agent_r.update_TD(helpers.state_to_index(previous_state_r),
                                  previous_action_r,
                                  R,
                                  helpers.state_to_index(State_r.state),
                                  action_r,
                                  end=end)

            elif METHOD_r == "Q-LEARNING":
                greedy_beam_r, greedy_action_r = Agent_r.greedy_adj(
                    helpers.state_to_index(State_r.state), previous_beam_nr_r, Nlr)

                TD_error_r = Agent_r.update_TD(helpers.state_to_index(previous_state_r),
                                  previous_action_r,
                                  R,
                                  helpers.state_to_index(State_r.state),
                                  greedy_action_r,
                                  end=end)

                if Agent_r.agent_type == 'wolf':
                    Agent_r.update_WoLF_PHC_adj(helpers.state_to_index(previous_state_r),
                                                helpers.state_to_index(previous_state_r)[0][-1],
                                                Nlr)

            else:
                raise Exception("Method not recognized for r")

            # Update Q-table
            if METHOD_t == "SARSA":
                TD_error_t = Agent_t.update_TD(helpers.state_to_index(previous_state_t),
                                  previous_action_t,
                                  R,
                                  helpers.state_to_index(State_t.state),
                                  action_t,
                                  end=end)

            elif METHOD_t == "Q-LEARNING":  # Note that next_action here is a direction index and not a beam number
                greedy_beam_t, greedy_action_t = Agent_t.greedy_adj(
                    helpers.state_to_index(State_t.state), previous_beam_nr_t, Nlt)

                TD_error_t = Agent_t.update_TD(helpers.state_to_index(previous_state_t),
                                  previous_action_t,
                                  R,
                                  helpers.state_to_index(State_t.state),
                                  greedy_action_t,
                                  end=end)

                if Agent_t.agent_type == 'wolf':
                    Agent_t.update_WoLF_PHC_adj(helpers.state_to_index(previous_state_t),
                                                helpers.state_to_index(previous_state_t)[1][-1],
                                                Nlt)

            else:
                raise Exception("Method not recognized for t")

            Agent_r.update_epsilon(n + 1, validate_weight, TD_error_r, helpers.state_to_index(previous_state_r))
            Agent_t.update_epsilon(n + 1, validate_weight, TD_error_t, helpers.state_to_index(previous_state_t))


            action_log_r[episode, n] = action_r[0]
            action_log_t[episode, n] = action_t[0]
            beam_log_r[episode, n] = beam_nr_r[0]
            beam_log_t[episode, n] = beam_nr_t[0]
            R_log[episode, n] = R_noiseless
            R_max_log[episode, n] = R_max
            R_min_log[episode, n] = R_min
            R_mean_log[episode, n] = R_mean
            TD_log_r[episode, n] = TD_error_r
            TD_log_t[episode, n] = TD_error_t


            previous_state_r = State_r.state
            previous_state_t = State_t.state
            previous_beam_nr_r = beam_nr_r
            previous_beam_nr_t = beam_nr_t
            previous_action_r = action_r
            previous_action_t = action_t

    # ------------------------------------- Run agent on validation data ---------------------------------
    print('Rewards for validation are now calculated')
    reward_start = time()

    Env.update_data(AoA_Local_validation, AoD_Global_validation, coeff_validation)
    Env.create_reward_matrix()

    print(f'Rewards tog {time() - reward_start} sekunder at regne')
    
    Agent_r = agent_classes.MultiAgent(action_space_r, agent_type='naive', eps=[f'{EPSILON_METHOD}', 0.05], alpha=0.05, gamma=0.7)

    Agent_t = agent_classes.MultiAgent(action_space_t, agent_type='naive', eps=[f'{EPSILON_METHOD}', 0.05], alpha=0.05, gamma=0.7)
    
    Agent_r.eps = validate_eps
    Agent_r.alpha = validate_alpha
    Agent_r.gamma = validate_gamma
    Agent_r.eps_method = EPSILON_METHOD
    Agent_r.reset_eps_table()

    Agent_t.eps = validate_eps
    Agent_t.alpha = validate_alpha
    Agent_t.gamma = validate_gamma
    Agent_t.eps_method = EPSILON_METHOD
    Agent_t.reset_eps_table()


    for episode in tqdm(range(Episodes_validation), desc="Episodes"):
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
        # Choose data
        path_idx = np.random.randint(0, M_validation)
        data_idx = np.random.randint(0, N_validation - chunksize) if (N_validation - chunksize) else 0

        # TODO dette skal ikke blot være beams men én beam og et antal tidligere "actions"

        if n_actions_r_r > 0:
            State_tmp_r = [[tuple([x]) for x in np.random.randint(0, Nbeam_tot_r, n_actions_r_r)]]
        else:
            State_tmp_r = [list("N/A")]

        if n_actions_t_r > 0:
            State_tmp_r.append([tuple([x]) for x in np.random.randint(0, Nbeam_tot_t, n_actions_t_r)])
        else:
            State_tmp_r.append(["N/A"])

        if DIST_r or LOCATION_r:
            State_tmp_r.append(list([dist_discrete_validation_r[0][0]]))
        else:
            State_tmp_r.append(["N/A"])

        if ORI_r:
            State_tmp_r.append(list(np.random.randint(0, ori_res_r, n_ori_r)))
        else:
            State_tmp_r.append(["N/A"])

        if LOCATION_r:
            State_tmp_r.append(list([angle_discrete_validation_r[0][0]]))
        else:
            State_tmp_r.append(["N/A"])

        # ------------ initiate state for transmitter -----------

        if n_actions_r_t > 0:
            State_tmp_t = [[tuple([x]) for x in np.random.randint(0, Nbeam_tot_r, n_actions_r_t)]]
        else:
            State_tmp_t = [list("N/A")]

        if n_actions_t_t > 0:
            State_tmp_t.append([tuple([x]) for x in np.random.randint(0, Nbeam_tot_t, n_actions_t_t)])
        else:
            State_tmp_t.append(["N/A"])

        if DIST_t or LOCATION_t:
            State_tmp_t.append(list([dist_discrete_validation_t[0][0]]))
        else:
            State_tmp_t.append(["N/A"])

        if ORI_t:
            State_tmp_t.append(list(np.random.randint(0, ori_res_t, n_ori_t)))
        else:
            State_tmp_t.append(["N/A"])

        if LOCATION_t:
            State_tmp_t.append(list([angle_discrete_validation_t[0][0]]))
        else:
            State_tmp_t.append(["N/A"])

        State_r = classes.State(State_tmp_r, ORI_r, DIST_r, LOCATION_r, n_actions_r_r, n_actions_t_r)
        State_t = classes.State(State_tmp_t, ORI_t, DIST_t, LOCATION_t, n_actions_r_t, n_actions_t_t)

        # Initiate the action
        previous_beam_nr_r, previous_action_r = Agent_r.e_soft_adj(helpers.state_to_index(State_r.state),
                                                                   State_r.state[0][-1],
                                                                   Nlr)

        previous_beam_nr_t, previous_action_t = Agent_t.e_soft_adj(helpers.state_to_index(State_t.state),
                                                                   State_t.state[1][-1],
                                                                   Nlt)

        #  FOR DEBUG PURPOSES, DO NOT REMOVE YET
        # beam_nr_list_r, action_list_r = Agent_r.get_action_list_adj(State_r.state[0][-1][0], Nlr, Agent_r.action_space)
        # beam_nr_list_t, action_list_t = Agent_t.get_action_list_adj(State_t.state[1][-1][0], Nlt, Agent_t.action_space)
        # for action_r, action_t in zip(action_list_r, action_list_t):
        #     Agent_r.Q[helpers.state_to_index(State_r.state), tuple([action_r])][2] = 0
        #     Agent_t.Q[helpers.state_to_index(State_t.state), tuple([action_t])][2] = 0

        previous_state_r = State_r.state
        previous_state_t = State_t.state

        end = False

        Agent_r.reset_epsilon()
        Agent_t.reset_epsilon()
        # Run the episode
        for n in range(chunksize):

            if n == chunksize - 1:
                end = True
            # Update the current state

            current_state_parameters_r = State_r.get_state_parameters(path_idx, data_idx + n,
                                                                      ori_discrete_validation_r,
                                                                      dist_discrete_validation_r,
                                                                      angle_discrete_validation_r)

            current_state_parameters_t = State_t.get_state_parameters(path_idx, data_idx + n,
                                                                      ori_discrete_validation_t,
                                                                      dist_discrete_validation_t,
                                                                      angle_discrete_validation_t)

            State_r.state = State_r.build_state(tuple([previous_beam_nr_r, previous_beam_nr_t]),
                                                current_state_parameters_r,
                                                tuple([previous_action_r, previous_action_t]))

            # Calculate the action
            beam_nr_r, action_r = Agent_r.e_soft_adj(helpers.state_to_index(State_r.state),
                                                     previous_beam_nr_r,
                                                     Nlr)  # TODO måske ændre sidste output til "limiting factors"

            State_t.state = State_t.build_state(tuple([previous_beam_nr_r, previous_beam_nr_t]),
                                                current_state_parameters_t,
                                                tuple([previous_action_r, previous_action_t]))

            beam_nr_t, action_t = Agent_t.e_soft_adj(helpers.state_to_index(State_t.state),
                                                     previous_beam_nr_t,
                                                     Nlt)  # TODO måske ændre sidste output til "limiting factors"

            # Get reward from performing action
            R, R_noiseless, R_max, R_min, R_mean = Env.take_action(path_idx, n + data_idx, beam_nr_r, beam_nr_t, P_n)

            # Update Q-table
            if METHOD_r == "SARSA":
                TD_error_r = Agent_r.update_TD(helpers.state_to_index(previous_state_r),
                                  previous_action_r,
                                  R,
                                  helpers.state_to_index(State_r.state),
                                  action_r,
                                  end=end)

            elif METHOD_r == "Q-LEARNING":
                greedy_beam_r, greedy_action_r = Agent_r.greedy_adj(
                    helpers.state_to_index(State_r.state), previous_beam_nr_r, Nlr)

                TD_error_r = Agent_r.update_TD(helpers.state_to_index(previous_state_r),
                                  previous_action_r,
                                  R,
                                  helpers.state_to_index(State_r.state),
                                  greedy_action_r,
                                  end=end)

                if Agent_r.agent_type == 'wolf':
                    Agent_r.update_WoLF_PHC_adj(helpers.state_to_index(previous_state_r),
                                                helpers.state_to_index(previous_state_r)[0][-1],
                                                Nlr)

            else:
                raise Exception("Method not recognized for r")

            # Update Q-table
            if METHOD_t == "SARSA":
                TD_error_t = Agent_t.update_TD(helpers.state_to_index(previous_state_t),
                                  previous_action_t,
                                  R,
                                  helpers.state_to_index(State_t.state),
                                  action_t,
                                  end=end)

            elif METHOD_t == "Q-LEARNING":  # Note that next_action here is a direction index and not a beam number
                greedy_beam_t, greedy_action_t = Agent_t.greedy_adj(
                    helpers.state_to_index(State_t.state), previous_beam_nr_t, Nlt)

                TD_error_t = Agent_t.update_TD(helpers.state_to_index(previous_state_t),
                                  previous_action_t,
                                  R,
                                  helpers.state_to_index(State_t.state),
                                  greedy_action_t,
                                  end=end)

                if Agent_t.agent_type == 'wolf':
                    Agent_t.update_WoLF_PHC_adj(helpers.state_to_index(previous_state_t),
                                                helpers.state_to_index(previous_state_t)[1][-1],
                                                Nlt)

            else:
                raise Exception("Method not recognized for t")

            Agent_r.update_epsilon(n + 1, validate_weight, TD_error_r, helpers.state_to_index(previous_state_r))
            Agent_t.update_epsilon(n + 1, validate_weight, TD_error_t, helpers.state_to_index(previous_state_t))

            action_log_r_validation[episode, n] = action_r[0]
            action_log_t_validation[episode, n] = action_t[0]
            beam_log_r_validation[episode, n] = beam_nr_r[0]
            beam_log_t_validation[episode, n] = beam_nr_t[0]
            R_log_validation[episode, n] = R_noiseless
            R_max_log_validation[episode, n] = R_max
            R_min_log_validation[episode, n] = R_min
            R_mean_log_validation[episode, n] = R_mean

            previous_state_r = State_r.state
            previous_state_t = State_t.state
            previous_beam_nr_r = beam_nr_r
            previous_beam_nr_t = beam_nr_t
            previous_action_r = action_r
            previous_action_t = action_t

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
        },
        'Validation': {
            'R_log': R_log_validation,
            'R_max': R_max_log_validation,
            'R_min': R_min_log_validation,
            'R_mean': R_mean_log_validation,
            'action_log_r': action_log_r_validation,
            'action_log_t': action_log_t_validation,
            'beam_log_r': beam_log_r_validation,
            'beam_log_t': beam_log_t_validation
        }
    }

    data_agent = {
        'Agent_r': Agent_r,
        'Agent_t': Agent_t,
        'agent_settings_r': agent_settings_r,
        'agent_settings_t': agent_settings_t,
        'channel_settings': channel_settings,
    }

    try:
        if "NLOS" in channel_settings["scenarios"][0]:
            # helpers.dump_pickle(data_agent, 'Results/', f'{CASE}_NLOS_{RESULT_NAME}_results.pickle')
            helpers.dump_hdf5_validate(data_reward, 'Results/', f'{CASE}_NLOS_{RESULT_NAME}_validated_{validate_eps}_{validate_alpha}_{validate_gamma}_{validate_weight}_results.hdf5')
        else:
            # helpers.dump_pickle(data_agent, 'Results/', f'{CASE}_LOS_{RESULT_NAME}_results.pickle')
            helpers.dump_hdf5_validate(data_reward, 'Results/', f'{CASE}_LOS_{RESULT_NAME}_validated_{validate_eps}_{validate_alpha}_{validate_gamma}_{validate_weight}_results.hdf5')
    except OSError as e:
        print(e)
        print("Saving to root folder instead")
        if "NLOS" in channel_settings["scenarios"][0]:
            # helpers.dump_pickle(data_agent, '', f'{CASE}_NLOS_{RESULT_NAME}_results.pickle')
            helpers.dump_hdf5_validate(data_reward, '', f'{CASE}_NLOS_{RESULT_NAME}_validated_{validate_eps}_{validate_alpha}_{validate_gamma}_{validate_weight}_results.hdf5')
        else:
            # helpers.dump_pickle(data_agent, '', f'{CASE}_LOS_{RESULT_NAME}_results.pickle')
            helpers.dump_hdf5_validate(data_reward, '', f'{CASE}_LOS_{RESULT_NAME}_validated_{validate_eps}_{validate_alpha}_{validate_gamma}_{validate_weight}_results.hdf5')
