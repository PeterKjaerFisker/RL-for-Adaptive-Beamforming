# -*- coding: utf-8 -*-
"""
@authors: Dennis Sand & Peter Fisker
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
    AGENT_SETTINGS = sys.argv[2]
    validate_eps = float(sys.argv[3])
    validate_alpha = float(sys.argv[4])
    validate_gamma = float(sys.argv[5])
    validate_weight = float(sys.argv[6])
else:
    CHANNEL_SETTINGS = "pedestrian_LOS_2_users_20000_steps_01"
    AGENT_SETTINGS = "SARSA_TFFT_2-2-0-0-2-32_7000_2000"
    validate_eps = 0
    validate_alpha = 0.05
    validate_gamma = 0.7
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
    with open(f'Settings/{AGENT_SETTINGS}.json', 'r') as fs:
        agent_settings = json.load(fs)

    # Load global parameters
    RESULT_NAME = agent_settings["RESULT_NAME"]
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

    # ----------- Validation Simulation Parameters ------------
    N_validation = validation_settings['N']  # Number of steps in an episode
    M_validation = validation_settings["M"]  # Number of episodes
    r_lim_validation = validation_settings["rlim"]  # Radius of the cell
    fc_validation = validation_settings["fc"]  # Center frequency
    P_t_validation = validation_settings["P_t"]  # Transmission power
    lambda_validation = 3e8 / fc  # Wave length
    Episodes_validation = 10000

    # ----------- Reinforcement Learning Parameters -----------
    METHOD = agent_settings["METHOD"]  # RL table update, "simple", "SARSA" or "Q-LEARNING"

    ADJ = agent_settings["ADJ"]  # Whether action space should be all beams ("False") or adjacent ("True")
    ORI = agent_settings["ORI"]  # Include the User Terminal orientation in the state
    DIST = agent_settings["DIST"]  # Include the distance between User Terminal and Base Station in the state
    LOCATION = agent_settings["LOCATION"]  # Include location of User Terminal in polar coordinates in the state
    n_actions_r = agent_settings["n_actions_r"]  # Number of previous actions
    n_actions_t = agent_settings["n_actions_t"]  # Number of previous actions
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
    Nbeam_tot_r = (2 ** (Nlr + 1)) - 2  # Total number of beams for the receiver
    Nbeam_tot_t = (2 ** (Nlt + 1)) - 2  # Total number of beams for the transmitter

    # ----------- Load the data -----------
    t_start = time()
    # Load the training data
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
    if ORI:
        ori_discrete = np.zeros([M, N])
        for m in range(M):
            ori_discrete[m, :] = helpers.discrete_ori(Orientation[m][0][2, :], ori_res)

        ori_discrete_validation = np.zeros([M_validation, N_validation])
        for m in range(M_validation):
            ori_discrete_validation[m, :] = helpers.discrete_ori(Orientation_validation[m][0][2, :], ori_res)
    else:
        ori_discrete = None
        ori_discrete_validation = None

    if DIST or LOCATION:
        dist_discrete = np.zeros([M, N])
        for m in range(M):
            dist_discrete[m, :] = helpers.discrete_dist(pos_log[m], dist_res, r_lim)

        dist_discrete_validation = np.zeros([M_validation, N_validation])
        for m in range(M_validation):
            dist_discrete_validation[m, :] = helpers.discrete_dist(pos_log_validation[m], dist_res, r_lim)
    else:
        dist_discrete = None
        dist_discrete_validation = None

    if LOCATION:
        angle_discrete = np.zeros([M, N])
        for m in range(M):
            angle_discrete[m, :] = helpers.discrete_angle(pos_log[m], angle_res)

        angle_discrete_validation = np.zeros([M_validation, N_validation])
        for m in range(M_validation):
            angle_discrete_validation[m, :] = helpers.discrete_angle(pos_log_validation[m], angle_res)
    else:
        angle_discrete = None
        angle_discrete_validation = None

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

    # Initializing arrays for logs.
    action_log_r_validation = np.zeros([Episodes_validation, chunksize])
    action_log_t_validation = np.zeros([Episodes_validation, chunksize])
    beam_log_r_validation = np.zeros([Episodes_validation, chunksize])
    beam_log_t_validation = np.zeros([Episodes_validation, chunksize])
    R_log_validation = np.zeros([Episodes_validation, chunksize])
    R_max_log_validation = np.zeros([Episodes_validation, chunksize])
    R_min_log_validation = np.zeros([Episodes_validation, chunksize])
    R_mean_log_validation = np.zeros([Episodes_validation, chunksize])

    EPSILON_METHOD = "constant"
    Agent = agent_classes.Agent(action_space_r, action_space_t, eps=[f'{EPSILON_METHOD}', 0.05], alpha=0.05)

    print('Rewards for training are now calculated')
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

        # TODO dette skal ikke blot være beams men én beam og et antal tidligere "retninger"

        if n_actions_r > 0:
            State_tmp = [[tuple([x]) for x in np.random.randint(0, Nbeam_tot_r, n_actions_r)]]
        else:
            State_tmp = [list("N/A")]

        if n_actions_t > 0:
            State_tmp.append([tuple([x]) for x in np.random.randint(0, Nbeam_tot_t, n_actions_t)])
        else:
            State_tmp.append(["N/A"])

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

        State = classes.State(State_tmp, ORI, DIST, LOCATION, n_actions_r, n_actions_t)

        # Initiate the action
        previous_beam_nr, previous_action = Agent.e_greedy_adj(helpers.state_to_index(State.state),
                                                               tuple([State.state[0][-1][0], State.state[1][-1][0]]),
                                                               Nlr,
                                                               Nlt)

        previous_state = State.state

        end = False

        Agent.reset_epsilon()
        # Run the episode
        for n in range(chunksize):
            if n == chunksize - 1:
                end = True

            # Update the current state
            current_state_parameters = State.get_state_parameters(path_idx, data_idx + n,
                                                                  ori_discrete,
                                                                  dist_discrete,
                                                                  angle_discrete)

            State.state = State.build_state(previous_beam_nr, current_state_parameters, previous_action)

            # Calculate the action
            # TODO måske ændre sidste output til "limiting factors"
            beam_nr, action = Agent.e_greedy_adj(helpers.state_to_index(State.state), previous_beam_nr, Nlr, Nlt)

            # Get reward from performing action
            R, R_max, R_min, R_mean = Env.take_action(path_idx, n + data_idx, beam_nr[0], beam_nr[1])

            # Update Q-table
            if METHOD == "SARSA":
                TD_error = Agent.update_TD(helpers.state_to_index(previous_state),
                                           previous_action,
                                           R,
                                           helpers.state_to_index(State.state),
                                           action,
                                           end=end)

            elif METHOD == "Q-LEARNING":
                greedy_beam, greedy_action = Agent.greedy_adj(helpers.state_to_index(State.state), previous_beam_nr,
                                                              Nlr, Nlt)

                TD_error = Agent.update_TD(helpers.state_to_index(previous_state),
                                           previous_action,
                                           R,
                                           helpers.state_to_index(State.state),
                                           greedy_action,
                                           end=end)
            else:
                raise Exception("Method not recognized")

            Agent.update_epsilon(n + 1, validate_weight, TD_error, helpers.state_to_index(previous_state))

            action_log_r[episode, n] = action[0]
            action_log_t[episode, n] = action[1]
            beam_log_r[episode, n] = beam_nr[0]
            beam_log_t[episode, n] = beam_nr[1]
            R_log[episode, n] = R
            R_max_log[episode, n] = R_max
            R_min_log[episode, n] = R_min
            R_mean_log[episode, n] = R_mean

            TD_log[episode, n] = TD_error

            previous_state = State.state
            previous_beam_nr = beam_nr
            previous_action = action




    # ------------------------------------- Run agent on validation data ---------------------------------

    print('Rewards for validation are now calculated')
    reward_start = time()

    Env.update_data(AoA_Local_validation, AoD_Global_validation, coeff_validation)
    Env.create_reward_matrix()

    print(f'Rewards tog {time() - reward_start} sekunder at regne')
    Agent.eps = validate_eps
    Agent.alpha = validate_alpha
    Agent.gamma = validate_gamma
    Agent.eps_method = 'adaptive'
    Agent.reset_eps_table()

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

        # TODO dette skal ikke blot være beams men én beam og et antal tidligere "retninger"

        if n_actions_r > 0:
            State_tmp = [[tuple([x]) for x in np.random.randint(0, Nbeam_tot_r, n_actions_r)]]
        else:
            State_tmp = [list("N/A")]

        if n_actions_t > 0:
            State_tmp.append([tuple([x]) for x in np.random.randint(0, Nbeam_tot_t, n_actions_t)])
        else:
            State_tmp.append(["N/A"])

        if DIST or LOCATION:
            State_tmp.append(list([dist_discrete_validation[0][0]]))
        else:
            State_tmp.append(["N/A"])

        if ORI:
            State_tmp.append(list(np.random.randint(0, ori_res, n_ori)))
        else:
            State_tmp.append(["N/A"])

        if LOCATION:
            State_tmp.append(list([angle_discrete_validation[0][0]]))
        else:
            State_tmp.append(["N/A"])

        State = classes.State(State_tmp, ORI, DIST, LOCATION, n_actions_r, n_actions_t)

        # Initiate the action
        previous_beam_nr, previous_action = Agent.e_greedy_adj(helpers.state_to_index(State.state),
                                                               tuple([State.state[0][-1][0], State.state[1][-1][0]]),
                                                               Nlr,
                                                               Nlt)

        previous_state = State.state

        end = False

        Agent.reset_epsilon()
        # Run the episode
        for n in range(chunksize):
            if n == chunksize - 1:
                end = True

            # Update the current state
            current_state_parameters = State.get_state_parameters(path_idx, data_idx + n,
                                                                  ori_discrete_validation,
                                                                  dist_discrete_validation,
                                                                  angle_discrete_validation)

            State.state = State.build_state(previous_beam_nr, current_state_parameters, previous_action)

            # Calculate the action
            # TODO måske ændre sidste output til "limiting factors"
            beam_nr, action = Agent.e_greedy_adj(helpers.state_to_index(State.state), previous_beam_nr, Nlr, Nlt)

            # Get reward from performing action

            R, R_max, R_min, R_mean = Env.take_action(path_idx, n + data_idx, beam_nr[0], beam_nr[1])

            # Update Q-table
            if METHOD == "SARSA":
                TD_error = Agent.update_TD(helpers.state_to_index(previous_state),
                                           previous_action,
                                           R,
                                           helpers.state_to_index(State.state),
                                           action,
                                           end=end)

            elif METHOD == "Q-LEARNING":
                greedy_beam, greedy_action = Agent.greedy_adj(helpers.state_to_index(State.state), previous_beam_nr,
                                                              Nlr, Nlt)

                TD_error = Agent.update_TD(helpers.state_to_index(previous_state),
                                           previous_action,
                                           R,
                                           helpers.state_to_index(State.state),
                                           greedy_action,
                                           end=end)
            else:
                raise Exception("Method not recognized")

            Agent.update_epsilon(n + 1, validate_weight, TD_error, helpers.state_to_index(previous_state))

            action_log_r_validation[episode, n] = action[0]
            action_log_t_validation[episode, n] = action[1]
            beam_log_r_validation[episode, n] = beam_nr[0]
            beam_log_t_validation[episode, n] = beam_nr[1]
            R_log_validation[episode, n] = R
            R_max_log_validation[episode, n] = R_max
            R_min_log_validation[episode, n] = R_min
            R_mean_log_validation[episode, n] = R_mean

            previous_state = State.state
            previous_beam_nr = beam_nr
            previous_action = action

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
        'Agent': Agent,
        'agent_settings': agent_settings,
        'channel_settings': channel_settings,
    }

    try:
        if "NLOS" in channel_settings["scenarios"][0]:
            helpers.dump_hdf5_validate(data_reward, 'Results/',
                                       f'{CASE}_NLOS_{RESULT_NAME}_validated_{validate_eps}_{validate_alpha}_{validate_gamma}_{validate_weight}_results.hdf5')
            # helpers.dump_pickle(data_agent, 'Results/', f'{CASE}_NLOS_{RESULT_NAME}_results.pickle')
        else:
            helpers.dump_hdf5_validate(data_reward, 'Results/',
                                       f'{CASE}_LOS_{RESULT_NAME}_validated_{validate_eps}_{validate_alpha}_{validate_gamma}_{validate_weight}_results.hdf5')
            # helpers.dump_pickle(data_agent, 'Results/', f'{CASE}_LOS_{RESULT_NAME}_results.pickle')
    except OSError as e:
        print(e)
        print("Saving to root folder instead")
        if "NLOS" in channel_settings["scenarios"][0]:
            helpers.dump_hdf5_validate(data_reward, '',
                                       f'{CASE}_NLOS_{RESULT_NAME}_validated_{validate_eps}_{validate_alpha}_{validate_gamma}_{validate_weight}_results.hdf5')
            # helpers.dump_pickle(data_agent, '', f'{CASE}_NLOS_{RESULT_NAME}_results.pickle')
        else:
            helpers.dump_hdf5_validate(data_reward, '',
                                       f'{CASE}_LOS_{RESULT_NAME}_validated_{validate_eps}_{validate_alpha}_{validate_gamma}_{validate_weight}_results.hdf5')
            # helpers.dump_pickle(data_agent, '', f'{CASE}_LOS_{RESULT_NAME}_results.pickle')