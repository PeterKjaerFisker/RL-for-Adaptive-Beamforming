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
    AGENT_SETTINGS_R = sys.argv[2]
    AGENT_SETTINGS_T = sys.argv[3]
else:
    CHANNEL_SETTINGS = "car_urban_LOS_16_users_10000_steps"
    AGENT_SETTINGS_R = "sarsa_TFFT_2-2-0-0-32-32_7000_2000"
    AGENT_SETTINGS_T = "sarsa_TFFT_2-2-0-0-32-32_7000_2000"

# %% main
if __name__ == "__main__":

    # Load Channel Settings for simulation
    with open(f'Settings/{CHANNEL_SETTINGS}.json', 'r') as fs:
        channel_settings = json.load(fs)

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

    

    RESULT_NAME = agent_settings_r["RESULT_NAME"].lstrip(f'_{chunksize}_{Episodes}') +'_'+agent_settings_t["RESULT_NAME"]
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
    METHOD_r = agent_settings_r["METHOD"]  # RL table update, "simple", "SARSA" or "Q-LEARNING"
    ADJ_r = agent_settings_r["ADJ"]  # Whether action space should be all beams ("False") or adjacent ("True")
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
    ADJ_t = agent_settings_t["ADJ"]  # Whether action space should be all beams ("False") or adjacent ("True")
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
    # precoder_codebook = helpers.codebook2(16,32)

    # Calculate hierarchical-codebook - Receiver
    combiner_codebook = helpers.codebook(Nr, Nlr, lambda_)
    # combiner_codebook = helpers.codebook2(8,8)

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
    else:
        ori_discrete_r = None
        
    if ORI_t:
        ori_discrete_t = np.zeros([M, N])
        for m in range(M):
            ori_discrete_t[m, :] = helpers.discrete_ori(Orientation[m][0][2, :], ori_res_t)
    else:
        ori_discrete_t = None

    if DIST_r or LOCATION_r:
        dist_discrete_r = np.zeros([M, N])
        for m in range(M):
            dist_discrete_r[m, :] = helpers.discrete_dist(pos_log[m], dist_res_r, r_lim)
    else:
        dist_discrete_r = None
        
    if DIST_t or LOCATION_t:
        dist_discrete_t = np.zeros([M, N])
        for m in range(M):
            dist_discrete_t[m, :] = helpers.discrete_dist(pos_log[m], dist_res_t, r_lim)
    else:
        dist_discrete_t = None

    if LOCATION_r:
        angle_discrete_r = np.zeros([M, N])
        for m in range(M):
            angle_discrete_r[m, :] = helpers.discrete_angle(pos_log[m], angle_res_r)
    else:
        angle_discrete_r = None
        
    if LOCATION_t:
        angle_discrete_t = np.zeros([M, N])
        for m in range(M):
            angle_discrete_t[m, :] = helpers.discrete_angle(pos_log[m], angle_res_t)
    else:
        angle_discrete_t = None

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

    Agent_r = classes.Agent(action_space_r, eps=0.05, alpha=["constant", 0.05])
    Agent_t = classes.Agent(action_space_t, eps=0.05, alpha=["constant", 0.05])

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

        # TODO dette skal ikke blot være beams men én beam og et antal tidligere "retninger"

        if n_actions_r_r > 0:
            State_tmp_r = [list(np.random.randint(0, Nbeam_tot_r, n_actions_r_r))]
        else:
            State_tmp_r = [list("N/A")]

        if n_actions_t_r > 0:
            State_tmp_r.append(list(np.random.randint(0, Nbeam_tot_t, n_actions_t_r)))
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
            
        if n_actions_r_t > 0:
            State_tmp_t = [list(np.random.randint(0, Nbeam_tot_r, n_actions_r_t))]
        else:
            State_tmp_t = [list("N/A")]

        if n_actions_t_t > 0:
            State_tmp_t.append(list(np.random.randint(0, Nbeam_tot_t, n_actions_t_t)))
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

        # TODO ===|||=== Hvis state skal være forskellig skal der ændres ting her ===|||===
        State_r = classes.State(State_tmp_r, ORI_r, DIST_r, LOCATION_r, n_actions_r_r, n_actions_t_r)
        State_t = classes.State(State_tmp_t, ORI_t, DIST_t, LOCATION_t, n_actions_r_t, n_actions_t_t)
        # TODO ===|||=== Hvis state skal være forskellig skal der ændres ting her ===|||===

        # Initiate the action
        beam_nr_r = tuple([np.random.choice(action_space_r)])  # TODO ændre action, i ADJ til at være retningen man går og ikke beam nr.
        beam_nr_t = tuple([np.random.choice(action_space_t)])
        adj_action_index_r = tuple([np.random.randint(0, 6)])  # TODO måske tilføj en seperat værdi der er beam nr. der ikke nødvendigivis er den del af state
        adj_action_index_t = tuple([np.random.randint(0, 6)])

        # TODO første action skal afhænge af initial state

        previous_state_r = State_r.state
        previous_state_t = State_t.state
        previous_beam_nr_r = beam_nr_r
        previous_beam_nr_t = beam_nr_t

        if ADJ_r:
            previous_action_r = adj_action_index_r
        else:
            previous_action_r = beam_nr_r
            
        if ADJ_t:
            previous_action_t = adj_action_index_t
        else:
            previous_action_t = beam_nr_t

        end = False
        # Run the episode
        for n in range(chunksize):

            # Update the current state
            # Check if the user terminal orientation is part of the state.
            if ORI_r:
                # Get the current discrete orientation of the user terminal
                ori_r = int(ori_discrete_r[path_idx, data_idx + n])
                # Check if current step is the last step in episode.
                # If not the last step, the next orientation of the user terminal is assigned to a variable
                if n < chunksize - 1:
                    next_ori_r = int(ori_discrete_r[path_idx, data_idx + n + 1])
            else:
                ori_r = "N/A"
                next_ori_r = "N/A"

            if DIST_r or LOCATION_r:
                dist_r = dist_discrete_r[path_idx, data_idx + n]
                if n < chunksize - 1:
                    next_dist_r = dist_discrete_r[path_idx, data_idx + n + 1]
            else:
                dist_r = "N/A"
                next_dist_r = "N/A"

            if LOCATION_r:
                angle_r = angle_discrete_r[path_idx, data_idx + n]
                if n < chunksize - 1:
                    next_angle_r = angle_discrete_r[path_idx, data_idx + n + 1]
            else:
                angle_r = "N/A"
                next_angle_r = "N/A"
                
            if ORI_t:
                # Get the current discrete orientation of the user terminal
                ori_t = int(ori_discrete_t[path_idx, data_idx + n])
                # Check if current step is the last step in episode.
                # If not the last step, the next orientation of the user terminal is assigned to a variable
                if n < chunksize - 1:
                    next_ori_t = int(ori_discrete_t[path_idx, data_idx + n + 1])
            else:
                ori_t = "N/A"
                next_ori_t = "N/A"

            if DIST_t or LOCATION_t:
                dist_t = dist_discrete_t[path_idx, data_idx + n]
                if n < chunksize - 1:
                    next_dist_t = dist_discrete_t[path_idx, data_idx + n + 1]
            else:
                dist_t = "N/A"
                next_dist_t = "N/A"

            if LOCATION_t:
                angle_t = angle_discrete_t[path_idx, data_idx + n]
                if n < chunksize - 1:
                    next_angle_t = angle_discrete_t[path_idx, data_idx + n + 1]
            else:
                angle_t = "N/A"
                next_angle_t = "N/A"

            if n == chunksize - 1:
                end = True

            current_state_parameters_r = [dist_r, ori_r, angle_r]
            current_state_parameters_t = [dist_t, ori_t, angle_t]

            # Calculate the action
            if ADJ_r:
                # State.state = State.build_state(previous_beam_nr, current_state_parameters, previous_action)
                # beam_nr, adj_action_index = Agent.e_greedy_adj(helpers.state_to_index(State.state), beam_nr, Nlr,
                #                                                Nlt)  # TODO måske ændre sidste output til "limiting factors"
                # action_index = adj_action_index
                State_r.state = State_r.build_state(tuple([previous_beam_nr_r, previous_beam_nr_t]), current_state_parameters_r, tuple([previous_action_r, previous_action_t]))
                # TODO ===|||=== Hvis state skal være forskellig skal der ændres ting her ===|||===
                beam_nr_r, adj_action_index_r = Agent_r.e_greedy_adj(helpers.state_to_index(State_r.state), beam_nr_r, Nlr)  # TODO måske ændre sidste output til "limiting factors"
                action_index_r = adj_action_index_r
                State_t.state = State_t.build_state(tuple([previous_beam_nr_r, previous_beam_nr_t]), current_state_parameters_t, tuple([previous_action_r, previous_action_t]))
                # TODO ===|||=== Hvis state skal være forskellig skal der ændres ting her ===|||===
                beam_nr_t, adj_action_index_t = Agent_t.e_greedy_adj(helpers.state_to_index(State_t.state), beam_nr_t, Nlt)  # TODO måske ændre sidste output til "limiting factors"
                action_index_t = adj_action_index_t
            else:
                # State.state = State.build_state(previous_beam_nr, current_state_parameters)
                # beam_nr = Agent.e_greedy(helpers.state_to_index(State.state))
                # action_index = beam_nr
                State_r.state = State_r.build_state(previous_beam_nr_r, current_state_parameters_r)
                beam_nr_r = Agent_r.e_greedy(helpers.state_to_index(State_r.state))
                action_index_r = beam_nr_r
                State_t.state = State_t.build_state(previous_beam_nr_t, current_state_parameters_t)
                beam_nr_t = Agent_t.e_greedy(helpers.state_to_index(State_t.state))
                action_index_t = beam_nr_t

            # Get reward from performing action

            R, R_max, R_min, R_mean = Env.take_action(path_idx, n+data_idx, beam_nr_r, beam_nr_t)

            # Update Q-table
            if METHOD_r == "SIMPLE":
                Agent_r.update_simple(helpers.state_to_index(State_r.state), action_index_r, R)

            elif METHOD_r == "SARSA":
                Agent_r.update_TD(helpers.state_to_index(previous_state_r),
                                previous_action_r,
                                R,
                                helpers.state_to_index(State_r.state),
                                action_index_r,
                                end=end)

            elif METHOD_r == "Q-LEARNING":
                if ADJ_r:  # Note that next_action here is a direction index and not a beam number
                    next_beam_r, next_action_r = Agent_r.greedy_adj(helpers.state_to_index(State_r.state), beam_nr_r, Nlr)
                else:  # Note that next_action is a beam number and not a direction index
                    next_action_r = Agent_r.greedy(helpers.state_to_index(State_r.state))

                Agent_r.update_TD(helpers.state_to_index(previous_state_r),
                                previous_action_r,
                                R,
                                helpers.state_to_index(State_r.state),
                                next_action_r,
                                end=end)
            else:
                raise Exception("Method not recognized for r")
                
            if METHOD_t == "SIMPLE":
                Agent_t.update_simple(helpers.state_to_index(State_t.state), action_index_t, R)

            elif METHOD_t == "SARSA":
                Agent_t.update_TD(helpers.state_to_index(previous_state_t),
                                previous_action_t,
                                R,
                                helpers.state_to_index(State_t.state),
                                action_index_t,
                                end=end)

            elif METHOD_t == "Q-LEARNING":
                if ADJ_t:  # Note that next_action here is a direction index and not a beam number
                    next_beam_t, next_action_t = Agent_t.greedy_adj(helpers.state_to_index(State_t.state), beam_nr_t, Nlt)
                else:  # Note that next_action is a beam number and not a direction index
                    next_action_t = Agent_t.greedy(helpers.state_to_index(State_t.state))

                Agent_t.update_TD(helpers.state_to_index(previous_state_t),
                                previous_action_t,
                                R,
                                helpers.state_to_index(State_t.state),
                                next_action_t,
                                end=end)
            else:
                raise Exception("Method not recognized for t")

            action_log_r[episode, n] = action_index_r[0]
            action_log_t[episode, n] = action_index_t[0]
            beam_log_r[episode, n] = beam_nr_r[0]
            beam_log_t[episode, n] = beam_nr_t[0]
            R_log[episode, n] = R
            R_max_log[episode, n] = R_max
            R_min_log[episode, n] = R_min
            R_mean_log[episode, n] = R_mean

            previous_state_r = State_r.state
            previous_state_t = State_t.state
            previous_beam_nr_r = beam_nr_r
            previous_beam_nr_t = beam_nr_t
            previous_action_r = action_index_r
            previous_action_t = action_index_t

    # %% Save pickle and hdf5
    data_reward = {
        'R_log': R_log,
        'R_max': R_max_log,
        'R_min': R_min_log,
        'R_mean': R_mean_log,
        'action_log_r': action_log_r,
        'action_log_t': action_log_t,
        'beam_log_r': beam_log_r,
        'beam_log_t': beam_log_t
    }
    
    data_agent = {
    'Agent_r': Agent_r, # TODO skal måske ikke kun være Agent_r, men også Agent_t
    'Agent_t': Agent_t,
    'agent_settings_r': agent_settings_r,
    'agent_settings_t': agent_settings_t,
    'channel_settings': channel_settings,
    }

    try:
        if "NLOS" in channel_settings["scenarios"][0]:
            # helpers.dump_pickle(data_agent, 'Results/', f'{CASE}_NLOS_{RESULT_NAME}_results.pickle')
            helpers.dump_hdf5(data_reward, 'Results/', f'{CASE}_NLOS_{RESULT_NAME}_results.hdf5')
        else:
            # helpers.dump_pickle(data_agent, 'Results/', f'{CASE}_LOS_{RESULT_NAME}_results.pickle')
            helpers.dump_hdf5(data_reward, 'Results/', f'{CASE}_LOS_{RESULT_NAME}_results.hdf5')
    except OSError as e:
        print(e)
        print("Saving to root folder instead")
        if "NLOS" in channel_settings["scenarios"][0]:
            # helpers.dump_pickle(data_agent, '', f'{CASE}_NLOS_{RESULT_NAME}_results.pickle')
            helpers.dump_hdf5(data_reward, '', f'{CASE}_NLOS_{RESULT_NAME}_results.hdf5')
        else:
            # helpers.dump_pickle(data_agent, '', f'{CASE}_LOS_{RESULT_NAME}_results.pickle')
            helpers.dump_hdf5(data_reward, '', f'{CASE}_LOS_{RESULT_NAME}_results.hdf5')
            
    
            
            
            
