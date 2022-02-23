# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

# %% Imports
import os
import sys

import dill as pickle
import numpy as np
import scipy.io as scio
import oct2py

import classes
import plots


# %% Functions
def dump_pickle(data, path, filename):
    with open(path + filename, 'wb+') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(path_to_pickle, filename):
    with open(path_to_pickle + filename, 'rb') as f:
        data = pickle.load(f)

    return data


def steering_vectors2d(direction, theta, N, lambda_):
    """
    Calculates the steering vector for a Standard ULA, with given antenna positions
    wave length and angles.
    Can be used for both receivers and transmitters by changing the direction parameter.
    :param direction: -1 for impinging direction, 1 for propagation direction
    :param theta: Angles in radians
    :param r: Coordinate positions of antennas in meters.
    :param lambda_: Wave length of carrier wave in meters
    :return: Array steering vector
    """

    r = np.zeros((2, N))
    r[0, :] = np.linspace(0, (N - 1) * lambda_ / 2, N)

    if isinstance(theta, np.ndarray):
        e = direction * np.matrix([np.cos(theta), np.sin(theta)])
        result = np.exp(-2j * (np.pi / lambda_) * e.T @ r)
    elif isinstance(theta, float):
        e = direction * np.array([[np.cos(theta), np.sin(theta)]])
        result = np.exp(-2j * (np.pi / lambda_) * e @ r)
    else:
        raise Exception("Theta is not an array or an ")

    return result


def codebook(N, k, lambda_):
    """
    Calculates the codebook based on the number of antennae and number of layers
    The bottom layer always has 2 beams, and the number of beams in each layer
    increases with a factor of 2
    :param N: Number of antennae
    :param k: Number of layers
    :return: Codebook matrix
    """

    codeword = np.zeros(N, dtype=np.complex64)

    if np.ceil(np.log2(N)) != np.floor(np.log2(N)):
        raise Exception("N not a power of 2")

    N_codeword = 2 ** (k + 1) - 2

    codebook = np.zeros((N_codeword, N), dtype=np.complex64)

    for i in range(1, k + 1):
        l = np.log2(N) - i
        M = int(2 ** (np.floor((l + 1) / 2)))
        Ns = int(N / M)

        if l % 2:
            Na = M / 2
        else:
            Na = M

        for m in range(1, M + 1):
            if m <= Na:
                fm = np.exp(-1j * m * ((Ns - 1) / Ns) * np.pi) * steering_vectors2d(1, np.arccos(
                    (((-1 + ((2 * m - 1) / Ns)) + 1) % 2) - 1), Ns, lambda_)

            else:
                fm = np.zeros(Ns)

            codeword[(m - 1) * Ns:m * Ns] = fm

        codebook[(2 ** i) - 2, :] = codeword

        for n in range(2, 2 ** i + 1):
            codebook[(2 ** i) - 3 + n, :] = codeword * (
                    np.sqrt(N) * steering_vectors2d(1, np.arccos((((2 * (n - 1) / (2 ** i)) + 1) % 2) - 1), N,
                                                    lambda_))

    for idx, row in enumerate(codebook):
        codebook[idx, :] = row * 1 / np.linalg.norm(row)

    return codebook


def angle_to_beam(AoA, W):
    """
    Finds the beam with highest gain for each angle in a given Angle of Arrival vector.
    Parameters
    ----------
    AoA : Vector with Angles of Arrival
    W : Codebook of beamformers

    Returns
    -------

    """
    beam_tmp = np.zeros([len(W), 1])
    beam = np.zeros([len(AoA), 1])

    for i in range(len(AoA)):
        A = (1 / np.sqrt(len(W[0, :]))) * np.exp(-1j * np.pi * np.cos(AoA[i]) * np.arange(0, len(W[0, :])))
        for j in range(len(W)):
            # The gain is found by multiplying the code-page with the steering vector
            beam_tmp[j] = np.abs(np.conjugate(W[j, :]).T @ A)

        beam[i] = np.argmax(beam_tmp)
    return beam


def get_local_angle(AoA, Ori):
    """
    Transforms angles in global Quadriga coordinate system,
    to angles in local antenna array coordinate system.
    :param AoA: Angles in global coordinate system
    :param Ori: Orientation of the antenna array in global coordinate system
    :return: Angles in local coordinate system
    """
    # Calculate local AoA
    AoA_Local = (np.pi / 2 - Ori)[:, np.newaxis] + AoA

    # Wrap where needed
    AoA_Local[AoA_Local < -np.pi] += 2 * np.pi
    AoA_Local[AoA_Local > np.pi] -= 2 * np.pi

    return AoA_Local


def discrete_ori(Ori, N):
    """
    Maps continuous orientation angles to N discrete angle values
    Parameters
    ----------
    Ori : Continuous orientation angles
    N : Number of discrete angles to map to

    Returns
    -------
    Orientation angles in discrete values
    """
    angles = [((n + 1) * np.pi) / N for n in range(N - 1)]

    Ori_abs = np.abs(Ori)
    Ori_discrete = np.zeros(np.shape(Ori))

    for n in range(1, N - 1):
        Ori_discrete[np.logical_and(Ori_abs > angles[n - 1],
                                    Ori_abs <= angles[n])] = n

    Ori_discrete[Ori_abs > angles[-1:]] = N - 1

    return Ori_discrete


def discrete_angle(pos, N):
    """
    Maps continuous position angle to N discrete angle values.
    Used for discretizing the angle between the user terminal and the base station.
    Assumes the base station is placed in origo.
    Parameters
    ----------
    pos : position of user terminal
    N : number of discrete angles to map to

    Returns
    -------
    Discrete position angles of the user terminal relative to the base station
    """
    Angle = np.arctan2(pos[1, :], pos[0, :])
    # Angle: [0 deg, 360 deg] in radians
    Angle[Angle < 0] += 2 * np.pi

    # Discrete angles
    angles = [(((n + 1) * 2 * np.pi) / N) for n in range(N - 1)]

    Angle_discrete = np.zeros(np.shape(Angle))

    for n in range(1, N - 1):
        Angle_discrete[np.logical_and(Angle > angles[n - 1],
                                      Angle <= angles[n])] = n

    Angle_discrete[Angle > angles[-1:]] = N - 1

    return Angle_discrete


def discrete_dist(pos, N, r_lim):
    """
    Maps continuous distances to N discrete distances given by concentric circles.
    Used to discretize the distance between the user terminal and the base station.
    Assumes the base station is placed in origo.
    Parameters
    ----------
    pos : Continuous positions as x-, y-coordinates
    N : Number of discrete distances to map to.
    r_lim : Range of the base stations signals

    Returns
    -------

    """
    pos_norm = np.linalg.norm(pos[0:2, :], axis=0)
    base = int(r_lim / N)
    return (base * np.round(pos_norm / base)).astype(int)


def misalignment_prob(R_db, R_max_db, x_db):
    # Create zeros vector with shape of R
    tmp = np.zeros(np.shape(R_db))

    # All places where the R values is less
    # than R_max - x_db is set to 1.
    tmp[R_db < R_max_db - x_db] = 1

    # Return the x_db misalignment probability
    return np.mean(tmp)


def noisy_ori(ori_vector):
    # "smooting" factor in random walk filter
    K = 21
    new_orientation = np.empty_like(ori_vector)
    for idx, episode in enumerate(ori_vector):
        z_axis = episode[0][2]

        # generate the random walk
        a_bar = np.zeros(len(z_axis) + K)
        for i in range(len(z_axis) + K - 1):
            a_bar[i + 1] = a_bar[i] + np.random.normal(0, 0.5 * np.pi / 180)

        # generate the additive orientation noise as MA filtered random walk
        a = np.zeros(len(z_axis))
        for i in range(len(z_axis)):
            a[i] = np.sum(a_bar[i:i + K]) / K

        # Add the noise to original data and wrap angles to range [-pi:pi] for correct signs
        res_z_axis = z_axis + a
        for i in range(len(a)):
            while res_z_axis[i] > np.pi:
                res_z_axis[i] -= 2 * np.pi

            while res_z_axis[i] < -1 * np.pi:
                res_z_axis[i] += 2 * np.pi

        res = np.zeros((3, len(z_axis)))
        res[2] = res_z_axis
        new_orientation[idx, 0] = res

    return new_orientation


def create_pos_log(case, para, pos_log_name):
    [N, M, r_lim, sample_period, scenarios] = para

    print("Creating track")

    # Create the class
    track = classes.Track(case=case, delta_t=sample_period, r_lim=r_lim)

    pos_log_done = False
    while pos_log_done is False:
        # Create the tracks
        pos_log = []
        for m in range(M):
            pos_log.append(track.run(N))

        plots.positions(False, pos_log, r_lim)

        user_input = input("Does the created track(s) look fine (yes/no/stop)")
        if user_input.lower() == "yes":
            pos_log_done = True
        if user_input.lower() == "stop":
            sys.exit("Program stopped by user")

    print('track done')
    # Save the data
    scio.savemat("Data_sets/" + pos_log_name, {"pos_log": pos_log, "scenarios": scenarios})


def load_data(pos_log_name, data_name):
    """
    Loads parameters from earlier simulations.
    :param pos_log_name: Name of data file containing positions and scenarios eg: "data_pos.mat"
    :param data_name: Name of data file containing parameters/coefficients from simulations eg: "data.mat"
    :return:
    """

    # Load the data
    try:
        print("Loading data")
        pos_log = scio.loadmat("Data_sets/" + pos_log_name)
        pos_log = pos_log["pos_log"]

    except IOError:
        print(f"Datafile {pos_log_name} not found")
        sys.exit()

    try:
        tmp = scio.loadmat("Data_sets/" + data_name)
        tmp = tmp["output"]

    except IOError:
        print(f"Datafile {data_name} not found")
        sys.exit()

    return tmp, pos_log


def quadriga_simulation(multi_user, ENGINE, pos_log_name, data_name, para):
    """
    Generates parameters for the channel model.
    Parameters are generated from Quadriga simulations.
    Either a MATLAB or Octave engine is used to run simulations.

    :param multi_user: Bool to determine if simulation should use multiple user terminals
    :param ENGINE: Which engine to use for simulations. "MATLAB" or "Octave"
    :param pos_log_name: Name of data file containing positions and scenarios eg: "data_pos.mat"
    :param data_name: Name of data file containing parameters/coefficients from simulations eg: "data.mat"
    :param para: List of simulation settings/parameters used in the simulations
    :return:
    """
    [fc, N, M, r_lim, sample_period, scenarios] = para

    if ENGINE == "octave":
        try:
            from oct2py import octave

        except ModuleNotFoundError:
            raise

        except OSError:
            raise OSError("'octave-cli' hasn't been added to path environment")

        print("Creating new data - octave")

        # Add Quadriga folder to octave path
        octave.addpath(octave.genpath(f"{os.getcwd()}/Quadriga"))

        # Run the scenario to get the simulated channel parameters
        if multi_user:
            if octave.get_data_multi_user(fc, pos_log_name, data_name, ENGINE):
                try:
                    simulation_data = scio.loadmat("Data_sets/" + data_name)
                    simulation_data = simulation_data["output"]
                except FileNotFoundError:
                    raise FileNotFoundError(f"Data file {data_name} not loaded correctly")
            else:
                raise Exception("Something went wrong")
        else:
            if octave.get_data(fc, pos_log_name, data_name, ENGINE):
                try:
                    simulation_data = scio.loadmat("Data_sets/" + data_name)
                    simulation_data = simulation_data["output"]
                except FileNotFoundError:
                    raise FileNotFoundError(f"Data file {data_name} not loaded correctly")
            else:
                raise Exception("Something went wrong")

    elif ENGINE == "MATLAB":
        try:
            import matlab.engine
            print("Creating new data - MATLAB")

        except ModuleNotFoundError:
            raise Exception("You don't have matlab.engine installed")

        # start MATLAB engine
        eng = matlab.engine.start_matlab()

        # Add Quadriga folder to path
        eng.addpath(eng.genpath(f"{os.getcwd()}/Quadriga"))

        if multi_user:
            if eng.get_data_multi_user(fc, pos_log_name, data_name, ENGINE):
                try:
                    simulation_data = scio.loadmat("Data_sets/" + data_name)
                    simulation_data = simulation_data["output"]

                except FileNotFoundError:
                    raise FileNotFoundError(f"Data file {data_name} not loaded correctly")

            else:
                raise Exception("Something went wrong")
        else:
            if eng.get_data(fc, pos_log_name, data_name, ENGINE):
                try:
                    simulation_data = scio.loadmat("Data_sets/" + data_name)
                    simulation_data = simulation_data["output"]

                except FileNotFoundError:
                    raise FileNotFoundError(f"Data file {data_name} not loaded correctly")
            else:
                raise Exception("Something went wrong")

        eng.quit()

    else:
        raise Exception("ENGINE name is incorrect")

    return simulation_data
