# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

# %% Imports
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

import classes


# %% Functions
def steering_vectors2d(direction, theta, r, lambda_):
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
    e = direction * np.matrix([np.cos(theta), np.sin(theta)])
    return np.exp(-2j * (np.pi / lambda_) * e.T @ r)


def plot_directivity(W, N, title):
    """
    Plots a directivity plot for a codebook
    :param W: Codebook
    :param N: Resolution of angles
    :param title: Title put on the figure
    :return: Nothing
    """
    # Calculate the directivity for a page in DFT-codebook
    beam = np.zeros((len(W), N))
    Theta = np.linspace(0, np.pi, N)
    # Sweep over range of angles, to calculate the normalized gain at each angle
    for i in range(N):
        # Hardcode the array steering vector for a ULA with len(W) elements
        A = (1 / np.sqrt(len(W[0, :]))) * np.exp(-1j * np.pi * np.cos(Theta[i]) * np.arange(0, len(W[0, :])))
        for j in range(len(W)):
            # The gain is found by multiplying the code-page with the steering vector
            beam[j, i] = np.abs(np.conjugate(W[j, :]).T @ A)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title(title)
    for j in range(len(W)):
        # Calculate the angle with max gain for each code-page.
        max_angle = np.pi - np.arccos(np.angle(W[j, 1]) / np.pi)

        # Plot the gain
        ax.plot(Theta, beam[j, :], label=f"{j}")
        ax.vlines(max_angle, 0, np.max(beam[j, :]),
                  colors='r', linestyles="dashed",
                  alpha=0.4)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=4)
    plt.show()


def angle_to_beam(AoA, W):
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


def disrete_ori(Ori, N):
    angles = [((n + 1) * np.pi) / N for n in range(N - 1)]

    Ori_abs = np.abs(Ori)
    Ori_discrete = np.zeros(np.shape(Ori))

    for n in range(1, N - 1):
        Ori_discrete[np.logical_and(Ori_abs > angles[n - 1],
                                    Ori_abs <= angles[n])] = n

    Ori_discrete[Ori_abs > angles[-1:]] = N - 1

    return Ori_discrete


def get_data(RUN, ENGINE, pos_log_name, data_name, para):
    """
    Generates parameters for the channel model.
    Parameters are either loaded from earlier simulations,
    or generated from Quadriga simulations.
    Either a MATLAB or Octave engine is used to run simulations.
    :param RUN: Bool to determine if load from files or run simulation
    :param ENGINE: Which engine to use for simulations. "MATLAB" or "Octave"
    :param pos_log_name: Name of data file containing positions and scenarios eg: "data_pos.mat"
    :param data_name: Name of data file containing parameters/coefficients from simulations eg: "data.mat"
    :param para: List of simulation settings/parameters used in the simulations
    :return:
    """
    [fc, N, M, r_lim, stepsize, scenarios] = para

    # Load the data
    if not RUN:
        try:
            print("Loading data")
            pos_log = scio.loadmat("Data_sets/"+pos_log_name)
            pos_log = pos_log["pos_log"]

        except IOError:
            RUN = True
            print(f"Datafile {pos_log_name} not found")

        try:
            tmp = scio.loadmat("Data_sets/"+data_name)
            tmp = tmp["output"]

        except IOError:
            RUN = True
            print(f"Datafile {data_name} not found")

    if RUN:
        print("Creating track")
        # Create the class
        track = classes.Track(r_lim, stepsize)

        # Create the tracks
        pos_log = []
        for m in range(M):
            pos_log.append(track.run(N))

        print('track done')
        # Save the data
        scio.savemat("Data_sets/"+pos_log_name, {"pos_log": pos_log, "scenarios": scenarios})

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
            if octave.get_data(fc, pos_log_name, data_name, ENGINE):
                try:
                    tmp = scio.loadmat("Data_sets/"+data_name)
                    tmp = tmp["output"]
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
            if eng.get_data(fc, pos_log_name, data_name, ENGINE):
                try:
                    tmp = scio.loadmat("Data_sets/"+data_name)
                    tmp = tmp["output"]

                except FileNotFoundError:
                    raise FileNotFoundError(f"Data file {data_name} not loaded correctly")

            else:
                raise Exception("Something went wrong")

            eng.quit()

        else:
            raise Exception("ENGINE name is incorrect")

    return tmp, pos_log
