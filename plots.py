# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

# %% Imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as plticker
import operator as o
import os

import seaborn as sns

import numpy as np


# %% Functions

def barplot(save, data_arrays, span, labels):
    '''
    Create a barchart for data across different categories with
    multiple conditions for each category.

    @param ax: The plotting axes from matplotlib.
    @param dpoints: The data set as an (n, 3) numpy array
    '''
    N = len(data_arrays)
    M = len(data_arrays[0])
    width = 1/(1+N)
    ind = np.arange(M)
    
    for dataset in range(N):
        xvals = data_arrays[dataset]
        bar = plt.bar(ind+width*dataset, xvals, width, label=labels[dataset])
        
    
    
    
    plt.xticks(ind+width, span)
    axes = plt.gca()
    y_min, y_max = axes.get_ylim()

    if len(span) > 6:
        x_offset = 1-(2*(1/(N+1)))+width
        
        plt.vlines(1+x_offset,0,y_max, colors='black', linestyles = 'dashed', label = 'Codepage seperator')
        plt.vlines(5+x_offset,0,y_max, colors='black', linestyles = 'dashed')
        
        plt.xlabel("Codeword number")
    else:
        plt.xlabel("Action")
        
    if len(span) > 14:
        plt.vlines(13+x_offset,0,y_max, colors='black', linestyles = 'dashed')
        
    
    # plt.title("Average misalignment for different action histories")
    plt.ylabel("Relative frequency")
    plt.legend()
    
    if save == True:
        plt.savefig("Figures/Barplot.pdf")
    plt.show()
    

def stability(save, data, average_window):
    """
    Plots the 'stability' for each episode, by first calculating a low-pass
    version of the signals, then using this as the mean at each point to
    calculate variance
    :param save: Whether to save or only show in IDE
    :param data: Data stability is found for
    :param average_window: Low-pass filtering is perforned by averaginge over a
                           window of this length
    :return: Nothing
    """
    est_mean = np.zeros((data.shape[0], data.shape[1]))

    # Adds copies of the last column to the end of the matrix, and the first column to the start
    padded_data = np.append(data, np.tile(data[:, -1], (int(np.ceil((average_window - 1) / 2)), 1)).T, axis=1)
    padded_data = np.append(np.tile(data[:, 0], (int(np.floor((average_window - 1) / 2)), 1)).T, padded_data, axis=1)

    # Calculate the mean at each point for all episodes
    for step in range(data.shape[1]):
        est_mean[:, step] = np.mean(padded_data[:, step:(step + average_window)], axis=1)

    # Calculate the sample variance for each episode and return vector of variances.
    stability = np.mean((data - est_mean) ** 2, axis=1)

    plt.figure()
    plt.title(f"Stability of episodes (Window size: {average_window} samples)")
    plt.plot(stability)
    plt.xlabel("Episode")
    plt.ylabel("Stability")
    if save == True:
        plt.savefig("Figures/Stability.pdf")
    plt.show()

def ECDF_bulk(save, data, span):
    return None

def ECDF(save, data, sections):

    """
    Plots the emperical cumulative distribution function for the x-db
    misalignment
    :param save: Whether to save or only show in IDE
    :param data: Data the misalignment should be found from
    :param sections: number of ECDFS to render
    :return: Nothing
    """
    
    data_len = data.shape[0]
    section_size = int(np.floor(data_len/sections))
    

    fig, ax = plt.subplots()
    ax.yaxis.tick_right()
    plt.axvline(-6, linestyle='--', color='black', label='-6 dB')
    plt.axvline(-3, linestyle='-.', color='black', label='-3 dB')
    plt.title("x-dB Mis-alignment probability - ECDF")
    for i in range(sections):

        data_section = data[i*section_size:(i+1)*section_size, :].flatten()
        sns.ecdfplot(data_section, label=f'{i*section_size} - {(i+1)*section_size}')

    plt.legend()
    loc = plticker.MultipleLocator(base=0.05)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.xlabel('Misalignment in dB')
    plt.ylabel("Probability")
    if save == True:
        plt.savefig("Figures/ECDF.pdf")
    plt.show()

def Relative_reward(save, mis_data, mis_mean, mis_min):
    """
    Plots the three normalized vectors of points
    :param save: Whether to save or only show in IDE
    :param mis_data: Vector of normalized points with respect to max
    :param mis_mean: Vector of normalized mean points with respect to max
    :param mis_min: Vector of normalized min points with respect to max
    :return: Nothing
    """
    plt.figure()
    plt.title("Relative difference - dB")
    plt.plot(mis_min, label="R_min")
    plt.plot(mis_data, label="R")
    plt.plot(mis_mean, label="R_mean")
    plt.legend()
    plt.xlabel("Number of Steps")
    plt.ylabel("Normalized reward")
    if save == True:
        plt.savefig("Figures/Relative_reward.pdf")
    plt.show()


def mean_reward(save, y1, y2, y3, y4, labels, title,
                x1=None, x2=None, x3=None, x4=None, db=False):
    """
    Plots y1 through y4 on a single plot, meant for
    plotting received reward vs max mean and min reward
    :param save: Whether to save or only show in IDE
    :param y1 - y4: Vectors of points, used as y values
    :param labels: Labels for each respective vector of points from y1 - y4
    :param title: Title put on the figure
    :param x1 - x4: Only use if x shouldn't default to positive whole numbers
    :return: Nothing
    """
    if x1 is None:
        x1 = np.arange(len(y1[0, :]))
    if x2 is None:
        x2 = np.arange(len(y2[0, :]))
    if x3 is None:
        x3 = np.arange(len(y3[0, :]))
    if x4 is None:
        x4 = np.arange(len(y4[0, :]))

    plt.figure()
    plt.title(title + f" - {len(y1)} Episodes")

    plt.plot(x1, np.mean(y1, axis=0), label=labels[0])
    plt.plot(x2, np.mean(y2, axis=0), label=labels[1])
    plt.plot(x3, np.mean(y3, axis=0), label=labels[2])
    plt.plot(x4, np.mean(y4, axis=0), label=labels[3])
    plt.legend()
    plt.xlabel("Number of Steps")
    plt.ylabel("Mean Reward")
    if db is False:
        plt.yscale('log')
    if save == True:
        plt.savefig(f"Figures/{title}.pdf")
    plt.show()


def directivity(save, W, N, title):
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
    ax.set_yticklabels([])
    ax.set_title(title)
    for j in range(len(W)):
        # Plot the gain
        ax.plot(Theta, beam[j, :], label=f"{j}")

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=4)
    if save == True:
        plt.savefig("Figures/Directivity.pdf")
    plt.show()


def positions(save, pos_log, r_lim):
    """
    Plots the paths given by the poslog as lines, and plots the circle
    containing these paths

    Parameters
    ----------
    save : Bool
        Whether to save the generated figure as pdf in the Figures folder
    pos_log : MATRIX
        Matrix of positions with rows being episodes and columns being time steps
    r_lim : FLOAT/INT
        The radius of the circle which the paths are restricted to

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots()
    # ax.set_title("Mobility traces")pos
    ax.set_ylabel("Distance from transmitter [m]")
    ax.set_xlabel("Distance from transmitter [m]")
    ax.add_patch(plt.Circle((0, 0), r_lim, color='r', alpha=0.1))

    for m in range(len(pos_log)):
        ax.plot(pos_log[m][0, :], pos_log[m][1, :], label=f"{m}")

    ax.set_xlim([-r_lim, r_lim])
    ax.set_ylim([-r_lim, r_lim])
    ax.plot(0, 0, 'X', label="Transmitter")
    # if len(pos_log) < 10:
        # plt.legend()

    ax.set_aspect('equal', adjustable='box')

    if save == True:
        plt.savefig("Figures/Paths.pdf")
    plt.show()
