# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""
# %% Imports
from collections import defaultdict
import numpy as np
from tqdm import tqdm

import helpers


# %% Track
class Track():
    def __init__(self, case, delta_t, r_lim):
        """
        Initialize an instance of the track class, which creates implement the
        mobility model from 'Smooth is better than sharp: a random mobility
        model for simulation of wireless networks'

        Parameters
        ----------
        case : String
            File containing a mobility pattern to generate paths from, ie. 'car_highway', 'car_urban' and 'pedestrian'
        delta_t : Float
            Sample period for position, etc.
        r_lim : Float
            Maximum radius of the cell, which the moving unit will restricted to stay within

        Returns
        -------
        None.

        """
        self.delta_t = delta_t
        self.env = case["environment"]
        self.vpref = case["vpref"]
        self.vmax = case["vmax"]
        self.vmin = case["vmin"]
        self.pvpref = case["pvpref"]
        self.pvuni = 1 - np.sum(case["pvpref"])
        self.pvchange = self.delta_t / case["vchange"]
        self.pdirchange = self.delta_t / case["dirchange"]
        self.pdirchange_stop = case["stop_dirchange"]
        self.mu_s = case["static_friction"]
        self.acc_max = case["acc_max"]
        self.dec_max = case["dec_max"]
        self.ctmax = case["curvetime"]["max"]
        self.ctmin = case["curvetime"]["min"]

        self.v_target = 0
        self.a = 0

        self.curve_time = 0
        self.curve_dt = 0
        self.delta_phi = 0
        self.v_stop = False
        self.vrmax = 0
        self.curve_slow = 0

        self.radius_limit = r_lim

    def set_acceleration(self, acc):
        """
        Calculate a velocity to accelerate/decelerate to the target velocity
        with

        Parameters
        ----------
        acc : Bool
            True if acceleration should occur otherwise False for deceleration

        Returns
        -------
        Float
            Acceleration/decelration value

        """
        if acc:
            return np.random.rand() * self.acc_max + 0.00001
        return - (np.random.rand() * self.dec_max + 0.00001)

    def change_velocity(self):
        """
        Calculates wheter a velocity change event should take place, and if
        what the new target speed is

        Returns
        -------
        Float
            Target velocity

        """
        p_uni = np.random.rand()
        p_pref = self.pvpref[0]
        l_pref = len(self.pvpref)

        # Checks if a pref. velocity should be chosen
        if p_uni < p_pref:
            return self.vpref[0]

        for i in range(1, l_pref):
            p_pref += self.pvpref[i]
            if (p_uni > p_pref - self.pvpref[i]) and (p_uni < p_pref):
                return self.vpref[i]

        # Return a velocity from a uniform dist. between set min and max
        return np.random.rand() * (self.vmax - self.vmin) + self.vmin

    def update_velocity(self, v):
        """
        Calculate the velocity the mobile unit should move at

        Parameters
        ----------
        v : Float
            Mobile terminals current velocity

        Returns
        -------
        v : Float
            The velocity the mobile terminal should move at

        """
        if np.random.rand() < self.pvchange:
            self.v_target = self.change_velocity()

            # Get an accelation / deccelation
            if self.v_target > v:
                self.a = self.set_acceleration(True)
            elif self.v_target < v:
                self.a = self.set_acceleration(False)
            else:
                self.a = 0

        # Update the velocity bases on target and accelation
        v = v + self.a * self.delta_t

        if (((self.a > 0) and (v > self.v_target)) or
                ((self.a < 0) and (v < self.v_target))):
            v = self.v_target
            self.a = 0

        return v

    def update_direction(self, phi, v):
        """
        Find the direction the mobile unit should point in

        Parameters
        ----------
        phi : Float
            Current direction
        v : Float
            Speed

        Returns
        -------
        phi : Float
            Direction after having rotated

        """
        # "Stop-turn-and-go" implemented here
        if v == 0:
            # Only changes the target delta phi once
            if not self.v_stop:
                if np.random.rand() < self.pdirchange_stop:
                    if np.random.rand() < 0.5:
                        delta_phi_target = np.pi / 2
                    else:
                        delta_phi_target = -np.pi / 2
                else:
                    delta_phi_target = 0

                # Calculat the number of time step the change in direction needs
                self.curve_time = np.floor((np.random.rand() * (self.ctmax - self.ctmin) + self.ctmin) / self.delta_t)

                # Resets the tracker
                self.curve_dt = 0

                # Calculate the delta direction change per time step
                self.delta_phi = delta_phi_target / self.curve_time

                self.v_stop = True

        else:
            self.v_stop = False

            # Change target delta_phi, while the user is moving
            if np.random.rand() < self.pdirchange:
                # Calculat the number of time step the change in direction needs
                self.curve_time = np.floor((np.random.rand() * (self.ctmax - self.ctmin) + self.ctmin) / self.delta_t)

                # Resets the tracker
                self.curve_dt = 0

                # Target direction change
                delta_phi_target = (np.random.rand() * 2 * np.pi - np.pi)

                # Calculate the delta direction change per time step
                self.delta_phi = delta_phi_target / self.curve_time

                # Calculate the maximum radius
                rc = self.v_target * self.curve_time * self.delta_t / np.abs(delta_phi_target)

                # Calculate the maximum velocity which can be taken
                self.vrmax = np.sqrt(self.mu_s * 9.81 * rc)

                if self.v_target > self.vrmax:
                    self.v_target = self.vrmax

                if v > self.vrmax:
                    self.a = self.set_acceleration(False)

                    self.curve_slow = np.ceil(((v - self.vrmax) / np.abs(self.a)) / self.delta_t)
                else:
                    self.curve_slow = 0

            # Updates the direction based on the target delta phi
            if self.curve_dt < self.curve_time + self.curve_slow:
                if self.curve_dt >= self.curve_slow:
                    phi = phi + self.delta_phi

                    # Checks for overflow
                    phi = self.angle_overflow(phi)

                self.curve_dt += 1

        return phi

    def update_pos(self, pos, v, phi):
        """
        Finds the next position by moving at speed v in direction phi

        Parameters
        ----------
        pos : Array
            x and y values of current position
        v : Float
            The speed at which the mobile unit moves
        phi : Float
            The direction the mobile unit moves

        Returns
        -------
        pos : Array
            x and y positions after having moved

        """
        # x-axis
        pos[0] = pos[0] + np.cos(phi) * v * self.delta_t

        # y-axis
        pos[1] = pos[1] + np.sin(phi) * v * self.delta_t

        return pos

    def angle_overflow(self, phi):
        """
        Restricts the angle from -pi to pi, by wraping around this interval

        Parameters
        ----------
        phi : Float
            Angle to be checked

        Returns
        -------
        phi : Float
            Angle in the interval (-pi;pi)

        """
        # Checks for overflow
        if phi > np.pi:
            phi -= 2 * np.pi
        if phi < -np.pi:
            phi += 2 * np.pi

        return phi

    def initialise_run(self):
        """
        Initialize parameters for the mobility model

        Returns
        -------
        v : Float
            Speed which the mobile unit starts with
        phi : Float
            Angle which the mobile unit points at the start
        pos : Array
            x and y positions the track starts at

        """
        # Velocity
        self.v_target = self.change_velocity()
        v = self.v_target

        # Position
        if self.env.lower() == "urban":
            pos = np.random.uniform(-self.radius_limit * np.sqrt(2), self.radius_limit * np.sqrt(2), size=2)

        elif self.env.lower() == "highway":
            # Choose a start position on the edge based on a random chosen angle
            egde_angle = (np.random.rand() * 2 * np.pi - np.pi)
            pos = self.radius_limit * np.array([np.cos(egde_angle), np.sin(egde_angle)])

        else:
            pos = np.array([0, 0])

        # Direction
        if self.env.lower() == "urban":
            phi = np.random.rand() * 2 * np.pi - np.pi

        elif self.env.lower() == "highway":
            # Limit the start direction so it does not go out of the circle at the start

            # Get the angle which points at the center
            dir_center = egde_angle + np.pi

            # Checks for overflow
            dir_center = self.angle_overflow(dir_center)

            # Draw from a uniform distribution around the center angle
            edge_max = np.pi / 6
            edge_min = -np.pi / 6
            phi = dir_center + np.random.rand() * (edge_max - edge_min) + edge_min

            # Checks for overflow
            phi = self.angle_overflow(phi)

        else:
            phi = 0

        return v, phi, pos

    def run(self, N):
        """
        Generate a single track of a certain number of steps

        Parameters
        ----------
        N : Int
            Number of steps, ie. descrete positions, the track consists of

        Returns
        -------
        pos : Array
            Array of positions in cartesian coordinates

        """
        # Create a empty array for the velocities
        v = np.zeros([N])
        phi = np.zeros([N])
        pos = np.zeros([3, N])
        pos[2, :] = 1.5

        # Get start values
        v[0], phi[0], pos[0:2, 0] = self.initialise_run()

        # Start running the "simulation"
        t = 1
        i = 0
        while (t < N):
            pos[0:2, t] = self.update_pos(pos[0:2, t - 1], v[t - 1], phi[t - 1])
            if np.linalg.norm(pos[0:2, t]) > self.radius_limit:
                # Restarts the run
                print(f'number of tries: {i}')
                print(f'How far we got: {t}')

                t = 1
                i += 1

                # Start with new values
                v[0], phi[0], pos[0:2, 0] = self.initialise_run()

            else:
                v[t] = self.update_velocity(v[t - 1])
                phi[t] = self.update_direction(phi[t - 1], v[t])
                t += 1

        return pos


# %% Environment Class
class Environment():
    def __init__(self, W, F, Nt, Nr,
                 fc, P_t):
        """
        Initialize the propagation environment

        Parameters
        ----------
        W : Matrix
            Combiner codebook
        F : Matrix
            Precoder codebook
        Nt : Int
            Numbers of transmitter antennas
        Nr : Int
            Number of receiver antennas
        fc : Float
            Center frequency of the signal being sent
        P_t : Float
            Transmission power

        Returns
        -------
        None.

        """
        self.AoA = 0
        self.AoD = 0
        self.Betas = 0
        self.W = W
        self.F = F
        self.Nt = Nt
        self.Nr = Nr
        self.lambda_ = 3e8 / fc
        self.P_t = P_t
        self.reward_matrix = 0

    def create_reward_matrix(self):
        episodes = len(self.AoA)
        total_steps = len(self.AoA[0])
        combiner_length = len(self.W[:, 0])
        precoder_length = len(self.F[:, 0])

        R = np.zeros((episodes, total_steps, precoder_length, combiner_length))
        for path_idx in range(episodes):
            for stepnr in tqdm(range(total_steps), desc="Steps"):
                Beta = self.Betas[path_idx][0][stepnr, :]

                # Calculate steering vectors for transmitter and receiver
                alpha_rx = helpers.steering_vectors2d(direction=-1, theta=self.AoA[path_idx][stepnr, :],
                                                      N=self.Nr, lambda_=self.lambda_)
                alpha_tx = helpers.steering_vectors2d(direction=1, theta=self.AoD[path_idx][0][stepnr, :],
                                                      N=self.Nt, lambda_=self.lambda_)

                alpha_rx = alpha_rx.reshape((len(alpha_rx), 1, self.Nr))
                alpha_tx = alpha_tx.reshape((len(alpha_tx), 1, self.Nt))

                H = helpers.get_H(
                    self.Nr,
                    self.Nt,
                    Beta,
                    alpha_rx,
                    alpha_tx)


                R[path_idx, stepnr] = helpers.jit_reward(self.W, self.F, H, self.P_t)
                
        self.reward_matrix = R

    def take_action(self, path_idx, stepnr, beam_nr_r, beam_nr_t, p_n = 0):
        """
        Calculates the reward (signal strength) maximum achievable reward,
        minimum achievable reward and average reward based on an action

        Parameters
        ----------
        path_idx : Int
            Path number from data set
        stepnr : Int
            Time step number
        beam_nr : Tuple of ints
            Beam numbers for receiver and transmitter respectively

        Returns
        -------
        reward : Int
            The reward for the chosen action
        max_reward : Int
            The maximum achievable reward
        min_reward : Int
            The minimum achievable reward
        mean_reward : Int
            The average reward

        """
        R = self.reward_matrix[path_idx, stepnr]
        n = np.random.normal(0,p_n/2,R.shape) + 1j * np.random.normal(0,p_n/2,R.shape)
        R = np.abs(np.sqrt(R)+n)**2
        return R[beam_nr_t, beam_nr_r], np.max(R), np.min(R), np.mean(R)

    def update_data(self, AoA, AoD, Betas):
        """
        Updates the environment information

        Parameters
        ----------
        AoA : TYPE
            Angle of arrival
        AoD : TYPE
            Angle of depature
        Betas : TYPE
            Channel parameters

        Returns
        -------
        None.

        """
        self.AoA = AoA
        self.AoD = AoD
        self.Betas = Betas


# %% State Class
class State:
    def __init__(self, intial_state, orientation_flag=False, distance_flag=False, location_flag=False,
                 r_hist_flag=False, t_hist_flag=False):
        """
        Initiate the state object, containing the current state

        Parameters
        ----------
        intial_state : Array
            The form the state takes
        orientation_flag : Bool, optional
            Whether orientation is included in the state. The default is False.
        distance_flag : Bool, optional
            Whether distance is included in the state. The default is False.
        location_flag : Bool, optional
            Whether location is included in the state. The default is False.

        Returns
        -------
        None.

        """
        self.state = intial_state
        self.orientation_flag = orientation_flag
        self.distance_flag = distance_flag
        self.location_flag = location_flag
        self.r_hist_flag = r_hist_flag
        self.t_hist_flag = t_hist_flag

    def build_state(self, beam_nr, para=[None, None, None], retning=None):
        """
        Builds a state object, in accordance with what is included in the state

        Parameters
        ----------
        action : Int
            The action for inclusion in the state.
        para : Arra, optional
            Distance, orientaion and angle for possible inclusion in the state. The default is [None, None, None].
        retning : Int, optional
            Retning of the chosen action for possible inclusion in the state. The default is None.

        Returns
        -------
        list
            A state

        """
        dist, ori, angle = para
        beam_r = beam_nr[0]
        beam_t = beam_nr[1]
        retning_r = retning[0]
        retning_t = retning[1]

        if self.r_hist_flag:
            if len(self.state[0]) > 1:
                if retning is not None:
                    state_r = self.state[0][1:-1]
                    state_r.append(retning_r)
                else:
                    state_r = self.state[0][1:]
                state_r.append(beam_r)
            else:
                state_r = self.state[0]
                state_r.pop()
                state_r.append(beam_r)
        else:
            state_r = ["N/A"]

        if self.t_hist_flag:
            if len(self.state[1]) > 1:
                if retning is not None:
                    state_t = self.state[1][1:-1]
                    state_t.append(retning_t)
                else:
                    state_t = self.state[1][1:]
                state_t.append(beam_t)
            else:
                state_t = self.state[1]
                state_t.pop()
                state_t.append(beam_t)
        else:
            state_t = ["N/A"]

        if self.distance_flag or self.location_flag:
            state_d = [dist]
        else:
            state_d = ["N/A"]

        if self.orientation_flag:
            state_o = self.state[3][1:]
            state_o.append(ori)
        else:
            state_o = ["N/A"]

        if self.location_flag:
            state_deg = [angle]
        else:
            state_deg = ["N/A"]

        return [state_r, state_t, state_d, state_o, state_deg]

    def get_state_parameters(self, path_idx, step_idx, ori_data, dist_data, angle_data):
        if self.orientation_flag:
            # Get the current discrete orientation of the user terminal
            ori = int(ori_data[path_idx, step_idx])
        else:
            ori = "N/A"

        if self.distance_flag or self.location_flag:
            dist = dist_data[path_idx, step_idx]
        else:
            dist = "N/A"

        if self.location_flag:
            angle = angle_data[path_idx, step_idx]
        else:
            angle = "N/A"

        return [dist, ori, angle]
