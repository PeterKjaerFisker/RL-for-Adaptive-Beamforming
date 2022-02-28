# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""
# %% Imports
from collections import defaultdict
import numpy as np

import helpers


# %% Track
class Track():
    def __init__(self, case, delta_t, r_lim):
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
        if acc:
            return np.random.rand() * self.acc_max + 0.00001
        return - (np.random.rand() * self.dec_max + 0.00001)

    def change_velocity(self):
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
        # x-axis
        pos[0] = pos[0] + np.cos(phi) * v * self.delta_t

        # y-axis
        pos[1] = pos[1] + np.sin(phi) * v * self.delta_t

        return pos

    def angle_overflow(self, phi):
        # Checks for overflow
        if phi > np.pi:
            phi -= 2 * np.pi
        if phi < -np.pi:
            phi += 2 * np.pi

        return phi

    def initialise_run(self):
        # Velocity
        self.v_target = self.change_velocity()
        v = self.v_target

        # Position
        if self.env.lower() == "urban":
            pos = np.random.uniform(-self.radius_limit*np.sqrt(2), self.radius_limit*np.sqrt(2), size=2)

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
        self.AoA = 0
        self.AoD = 0
        self.Beta = 0
        self.W = W
        self.F = F
        self.Nt = Nt
        self.Nr = Nr
        self.lambda_ = 3e8 / fc
        self.P_t = P_t

    def _get_reward(self, stepnr, action):
        # Calculate steering vectors for transmitter and receiver
        alpha_rx = helpers.steering_vectors2d(direction=-1, theta=self.AoA[stepnr, :],
                                              N=self.Nr, lambda_=self.lambda_)
        alpha_tx = helpers.steering_vectors2d(direction=1, theta=self.AoD[stepnr, :],
                                              N=self.Nt, lambda_=self.lambda_)

        # Calculate channel matrix H
        H = np.zeros((self.Nr, self.Nt), dtype=np.complex128)
        for i in range(len(self.Beta[stepnr, :])):
            H += self.Beta[stepnr, i] * (alpha_rx[i].T @ np.conjugate(alpha_tx[i]))
        H = H * np.sqrt(self.Nr * self.Nt)

        # Calculate the reward
        R = np.zeros([len(self.F[:, 0]), len(self.W[:, 0])])
        for p in range(len(self.F[:, 0])):
            for q in range(len(self.W[:, 0])):
                R[p, q] = np.linalg.norm(np.sqrt(self.P_t) * np.conjugate(self.W[q, :]).T
                                         @ H @ self.F[p, :]) ** 2

        return np.max(R[:, action]), np.max(R), np.min(np.max(R, axis=0)), np.mean(np.max(R, axis=0))

    def take_action(self, stepnr, action):
        reward, max_reward, min_reward, mean_reward = self._get_reward(stepnr, action)

        return reward, max_reward, min_reward, mean_reward

    def update_data(self, AoA, AoD, Beta):
        self.AoA = AoA
        self.AoD = AoD
        self.Beta = Beta


# %% State Class
class State:
    def __init__(self, intial_state, orientation_flag=False, distance_flag=False, location_flag=False):
        self.state = intial_state
        self.orientation_flag = orientation_flag
        self.distance_flag = distance_flag
        self.location_flag = location_flag

    def build_state(self, action, para=[None, None, None], retning=None):
        dist, ori, angle = para

        if retning is not None:
            state_a = self.state[0][1:-1]
            state_a.append(retning)
        else:
            state_a = self.state[0][1:]

        state_a.append(action)

        if self.distance_flag or self.location_flag:
            state_d = [dist]
        else:
            state_d = ["N/A"]

        if self.orientation_flag:
            state_o = self.state[2][1:]
            state_o.append(ori)
        else:
            state_o = ["N/A"]

        if self.location_flag:
            state_deg = [angle]
        else:
            state_deg = ["N/A"]

        return [state_a, state_d, state_o, state_deg]


# %% Agent Class
class Agent:
    def __init__(self, action_space, alpha=["constant", 0.7], eps=0.1, gamma=0.7, c=200):
        self.action_space = action_space  # Number of beam directions
        self.alpha_start = alpha[1]
        self.alpha_method = alpha[0]
        self.alpha = defaultdict(self._initiate_dict(alpha[1]))
        self.eps = eps
        self.gamma = gamma
        self.c = c
        self.Q = defaultdict(self._initiate_dict(0.001))
        self.accuracy = np.zeros(1)

    def _initiate_dict(self, value1, value2=0):
        """
        Small function used when initiating the dicts.
        For the alpha dict, value1 is alphas starting value.
        Value2 should be set to 0 as it is used to log the number of times it has been used.

        Parameters
        ----------
        value1 : FLOAT
            First value in the array.
        value2 : FLOAT, optional
            Second value in the array. The default is 0.

        Returns
        -------
        TYPE
            An iterative type which defaultdict can use to set starting values.

        """
        return lambda: [value1, value2]

    def _update_alpha(self, state, action):
        """
        Updates the alpha values if method "1/n" has been chosen

        Parameters
        ----------
        state : ARRAY
            Current position (x,y).
        action : INT
            Current action taken.

        Returns
        -------
        None.

        """
        if self.alpha_method == "1/n":
            if self.alpha[state, action][1] == 0:
                self.alpha[state, action] = [self.alpha_start * (1 / 1),
                                             1 + self.alpha[state, action][1]]
            else:
                self.alpha[state, action] = [self.alpha_start * (1 / self.alpha[state, action][1]),
                                             1 + self.alpha[state, action][1]]

    def greedy(self, state):
        """
        Calculate which action is expected to be the most optimum.

        Parameters
        ----------
        state : ARRAY
            The current state

        Returns
        -------
        INT
            The chosen action.

        """
        beam_dir = np.random.choice(self.action_space)
        r_est = self.Q[state, beam_dir][0]

        for action in self.action_space:
            if self.Q[state, action][0] > r_est:
                beam_dir = action
                r_est = self.Q[state, action][0]

        return beam_dir

    def e_greedy(self, state):
        """
        Return a random action in the action space based on the epsilon value.
        Else return the same value as the greedy function

        Parameters
        ----------
        state : ARRAY
            Which position on the grid you are standing on (x,y).

        Returns
        -------
        INT
            The chosen action.

        """
        if np.random.random() > self.eps:
            return self.greedy(state)
        else:
            return np.random.choice(self.action_space)

    def get_action_list_adj(self, last_action, Nlayers):
        current_layer = int(np.floor(np.log2(last_action + 2)))  # Wrap around for left/right

        # Check if the codeword to the "right" is still in the same layer
        if current_layer == int(np.floor(np.log2(last_action + 3))):
            action_Right = last_action + 1
        else:
            action_Right = int((last_action - 1) / 2)

        # Check if the codeword to the "left" is still in the same layer
        if current_layer == int(np.floor(np.log2(last_action + 1))):
            action_Left = last_action - 1
        else:
            action_Left = int((last_action * 2) + 1)

        # Limits the agent to taking appropriate actions
        actions = [self.action_space[last_action],  # Stay
                   self.action_space[action_Right],  # Right
                   self.action_space[action_Left]]  # Left

        # Check if current layer is between the bottom and top layers
        if current_layer != 1 and current_layer != Nlayers:
            actions.append(self.action_space[int(np.floor((last_action - 2) / 2))])  # Down
            actions.append(self.action_space[int((last_action * 2) + 3)])  # Up Right
            actions.append(self.action_space[int((last_action * 2) + 2)])  # Up Left

            dir_list = [0, 1, 2, 3, 4, 5]

        elif current_layer != 1:  # Check if on bottom layer
            actions.append(self.action_space[int(np.floor((last_action - 2) / 2))])  # Down

            dir_list = [0, 1, 2, 3]

        else:  # Check if on uppermost layer
            actions.append(self.action_space[int((last_action * 2) + 3)])  # Up Right
            actions.append(self.action_space[int((last_action * 2) + 2)])  # Up Left

            dir_list = [0, 1, 2, 4, 5]

        return actions, dir_list

    def greedy_adj(self, state, last_action, Nlayers):

        actions, dir_list = self.get_action_list_adj(last_action, Nlayers)

        choice = np.random.randint(0, len(dir_list))
        next_action = actions[choice]
        next_dir = dir_list[choice]
        r_est = self.Q[state, next_action][0]

        for idx, last_action in enumerate(actions):
            if self.Q[state, last_action][0] > r_est:
                next_action = last_action
                next_dir = dir_list[choice]
                r_est = self.Q[state, last_action][0]

        return next_action, next_dir

    def e_greedy_adj(self, state, last_action, Nlayers):
        if np.random.random() > self.eps:
            next_action, next_dir = self.greedy_adj(state, last_action, Nlayers)
        else:
            actions, dir_list = self.get_action_list_adj(last_action, Nlayers)

            choice = np.random.randint(0, len(dir_list))
            next_action = actions[choice]
            next_dir = dir_list[choice]  # TODO NO LONGER VALID -1 = Left, 0 = Stay, +1 = Right

        return next_action, next_dir

    def update_simple(self, state, action, reward):
        """
        Update the Q table for the given state and action based on equation (2.5)
        in the book:
        Reinforcement Learning - An introduction.
        Second edition by Richard S. Sutton and Andrew G. Barto

        Parameters
        ----------
        state : ARRAY
            Which position on the grid you are standing on (x,y).
        action : INT
            The action you are taking.
        reward : MATRIX
            The reward matrix.

        Returns
        -------
        None.

        """

        self.Q[state, action] = [
            (self.Q[state, action][0] + self.alpha[state, action][0] * (reward - self.Q[state, action][0])),
            self.Q[state, action][1] + 1]
        self._update_alpha(state, action)

    def update_TD(self, State, action, R, next_state, next_action, end=False):
        """
        Update the Q table for the given state and action based on TD(0).
        Based on the book:
        Reinforcement Learning - An introduction.
        Second edition by Richard S. Sutton and Andrew G. Barto
        Parameters
        ----------
        State : State object
        action : The action (beam number) chosen based on the current state
        R : The reward
        next_state :
        next_action :
        end :

        Returns
        -------

        """
        state = helpers.state_to_index(State.state)
        next_state = helpers.state_to_index(next_state)
        if end is False:
            next_Q = self.Q[next_state, next_action][0]
        else:
            next_Q = 0

        self.Q[state, action] = [self.Q[state, action][0] + self.alpha[state, action][0] *
                                 (R + self.gamma * next_Q - self.Q[state, action][0]),
                                 self.Q[state, action][1] + 1]
        self._update_alpha(state, action)
