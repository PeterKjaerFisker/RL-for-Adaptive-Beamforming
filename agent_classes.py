# %% Imports
from collections import defaultdict
import numpy as np
from tqdm import tqdm

import helpers


# %% Multi Agent Class
class MultiAgent:
    def __init__(self, action_space, agent_type='naive', alpha=0.01, eps=["constant", 0.01], gamma=0.6):
        """
        Initiate a reinforcement learning agent

        Parameters
        ----------
        action_space : Array
            All possible actions
        alpha : Float, optional
            Learning rate, options are 'constant' and '1/n'. The default is ["constant", 0.7].
        eps : Float, optional
            Probability of choosing random action for epsilon greedy policy. The default is 0.1.
        gamma : Float, optional
            Forgetting factor. The default is 0.7.
        c : Float, optional
            c value for UCB, balances exploration. The default is 200.

        Returns
        -------
        None.

        """
        self.action_space = action_space
        self.agent_type = agent_type
        self.alpha = alpha

        self.eps_method = eps[0]
        self.eps = eps[1]
        self.eps_table = defaultdict(lambda: 1)
        self.delta = 1 / 5  # The inverse of the expected amount of actions in any state. For adaptive epsilon

        self.gamma = gamma

        self.Q = defaultdict(self._initiate_dict(0.001))
        self.state_counter = defaultdict(lambda: 0)

        self.delta_w = 0.15
        self.delta_l = 0.60

    def update_epsilon(self, timestep, weight, td_error, state):
        if self.eps_method == "constant":
            pass
        elif self.eps_method == "decaying":
            self.eps = np.exp(-timestep / weight)
        elif self.eps_method == "adaptive":
            td_error = float(td_error)
            ratio = (1 - np.exp(-np.abs(self.alpha * td_error) / (10 ** -6 * weight))) / (
                    1 + np.exp(-np.abs(self.alpha * td_error) / (10 ** -6 * weight)))
            self.eps_table[state] = self.delta * ratio + (1 - self.delta) * self.eps_table[state]
        elif self.eps_method == 'sigmoid':
            offset = 1 / (1 + np.exp((-self.alpha * (0 - 0.001)) / (10 ** -6 * weight)))
            ratio = 1 / (1 + np.exp((-self.alpha * (np.abs(td_error) - 0.001)) / (10 ** -6 * weight))) - offset
            self.eps_table[state] = self.delta * ratio + (1 - self.delta) * self.eps_table[state]

    def reset_epsilon(self):
        if self.eps_method == "constant":
            pass
        elif self.eps_method == "decaying":
            self.eps = 1

    def reset_eps_table(self):
        self.eps_table = defaultdict(lambda: 1)

    def _initiate_dict(self, value_est, visit_count=0, policy_prob=0, exp_policy_prob=0):
        """
        Small function used when initiating the dicts.
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
        if self.agent_type == 'wolf':
            return lambda: [value_est, np.uint16(visit_count), policy_prob, exp_policy_prob]
        else:
            return lambda: [value_est, np.uint16(visit_count)]

    def get_action_list_adj(self, current_beam_nr, Nlayers, action_space):
        """
        Calculate the beam numbers of adjecent beams

        Parameters
        ----------
        last_action : Int
            The beam number of the current beam
        Nlayers : Int
            Number of layers, for restricting avaliable actions

        Returns
        -------
        actions : List
            List of beam numbers for adjecent beams
        action_list : Array
            Array of directions avaliable at current beam

        """
        current_layer = int(np.floor(np.log2(current_beam_nr + 2)))  # Wrap around for left/right

        # Check if the codeword to the "right" is still in the same layer
        if current_layer == int(np.floor(np.log2(current_beam_nr + 3))):
            beam_nr_Right = current_beam_nr + 1
        else:
            beam_nr_Right = int((current_beam_nr - 1) / 2)

        # Check if the codeword to the "left" is still in the same layer
        if current_layer == int(np.floor(np.log2(current_beam_nr + 1))):
            beam_nr_Left = current_beam_nr - 1
        else:
            beam_nr_Left = int((current_beam_nr * 2) + 1)

        # Limits the agent to taking appropriate actions
        beam_nr_list = [action_space[current_beam_nr],  # Stay
                        action_space[beam_nr_Right],  # Right
                        action_space[beam_nr_Left]]  # Left

        if Nlayers > 1:
            # Check if current layer is between the bottom and top layers
            if current_layer != 1 and current_layer != Nlayers:
                beam_nr_list.append(action_space[int(np.floor((current_beam_nr - 2) / 2))])  # Down
                beam_nr_list.append(
                    action_space[
                        int((current_beam_nr * 2) + 3)])  # TODO Up Right TODO måske byt om så ting er i rækkefølge
                beam_nr_list.append(action_space[int((current_beam_nr * 2) + 2)])  # Up Left

                action_list = [0, 1, 2, 3, 4, 5]

            elif current_layer != 1:  # Check if on bottom layer
                beam_nr_list.append(action_space[int(np.floor((current_beam_nr - 2) / 2))])  # Down

                action_list = [0, 1, 2, 3]

            else:  # Check if on uppermost layer
                beam_nr_list.append(action_space[int((current_beam_nr * 2) + 3)])  # Up Right
                beam_nr_list.append(action_space[int((current_beam_nr * 2) + 2)])  # Up Left

                action_list = [0, 1, 2, 4, 5]
        else:
            action_list = [0, 1, 2]

        return beam_nr_list, action_list

    def greedy_adj(self, state, current_beam_nr, Nl):
        """
        Calculates the optimal action according to the greedy policy
        when actions are restricted to choosing adjecent beams

        Parameters
        ----------
        state : Array
            The state
        last_action : Int
            The number of the last chosen beam, to calculate adjecent beams
        Nlayers : Int
            Numbers of layers in the hierarchical codebook, to calculate when to limit actions

        Returns
        -------
        next_action : Int
            Beam number the action corresponds to
        next_dir : Int
            Which direction was chosen

        """

        beam_nr_list, action_list = self.get_action_list_adj(current_beam_nr[0], Nl, self.action_space)

        choice = np.random.randint(0, len(action_list))
        next_beam = tuple([beam_nr_list[choice]])
        next_action = tuple([action_list[choice]])
        r_est = self.Q[state, next_action][0]

        for idx, last_action in enumerate(action_list):
            if self.Q[state, tuple([last_action])][0] > r_est:
                next_beam = tuple([beam_nr_list[idx]])
                next_action = tuple([last_action])
                r_est = self.Q[state, tuple([last_action])][0]

        return next_beam, next_action

    def get_action_adj(self, state, current_beam_nr, Nl):
        """
        Calculates the optimal action according to the greedy policy
        when actions are restricted to choosing adjecent beams

        Parameters
        ----------
        state : Array
            The state
        last_action : Int
            The number of the last chosen beam, to calculate adjecent beams
        Nlayers : Int
            Numbers of layers in the hierarchical codebook, to calculate when to limit actions

        Returns
        -------
        next_action : Int
            Beam number the action corresponds to
        next_dir : Int
            Which direction was chosen

        """

        beam_nr_list, action_list = self.get_action_list_adj(current_beam_nr[0], Nl, self.action_space)

        visited = False
        for action in action_list:
            if self.Q[state, tuple([action])][1] > 0:
                visited = True

        if not visited:
            for action in action_list:
                self.Q[state, tuple([action])][2] = 1 / len(action_list)

        rng = np.random.uniform(0, 1)
        acc_prob = 0
        for idx, action in enumerate(action_list):
            acc_prob += self.Q[state, tuple([action])][2]
            if rng < acc_prob:
                next_beam = tuple([beam_nr_list[idx]])
                next_action = tuple([action])
                return next_beam, next_action

        raise Exception('ERROR: Probabilities did not add up to 1. Most likely')

    def e_soft_adj(self, state, current_beam_nr, Nl):
        """
        Calculates the optimal action according to the epsilon greedy policy
        when actions are restricted to choosing adjecent beams

        Parameters
        ----------
        state : Array
            The state
        last_action : Int
            The number of the last chosen beam, to calculate adjecent beams
        Nlayers : Int
            Numbers of layers in the hierarchical codebook, to calculate when to limit actions

        Returns
        -------
        next_action : Int
            Beam number the action corresponds to
        next_dir : Int
            Which direction was chosen

        """
        if self.eps_method == "adaptive":
            epsilon = self.eps_table[state]
        else:
            epsilon = self.eps

        if np.random.random() > epsilon:
            if self.agent_type == 'wolf':
                next_beam, next_action = self.get_action_adj(state, current_beam_nr, Nl)
            else:
                next_beam, next_action = self.greedy_adj(state, current_beam_nr, Nl)
        else:
            beam_nr_list, action_list = self.get_action_list_adj(current_beam_nr[0], Nl, self.action_space)

            if self.agent_type == 'wolf':
                visited = False
                for action in action_list:
                    if self.Q[state, tuple([action])][1] > 0:
                        visited = True

                if not visited:
                    for action in action_list:
                        self.Q[state, tuple([action])][2] = 1 / len(action_list)

            choice = np.random.randint(0, len(action_list))

            next_beam = tuple([beam_nr_list[choice]])
            next_action = tuple([action_list[choice]])

        return next_beam, next_action

    def update_WoLF_PHC_adj(self, state, current_beam_nr, Nl):

        beam_nr_list, action_list = self.get_action_list_adj(current_beam_nr[0], Nl, self.action_space)

        choice = np.random.randint(0, len(action_list))
        greedy_action = action_list[choice]
        r_est = self.Q[state, tuple([greedy_action])][0]
        state_counter = self.state_counter[state]

        temp1 = 0
        temp2 = 0
        for idx, action in enumerate(action_list):
            current_est = self.Q[state, tuple([action])][0]
            # Update the expected policy probability for the state-action pair and Determine which learning rate to use
            self.Q[state, tuple([action])][3] += (self.Q[state, tuple([action])][2] -
                                                  self.Q[state, tuple([action])][3]) / state_counter
            temp1 += self.Q[state, tuple([action])][2] * current_est
            temp2 += self.Q[state, tuple([action])][3] * current_est

            if current_est > r_est:
                greedy_action = action
                r_est = current_est

        # Update the expected policy probability for the state-action pair and Determine which learning rate to use
        # temp1 = 0
        # temp2 = 0
        # for action in action_list:
        #     self.Q[state, tuple([action])][3] += (self.Q[state, tuple([action])][2] -
        #                                           self.Q[state, tuple([action])][3]) / state_counter
        #     temp1 += self.Q[state, tuple([action])][2] * self.Q[state, tuple([action])][0]
        #     temp2 += self.Q[state, tuple([action])][3] * self.Q[state, tuple([action])][0]

        if temp1 > temp2:
            delta = self.delta_w
        else:
            delta = self.delta_l

        # gradients = np.zeros(len(action_list))
        gradients2 = np.zeros(len(action_list))

        for idx, action in enumerate(action_list):
            gradients2[idx] = self.get_delta(state, tuple([action]), delta, len(action_list))
        sumt = np.sum(gradients2)

        for idx, action in enumerate(action_list):
            if action == greedy_action:
                self.Q[state, tuple([action])][2] += sumt - gradients2[idx]
            else:
                self.Q[state, tuple([action])][2] += -gradients2[idx]

        # for idx, action_i in enumerate(action_list):
        #     if action_i == greedy_action:
        #         for action_j in [x for x in action_list if x != action_i]:
        #             gradients[idx] += self.get_delta(state, tuple([action_j]), delta, len(action_list))
        #     else:
        #         gradients[idx] = -self.get_delta(state, tuple([action_i]), delta, len(action_list))
        #
        # for idx, action in enumerate(action_list):
        #     self.Q[state, tuple([action])][2] += gradients[idx]

    def get_delta(self, state, action, delta, length):
        return np.min([self.Q[state, action][2], delta / (length - 1)])

    def update_TD(self, state, action, R, next_state, next_action, end=False):
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
        next_state = helpers.state_to_index(next_state)
        if end is False:
            next_Q = self.Q[next_state, next_action][0]
        else:
            next_Q = 0

        TD_error = (R + self.gamma * next_Q - self.Q[state, action][0])

        self.Q[state, action][0] += self.alpha * (R + self.gamma * next_Q - self.Q[state, action][0])
        self.Q[state, action][1] += 1

        self.state_counter[state] += 1
        return TD_error


# %% Agent Class
class Agent:
    def __init__(self, action_space_r, action_space_t, alpha=0.01, eps=["constant", 0.01], gamma=0.7):
        """
        Initiate a reinforcement learning agent

        Parameters
        ----------
        action_space : Array
            All possible actions
        alpha : Float, optional
            Learning rate, options are 'constant' and '1/n'. The default is ["constant", 0.7].
        eps : Float, optional
            Probability of choosing random action for epsilon greedy policy. The default is 0.1.
        gamma : Float, optional
            Forgetting factor. The default is 0.7.
        c : Float, optional
            c value for UCB, balances exploration. The default is 200.

        Returns
        -------
        None.

        """
        self.action_space_r = action_space_r  # Number of beam directions for receiver
        self.action_space_t = action_space_t  # Number of beam directions for transmitter
        self.alpha = alpha

        self.eps_method = eps[0]
        self.eps = eps[1]
        self.eps_table = defaultdict(lambda: 1)
        self.delta = 1 / 4  # The inverse of the expected amount of actions in any state. For adaptive epsilon

        self.gamma = gamma
        self.Q = defaultdict(self._initiate_dict(0.001))

    def update_epsilon(self, timestep, weight, td_error, state):
        if self.eps_method == "constant":
            pass
        elif self.eps_method == "decaying":
            self.eps = np.exp(-timestep / weight)
        elif self.eps_method == "adaptive":
            ratio = (1 - np.exp(-np.abs(self.alpha * td_error) / (10 ** -6 * weight))) / (
                    1 + np.exp(-np.abs(self.alpha * td_error) / (10 ** -6 * weight)))
            self.eps_table[state] = self.delta * ratio + (1 - self.delta) * self.eps_table[state]
        elif self.eps_method == 'sigmoid':
            offset = 1 / (1 + np.exp((-self.alpha * (0 - 0.001)) / (10 ** -6 * weight)))
            ratio = 1 / (1 + np.exp((-self.alpha * (np.abs(td_error) - 0.001)) / (10 ** -6 * weight))) - offset
            self.eps_table[state] = self.delta * ratio + (1 - self.delta) * self.eps_table[state]

    def reset_epsilon(self):
        if self.eps_method == "constant":
            pass
        elif self.eps_method == "decaying":
            self.eps = 1

    def reset_eps_table(self):
        self.eps_table = defaultdict(lambda: 1)

    def _initiate_dict(self, value_est, visit_count=0):
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
        return lambda: [value_est, visit_count]

    def get_action_list_adj(self, current_beam_nr, Nlayers, action_space):
        """
        Calculate the beam numbers of adjecent beams

        Parameters
        ----------
        last_action : Int
            The beam number of the current beam
        Nlayers : Int
            Number of layers, for restricting avaliable actions

        Returns
        -------
        actions : List
            List of beam numbers for adjecent beams
        action_list : Array
            Array of directions avaliable at current beam

        """
        current_layer = int(np.floor(np.log2(current_beam_nr + 2)))  # Wrap around for left/right

        # Check if the codeword to the "right" is still in the same layer
        if current_layer == int(np.floor(np.log2(current_beam_nr + 3))):
            beam_nr_Right = current_beam_nr + 1
        else:
            beam_nr_Right = int((current_beam_nr - 1) / 2)

        # Check if the codeword to the "left" is still in the same layer
        if current_layer == int(np.floor(np.log2(current_beam_nr + 1))):
            beam_nr_Left = current_beam_nr - 1
        else:
            beam_nr_Left = int((current_beam_nr * 2) + 1)

        # Limits the agent to taking appropriate actions
        beam_nr_list = [action_space[current_beam_nr],  # Stay
                        action_space[beam_nr_Right],  # Right
                        action_space[beam_nr_Left]]  # Left

        if Nlayers > 1:
            # Check if current layer is between the bottom and top layers
            if current_layer != 1 and current_layer != Nlayers:
                beam_nr_list.append(action_space[int(np.floor((current_beam_nr - 2) / 2))])  # Down
                beam_nr_list.append(
                    action_space[
                        int((current_beam_nr * 2) + 3)])  # TODO Up Right TODO måske byt om så ting er i rækkefølge
                beam_nr_list.append(action_space[int((current_beam_nr * 2) + 2)])  # Up Left

                action_list = [0, 1, 2, 3, 4, 5]

            elif current_layer != 1:  # Check if on bottom layer
                beam_nr_list.append(action_space[int(np.floor((current_beam_nr - 2) / 2))])  # Down

                action_list = [0, 1, 2, 3]

            else:  # Check if on uppermost layer
                beam_nr_list.append(action_space[int((current_beam_nr * 2) + 3)])  # Up Right
                beam_nr_list.append(action_space[int((current_beam_nr * 2) + 2)])  # Up Left

                action_list = [0, 1, 2, 4, 5]
        else:
            action_list = [0, 1, 2]

        return beam_nr_list, action_list

    def greedy_adj(self, state, current_beam_nr, Nlr, Nlt):
        """
        Calculates the optimal action according to the greedy policy
        when actions are restricted to choosing adjecent beams

        Parameters
        ----------
        state : Array
            The state
        last_action : Int
            The number of the last chosen beam, to calculate adjecent beams
        Nlayers : Int
            Numbers of layers in the hierarchical codebook, to calculate when to limit actions

        Returns
        -------
        next_action : Int
            Beam number the action corresponds to
        next_dir : Int
            Which direction was chosen

        """

        beam_nr_list_r, action_list_r = self.get_action_list_adj(current_beam_nr[0], Nlr, self.action_space_r)
        beam_nr_list_t, action_list_t = self.get_action_list_adj(current_beam_nr[1], Nlt, self.action_space_t)

        choice_r = np.random.randint(0, len(action_list_r))
        choice_t = np.random.randint(0, len(action_list_t))

        next_beam = tuple((beam_nr_list_r[choice_r], beam_nr_list_t[choice_t]))
        next_action = tuple((action_list_r[choice_r], action_list_t[choice_t]))
        r_est = self.Q[state, next_action][0]

        for idx_r, action_r in enumerate(action_list_r):
            for idx_t, action_t in enumerate(action_list_t):
                if self.Q[state, tuple((action_r, action_t))][0] > r_est:
                    next_beam = tuple((beam_nr_list_r[idx_r], beam_nr_list_t[idx_t]))
                    next_action = tuple((action_r, action_t))
                    r_est = self.Q[state, tuple((action_r, action_t))][0]

        return next_beam, next_action

    def e_greedy_adj(self, state, current_beam_nr, Nlr, Nlt):
        """
        Calculates the optimal action according to the epsilon greedy policy
        when actions are restricted to choosing adjecent beams

        Parameters
        ----------
        state : Array
            The state
        last_action : Int
            The number of the last chosen beam, to calculate adjecent beams
        Nlayers : Int
            Numbers of layers in the hierarchical codebook, to calculate when to limit actions

        Returns
        -------
        next_action : Int
            Beam number the action corresponds to
        next_dir : Int
            Which direction was chosen

        """
        if self.eps_method == "adaptive":
            epsilon = self.eps_table[state]
        else:
            epsilon = self.eps

        if np.random.random() > epsilon:
            next_beam, next_action = self.greedy_adj(state, current_beam_nr, Nlr, Nlt)
        else:
            beam_nr_list_r, action_list_r = self.get_action_list_adj(current_beam_nr[0], Nlr, self.action_space_r)
            beam_nr_list_t, action_list_t = self.get_action_list_adj(current_beam_nr[1], Nlt, self.action_space_t)

            choice_r = np.random.randint(0, len(action_list_r))
            choice_t = np.random.randint(0, len(action_list_t))

            next_beam = tuple((beam_nr_list_r[choice_r], beam_nr_list_t[choice_t]))
            next_action = tuple((action_list_r[choice_r], action_list_t[choice_t]))

        return next_beam, next_action

    def update_TD(self, state, action, R, next_state, next_action, end=False):
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
        next_state = helpers.state_to_index(next_state)
        if end is False:
            next_Q = self.Q[next_state, next_action][0]
        else:
            next_Q = 0

        TD_error = (R + self.gamma * next_Q - self.Q[state, action][0])

        self.Q[state, action][0] += self.alpha * (R + self.gamma * next_Q - self.Q[state, action][0])
        self.Q[state, action][1] += 1

        return TD_error
