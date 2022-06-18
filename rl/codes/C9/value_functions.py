#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# code implementation for some value functions described in the book.
# all the functions are wrapped in a class
# each class has two methods: value and learn
# we don't care about the value and learn for terminal states

import abc
import numpy as np
import environment


# define the abstract class for value function
class ValueFunction:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    # value method returns the @state's current estimate
    # @state: the state of current example
    @abc.abstractmethod
    def value(self, state):
        pass

    # update method make the update in an individual way and returns nothing
    # @state: the state of current example
    # @target: the target of trained example
    # @alpha: the step-size
    @abc.abstractmethod
    def learn(self, state, target, alpha):
        pass


# a wrapper class for tabular value function
class TabularValueFunction(ValueFunction):
    # @values: initial value all the state with a size of N_STATES+1
    def __init__(self, values=None):
        if values is not None:
            self.values = np.asarray(values)
        else:
            # default initialization
            self.values = np.asarray([0 for i in range(environment.N_STATES + 1)])

    # just return the value in the table
    def value(self, state):
        if state in environment.TERMINAL_STATES:
            return 0
        else:
            return self.values[state]

    # in the tabular DP alpha often set to 1
    def learn(self, state, target, alpha=1):
        if state not in environment.TERMINAL_STATES:
            self.values[state] += alpha * (target - self.values[state])


# a wrapper class for aggregation value function
class AggregationValueFunction(ValueFunction):
    # @num_of_groups: # of aggregations
    def __init__(self, num_of_groups):
        self.num_of_groups = num_of_groups
        self.group_size = environment.N_STATES // num_of_groups

        # weight for each group and each state in the same group shares the same weight
        self.group_weights = np.zeros(environment.N_STATES + 1)

    def value(self, state):
        if state in environment.TERMINAL_STATES:
            return 0
        else:
            group_index = (state - 1) // self.group_size
            return self.group_weights[group_index]

    def learn(self, state, target, alpha):
        if state not in environment.TERMINAL_STATES:
            group_index = (state - 1) // self.group_size
            self.group_weights[group_index] += alpha * (target - self.group_weights[group_index])


# a wrapper class for polynomial- or fourier- basis value function
# we only consider the one-dimension state space case
POLYNOMIAL_BASIS = 0
FOURIER_BASIS = 1


class BasisValueFunction(ValueFunction):
    # @order: # of bases, each function also has one more constant parameter(called bias in machine learning)
    # @type: polynomial bases or Fourier bases
    def __init__(self, order, basis_type):
        self.order = order
        self.basis_type = basis_type
        self.weights = np.zeros(order + 1)
        self.basis = []
        if basis_type == POLYNOMIAL_BASIS:
            for i in range(order + 1):
                self.basis.append(lambda s, i=i: np.power(s, i))
        elif basis_type == FOURIER_BASIS:
            for i in range(order + 1):
                self.basis.append(lambda s, i=i: np.cos(i * np.pi * s))

    def value(self, state):
        if state in environment.TERMINAL_STATES:
            return 0
        else:
            # map the state space into (0,1]
            state /= float(environment.N_STATES)
            # get the feature vector
            features = np.asarray([func(state) for func in self.basis])
            return np.dot(self.weights, features)

    def learn(self, state, target, alpha):
        if state not in environment.TERMINAL_STATES:
            # map the state space into (0,1]
            state /= environment.N_STATES
            # get derivative value
            derivative_value = np.asarray([func(state) for func in self.basis])
            state_value = np.dot(self.weights, derivative_value)
            if self.basis_type == POLYNOMIAL_BASIS:
                self.weights += (target - state_value) * alpha * derivative_value
            # we will set the step-size in the recommended way in the book
            elif self.basis_type == FOURIER_BASIS:
                alphas = np.ones(self.order + 1) * alpha / np.asarray([1] + list(range(1, self.order + 1)))
                self.weights += (target - state_value) * alphas * derivative_value


# a wrapper class for tiling value function
# we only consider the one-dimension case for state space
class TilingValueFunction(ValueFunction):
    # @num_of_tilings: # of tilings
    # @tile_size: each tiling has several tiles, this parameter specifies the size of each tile
    #   while in this case it just means the width of each tile
    # @tiling_offset: specifies how tilings offset from each other
    def __init__(self, num_of_tilings, tile_size, tiling_offset):
        self.num_of_tilings = num_of_tilings
        self.tile_size = tile_size
        self.tiling_offset = tiling_offset
        # the number of tiles in each tiling
        # we need one more tile for each tiling to ensure all states will be covered by each tiling
        self.tiling_size = environment.N_STATES // tile_size + 1
        # the start for each tiling, we assume they will be in the range of (-tile size + 1, 1]
        self.tiling_starts = np.arange(-num_of_tilings + 1, 1) * tiling_offset
        # weight for each tile and each state in the same tile shares the same weight
        self.tile_weights = np.zeros((num_of_tilings, self.tiling_size))

    def value(self, state):
        if state in environment.TERMINAL_STATES:
            return 0
        else:
            # index of the active tile for each tiling
            tile_indexes = [(state - start) // self.tile_size for start in self.tiling_starts]
            # weight of active tile for each tiling
            active_tile_weights = np.asarray(
                [self.tile_weights[i][tile_indexes[i]] for i in range(self.num_of_tilings)])
            return np.sum(active_tile_weights)

    def learn(self, state, target, alpha):
        if state not in environment.TERMINAL_STATES:
            # we treat each tiling equally and the state is covered in each tiling
            # so the alpha should be divided equally into each tiling
            alpha /= self.num_of_tilings
            tile_indexes = []
            state_value = 0.0
            # traverse through all tilings
            for i in range(self.num_of_tilings):
                # index of the active tile for each tiling
                tile_index = (state - self.tiling_starts[i]) // self.tile_size
                state_value += self.tile_weights[i][tile_index]
                tile_indexes.append(tile_index)
            # calculate the updated delta
            delta = alpha * (target - state_value)
            # each tile equally updated
            for i, tile_index in enumerate(tile_indexes):
                self.tile_weights[i][tile_index] += delta
