#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# code implementation for the state-action value functions used to plot.
# the main methods are: value and learn

import abc
import numpy as np
from environment import POSITION_MIN, POSITION_MAX, VELOCITY_MIN, VELOCITY_MAX, ACTIONS
import lib.tile_coding as tc


# define the abstract class for value function
class ValueFunction:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    # value method returns the @state-@action's current estimate
    # @state: the state of current example
    # @action: the action of current example
    @abc.abstractmethod
    def value(self, position, velocity, action):
        pass

    # learn the target in an individual way
    # @position, velocity: the state of current example
    # @action: the action of current example
    # @target: target that need to learn
    @abc.abstractmethod
    def learn(self, position, velocity, action, target):
        pass

    # give all actions' indexes that maximum the value of given state
    # @position, velocity: the state parameters
    @abc.abstractmethod
    def argmax_actions(self, position, velocity):
        pass

    # return the maximum value of given state among actions
    # @position, velocity: the state parameters
    @abc.abstractmethod
    def max_value(self, position, velocity):
        pass


# a wrapper class for tiling value function
# we only consider the one-dimension case for state space
# wrapper class for state action value function
class TilingValueFunction:
    # In this example I use the tiling software instead of implementing standard tiling by myself
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/sutton/tiles/tiles3.html
    # @max_size: the maximum # of indices
    def __init__(self, dimension, ints=None, alpha=1.0):
        self.dimension = dimension

        self.num_of_tilings = tc.min_power_exponent(4 * dimension)
        self.max_size = tc.compute_max_size(dimension, ints)
        self.alpha = alpha / self.num_of_tilings

        self.hash_table = tc.IndexHashTable(self.max_size)

        # weight for each tile
        self.weights = np.zeros(self.max_size)

        # position and velocity needs scaling to satisfy the tile software
        self.position_scale = self.num_of_tilings / (POSITION_MAX - POSITION_MIN)
        self.velocity_scale = self.num_of_tilings / (VELOCITY_MAX - VELOCITY_MIN)

    # get indices of active tiles for given state and action
    def get_active_tiles(self, position, velocity, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        active_tiles = tc.tiles(self.hash_table, self.num_of_tilings,
                                [self.position_scale * position, self.velocity_scale * velocity],
                                [action])
        return active_tiles

    # estimate the value of given state and action
    def value(self, position, velocity, action):
        if position == POSITION_MAX:
            return 0.0
        active_tiles = self.get_active_tiles(position, velocity, action)
        return np.sum(self.weights[active_tiles])

    # learn with the given state, action and target
    def learn(self, position, velocity, action, target):
        active_tiles = self.get_active_tiles(position, velocity, action)
        estimation = np.sum(self.weights[active_tiles])
        delta = self.alpha * (target - estimation)
        for active_tile in active_tiles:
            self.weights[active_tile] += delta

    def argmax_actions(self, position, velocity):
        action_values = [self.value(position, velocity, action) for action in ACTIONS]
        return np.where(action_values == np.max(action_values))[0]

    def max_value(self, position, velocity):
        action_values = [self.value(position, velocity, action) for action in ACTIONS]
        return np.max(action_values)

# return the recommended number of tilings for the given dimension
    @staticmethod
    def get_num_of_tilings(dimension):
        return tc.min_power_exponent(4 * dimension)
