#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# represent the environment constructed by the mountain car task

import numpy as np

# the bound of the position
POSITION_MIN = -1.2
POSITION_MAX = 0.5

# the bound of the velocity
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

# the bound of start position
START_POSITION_MIN = -0.6
START_POSITION_MAX = -0.4

# possible actions
ACTION_FORWARD = +1
ACTION_REVERSE = -1
ACTION_ZERO = 0
ACTIONS = [ACTION_REVERSE, ACTION_ZERO, ACTION_FORWARD]
ACTION_SIZE = 3


# return the start position for the episode
def get_start_position():
    return np.random.uniform(START_POSITION_MIN, START_POSITION_MAX)


# return the start state with position and velocity
def get_start_state():
    return get_start_position(), 0.0


# take the @action at the @state, and return the new state after this transition
def step(position, velocity, action):
    new_velocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
    new_velocity = min(max(new_velocity, VELOCITY_MIN), VELOCITY_MAX)
    new_position = position + new_velocity
    new_position = min(max(new_position, POSITION_MIN), POSITION_MAX)
    if new_position == POSITION_MIN:
        new_velocity = 0.0
    reward = -1.0
    return new_position, new_velocity, reward

