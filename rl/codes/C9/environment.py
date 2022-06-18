#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# represent the environment constructed by the 1000-states random walk task

import numpy as np

# the number of states expect the terminal states
N_STATES = 1000
# all states including the terminal states
STATES = np.arange(N_STATES + 2)
# the terminal states
TERMINAL_STATES = [0, N_STATES + 1]
# the start state
START_STATE = 500

# define directions for action
TO_LEFT = -1
TO_RIGHT = 1
DIRECTIONS = [TO_LEFT, TO_RIGHT]

# the max stride for one step
global STEP_RANGE


# take the @action at the @state, return the new state and reward for this transition
def step(state, action):
    state = max(min(state + action, N_STATES + 1), 0)
    if state == 0:
        reward = -1
    elif state == N_STATES + 1:
        reward = 1
    else:
        reward = 0
    return state, reward


# in different tasks the state range will be different
# so we should have this function to dynamically set it
def set_state_range(state_range):
    global STEP_RANGE
    STEP_RANGE = state_range
