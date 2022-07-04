#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# represent the agent's behaviors, such as the action selection policy

import numpy as np
import environment

# get the action for next step, following the epsilon-greedy policy


def get_action(position, velocity, argmax_actions, epsilon=0.0):
    if np.random.binomial(1, epsilon) == 0:
        return environment.ACTIONS[np.random.choice(argmax_actions)]
    else:
        return environment.ACTIONS[np.random.randint(environment.ACTION_SIZE)]
        # return np.random.choice(environment.ACTIONS)