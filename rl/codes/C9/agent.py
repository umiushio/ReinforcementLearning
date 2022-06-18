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


# get the direction, following the random policy
def get_direction():
    if np.random.binomial(1, 0.5) == 1:
        return environment.TO_RIGHT
    return environment.TO_LEFT


# get the action for next step, following the random policy
def get_action():
    direction = get_direction()
    step = np.random.randint(environment.STEP_RANGE)
    return direction * step
