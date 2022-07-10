#######################################################################
# Copyright (C)                                                       #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# the information of agent's policy or related function and data for Baird's counterexample

import numpy.random
from environment import ACTIONS


# the action probability distribution of target and behavior policy
target_policy_distribution = [1.0, 0]
behavior_policy_distribution = [1 / 7, 6 / 7]


# the action selected followed the target policy
def target_policy(state):
    return numpy.random.choice(ACTIONS, p=target_policy_distribution)


# the action selected followed the behavior policy
def behavior_policy(state):
    return numpy.random.choice(ACTIONS, p=behavior_policy_distribution)


# the importance sampling ratio of action
def rho(action):
    return target_policy_distribution[action] / behavior_policy_distribution[action]
