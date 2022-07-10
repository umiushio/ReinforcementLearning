#######################################################################
# Copyright (C)                                                       #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# the environment setting of Baird's counterexample

import numpy

# we set the states in a sightly different way from the book's
# we suppose the lower state is 0, while the above states is 1 to 6
# while the action definition is same as the book's
# action
SOLID = 0
DASHED = 1
ACTIONS = [SOLID, DASHED]
# state
NUM_OF_STATES = 7
STATES = range(NUM_OF_STATES)
# on-policy distribution
state_probabilities = numpy.ones(NUM_OF_STATES) / NUM_OF_STATES
# action transfer probability matrix
# as the transfer only depend on the action
# we do not need to consider the original state
transfer_probabilities = [[1, 0, 0, 0, 0, 0, 0],
                          [0, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6]]


# take a step from the state and action taken
# though the result is independent of the state, we just unify the form
def step(state, action):
    return numpy.random.choice(STATES, p=transfer_probabilities[action]), 0.0


# return the expected reward from the state and action taken
# here the return always 1
# this function can be used in expected algorithms
def expected_reward(state, action):
    return 0.0


# give the start state of one episode
def begin():
    return numpy.random.choice(STATES)
