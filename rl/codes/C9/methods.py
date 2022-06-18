#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# code implementation for some function approximation methods described in the book
# contains gradient MC and semi-gradient n-step TD
# also including tabular DP to compute the true value of states

import numpy as np
import environment
import agent
import value_functions


# tabular Dynamic Programming method and perform a traversal update
# @value_function: an instance of class ValueFunction
def tabular_DP(value_function: value_functions.ValueFunction):
    # traverse all states except for terminal states
    for state in environment.STATES[1:-1]:
        # set the value of the state to 0
        target = 0
        # traverse through all actions
        for direction in environment.DIRECTIONS:
            for step in range(environment.STEP_RANGE):
                new_state, reward = environment.step(state, direction * step)
                # add the update target of each transition and for obtaining the expectation value
                # so each update need to be divided by (direction number * step number) times
                target += reward + value_function.value(new_state)
        value_function.learn(state, target / (2 * environment.STEP_RANGE))


# use the tabular TD method to compute the true values of all states
# @epsilon: threshold of update difference
# @value_function: [optional] initial value function, which may accelerate the process
def compute_true_values(value_function=None, epsilon=1e-3):
    if value_function is None:
        # default initialization
        value_function = value_functions.TabularValueFunction()
    values = np.asarray([value_function.value(state) for state in environment.STATES[1:-1]])
    while True:
        tabular_DP(value_function)
        new_values = np.asarray([value_function.value(state) for state in environment.STATES[1:-1]])
        # the difference between before and after update
        error = np.sum(np.abs(new_values - values))
        if error < epsilon:
            break
        values = new_values


# gradient Monte Carlo method
# @value_function: an instance of class ValueFunction
# @alpha: step size
# @distribution: [optional] array to store the distribution statistics
def gradient_MC(value_function: value_functions.ValueFunction, alpha, distribution=None):
    # initial starting state
    state = environment.START_STATE
    trajectory = [state]
    while state not in environment.TERMINAL_STATES:
        # due to reward will be 1 or -1 only at the last transition, otherwise 0
        # thus we need not to store the trajectory of reward
        state, reward = environment.step(state, agent.get_action())
        trajectory.append(state)
    # gradient update for each state in the trajectory
    for state in trajectory[:-1]:
        returns = reward
        value_function.learn(state, returns, alpha)
    if distribution is not None:
        for state in trajectory[:-1]:
            distribution[state] += 1


# semi-gradient n-step temporal difference method
# @valueFunction: an instance of class ValueFunction
# @alpha: step size
# @n: # of steps
def semi_gradient_TD(value_function: value_functions.ValueFunction, n, alpha, gamma=1):
    # initial starting state
    state = environment.START_STATE
    # arrays to store states and rewards encountered in a episode
    # the space is not large, so we don't use the modular thick
    states = [state]
    rewards = [0]

    # end time T, the length of the episode
    end_time = float('inf')
    # time t, track the time
    t = 0
    while True:
        if t < end_time:
            # choose an action and make a transition
            action = agent.get_action()
            next_state, reward = environment.step(state, action)
            # store the new state and reward
            states.append(next_state)
            rewards.append(reward)
            if next_state in environment.TERMINAL_STATES:
                end_time = t + 1

        # the time of state to update
        tau = t - n + 1
        if tau >= 0:
            # calculate the corresponding return
            if tau + n < end_time:
                returns = np.sum(rewards[tau + 1: tau + n + 1]) + value_function.value(next_state)
            else:
                returns = np.sum(rewards[tau + 1:])
            # update the value function
            value_function.learn(states[tau], returns, alpha)
        if tau == end_time - 1:
            break
        t += 1
        state = next_state
