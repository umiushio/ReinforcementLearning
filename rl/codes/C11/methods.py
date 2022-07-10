#######################################################################
# Copyright (C)                                                       #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# the methods that used to accomplish the task of the Baird's counterexample

import numpy
import environment
import agent


# compute the bellman error of given state under the target policy
def compute_bellman_error(value_function, state, gamma):
    targets = []
    for action in environment.ACTIONS:
        reward = environment.expected_reward(state, action)
        value = numpy.dot(environment.transfer_probabilities[action],
                          [value_function.value(s) for s in environment.STATES])
        targets.append(reward + gamma * value)
    return numpy.dot(agent.target_policy_distribution, targets) - value_function.value(state)


# compute the root mean square value error of the current value function
# the true value function is zero vector by default
# in this task we just need the default case
def compute_rmsVE(value_function, true_values=None):
    if true_values is None:
        true_values = numpy.zeros(environment.NUM_OF_STATES)
    values = numpy.asarray([value_function.value(state) for state in environment.STATES])
    return numpy.sqrt(numpy.dot(numpy.power(values - true_values, 2), environment.state_probabilities))


# compute the root mean square projected bellman error of the current value function
# as the formula in the book, we compute two terms of it by matrix operation
# note that we need the pseudo inverse of the second term in this task
# so it may be expensive when the state space is large
def compute_rmsPBE(value_function, gamma):
    bellman_errors = numpy.asarray([compute_bellman_error(value_function, state, gamma) * value_function.features[state]
                                    for state in environment.STATES])
    first_term = numpy.dot(bellman_errors.T, environment.state_probabilities)
    D = numpy.diag(environment.state_probabilities)
    middle_term = numpy.linalg.pinv(numpy.dot(value_function.features.T, numpy.dot(D, value_function.features)))
    return numpy.sqrt(numpy.dot(first_term.T, numpy.dot(middle_term, first_term)))


# semi-gradient off-policy TD algorithm
# @value_function: value function that need to learn
# @gamma: discount factor
# @steps: # of steps for learning
# @alpha: step size
# @weights_trajectory: the trajectory of all the weights over steps
def semi_gradient_off_policy_temporal_difference(value_function, gamma, steps, alpha=None, weights_trajectory=None):
    state = environment.begin()
    for step in range(steps):
        action = agent.behavior_policy(state)
        new_state, reward = environment.step(state, action)
        target = reward + gamma * value_function.value(new_state)
        rho = agent.rho(action)
        value_function.learn(state, target, rho, alpha)
        if weights_trajectory is not None:
            for i, weight in enumerate(value_function.weights):
                weights_trajectory[i].append(weight)
        state = new_state


# semi-gradient DP algorithm
# @value_function: value function that need to learn
# @gamma: discount factor
# @sweeps: # of sweeps for learning
# @alpha: step size
# @weights_trajectory: the trajectory of all the weights over steps
def semi_gradient_dynamic_programming(value_function, gamma, sweeps, alpha=None, weights_trajectory=None):
    for sweep in range(sweeps):
        for state in environment.STATES:
            value_function.update(state, compute_bellman_error(value_function, state, gamma), alpha=alpha)
        if weights_trajectory is not None:
            for i, weight in enumerate(value_function.weights):
                weights_trajectory[i].append(weight)


# TDC algorithm
# @value_function: value function that need to learn
# @gamma: discount factor
# @steps: # of steps for learning
# @alpha: step size
# @beta: second step size
# @weights_trajectory: the trajectory of all the weights over steps
# @value_errors_trajectory: the trajectory of all the VE over steps
# @bellman_errors_trajectory: the trajectory of all the PBE over steps
def temporal_difference_with_gradient_correction(value_function, gamma, steps, alpha=None, beta=None,
                                                 weights_trajectory=None,
                                                 value_errors_trajectory=None,
                                                 bellman_errors_trajectory=None):
    state = environment.begin()
    for step in range(steps):
        action = agent.behavior_policy(state)
        new_state, reward = environment.step(state, action)
        rho = agent.rho(action)
        delta = reward + gamma * value_function.value(new_state) - value_function.value(state)
        # we update the value function in two parts
        value_function.update(state, delta, rho)
        # update the correction vector and get its value
        correction = -gamma * value_function.correct(state, delta, rho)
        # update with correction
        value_function.update(new_state, correction, rho)
        if weights_trajectory is not None:
            for i, weight in enumerate(value_function.weights):
                weights_trajectory[i].append(weight)
        if value_errors_trajectory is not None:
            value_errors_trajectory.append(compute_rmsVE(value_function))
        if bellman_errors_trajectory is not None:
            bellman_errors_trajectory.append(compute_rmsPBE(value_function, gamma))
        state = new_state


# expected TDC algorithm
# @value_function: value function that need to learn
# @gamma: discount factor
# @sweeps: # of sweeps for learning
# @alpha: step size
# @beta: second step size
# @weights_trajectory: the trajectory of all the weights over steps
# @value_errors_trajectory: the trajectory of all the VE over steps
# @bellman_errors_trajectory: the trajectory of all the PBE over steps
def expected_temporal_difference_with_gradient_correction(value_function, gamma, sweeps, alpha=None, beta=None,
                                                          weights_trajectory=None,
                                                          value_errors_trajectory=None,
                                                          bellman_errors_trajectory=None):
    for sweep in range(sweeps):
        for state in environment.STATES:
            total_deltas = []
            deltas = []
            for action in environment.ACTIONS:
                new_state, reward = environment.step(state, action)
                delta = reward + gamma * value_function.value(new_state) - value_function.value(state)
                # we only get the value and do not change the correction vector
                correction = value_function.correct(state, delta, 1, write=False)
                total_deltas.append(delta * value_function.features[state] -
                                    gamma * correction * value_function.features[new_state])
                deltas.append(delta)
            # we compute the total update delta and the TD error
            total_delta = numpy.dot(numpy.asarray(total_deltas).T, agent.target_policy_distribution)
            delta = numpy.dot(numpy.asarray(deltas), agent.target_policy_distribution)
            # we correct the v vector and update directly on the weight vector
            value_function.correct(state, delta, 1)
            value_function.update_directly(total_delta)
        if weights_trajectory is not None:
            for i, weight in enumerate(value_function.weights):
                weights_trajectory[i].append(weight)
        if value_errors_trajectory is not None:
            value_errors_trajectory.append(compute_rmsVE(value_function))
        if bellman_errors_trajectory is not None:
            bellman_errors_trajectory.append(compute_rmsPBE(value_function, gamma))


# expected TDC algorithm
# @value_function: value function that need to learn
# @gamma: discount factor
# @sweeps: # of sweeps for learning
# @alpha: step size
# @beta: second step size
# @weights_trajectory: the trajectory of all the weights over steps
# @value_errors_trajectory: the trajectory of all the VE over steps
def expected_emphatic_temporal_difference(value_function, gamma, sweeps, alpha=None,
                                          weights_trajectory=None,
                                          value_errors_trajectory=None):
    # interest
    I = 1.0
    # emphasis
    M = 0.0
    for sweep in range(sweeps):
        for state in environment.STATES:
            # as the rho in the emphasis update equation is at previous step
            # we need compute its expected value
            prior_probabilities = numpy.asarray([agent.behavior_policy_distribution[action] *
                                                 environment.transfer_probabilities[action][state]
                                                 for action in environment.ACTIONS])
            prior_probabilities /= numpy.sum(prior_probabilities)
            expected_previous_rho = numpy.dot(prior_probabilities,
                                              [agent.rho(action) for action in environment.ACTIONS])
            deltas = []
            for action in environment.ACTIONS:
                reward = environment.expected_reward(state, action)
                value = numpy.dot(environment.transfer_probabilities[action],
                                  [value_function.value(s) for s in environment.STATES])
                deltas.append(reward + gamma * value - value_function.value(state))
            # compute expected weight update for current state
            expected_delta = numpy.dot(agent.target_policy_distribution,
                                       (gamma * expected_previous_rho * M + I) * numpy.asarray(deltas))
            value_function.update(state, expected_delta, alpha=alpha)
        # update the emphasis
        M = gamma * M + I
        if weights_trajectory is not None:
            for i, weight in enumerate(value_function.weights):
                weights_trajectory[i].append(weight)
        if value_errors_trajectory is not None:
            value_errors_trajectory.append(compute_rmsVE(value_function))
