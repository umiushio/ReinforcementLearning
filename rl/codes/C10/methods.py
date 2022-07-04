#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# code implementation for the function approximation methods used to plot
# contains semi_gradient_sarsa and semi_gradient_n_step_sarsa methods
# in fact semi_gradient_sarsa method is a special case of semi_gradient_n_step_sarsa method

import numpy as np
import environment
import agent
import value_functions


# semi-gradient Sarsa method
# @valueFunction: an instance of class ValueFunction
# it is implemented simply that give the intuition for semi_gradient_sarsa
# @alpha: step size
# @epsilon: the soft-greedy policy parameter
# @gamma: the discounting factor
def semi_gradient_sarsa(value_function: value_functions.ValueFunction, alpha=None, epsilon=0, gamma=1):
    # get initial state and action
    position, velocity = environment.get_start_state()
    action = agent.get_action(position, velocity, value_function.argmax_actions(position, velocity), epsilon)
    while position != environment.POSITION_MAX:
        new_position, new_velocity, reward = environment.step(position, velocity, action)
        new_action = agent.get_action(position, velocity, value_function.argmax_actions(position, velocity), epsilon)
        target = reward + gamma * value_function.value(position, velocity, action)
        value_function.learn(position, velocity, action, target, alpha)
        position = new_position
        velocity = new_velocity
        action = new_action


# semi-gradient n-step Sarsa method
# it will return the steps of the episode
# @valueFunction: an instance of class ValueFunction
# @n: # of steps
# @alpha: step size
# @epsilon: the soft-greedy policy parameter
def semi_gradient_n_step_sarsa(value_function: value_functions.ValueFunction, n=1, alpha=None, epsilon=0):
    # As some alphas or ns may cause the episode continue forever, so we set the max_steps to avoid such cases
    max_steps = int(1e6)
    # initial starting state and action
    position, velocity = environment.get_start_state()
    action = agent.get_action(position, velocity, value_function.argmax_actions(position, velocity))
    # arrays to store states and rewards encountered in a episode
    # the space is not large, so we don't use the modular thick
    positions = [position]
    velocities = [velocity]
    actions = [action]
    rewards = [0.0]

    # end time T, the length of the episode
    # end_time = float('inf')
    end_time = max_steps
    # time t, track the time
    t = 0
    while True:
        # if t > max_steps:
        #     break
        if t < end_time:
            # make a transition and get the reward and new state
            next_position, next_velocity, reward = environment.step(position, velocity, action)
            next_action = agent.get_action(position, velocity,
                                           value_function.argmax_actions(position, velocity), epsilon)
            # store the new state and reward
            positions.append(next_position)
            velocities.append(next_velocity)
            actions.append(next_action)
            rewards.append(reward)
            if next_position == environment.POSITION_MAX:
                end_time = t + 1

        # the time of state to update
        tau = t - n + 1
        if tau >= 0:
            # calculate the corresponding return
            if tau + n < end_time:
                returns = np.sum(rewards[tau + 1: tau + n + 1]) + \
                          value_function.value(next_position, next_velocity, next_action)
            else:
                returns = np.sum(rewards[tau + 1:])
            # update the value function
            value_function.learn(positions[tau], velocities[tau], actions[tau], returns)
        if tau == end_time - 1:
            break
        t += 1
        position = next_position
        velocity = next_velocity
        action = next_action

    return end_time
