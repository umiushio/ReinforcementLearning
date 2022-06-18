#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# plot the figures that showed in the book

from environment import N_STATES, STATES, set_state_range
import value_functions
import methods
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.use('Agg')

states = STATES[1:-1]
global true_values


# get true values of states
def compute_true_values():
    set_state_range(100)

    global true_values
    true_values = np.arange(-N_STATES - 1, N_STATES + 1, 2) / N_STATES
    value_function = value_functions.TabularValueFunction(true_values)
    methods.compute_true_values(value_function)
    true_values = np.asarray([value_function.value(state) for state in states])


# Figure 9.1, use gradient MC method
def figure_9_1():
    set_state_range(100)

    # set the repeated episodes and step-size
    episodes = int(1e5)
    alpha = 2e-5

    # set 10 aggregations in this example
    num_of_groups = 10
    value_function = value_functions.AggregationValueFunction(num_of_groups)
    distribution = np.zeros(N_STATES + 1)
    for episode in tqdm(range(episodes)):
        methods.gradient_MC(value_function, alpha, distribution)

    state_values = np.asarray([value_function.value(state) for state in states])
    distribution /= np.sum(distribution)

    plt.figure(figsize=(20, 20))

    plt.subplot(2, 1, 1)
    plt.plot(states, true_values, label='true value')
    plt.plot(states, state_values, label='approximate MC value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(states, distribution[1:], label='state distribution')
    plt.xlabel('State')
    plt.ylabel('distribution')
    plt.legend()

    plt.savefig('../images/figure_9_1.png')
    plt.close()


# semi-gradient TD in 1000-state random walk, similar with the gradient MC
def figure_9_2_left():
    set_state_range(100)

    episodes = int(1e5)
    alpha = 2e-4
    num_of_groups = 10
    value_function = value_functions.AggregationValueFunction(num_of_groups)
    for episode in tqdm(range(episodes)):
        methods.semi_gradient_TD(value_function, 1, alpha)
    state_values = np.asarray([value_function.value(state) for state in states])

    plt.plot(states, state_values, label='Approximate TD value')
    plt.plot(states, true_values, label='True value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()


# compare different alphas and steps for semi-gradient TD
def figure_9_2_right():
    set_state_range(50)

    # all possible steps and alphas
    steps = np.power(2, np.arange(11))
    alphas = np.arange(0, 1.1, 0.1)
    # each run has 10 episodes while we need perform 100 independent runs
    episodes = 10
    runs = 100
    # set the number of groups, this case we have 20 aggregations
    num_of_groups = 20
    # track the errors for each case of (step, alpha) combination
    errors = np.zeros((len(steps), len(alphas)))
    for run in tqdm(range(runs)):
        for step_index, step in enumerate(steps):
            for alpha_index, alpha in enumerate(alphas):
                value_function = value_functions.AggregationValueFunction(num_of_groups)
                for episode in range(episodes):
                    methods.semi_gradient_TD(value_function, step, alpha)
                    state_values = np.asarray([value_function.value(state) for state in states])
                    # calculate the RMS error
                    errors[step_index, alpha_index] += np.sqrt(np.mean(np.power(true_values - state_values, 2)))
    # take average
    errors /= episodes * runs
    # truncate the error
    for i in range(len(steps)):
        plt.plot(alphas, errors[i, :], label='n = ' + str(steps[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()


def figure_9_2():
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    figure_9_2_left()
    plt.subplot(1, 2, 2)
    figure_9_2_right()

    plt.savefig('../images/figure_9_2.png')
    plt.close()


# compare the basis of Polynomials and Fourier
def figure_9_5():
    set_state_range(100)

    # my machine can only afford 3 run
    runs = 3
    episodes = 5000
    # the order of bases
    orders = [5, 10, 20]
    # alpha for Polynomials and Fourier
    alphas = [1e-4, 5e-5]

    labels = ['polynomial basis', 'fourier basis']
    # track errors for each episode
    errors = np.zeros((len(alphas), len(orders), episodes))
    for run in tqdm(range(runs)):
        for i in range(len(orders)):
            basis_value_functions = [value_functions.BasisValueFunction(orders[i], value_functions.POLYNOMIAL_BASIS),
                                     value_functions.BasisValueFunction(orders[i], value_functions.FOURIER_BASIS)]
            for j in range(len(basis_value_functions)):
                for episode in range(episodes):
                    # gradient MC algorithm
                    methods.gradient_MC(basis_value_functions[j], alphas[j])
                    # get approximating state values
                    state_values = np.asarray([basis_value_functions[j].value(state) for state in states])
                    # calculate the RMS error
                    errors[j, i, episode] += np.sqrt(np.mean(
                        np.power(true_values - state_values, 2)))
    # take average
    errors /= runs

    for i in range(len(alphas)):
        for j in range(len(orders)):
            plt.plot(errors[i, j, :], label='%s order = %d' % (labels[i], orders[j]))
    plt.xlabel('Episodes')
    plt.ylabel('RMSE')
    plt.legend()

    plt.savefig('../images/figure_9_5.png')
    plt.close()


# compare the tile coding with the state aggregation
def figure_9_10():
    set_state_range(50)

    # my machine can only afford 3 run
    runs = 3
    episodes = 5000
    # set the parameters of tile coding
    num_of_tilings = 50
    tile_width = 200
    tiling_offset = 4

    labels = ['tile coding (50 tilings)', 'state aggregation (one tiling)']
    # track errors for each episode
    errors = np.zeros((len(labels), episodes))
    for run in tqdm(range(runs)):
        tiling_value_functions = [value_functions.TilingValueFunction(num_of_tilings, tile_width, tiling_offset),
                                  value_functions.AggregationValueFunction(N_STATES // tile_width)]

        for i in range(len(tiling_value_functions)):
            for episode in range(episodes):
                # references the way to set step-size parameter from the source code of Shangtong Zhang
                # if just use alpha = 0.0001, one will get the effect showed in figure_9_10_abv.png
                # we can see the effect is also fine at the second half
                alpha = 1e-4
                # gradient MC algorithm
                methods.gradient_MC(tiling_value_functions[i], alpha)
                state_values = state_values = np.asarray([tiling_value_functions[i].value(state) for state in states])
                # calculate the RMS error
                errors[i][episode] += np.sqrt(np.mean(np.power(true_values - state_values, 2)))
    # take average
    errors /= runs

    for i in range(len(labels)):
        plt.plot(errors[i], label=labels[i])
    plt.xlabel('Episodes')
    plt.ylabel('RMSE')
    plt.legend()

    plt.savefig('../images/figure_9_10_abv.png')
    plt.close()


if __name__ == '__main__':
    compute_true_values()
    figure_9_1()
    figure_9_2()
    figure_9_5()
    figure_9_10()
