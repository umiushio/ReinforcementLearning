#######################################################################
# Copyright (C)                                                       #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# reproduce the figures in the book of Baird's counterexample task

import numpy
import value_functions
import methods
import matplotlib.pyplot as plt

# the initial weight vector and fixed feature function
weights = [10.0, 1, 1, 1, 1, 1, 1, 1]
features = [[1.0, 0, 0, 0, 0, 0, 0, 2],
            [0, 2, 0, 0, 0, 0, 0, 1],
            [0, 0, 2, 0, 0, 0, 0, 1],
            [0, 0, 0, 2, 0, 0, 0, 1],
            [0, 0, 0, 0, 2, 0, 0, 1],
            [0, 0, 0, 0, 0, 2, 0, 1],
            [0, 0, 0, 0, 0, 0, 2, 1]]
# the discounting factor
gamma = 0.99


# plot figure 11.2
# we use the lambda to unify the two subplots while the only difference is the algorithm adopted
def figure_11_2():
    alpha = 0.01

    value_function_vector = [value_functions.ValueFunction(numpy.asarray(weights), numpy.asarray(features), alpha),
                             value_functions.ValueFunction(numpy.asarray(weights), numpy.asarray(features), alpha)]
    semi_gradient_methods = [lambda value_function, loop, trajectory:
                             methods.semi_gradient_off_policy_temporal_difference(
                                 value_function, gamma, loop, weights_trajectory=trajectory),
                             lambda value_function, loop, trajectory:
                             methods.semi_gradient_dynamic_programming(
                                 value_function, gamma, loop, weights_trajectory=trajectory)]
    steps = 1000
    sweeps = 1000
    loops = [steps, sweeps]
    titles = ['Semi-gradient Off-policy TD', 'Semi-gradient DP']

    plt.figure(figsize=(20, 10))

    for i in range(len(titles)):
        weights_trajectory = [[weight] for weight in weights]
        semi_gradient_methods[i](value_function_vector[i], loops[i], weights_trajectory)
        plt.subplot(1, 2, i + 1)
        for j in range(len(weights)):
            plt.plot(range(loops[i] + 1), weights_trajectory[j], label='w' + str(j))
        plt.xlabel('Steps')
        plt.ylabel('Values')
        plt.title(titles[i])
        plt.legend()

    plt.savefig('../images/figure_11_2.png')
    plt.close()


# plot the figure 11.5 left part
def figure_11_5_left():
    alpha = 0.005
    beta = 0.05
    v = [0.0, 0, 0, 0, 0, 0, 0, 0]
    steps = 1000

    value_function = value_functions.CorrectionValueFunction(
        numpy.asarray(weights), numpy.asarray(features), numpy.asarray(v), alpha, beta)
    weights_trajectory = [[weight] for weight in weights]
    value_errors_trajectory = [methods.compute_rmsVE(value_function)]
    bellman_errors_trajectory = [methods.compute_rmsPBE(value_function, gamma)]
    methods.temporal_difference_with_gradient_correction(value_function, gamma, steps,
                                                         weights_trajectory=weights_trajectory,
                                                         value_errors_trajectory=value_errors_trajectory,
                                                         bellman_errors_trajectory=bellman_errors_trajectory)

    for i in range(len(weights)):
        plt.plot(range(steps + 1), weights_trajectory[i], label='w' + str(i))
    plt.plot(range(steps + 1), value_errors_trajectory, label='rmsVE')
    plt.plot(range(steps + 1), bellman_errors_trajectory, label='rmsPBE')
    plt.xlabel('Steps')
    plt.ylabel('Values')
    plt.title('TDC')
    plt.legend()


# plot the figure 11.5 right part
def figure_11_5_right():
    alpha = 0.005
    beta = 0.05
    v = [0.0, 0, 0, 0, 0, 0, 0, 0]
    sweeps = 1000

    value_function = value_functions.CorrectionValueFunction(
        numpy.asarray(weights), numpy.asarray(features), numpy.asarray(v), alpha, beta)
    weights_trajectory = [[weight] for weight in weights]
    value_errors_trajectory = [methods.compute_rmsVE(value_function)]
    bellman_errors_trajectory = [methods.compute_rmsPBE(value_function, gamma)]
    methods.expected_temporal_difference_with_gradient_correction(value_function, gamma, sweeps,
                                                                  weights_trajectory=weights_trajectory,
                                                                  value_errors_trajectory=value_errors_trajectory,
                                                                  bellman_errors_trajectory=bellman_errors_trajectory)

    for i in range(len(weights)):
        plt.plot(range(sweeps + 1), weights_trajectory[i], label='w' + str(i))
    plt.plot(range(sweeps + 1), value_errors_trajectory, label='rmsVE')
    plt.plot(range(sweeps + 1), bellman_errors_trajectory, label='rmsPBE')
    plt.xlabel('Steps')
    plt.ylabel('Values')
    plt.title('Expected TDC')
    plt.legend()


# plot the figure 11.5
def figure_11_5():
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    figure_11_5_left()
    plt.subplot(1, 2, 2)
    figure_11_5_right()

    plt.savefig('../images/figure_11_5.png')
    plt.close()


# plot the figure 11.6
# as the performance is much greater than the one in the book
# we only use 200 sweeps to get the similar shape of the figure
def figure_11_6():
    alpha = 0.03
    sweeps = 200
    value_function = value_functions.ValueFunction(numpy.asarray(weights), numpy.asarray(features), alpha)
    weights_trajectory = [[weight] for weight in weights]
    value_errors_trajectory = [methods.compute_rmsVE(value_function)]

    methods.expected_emphatic_temporal_difference(value_function, gamma, sweeps,
                                                  weights_trajectory=weights_trajectory,
                                                  value_errors_trajectory=value_errors_trajectory)
    for i in range(len(weights)):
        plt.plot(range(sweeps + 1), weights_trajectory[i], label='w' + str(i))
    plt.plot(range(sweeps + 1), value_errors_trajectory, label='rmsVE')
    plt.xlabel('Steps')
    plt.ylabel('Values')
    plt.title('Expected emphatic TD')
    plt.legend()

    plt.savefig('../images/figure_11_6.png')
    plt.close()


if __name__ == "__main__":
    figure_11_2()
    figure_11_5()
    figure_11_6()
