#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# plot the figures for the mountain car's task that showed in the book

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

import environment
import value_functions
import methods
matplotlib.use('Agg')


# plot the cost, that is -max_value, for all states
def plot_cost(value_function: value_functions.ValueFunction, episode, ax):
    grid_size = 40
    positions = np.linspace(environment.POSITION_MIN, environment.POSITION_MAX, grid_size)
    velocities = np.linspace(environment.VELOCITY_MIN, environment.VELOCITY_MAX, grid_size)
    costs = []
    for position in positions:
        for velocity in velocities:
            costs.append(-value_function.max_value(position, velocity))
    axis_x, axis_y = np.meshgrid(positions, velocities)
    ax.scatter(axis_x, axis_y, costs)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('cost')
    ax.set_title('Episode %d' % episode)


# Figure 10.1, use the semi-gradient SARSA with epsilon-greedy policy
def figure_10_1():
    # get approximate MC value of states
    episodes = int(1e4)
    alpha = 0.3

    dimension = 2
    value_function = value_functions.TilingValueFunction(dimension, ints=[environment.ACTION_SIZE], alpha=alpha)

    # plot the cost for specific episodes
    mark_episodes = [1, 10, 100, 1000, 10000]
    figure = plt.figure(figsize=(30, 20))
    axes = [figure.add_subplot(2, 3, i + 1, projection='3d') for i in range(len(mark_episodes))]

    for episode in tqdm(range(episodes)):
        methods.semi_gradient_n_step_sarsa(value_function)
        if episode + 1 in mark_episodes:
            plot_cost(value_function, episode + 1, axes[mark_episodes.index(episode + 1)])

    plt.savefig('../images/figure_10_1.png')
    plt.close()


# Figure 10.2, the learning curves in different alphas
def figure_10_2():
    # set the episodes, alphas and runs
    episodes = 500
    alphas = [0.1, 0.2, 0.5]
    runs = 100

    steps = np.zeros((len(alphas), episodes))

    dimension = 2

    for run in tqdm(range(runs)):
        for i, alpha in enumerate(alphas):
            value_function = value_functions.TilingValueFunction(dimension, ints=[environment.ACTION_SIZE], alpha=alpha)
            for episode in range(episodes):
                steps[i][episode] += methods.semi_gradient_n_step_sarsa(value_function)

    steps /= runs

    num_of_tilings = value_functions.TilingValueFunction.get_num_of_tilings(dimension)
    for i, alpha in enumerate(alphas):
        plt.plot(steps[i][:], label='alpha = ' + str(alpha) + '/' + str(num_of_tilings))

    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.legend()

    plt.savefig('../images/figure_10_2.png')
    plt.close()


# Figure 10.3, learning performance for different n
def figure_10_3():
    runs = 100
    episodes = 500
    alphas = [0.5, 0.3]
    ns = [1, 8]
    dimension = 2
    steps = np.zeros((len(ns), episodes))

    for run in tqdm(range(runs)):
        for i in range(len(ns)):
            value_function = value_functions.TilingValueFunction(
                dimension, ints=[environment.ACTION_SIZE], alpha=alphas[i])
            for episode in range(episodes):
                steps[i, episode] += methods.semi_gradient_n_step_sarsa(value_function, n=ns[i])

    steps /= runs

    for i in range(len(ns)):
        plt.plot(steps[i][:], label='n = ' + str(ns[i]))
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.legend()

    plt.savefig('../images/figure_10_3.png')
    plt.close()


# Figure 10.4, effect of the alpha and n on early performance
def figure_10_4():
    runs = 50
    episodes = 50
    ns = np.power(2, np.arange(5))
    alphas = np.arange(0.2, 1.7, 0.1)
    max_steps = 300
    dimension = 2

    steps = np.zeros((len(alphas), len(ns)))

    for run in tqdm(range(runs)):
        for j, n in enumerate(ns):
            for i, alpha in enumerate(alphas):
                if (n == 2 and (alpha <= 0.4 or alpha >= 1.6)) or (n == 4 and alpha >= 1.4)\
                        or (n == 8 and alpha >= 1) or (n == 16 and alpha >= 0.8):
                    steps[i][j] += max_steps * episodes
                    continue
                value_function = value_functions.TilingValueFunction(
                    dimension, ints=[environment.ACTION_SIZE], alpha=alpha)
                for episode in range(episodes):
                    steps[i][j] += methods.semi_gradient_n_step_sarsa(value_function, n=n)
    steps /= runs * episodes

    for i, n in enumerate(ns):
        plt.plot(alphas, steps[:, i], label='n = ' + str(n))
    plt.xlabel('Alphas')
    plt.ylabel('Steps per Episode')
    plt.ylim(220, max_steps)
    plt.legend()

    plt.savefig('../images/figure_10_4.png')
    plt.close()


if __name__ == '__main__':
    figure_10_1()
    figure_10_2()
    figure_10_3()
    figure_10_4()
