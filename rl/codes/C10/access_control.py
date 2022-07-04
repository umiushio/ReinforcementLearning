#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# code implement for the access control example

import collections
import lib.tile_coding as tc
import seaborn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# the properties of environment or model
NUM_OF_SERVERS = 10
PRIORITIES = [1, 2, 4, 8]
REJECT_ACTION = 0
ACCEPT_ACTION = 1
ACTIONS = [REJECT_ACTION, ACCEPT_ACTION]
FREE_PROBABILITY = 0.06


# the environment of access control task
class Environment:
    # as the queue never empties, we actually do not need queue
    # we only need provide a custom with priority at each time step
    # for extension to the random queue, here we still keep it
    # similarly, there are also some other settings that can been reduced
    def __init__(self):
        self.queue = None
        self.max_servers = 0
        self.busy_servers = 0
        self.priority_probabilities = np.ones(len(PRIORITIES)) / len(PRIORITIES)
        self.current_custom_priority = 0
        self.free_probability = FREE_PROBABILITY
        # here the probability of custom arrival always 1
        self.arrive_probability = 1

    # begin the task, return the initial state
    def begin(self):
        self.queue = collections.deque()
        self.max_servers = NUM_OF_SERVERS
        self.busy_servers = 0
        self.update()
        return self.get_num_of_free_servers(), self.current_custom_priority

    # update the model at one time step
    def update(self):
        self.free_servers()
        self.add_custom()
        if self.queue:
            self.current_custom_priority = self.queue[0]
            self.queue.popleft()
        else:
            self.current_custom_priority = 0

    # add custom to queue at one time step
    # in this case, always add one custom per time step
    def add_custom(self):
        if np.random.binomial(1, self.arrive_probability) == 1:
            self.queue.append(np.random.choice(PRIORITIES, p=self.priority_probabilities))

    # get the priority of the custom at head of the queue
    def get_current_custom_priority(self):
        return self.current_custom_priority

    # get the number of free servers
    def get_num_of_free_servers(self):
        return self.max_servers - self.busy_servers

    # the busy servers may be free per time step
    def free_servers(self):
        self.busy_servers -= np.random.binomial(self.busy_servers, self.free_probability)

    # determine whether the servers are all busy
    def is_full_busy(self):
        if self.busy_servers >= self.max_servers:
            return True
        return False

    # make a step by taking the given action
    def step(self, action):
        if action == ACCEPT_ACTION:
            self.busy_servers += 1
            reward = self.current_custom_priority
        else:
            reward = 0
        self.update()
        return self.get_num_of_free_servers(), self.current_custom_priority, reward


# get the action followed the epsilon-greedy policy
# @model: the model of task
# @epsilon: the parameter of soft-greedy policy
# @argmax_actions: all actions that maximize the state value
def get_action(model: Environment, epsilon, argmax_actions):
    if model.is_full_busy():
        return REJECT_ACTION
    else:
        if np.random.binomial(1, epsilon) == 0:
            return np.random.choice(argmax_actions)
        else:
            return np.random.choice(ACTIONS)


# as the book says, we here only use the tabular value function that uses the IHT
# we also can rewrite this to the tiling value function to adapt the complicated situations
class ValueFunction:
    def __init__(self, ints, alpha=1):
        self.max_size = tc.compute_max_size(0, ints)
        self.hash_table = tc.IndexHashTable(self.max_size)
        self.weights = np.zeros(self.max_size)
        self.alpha = alpha

    # get the index for the given state
    def get_index(self, num_of_free_servers, current_custom_priority, action):
        return tc.tiles(self.hash_table, 1, [], ints=[num_of_free_servers, current_custom_priority, action])

    # estimate the state-action value
    def value(self, num_of_free_servers, current_custom_priority, action):
        index = self.get_index(num_of_free_servers, current_custom_priority, action)
        return self.weights[index]

    # update the state-action value by delta
    # here we use update not learn as it will be more convenient for the differential algorithm
    def update(self, num_of_free_servers, current_custom_priority, action, delta, alpha=None):
        index = self.get_index(num_of_free_servers, current_custom_priority, action)
        if alpha is None:
            alpha = self.alpha
        self.weights[index] += alpha * delta

    # return the max state value among actions
    def max_value(self, num_of_free_servers, current_custom_priority):
        if num_of_free_servers == 0:
            return self.value(num_of_free_servers, current_custom_priority, REJECT_ACTION)
        else:
            return max([self.value(num_of_free_servers, current_custom_priority, action) for action in ACTIONS])

    # return the actions that maximize the state value
    def argmax_actions(self, num_of_free_servers, current_custom_priority):
        if num_of_free_servers == 0:
            return [REJECT_ACTION]
        else:
            action_values = [self.value(num_of_free_servers, current_custom_priority, action) for action in ACTIONS]
            return np.where(action_values == np.max(action_values))[0]


# the differential semi-gradient Sarsa algorithm
# I think the mean reward is unique to this method that I define it in methods not in value functions
# @value_function: an instance of class ValueFunction
# @alpha: the step-size for learning target
# @beta: the step-size for updating mean reward
# @epsilon: the parameter of soft-greedy policy
# @max_steps: the maximum steps for the task
def differential_semi_gradient_sarsa(value_function: ValueFunction, alpha=None, beta=1, epsilon=0, max_steps=int(1e6)):
    # initialize the model and mean reward
    model = Environment()
    mean_reward = 0

    # start the task
    num_of_free_servers, current_custom_priority = model.begin()
    argmax_actions = value_function.argmax_actions(num_of_free_servers, current_custom_priority)
    action = get_action(model, epsilon, argmax_actions)
    for s in tqdm(range(max_steps)):
        new_num_of_free_servers, new_custom_priority, reward = model.step(action)
        argmax_actions = value_function.argmax_actions(num_of_free_servers, current_custom_priority)
        new_action = get_action(model, epsilon, argmax_actions)
        delta = reward - mean_reward + value_function.value(new_num_of_free_servers, new_custom_priority, new_action)\
                - value_function.value(num_of_free_servers, current_custom_priority, action)
        mean_reward += beta * delta
        value_function.update(num_of_free_servers, current_custom_priority, action, delta, alpha)
        num_of_free_servers = new_num_of_free_servers
        current_custom_priority = new_custom_priority
        action = new_action


def figure_10_5():
    # set the necessary parameters
    alpha = 0.01
    beta = 0.01
    epsilon = 0.1
    max_steps = int(2e6)

    # learn the value function
    value_function = ValueFunction([len(PRIORITIES), NUM_OF_SERVERS + 1, len(ACTIONS)], alpha)
    differential_semi_gradient_sarsa(value_function, beta=beta, epsilon=epsilon, max_steps=max_steps)

    # plot the differential values for all states
    values = np.zeros((len(PRIORITIES), NUM_OF_SERVERS + 1))
    fig = plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)

    for i, priority in enumerate(PRIORITIES):
        for free_servers in range(NUM_OF_SERVERS + 1):
            values[i, free_servers] = value_function.max_value(free_servers, priority)
        plt.plot(range(NUM_OF_SERVERS + 1), values[i, :], label='priority ' + str(priority))

    plt.xlabel('Number of free servers')
    plt.ylabel('Differential value of best action')
    plt.legend()

    # plot the policy for selecting the best action for all states
    # as few experience in 10-free servers case, the best action in this case may be different from the book's
    ax = fig.add_subplot(2, 1, 2)
    policy = np.zeros((len(PRIORITIES), NUM_OF_SERVERS + 1))
    for i, priority in enumerate(PRIORITIES):
        for free_servers in range(NUM_OF_SERVERS + 1):
            policy[i, free_servers] = value_function.argmax_actions(free_servers, priority)[0]

    fig = seaborn.heatmap(policy, cmap="YlGnBu", ax=ax, xticklabels=range(NUM_OF_SERVERS + 1), yticklabels=PRIORITIES)
    fig.set_title('Policy (Reject 0, Accept 1)')
    fig.set_xlabel('Number of free servers')
    fig.set_ylabel('Priority')

    plt.savefig('../images/figure_10_5.png')
    plt.close()


if __name__ == '__main__':
    figure_10_5()
    