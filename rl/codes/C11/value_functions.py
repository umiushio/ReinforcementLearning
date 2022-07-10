#######################################################################
# Copyright (C)                                                       #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# value functions needed to solve the task of Baird's counterexample

import numpy


# the ordinary value function for linear function approximation:
# only has the weights and features or alphas,together with the value function, learn and update methods
class ValueFunction:
    # initialize with given weights and features and optional term alpha
    # the weights and features must in numpy.array type
    def __init__(self, weights, features, alpha=None):
        self.weights = weights
        self.features = features
        self.alpha = alpha

    # return the current estimated value of given state
    def value(self, state):
        return numpy.dot(self.features[state], self.weights)

    # learn the target for the weight vector by using the usual update rule
    def learn(self, state, target, rho=1, alpha=None):
        if alpha is None:
            alpha = self.alpha
        self.weights += alpha * rho * (target - numpy.dot(self.features[state], self.weights)) * self.features[state]

    # update the weights by given delta in a usual update rule
    def update(self, state, delta, rho=1, alpha=None):
        if alpha is None:
            alpha = self.alpha
        self.weights += alpha * rho * delta * self.features[state]


# the correction version of value function for the GTD method
# it has the gradient correction and the common functions
class CorrectionValueFunction:
    # initialize with given weights, features and corrections denoted as v
    # and optional terms of alpha and beta
    # the weights, features and corrections must in numpy.array type
    def __init__(self, weights, features, v, alpha=None, beta=None):
        self.weights = weights
        self.features = features
        self.v = v
        self.alpha = alpha
        self.beta = beta

    # return the current estimated value of given state
    def value(self, state):
        return numpy.dot(self.weights, self.features[state])

    # update the corrections and return its value
    # there are two forms that both perform successfully in this task
    # I apt the one commented that maybe have a better performance
    # moreover, it enable return the value but not update the corrections by setting change=True
    def correct(self, state, delta, rho, beta=None, write=True):
        if beta is None:
            beta = self.beta
        if write:
            v_copy = self.v
        else:
            v_copy = self.v.copy()
        # v_copy += beta * (rho * delta - numpy.dot(self.v, self.features[state])) * self.features[state]
        v_copy += beta * rho * (delta - numpy.dot(self.v, self.features[state])) * self.features[state]
        return numpy.dot(v_copy, self.features[state])

    # update the weights by given delta in a usual update rule
    def update(self, state, delta, rho=1, alpha=None):
        if alpha is None:
            alpha = self.alpha
        self.weights += alpha * rho * delta * self.features[state]

    # update the weights directly by given delta vector
    def update_directly(self, delta_vector, alpha=None):
        if alpha is None:
            alpha = self.alpha
        self.weights += alpha * delta_vector

    # learn the target for the weight vector by using the usual update rule
    def learn(self, state, target, rho=1, alpha=None):
        if alpha is None:
            alpha = self.beta
        self.weights += alpha * rho * (target - numpy.dot(self.weights, self.features[state])) * self.features[state]

