import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


def hyperbolic_tangent(z):
    return np.tanh(z)


def hyperbolic_tangent_prime(z):
    return 1 - (z * z)


def hyperbolic_tangent_prime0(z):
    return 1 - (np.tanh(z) * np.tanh(z))


def soft_plus(z):
    return np.log(1 + np.exp(z))


def soft_max(z):
    return np.exp(z) / np.sum(np.exp(z))