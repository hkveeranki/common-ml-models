import numpy as np
from random import random


class Neuron:
    """
    Neuron implements the functionality of a neuron (node) in multilayer
    neural network.
    """

    def __init__(self, size):
        self.delta = 0
        self.__size = size
        self.weights = [random() for i in range(size)]
        self.bias = random()

    def update_weights(self, inputs, eta):
        """
        Update weights of the neuron
        weight_i = weight_i + eta * delta * input_i
        :param inputs: input given to this node
        :param eta: learning rate
        """
        self.weights = self.weights + np.multiply(self.weights,
                                                  inputs) * self.delta * eta
        self.bias = eta * self.delta

    def activate(self, sample):
        return self.bias + sum(np.multiply(self.weights, sample))

    def propagate_delta(self, index):
        return self.weights[index] * self.delta
