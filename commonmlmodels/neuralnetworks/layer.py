import sys

import numpy as np

from .activationfunctions.activation_function import \
    ActivationFunction


class Layer:
    """
    Layer implements the functionality of a layer in the neural network.
    """

    def __init__(self, size, activation_function):
        self.size = size
        if not isinstance(activation_function, ActivationFunction):
            sys.stderr.write((
                'activation function should implement'
                ' base class ActivationFunction'))
            sys.exit(-1)
        self.activation_function = activation_function
        self.__weights = None
        self.__bias = None
        self.__errors = None
        self.__acc_weight_diffs = None
        self.__acc_bias_diffs = None
        self.__output = None

    def make_weights(self, num_weights):
        """
        Make the nodes in the given layer.
        :param num_weights: number of weights needed at each node.
        """
        self.__weights = np.random.rand(self.size, num_weights)
        self.__bias = np.random.rand(self.size, 1)
        self.__initialize()

    def get_output(self, inputs=None):
        """
        Calculate the output from the layer.
        :param inputs: inputs if provided will be used to calculate the output.
        :return: the output from the layer
        """
        if inputs is not None:
            net_activation = np.dot(self.__weights, inputs) + self.__bias
            self.__output = self.activation_function.forward(net_activation)
        return np.array(self.__output)

    def get_delta_from_expected(self, expected):
        """
        Get error from expected value. This is done for the last layer.
        :param expected: expected output from the layer
        :return: the error
        """
        return self.__output - expected

    def get_delta_from_layer(self):
        """Calculate the error to be propagated to the previous layer."""
        return np.dot(self.__weights.T, self.__errors)

    def update_error(self, deltas, inputs):
        """
        Update delta for all the nodes in the neural network
        error = delta * derivative(output)
        :param deltas: value of delta in the above formula
        """
        self.__errors = deltas * self.activation_function.derivative(
            self.__output)
        self.__acc_weight_diffs += np.dot(self.__errors, inputs.T)
        self.__acc_bias_diffs += self.__errors

    def update_weights(self, eta):
        """
        Update weights to all nodes of this layer
        :param eta: learning rate
        """
        self.__weights -= eta * self.__acc_weight_diffs
        self.__bias -= eta * self.__acc_bias_diffs
        # Normalize the values.
        new_weights = np.concatenate((self.__weights, self.__bias), axis=1)
        norm = np.linalg.norm(new_weights, axis=1, ord=2)
        new_weights = new_weights.astype(np.float) / norm[:, None]
        self.__bias = new_weights[:, -1][:, None]
        self.__weights = new_weights[:, :-1]
        self.__initialize()

    def print_weights(self):
        print('weights', self.__weights)
        print('bias', self.__bias)

    def __initialize(self):
        self.__acc_weight_diffs = np.zeros(self.__weights.shape)
        self.__acc_bias_diffs = np.zeros((self.size, 1))
