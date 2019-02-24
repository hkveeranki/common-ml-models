import sys

from neuralnetworks.activationfunctions.activation_function import \
    ActivationFunction
from .neuron import Neuron


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
        self.activation_function = activation_function
        self.nodes = []
        self.__output = None

    def make_nodes(self, num_weights):
        """
        Make the nodes in the given layer
        :param num_weights: number of weights needed at each node.
        """
        for i in range(self.size):
            self.nodes.append(Neuron(num_weights))

    def get_output(self, inputs=None):
        """
        Calculate the output from the layer.
        :param inputs: inputs if provided will be used to calculate the output.
        :return: the output from the layer
        """
        if inputs is None:
            if self.__output is None:
                sys.stderr.write('Feed forward has not been done.')
                sys.exit(-1)
            return self.__output
        outputs = []
        for i in range(self.size):
            node = self.nodes[i]
            net_activation = node.activate(inputs)
            output = self.activation_function.forward(net_activation)
            outputs.append(output)
        self.__output = outputs
        return outputs

    def get_error_from_expected(self, expected):
        """
        Get error from expected value. This is done for the last layer.
        :param expected: expected output from the layer
        :return: the error
        """
        errors = []
        for i in range(self.size):
            errors.append(expected[i] - self.__output[i])
        return errors

    def get_error_from_layer(self, layer):
        """Get the error required from the next layer."""
        errors = []
        for i in range(self.size):
            errors.append(layer.get_error_for_previous_layer(i))
        return errors

    def get_error_for_previous_layer(self, index):
        """Calculate the error to be propagated to the previous layer."""
        error = 0.0
        for i in range(self.size):
            error += self.nodes[i].propagate_delta(index)
        return error

    def update_delta(self, deltas):
        """
        Update delta for all the nodes in the neural network
        error = delta * derivative(output)
        :param deltas:
        :return:
        """
        for i in range(self.size):
            node = self.nodes[i]
            node.delta = deltas[i] * self.activation_function.derivative(
                self.__output[i])

    def update_node_weights(self, eta, inputs):
        """
        Update weights to all nodes of this layer
        :param eta: learning rate
        :param inputs: inputs to the layer
        """
        for j in range(self.size):
            self.nodes[j].update_weights(eta=eta, inputs=inputs)
