from .activation_function import ActivationFunction
import numpy as np


class Softmax(ActivationFunction):
    """
    Softmax activation function.
    f(x) = e ^ (x - max(x)) / sum(e^(x - max(x))
    derivative = f*(1-f)
    """

    def forward(self, net):
        max_input = np.max(net)
        return np.exp(net - max_input)/sum(np.exp(net - max_input))

    def derivative(self, output):
        return output * (1 - output)
