import scipy

from .activation_function import ActivationFunction
from numpy import exp
import numpy as np


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.
    f(x) = 1/(1+e^(-x))
    derivative = f*(1-f)
    """

    def forward(self, net):
        return 1/(1 + np.exp(-net))

    def derivative(self, output):
        return output * (1 - output)
