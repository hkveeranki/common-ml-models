from .activation_function import ActivationFunction
from numpy import exp


class TanH(ActivationFunction, object):
    """
    tan hyperbolic as activation function.
    tanh(x) = (e^(-x) - e^(x))/(e^(-x) + e^(x))
    derivative = 1 - (tanh(x)*tanh(x)).
    """

    def forward(self, net):
        return (exp(net) - exp(-net)) / (exp(net) + exp(-net))

    def derivative(self, output):
        return 1 - (output * output)
