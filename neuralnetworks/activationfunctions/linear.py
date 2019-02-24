from .activation_function import ActivationFunction


class Linear(ActivationFunction):
    """
    Linear activation function
    f(x) = x
    derivative = 1
    """

    def forward(self, net):
        return net

    def derivative(self, output):
        return 1
