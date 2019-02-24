from .activation_function import ActivationFunction


class ReLU(ActivationFunction):
    """
    ReLU activation function.
    f(x) = max(0,x)
    derivative = 0 if x < 0 else 1
    """

    def forward(self, net):
        return net * (net > 0)

    def derivative(self, output):
        if output < 0:
            return 0
        return 1
