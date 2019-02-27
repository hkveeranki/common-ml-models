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
        output[output<=0] = 0
        output[output>0] = 1
        return output
