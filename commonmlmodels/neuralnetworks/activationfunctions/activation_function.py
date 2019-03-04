class ActivationFunction:
    """Base class for all activation functions."""

    def forward(self, net):
        """
        Transfer the net activation to next layer.
        :param net: net activation received from previous layer
        :return: value to transfer to next layer.
        """
        raise NotImplementedError(
            "forward operation should be implemented by sub class")

    def derivative(self, output):
        """
        Back propagate the derivative based on the output received.
        :param output: output received from previous layer
        :return: Value to be back propagated.
        """
        raise NotImplementedError(
            "back propagate operation should be implemented by sub class")
