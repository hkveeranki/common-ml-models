import sys
import time

from .layer import Layer


def get_error(actual, expected):
    return sum([(expected[i] - actual[i]) ** 2 for i in range(len(expected))])


class MultiLayerNeuralNetwork:
    """
    This class implements multi layer neural network.
    """

    def __init__(self, n_input, n_output):
        self.network = []
        self.input_length = n_input
        self.output_length = n_output
        self.__compiled = False
        self.__last_added_layer_size = None
        self.__num_layers = 0

    def add(self, layer):
        """
        Adds a given layer to the neural network
        :param layer: input layer which is an object of `Layer` class
        """
        if self.__last_added_layer_size is None:
            num_weights = self.input_length
        else:
            num_weights = self.__last_added_layer_size
        layer.make_nodes(num_weights)
        self.network.append(layer)
        self.__last_added_layer_size = layer.size
        self.__num_layers += 1

    def compile(self, output_activation_function):
        """
        Compile the given neural network by adding the final layer.
        :param output_activation_function: activation function for the
               final layer.
        """
        output_layer = Layer(size=self.output_length,
                             activation_function=output_activation_function)
        self.add(output_layer)
        self.__compiled = True

    def train(self, x_train, y_train, n_epochs, eta):
        """
        Train the neural network for given number of epochs using given data.
        :param x_train: training sample data
        :param y_train: training label data
        :param n_epochs: number of epochs
        :param eta: learning rate
        """
        if not self.__compiled:
            sys.stderr.write(
                "Please compile the network before calling train.\n")
        sample_length = len(x_train[0])
        if self.input_length != sample_length:
            sys.stderr.write(
                ("Invalid training sample. input length for the neural network "
                 "is %d and the size of given training sample is %d\n" % (
                     self.input_length, sample_length)))
            sys.exit(-1)

        for epoch in range(n_epochs):
            t = time.time()
            self.train_iteration(x_train, y_train, eta)
            sys.stderr.write(
                "iteration %d took %.3f seconds\n" % (epoch, time.time() - t))

    def train_iteration(self, x_train, y_train, eta):
        """
        Perform an iteration of training on the neural network.
        :param x_train: training samples
        :param y_train: training labels
        :param eta: learning rate
        """
        total_error = 0
        t = time.time()
        num_samples = len(x_train)
        sys.stderr.write("Number of training samples is %d\n" % num_samples)
        for i in range(num_samples):
            # print("using sample", i)
            sample = x_train[i]
            label = y_train[i]
            predicted = self.feed_forward(sample)
            total_error += get_error(label, predicted)
            self.back_propagate(label)
            self.update(sample, eta)
            if i % 100 == 0:
                nth = i // 100
                print("%d00 - %d00 samples took %.3f seconds" %
                      (nth, nth + 1, time.time() - t))
                t = time.time()
            # print("Done.")

    def feed_forward(self, sample):
        """
        Perform the feed forward operation in a neural network
        :param sample: the input to the neural network
        :return: output from the neural network
        """
        inputs = sample
        for i in range(self.__num_layers):
            # outputs of this layer are input to next layer
            inputs = self.network[i].get_output(inputs=inputs)
        return inputs

    def back_propagate(self, expected):
        """
        Run the back propagation through the neural network. The formula for
        back propagation is given by the following equation.
        error_n = delta_n * derivative(actual)
        actual is the actual output for the given sample.
        For last layer delta is (actual - expected)
        other wise delta_k for a node k is sum(weight_k * error_j) for all j
        where error_j is error from jth neuron in the next layer and weight_k is
        weight that connects kth neuron of current layer to the jth neuron.
        :param expected: the actual label of the sample.
        """
        for i in range(self.__num_layers - 1, 0, -1):
            layer = self.network[i]
            if i == self.__num_layers - 1:
                # get the error from the expected
                err = layer.get_error_from_expected(expected)
            else:
                # get error from the layer i+1
                err = layer.get_error_from_layer(layer=self.network[i + 1])
            # update delta of the neurons from the layer
            layer.update_delta(err)

    def update(self, sample, eta):
        """
        Update weights of the neural network.
        :param sample: the input sample used in the iteration
        :param eta: learning rate
        """
        for i in range(self.__num_layers):
            if i == 0:
                # input is the sample itself
                inputs = sample
            else:
                # input is the output of the previous layer
                inputs = self.network[i - 1].get_output()
            # Apply it for each neuron in the layer
            self.network[i].update_node_weights(eta=eta, inputs=inputs)

    def predict(self, sample):
        """
        Predict the label for a sample
        :param sample: sample vector
        :return: predicted label
        """
        outputs = self.feed_forward(sample)
        return outputs.index(max(outputs))

    def save_weights(self, save_path):
        """
        Save the network into a pickle file
        :param save_path: path of the file
        """
        import pickle
        with open(save_path, "wb") as f:
            pickle.dump(self.network, f)

    def load_weights(self, load_path):
        """
        Load the network from a pickle dump
        :param load_path: path to the pickle file
        """
        import pickle
        with open(load_path, "rb") as f:
            self.network = pickle.load(f)

    def accuracy(self, x_test, y_test):
        acc = 0.0
        num_samples = len(x_test)
        for i in range(num_samples):
            y_pred = self.predict(x_test[i])
            if y_pred == y_test[i]:
                acc += 1
        return acc / num_samples
