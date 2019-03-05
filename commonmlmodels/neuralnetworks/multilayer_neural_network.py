import sys
import time
import random
import numpy as np

from .layer import Layer


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
        layer.make_weights(num_weights)
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

    def train(self, x_train, y_train, x_val=None, y_val=None, n_epochs=100,
              eta=0.001, batch_size=64):
        """
        Train the neural network for given number of epochs using given data.
        :param y_val: Validation data samples
        :param x_val: Validation data labels
        :param batch_size: batch size to be used
        :param x_train: training sample data
        :param y_train: training label data
        :param n_epochs: number of epochs
        :param eta: learning rate
        """
        if not self.__compiled:
            sys.stderr.write(
                "Please compile the network before calling train.\n")
        sample_length = len(x_train[0])
        if not x_val or not y_val:
            x_val, y_val = x_train, y_train

        if self.input_length != sample_length:
            sys.stderr.write(
                ("Invalid training sample. input length for the neural network "
                 "is %d and the size of given training sample is %d\n" % (
                     self.input_length, sample_length)))
            sys.exit(-1)
        num_samples = len(x_train)
        for epoch in range(n_epochs):
            t = time.time()
            err = self.train_iteration(x_train, y_train, eta, batch_size)
            sys.stderr.write(
                "iteration %d took %.3f seconds and total_loss = %.3f\n" % (
                    epoch, time.time() - t, err / num_samples))
            print(self.accuracy(x_test=x_val, y_test=y_val))

    def train_iteration(self, x_train, y_train, eta, batch_size):
        """
        Perform an iteration of training on the neural network.
        :param batch_size: batch_size to be used
        :param x_train: training samples
        :param y_train: training labels
        :param eta: learning rate
        """

        def calculate_loss(actual, expected):
            return sum(
                [(expected[i] - actual[i]) ** 2 for i in range(len(expected))])

        total_error = 0
        num_samples = len(x_train)
        sys.stderr.write("Number of training samples is %d\n" % num_samples)
        num_iter = num_samples // batch_size
        for b in range(num_iter):
            t = time.time()
            for _ in range(batch_size):
                i = random.randint(0, num_samples - 1)
                sample = x_train[i].reshape(x_train[i].size, 1)
                label = y_train[i]
                label = label.reshape(label.size, 1)
                predicted = self.feed_forward(sample)
                total_error += calculate_loss(label, predicted)
                self.back_propagate(label, sample)
            self.update(eta=eta)
            print("batch %d of %d samples took %.3f seconds with loss %.3f" %
                  (b, batch_size, time.time() - t, total_error / num_samples))
        self.print_weights()
        return total_error

    def feed_forward(self, sample):
        """
        Perform the feed forward operation in the neural network.
        :param sample: the input to the neural network
        :return: output from the neural network
        """
        outputs = self.network[0].get_output(sample)
        for layer in self.network[1:]:
            # outputs of this layer are input to next layer
            outputs = layer.get_output(inputs=outputs)
        return np.array(outputs)

    def back_propagate(self, expected, sample):
        """
        Run the back propagation through the neural network. The formula for
        back propagation is given by the following equation
        error_n = delta_n * derivative(output)
        actual is the actual output for the given sample.
        For last layer delta = (actual - expected)
        other wise delta_k for a node k is sum(weight_k * error_j) for all j
        where error_j is error from jth neuron in the next layer and weight_k is
        weight that connects kth neuron of current layer to the jth neuron.
        :param expected: the actual label of the sample.
        :param sample: input used in the iteration.
        """
        delta = self.network[-1].get_delta_from_expected(expected)
        self.network[-1].update_error(deltas=delta,
                                      inputs=self.network[-2].get_output())

        for i in range(self.__num_layers - 2, 0, -1):
            layer = self.network[i]
            # get error from the layer i+1
            err = self.network[i + 1].get_delta_from_layer()
            # update delta of the neurons from the layer
            layer.update_error(deltas=err,
                               inputs=self.network[i - 1].get_output())

        err = self.network[1].get_delta_from_layer()
        self.network[0].update_error(deltas=err, inputs=sample)

    def update(self, eta):
        """
        Update weights of the neural network.
        :param sample: the input sample used in the iteration
        :param eta: learning rate
        """
        for i in range(self.__num_layers):
            self.network[i].update_weights(eta=eta)

    def predict_label(self, sample):
        """
        Predict the label for a sample
        :param sample: sample vector
        :return: predicted label
        """
        outputs = self.feed_forward(sample)
        # print(outputs)
        return np.argmax(outputs)

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
        preds = set()
        for i in range(num_samples):
            actual = np.argmax(y_test[i])
            y_pred = self.predict_label(x_test[i][:, None])
            preds.add(y_pred)
            if y_pred == actual:
                acc += 1
        print('Predicted', preds)
        return acc / num_samples

    def print_weights(self):
        itr = 0
        for layer in self.network:
            print('---------------- Layer', itr + 1, ' -----------------')
            layer.print_weights()
            print('---------------- Done -------------------')
            itr += 1
