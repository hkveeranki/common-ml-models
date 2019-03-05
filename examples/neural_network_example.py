import numpy as np
from sklearn.preprocessing import OneHotEncoder

from commonmlmodels.neuralnetworks.activationfunctions.relu import ReLU
from commonmlmodels.neuralnetworks.activationfunctions.sigmoid import Sigmoid
from commonmlmodels.neuralnetworks.activationfunctions.softmax import Softmax
from commonmlmodels.neuralnetworks.activationfunctions.tanh import TanH
from commonmlmodels.neuralnetworks.layer import Layer
from commonmlmodels.neuralnetworks.multilayer_neural_network import \
    MultiLayerNeuralNetwork

START_LINE = 21
INP_LEN = 32


def process_data(filename):
    """ Process the data from the given file."""
    inputs = []
    outputs = []
    file_reader = open(filename, 'r')
    lines = file_reader.readlines()
    cur = START_LINE
    data_len = len(lines)
    while cur + INP_LEN < data_len:
        data = lines[cur:cur + INP_LEN]
        digit = int(lines[cur + INP_LEN])
        cur += INP_LEN + 1
        data = convert_data(data)
        inputs.append(data)
        outputs.append(digit)
    return np.array(inputs), np.array(outputs)


def convert_data(data):
    """Convert the data into training sample."""
    new_data = []
    for i in range(len(data)):
        row = []
        data[i] = data[i].strip(' \n')
        for j in range(len(data[0])):
            row.append(int(data[i][j]))
        new_data.append(row)
    return np.ravel(np.array(new_data))


x_train, y_train = process_data('neural_network_example_train.data')
x_test, y_test = process_data('neural_network_example_test.data')
integer_encoded = y_train.reshape(len(y_train), 1)
one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
y_train = one_hot_encoder.fit_transform(integer_encoded)
y_test = one_hot_encoder.fit_transform(y_test.reshape(len(y_test), 1))
input_len = len(x_train[0])
output_len = len(y_train[0])
nn = MultiLayerNeuralNetwork(n_input=input_len, n_output=output_len)
sigmoid = Sigmoid()
softmax = Softmax()
relu = ReLU()
tanh = TanH()
nn.add(Layer(size=1024, activation_function=sigmoid))
nn.add(Layer(size=512, activation_function=tanh))
nn.compile(output_activation_function=softmax)
nn.train(x_train, y_train, n_epochs=100, batch_size=64)
