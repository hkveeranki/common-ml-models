import numpy as np
import random as ran

import sys

selected_chars = [0, 2, 5]

actual_outputs = {
    0: (0, 0, 0),
    2: (0, 1, 0),
    5: (1, 0, 1)
}

START_LINE = 21
INP_LEN = 32
inputs = []
outputs = []
test_inputs = []
test_outputs = []

final_needed = {}

eta = 0.01
decay = 0.999
input_size = 64
hidden_size = int(sys.argv[1])
output_size = 3

biasji = np.zeros((hidden_size, 1))
biaskj = np.zeros((output_size, 1))

weightsji = np.random.randn(hidden_size, input_size)
weightskj = np.random.randn(output_size, hidden_size)


def f(x):
    """ Activation function 
    :rtype: np.array
    """
    return 1 / (1 + np.exp(-x))


def same_result(ycap, y):
    for i in range(len(y)):
        if y[i] != ycap[i]:
            return False
    return True


def process_data(filename, test):
    """ Process the data"""
    file_reader = open(filename, 'r')
    lines = file_reader.readlines()
    cur = START_LINE
    data_len = len(lines)
    while cur + INP_LEN < data_len:
        data = lines[cur:cur + INP_LEN]
        digit = int(lines[cur + INP_LEN])
        cur += INP_LEN + 1
        if digit in selected_chars:
            if test:
                test_outputs.append(actual_outputs[digit])
            else:
                outputs.append(actual_outputs[digit])
            down_sample(data, test)


def down_sample(data, test):
    """ Reduce the size of the data """
    new_data = []
    box_len = 4
    number_of_boxes = int(INP_LEN / box_len)
    for i in range(number_of_boxes):
        for j in range(number_of_boxes):
            cnt = 0
            for k in range(box_len):
                for l in range(box_len):
                    cnt += int(data[box_len * i + k][box_len * j + l])
            new_data.append(cnt / 4.0)
    if test:
        test_inputs.append(new_data)
    else:
        inputs.append(new_data)


def predict(inp_data):
    """ Forward Propogation """
    inp = np.zeros((input_size, 1))
    for j in range(input_size):
        inp[j] = inp_data[j]
    Yj = f(np.dot(weightsji, inp) + biasji)
    Zk = f(np.dot(weightskj, Yj) + biaskj)
    return Zk


def caluclate_sensitivities(inp_data, out_data):
    """ Back Propagation Calculate sensitivities"""
    X = np.zeros((input_size, 1))
    Tk = np.zeros((output_size, 1))
    for i in range(input_size):
        X[i] = inp_data[i]
    for i in range(output_size):
        Tk[i] = out_data[i]
    Yj = f(np.dot(weightsji, X) + biasji)
    Zk = f(np.dot(weightskj, Yj) + biaskj)
    delk = (Zk - Tk) * (Zk) * (1 - Zk)  # Derivative of sigmoid is y*(1-y)
    deltakj = np.dot(delk, Yj.T)
    delj = np.dot(weightskj.T, delk) * Yj * (1 - Yj)  # Senisitivities of hidden nodes
    deltaji = np.dot(delj, X.T)
    return deltaji, deltakj, delj, delk


def train_iteration(data, out):
    global weightsji, weightskj, biasji, biaskj
    deltaji, deltakj, dbiasji, dbiaskj = caluclate_sensitivities(data, out)
    weightsji -= eta * deltaji
    weightskj -= eta * deltakj
    dbiasji -= eta * dbiasji
    dbiaskj -= eta * dbiaskj


def train_nn():
    iterations = 1000 * len(inputs)
    for i in range(iterations):
        ind = ran.randint(0, len(inputs) - 1)
        train_iteration(inputs[ind], outputs[ind])


def caluclate_final_accuracy(inp, out):
    misclass = 0
    for i in range(len(inp)):
        res = predict(inp[i])
        ycap = []
        for j in range(len(out[i])):
            if res[j] >= 0.5:
                ycap.append(1)
            else:
                ycap.append(0)
        if not same_result(ycap, out[i]):
            misclass += 1
    return (1 - misclass / len(inp)) * 100


process_data('neural_network_example_train.data', False)
process_data('neural_network_example_test.data', True)

train_nn()
print("nH:", hidden_size)
print("hidden-output weights:", np.round(weightskj, 6))
print("input-hidden weights", np.round(weightsji, 2))
print("hidden-output bias", biaskj)
print("input-hidden bias", biasji)

print("Accuracy:", caluclate_final_accuracy(test_inputs, test_outputs))
for j in actual_outputs.keys():
    for i in range(len(inputs)):
        if outputs[i] == actual_outputs[j]:
            Yj = np.round(f(np.dot(weightsji, inputs[i]) + biasji), 3)
            Zk = np.round(f(np.dot(weightskj, Yj) + biaskj), 3)
            print('For ', j)
            print("Intermediate Outputs", Yj)
            print("Final Outputs", Zk)
            break
