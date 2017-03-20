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
    """ Activation function """
    return 1 / (1 + np.exp(-x))


def same_result(ycap, y):
    for i in range(len(y)):
        if y[i] != ycap[i]:
            return False
    return True


def process_data(filename):
    """ Process the data"""
    f = open(filename, 'r')
    lines = f.readlines()
    cur = START_LINE
    data_len = len(lines)
    while cur + INP_LEN < data_len:
        data = lines[cur:cur + INP_LEN]
        digit = int(lines[cur + INP_LEN])
        cur += INP_LEN + 1
        if digit in selected_chars:
            outputs.append(actual_outputs[digit])
            down_sample(data)


def down_sample(data):
    """ Reduce the size of the data """
    new_data = []
    box_len = 4
    number_of_boxes = int(INP_LEN / box_len)
    for i in range(number_of_boxes):
        for j in range(number_of_boxes):
            sum = 0
            for k in range(box_len):
                for l in range(box_len):
                    sum += int(data[box_len * i + k][box_len * j + l])
            new_data.append(sum / 4.0)
    inputs.append(new_data)


def predict(inp_data):
    """ Forward Propogation """
    inp = np.zeros((input_size, 1))
    for j in range(input_size):
        inp[j] = inp_data[j]
    hidden_nodes = f(np.dot(weightsji, inp) + biasji)
    output_nodes = f(np.dot(weightskj, hidden_nodes) + biaskj)
    return output_nodes


def caluclate_sensitivities(inp_data, out_data):
    """ Back Propogation Caluclate sensitivities"""
    inp = np.zeros((input_size, 1))
    out = np.zeros((output_size, 1))
    for i in range(input_size):
        inp[i] = inp_data[i]
    for i in range(output_size):
        out[i] = out_data[i]
    hidden_nodes = f(np.dot(weightsji, inp) + biasji)
    output_nodes = f(np.dot(weightskj, hidden_nodes) + biaskj)
    dy = output_nodes - out
    dy = output_nodes * (1 - output_nodes) * dy
    dbiaskj = dy
    dweightskj = np.dot(dy, hidden_nodes.T)
    dh = np.dot(weightskj.T, dy)
    dh_orig = dh * (1 - hidden_nodes) * hidden_nodes
    dbiasji = dh_orig
    dweightsji = np.dot(dh_orig, inp.T)
    return dweightsji, dweightskj, dbiasji, dbiaskj


def train_iteration(data, out):
    global weightsji, weightskj, biasji, biaskj
    dweightsji, dweightskj, dbiasji, dbiaskj = caluclate_sensitivities(data, out)
    weightsji -= eta * dweightsji
    weightskj -= eta * dweightskj
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
            if res[j] > 0.5:
                ycap.append(1)
            else:
                ycap.append(0)
        if not same_result(ycap, out[i]):
            misclass += 1
   # print(misclass)
    return (1 - misclass / len(inp)) * 100


process_data('optdigits-orig.cv')
train_nn()
# print(np.round(weightskj, 6))
# print(np.round(weightsji, 6))
# print(biaskj)
# print(biasji)

print(hidden_size, caluclate_final_accuracy(inputs, outputs))
