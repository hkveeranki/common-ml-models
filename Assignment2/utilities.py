"""
Custom Functions to assist perceptron and voted perceptron
"""
import sys


def dot_product(x, y):
    """ Returns dot product of the two vectors present"""
    prod = 0
    for i in range(len(x)):
        prod += x[i] * y[i]
    # prod = round(prod, 2)
    return prod


def make_data(current_data):
    """ Formats the given data into the required form """
    x = []
    n = len(current_data)
    for i in range(n - 1):
        x.append(current_data[i])
    x.append(1)
    return x, current_data[n - 1]


def update_w(w, x, y):
    """ Updates w after a misclassification """
    newW = []
    for i in range(len(w)):
        newW.append(w[i] + y * x[i])
        #  w[i] = round(w[i], 10)
    return newW


def crunch_data(filename, classdecider, type_converter, start):
    """ Crunches the data into requirement """
    try:
        f = open(filename, 'r')
        data = []
        n = 1
        lines = f.readlines()
        for line in lines:
            s = line.strip('\n').split(',')
            n = len(s)
            if '?' not in s:
                x = s[start:(n - 1)]
                for i in range(len(x)):
                    x[i] = type_converter(x[i])
                x.append(classdecider(s[n - 1]))
                data.append(x)
        return data, n - 1 - start
    except IOError:
        sys.stderr.write('unable to open the given data file')
