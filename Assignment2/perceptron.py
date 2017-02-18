"""
This file contains the algorithm for perceptron
"""


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
    for i in range(len(current_data) - 1):
        x.append(current_data[i])
    x.append(1)
    return x, current_data[2]


def update_w(w, x, y):
    """ Updates w after a misclassification """
    for i in range(len(w)):
        w[i] += y * x[i]
        #  w[i] = round(w[i], 10)
    return w


class Perceptron:
    """Class that enables us to run Perceptron"""

    def __init__(self, w, b):
        """ Constructor to initialise"""
        self.w = w
        self.w.append(b)
        self.steps_taken = 0

    def run(self, data):
        i = 0
        cnt = 0
        n = len(data)
        while True:
            if cnt == n:
                break
            x, y = make_data(data[i])
            c = dot_product(self.w, x)
            val = round(y * c, 4)
            if val <= 0:
                print("Updated ", self.w)
                self.w = update_w(self.w, x, y)
                prev_i = i
                print(" to ", self.w, "for", x, y)
                cnt = 0
            else:
                print('for', data[i], 'y = ', y, 'c = ', c)
                cnt += 1
            i += 1
            self.steps_taken += 1
            i %= n

    def print_result(self):
        print("converged with", self.w, 'in', self.steps_taken, 'steps')
