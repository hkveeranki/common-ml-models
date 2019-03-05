"""
This file contains the algorithm for perceptron
"""

from .utilities import dot_product, make_data, update_w


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
            if y * c <= 0:
                # print("Updated ", self.w)
                self.w = update_w(self.w, x, y)
                # print(" to ", self.w, "for", x, y)
                cnt = 0
            else:
                # print('for', data[i], 'y = ', y, 'c = ', c)
                cnt += 1
            i += 1
            self.steps_taken += 1
            i %= n

    def print_summary(self):
        print("converged at ", self.w, 'in', self.steps_taken, 'steps')
