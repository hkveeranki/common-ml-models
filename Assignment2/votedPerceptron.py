"""
This File contains algorithm for Voted Perceptron
"""

from utilities import dot_product, make_data, update_w
import copy


class votedPerceptron:
    """ This class implements Voted Perceptron """

    def __init__(self, w, b, epochs):
        """Default Constructor"""
        self.w = copy.deepcopy(w)
        self.w.append(b)
        self.votes = {}
        self.weights = []
        self.epochs = epochs

    def run_voted(self, data):
        m = len(data)
        curW = self.w
        n = 0
        self.weights.append(curW)
        self.votes[n] = 1
        for _ in range(self.epochs):
            for i in range(m):
                x, y = make_data(data[i])
                c = dot_product(curW, x)
                if y * c <= 0:
                    n = n + 1
                    curW = update_w(curW, x, y)
                    self.weights.append(curW)
                    # print(prevW, 'with votes', self.votes[n - 1], " => ", self.weights[n], "for", y, c)
                    self.votes[n] = 1
                self.votes[n] += 1

    def run_normal(self, data):
        """Runs normal perceptron with iterations"""
        m = len(data)
        curW = self.w
        for _ in range(self.epochs):
            for i in range(m):
                x, y = make_data(data[i])
                c = dot_product(curW, x)
                if y * c <= 0:
                    curW = update_w(curW, x, y)
        return curW

    def print_summary(self):
        # for i in range(len(self.weights)):
        #    print(self.weights[i], self.votes[i])
        print(len(self.weights))
