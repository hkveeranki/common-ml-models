from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from utilities import crunch_data, error


def class_decider(x):
    return int(x)


def generate_graph_data(given_data):
    tmpx1 = []
    tmpy1 = []
    tmpx2 = []
    tmpy2 = []
    for i in range(len(given_data)):
        tmp = given_data[i]
        if tmp[2] == -1:
            tmpx2.append(tmp[0])
            tmpy2.append(tmp[1])
        else:
            tmpx1.append(tmp[0])
            tmpy1.append(tmp[1])
    return tmpx1, tmpy1, tmpx2, tmpy2


data, n = crunch_data('table1_2.data', class_decider, float)
x1, y1, x2, y2 = generate_graph_data(data)
init_w = [0 for i in range(n)]
Algo = Perceptron(init_w, 0)
Algo.run(data)
Algo.print_summary()
# Plotting Graphs
plt.figure(1)
plt.scatter(x1, y1, c='blue')
plt.scatter(x2, y2, c='green')
w = Algo.w
a = -w[0] / w[1]
xx = np.linspace(-5, 9, 28)
yy = a * xx - (w[2]) / w[1]

plt.plot(xx, yy, 'r--')

plt.show()
