from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np


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


data = [[0.1, 1.1, 1], [6.8, 7.1, 1], [-3.5, -4.1, 1], [2.0, 2.7, 1], [4.1, 2.8, 1], [3.1, 5.0, 1], [-0.8, -1.3, 1],
        [0.9, 1.2, 1], [5.0, 6.4, 1], [3.9, 4.0, 1], [7.1, 0.2, -1], [-1.4, -4.3, -1], [4.5, 0.0, -1], [6.3, 1.6, -1],
        [4.2, 1.9, -1], [1.4, -3.2, -1], [2.4, -4.0, -1], [2.5, -6.1, -1], [8.4, 3.7, -1], [4.1, -2.2, -1]]

x1, y1, x2, y2 = generate_graph_data(data)
Algo = Perceptron([0, 0], 0)
Algo.run(data)
Algo.print_result()

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
# Algo.print_result()
