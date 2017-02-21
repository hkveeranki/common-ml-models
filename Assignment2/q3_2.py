from utilities import crunch_data, error
from Fisher import Fisher
from leastsquare import LeastSquare
import matplotlib.pyplot as plt
import numpy as np


def class_decider(x):
    return int(x)


def class_divider(tmpdata):
    class1 = []
    class2 = []
    for i in tmpdata:
        if i[-1] == 1:
            class1.append(i[:-1])
        elif i[-1] == -1:
            class2.append(i[:-1])
        else:
            print(i)
            error('Bad data')
            exit(0)
    return class1, class2


data, n = crunch_data('table3.data', class_decider, float)
data1, data2 = class_divider(data)
Algo = Fisher()

w = Algo.run(data1, data2)

print("Fishers LDA Classifier", w)

a = -w[0] / w[1]
xx = np.linspace(-2, 4, 20)
yy = a * xx - (w[2]) / w[1]

plt.plot(xx, yy, c='black', ls='dashed', label='Fisher LDA classifier')

c1x = [data1[i][0] for i in range(len(data1))]
c1y = [data1[i][1] for i in range(len(data1))]
c2x = [data2[i][0] for i in range(len(data2))]
c2y = [data2[i][1] for i in range(len(data2))]

plt.scatter(c1x, c1y, c='blue')
plt.scatter(c2x, c2y, c='green')
ls = LeastSquare()
ls.run(data)

w = ls.classifier
print("Least Squares Approach", w)

a = -w[0] / w[1]
xx = np.linspace(-2, 4, 20)
yy = a * xx - (w[2]) / w[1]

plt.plot(xx, yy, 'r--', label='least squares classifier')
plt.legend()

plt.show()
