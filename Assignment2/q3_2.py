from utilities import crunch_data, error
from Fisher import Fisher
from leastsquare import LeastSquare


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
print(Algo.run(data1, data2))
ls = LeastSquare()
print(ls.run(data))

c1x = [data1[i][0] for i in range(len(data1))]
c1y = [data1[i][1] for i in range(len(data1))]
c2x = [data2[i][0] for i in range(len(data2))]
c2y = [data2[i][1] for i in range(len(data2))]

plt.scatter(c1x, c1y, c='blue')
plt.scatter(c2x, c2y, c='green')

plt.show()
