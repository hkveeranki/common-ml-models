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


data, n = crunch_data('table2.data', class_decider, int)
data1, data2 = class_divider(data)
Algo = Fisher()
print(Algo.run(data1, data2))
ls = LeastSquare()
print(ls.run(data))
