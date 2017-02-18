from utilities import crunch_data
from votedPerceptron import votedPerceptron
from perceptron import Perceptron
from kFoldCV import kfoldcv
import sys
from random import shuffle

cancerfile = 'breast-cancer-wisconsin.data'
ionospherefile = 'ionosphere.data'


def cancer_decider(attr):
    if attr == '2':
        return 1
    elif attr == '4':
        return -1
    else:
        sys.stderr.write('Data Error')


def ionosphere_decider(attr):
    if attr == 'b':
        return -1
    elif attr == 'g':
        return 1
    else:
        sys.stderr.write('Data Error')


def chunk_data(lst, n):
    increment = len(lst) / float(n)
    last = 0
    i = 1
    results = []
    while last < len(lst):
        idx = int(round(increment * i))
        results.append(lst[last:idx])
        last = idx
        i += 1
    return results


cancer_data, attrn_cancer = crunch_data(cancerfile, cancer_decider, int, 1)
ionosphere_data, attrn_ionosphere = crunch_data(ionospherefile, ionosphere_decider, float, 0)

shuffle(cancer_data)
shuffle(ionosphere_data)
cancer_chunks = chunk_data(cancer_data, 10)
ionosphere_chunks = chunk_data(ionosphere_data, 10)

cancer_w = [0 for i in range(attrn_cancer)]
ionosphere_w = [0 for j in range(attrn_ionosphere)]

epochs = [10, 15, 20, 25, 30, 35, 40, 45, 50]
cancer_acc = []
ionosphere_acc = []
for epoch in epochs:
    correct = kfoldcv(cancer_chunks, 10, cancer_w, epoch)
    cancer_acc.append(float(correct) / len(cancer_data))
    print(cancer_acc[-1])
    correct = kfoldcv(ionosphere_chunks, 10, ionosphere_w, epoch)
    ionosphere_acc.append(float(correct) / len(ionosphere_data))
    print(ionosphere_acc[-1])
