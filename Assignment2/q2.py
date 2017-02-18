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
cancer_acc_voted = []
ionosphere_acc_voted = []
cancer_acc_normal = []
ionosphere_acc_normal = []
for epoch in epochs:
    correct = kfoldcv(cancer_chunks, 10, cancer_w, epoch)
    cancer_acc_voted.append(float(correct[0]) / len(cancer_data))
    cancer_acc_normal.append(float(correct[1]) / len(cancer_data))
    correct = kfoldcv(ionosphere_chunks, 10, ionosphere_w, epoch)
    ionosphere_acc_voted.append(float(correct[0]) / len(ionosphere_data))
    ionosphere_acc_normal.append(float(correct[1]) / len(ionosphere_data))
    print("done for ", epoch)

print(ionosphere_acc_normal, ionosphere_acc_voted)
print(cancer_acc_normal, cancer_acc_voted)
