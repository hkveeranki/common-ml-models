from numpy import log
from random import shuffle

EXCLUDE = [0, 5, 16, 17, 18, 24, 39]
INIT_CNT = 0.00000000000001  # to avoid division by zero error
DEBUG = 0


def format_string(data_string):
    return data_string.strip(' \n')


def give_needed(data):
    """ Only extract needed attributes and returns a list"""
    needed = []
    data = format_string(data)
    data = data.split(',')
    for _ in range(len(data)):
        if _ not in EXCLUDE:
            needed.append(format_string(data[_]))
    return needed


def process_data(filename):
    """ Process the data from the file"""
    f = open(filename, 'r')
    lines = f.readlines()
    for index in range(len(lines)):
        lines[index] = format_string(lines[index])
    f.close()
    return lines


def train_classifier(raw_data):
    """ Train the classifier from the given crude_data"""
    attr_len = len(give_needed(raw_data[0])) - 1
    counts = [{} for _ in range(attr_len)]
    prob_Y = [0, 0]
    n = len(raw_data)
    for ind in range(len(raw_data)):
        final_data = give_needed(raw_data[ind])
        Y = final_data[-1]
        if Y == '- 50000.':
            prob_Y[0] += 1
        else:
            prob_Y[1] += 1
        for Xi in range(len(final_data) - 1):
            attrib = final_data[Xi]
            if attrib == '?' and len(counts[Xi]) != 0:
                attrib = max(counts[Xi], key=counts[Xi].get)
            if attrib not in counts[Xi]:
                counts[Xi][attrib] = [INIT_CNT, INIT_CNT]
            if Y == '- 50000.':
                counts[Xi][attrib][0] += 1

            else:
                counts[Xi][attrib][1] += 1
    div = log(n)
    prob_Y[0] = log(prob_Y[0]) - div
    prob_Y[1] = log(prob_Y[1]) - div
    for Xi in range(attr_len):
        cnts = counts[Xi]
        for key in cnts.keys():
            cnts[key][0] = log(cnts[key][0]) - div - prob_Y[0]
            cnts[key][1] = log(cnts[key][1]) - div - prob_Y[1]

    return counts, prob_Y


def chunk_data(lst, n):
    """ Chunk the data into lists to k fold cv"""
    increment = len(lst) / float(n)
    last = 0
    index = 1
    results = []
    while last < len(lst):
        idx = int(round(increment * index))
        results.append(lst[last:idx])
        last = idx
        index += 1
    return results


def get_label(inp, class_conditional, class_prob):
    """ Predict the label from the given prior probabilites and class conditional densities"""
    c0 = class_prob[0]
    c1 = class_prob[1]
    label = 0
    for Xj in range(len(class_conditional)):
        attrib = inp[Xj]
        dist = class_conditional[Xj]
        if attrib == '?' or attrib not in dist.keys():
            maxp = -1000000
            mode = 0
            for key in dist.keys():
                p = sum(dist[key])
                if p > maxp:
                    mode = key
            attrib = mode

        c0 += dist[attrib][0]
        c1 += dist[attrib][1]
    if c0 < c1:
        label = 1
    return label


def kfoldcv(input_data, fold):
    """ Run the the Kfold cross validation"""
    accur = 0
    for curfold in range(fold):
        training_inputs = []
        testing_inputs = []
        for tmpj in range(fold):
            for inp in input_data[tmpj]:
                if tmpj != curfold:
                    training_inputs.append(inp)
                else:
                    testing_inputs.append(inp)

        accur += test_classifier(training_inputs, testing_inputs)
    return accur / fold


def test_classifier(training_inputs, testing_inputs):
    labels = ['- 50000.', '50000+.']
    corclass = 0
    num_data_points = len(testing_inputs)
    if DEBUG == 1:
        print('Training...')
    conditional_probs, class_probs = train_classifier(training_inputs)
    if DEBUG == 1:
        print('Done...\n Testing')
    for test_j in testing_inputs:
        final_data = give_needed(test_j)
        Y = final_data[-1]
        Ycap = get_label(final_data, conditional_probs, class_probs)
        if labels[Ycap] == Y:
            corclass += 1
    if DEBUG == 1:
        print('Done.')
    return (corclass / num_data_points) * 100


print("Reading data...")
crude_data = process_data('census-income.data')
accuracies = []
data_chunks = []
# Testing part with provided test data
# test_data = process_data('census-income.test')
# print('Done.\nTesting Accuracy:', test_classifier(crude_data, test_data))

print('Done.\nRunning 10 Fold CV...')
for i in range(30):
    shuffle(crude_data)
    data_chunks = chunk_data(crude_data, 10)
    acc = kfoldcv(data_chunks, 10)
    accuracies.append(round(acc, 4))
    print(i, 'Done acc:', acc)
mean_acc = sum(accuracies) / len(accuracies)
mean_acc = round(mean_acc, 6)
sd_acc = 0
for j in accuracies:
    sd_acc += round((j - mean_acc) ** 2, 6)
print('Done.\n Mean Accuracy is ', mean_acc, 'Standard deviation is ', sd_acc)
