from utilities import dot_product, make_data
from votedPerceptron import votedPerceptron


def sign(val):
    if val < 0:
        return -1
    elif val > 0:
        return 1
    return 0


def get_value(classifiers, votes, data):
    val = 0
    for i in range(len(classifiers)):
        val += votes[i] * sign(dot_product(classifiers[i], data))
    return val


def validate_set(training_data, testing_data, w_init, epochs):
    """ Runs the verification for training set and testing set"""
    Algo = votedPerceptron(w_init, 0, epochs)
    Algo.run_voted(training_data)
    classifiers = Algo.weights
    votes = Algo.votes
    cnt = 0
    for data in testing_data:
        x, y = make_data(data)
        c = get_value(classifiers, votes, x)
        if sign(c) == y:
            cnt += 1
    return cnt


def kfoldcv(data_chunks, fold, w_init, epochs):
    """ Runs the K fold Cross validation"""
    cnt = 0
    for i in range(fold):
        training_data = []
        for j in range(fold):
            if i != j:
                for k in data_chunks[j]:
                    training_data.append(k)
        cnt += validate_set(training_data, data_chunks[i], w_init, epochs)
    return cnt
