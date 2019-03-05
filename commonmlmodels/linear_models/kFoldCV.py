from .utilities import dot_product, make_data
from .voted_perceptron import VotedPerceptron


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
    Algo = VotedPerceptron(w_init, 0, epochs)
    Algo.run_voted(training_data)
    classifiers = Algo.weights
    votes = Algo.votes
    cnt1 = 0
    for data in testing_data:
        x, y = make_data(data)
        c = get_value(classifiers, votes, x)
        if sign(c) == y:
            cnt1 += 1
    cnt2 = 0
    for data in testing_data:
        x, y = make_data(data)
        c = dot_product(classifiers[-1], x)
        if sign(c) == y:
            cnt2 += 1

    return cnt1, cnt2


def kfoldcv(data_chunks, fold, w_init, epochs):
    """ Runs the K fold Cross validation"""
    cnt1 = 0
    cnt2 = 0
    for i in range(fold):
        training_data = []
        for j in range(fold):
            if i != j:
                for k in data_chunks[j]:
                    training_data.append(k)
        cnt = validate_set(training_data, data_chunks[i], w_init, epochs)
        cnt1 += cnt[0]
        cnt2 += cnt[1]
    return cnt1, cnt2
