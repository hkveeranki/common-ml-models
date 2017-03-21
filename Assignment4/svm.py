from random import shuffle

from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

inputs = []
outputs = []
START_LINE = 1


def divide(data):
    y = int(data[-1])
    data = data[:-1]
    x = []
    for _ in range(len(data)):
        x.append(float(data[_]))
    return x, y


def process_data(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    data = []
    n = len(lines)
    for ind in range(START_LINE, n):
        tmp = lines[ind].strip(' \n').split(',')
        x_data, y_data = divide(tmp)
        x_data.append(y_data)
        data.append(x_data)
    return data


def seperate_data(data):
    label = data[-1]
    features = data[:-1]
    return features, label


def chunk_data(lst, n):
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


def kfoldcv(data, fold, c, deg=None, gamma=None):
    corclass_poly = 0
    corclass_rbf = 0
    for curfold in range(fold):
        training_inputs = []
        training_outputs = []
        testing_inputs = []
        testing_outputs = []
        for j in range(fold):
            for inp in data[j]:
                x, y = seperate_data(inp)
                if j != curfold:
                    training_inputs.append(x)
                    training_outputs.append(y)
                else:
                    testing_inputs.append(x)
                    testing_outputs.append(y)
        classifier1 = svm.SVC(C=c, kernel='poly', degree=deg)
        classifier2 = svm.SVC(C=c, kernel='rbf', gamma=gamma)
        classifier1.fit(training_inputs, training_outputs)
        classifier2.fit(training_inputs, training_outputs)
        for j in range(len(testing_inputs)):
            res1 = classifier1.predict([testing_inputs[j]])
            res2 = classifier2.predict([testing_inputs[j]])
            if res1[0] == testing_outputs[j]:
                corclass_poly += 1
            if res2[0] == testing_outputs[j]:
                corclass_rbf += 1
    return corclass_poly, corclass_rbf


crude_data = process_data('Data_SVM.csv')
print("Processing Done...")
degree_min = 2
degree_max = 3
C_min = 0.5
sigma_min = 0.2
sigma_max = 0.5
num_iter = 3
acc_poly = {}
acc_rbf = {}
sd_poly = {}
sd_rbf = {}
print('Running 10 Fold Validation')
for _ in range(num_iter):
    C = C_min + _ * 0.5
    for step in range(num_iter):
        degree = degree_min + step
        sigma = round(sigma_min + step * 0.1, 3)
        poly_accuracies = []
        rbf_accuracies = []
        for i in range(30):
            shuffle(crude_data)
            chunks = chunk_data(crude_data, 10)
            poly_cor, rbf_cor = kfoldcv(chunks, 10, 1.0, degree, sigma)
            poly_accuracies.append((poly_cor / len(crude_data)) * 100)
            rbf_accuracies.append((rbf_cor / len(crude_data)) * 100)

        mean_poly = round(sum(poly_accuracies) / 30, 3)
        mean_rbf = round(sum(rbf_accuracies) / 30, 3)
        poly_sd = rbf_sd = 0
        for j in range(30):
            poly_sd += (poly_accuracies[i] - mean_poly) ** 2
            rbf_sd += (rbf_accuracies[i] - mean_rbf) ** 2
        poly_sd /= 30
        rbf_sd /= 30
        poly_sd = round(poly_sd, 4)
        rbf_sd = round(rbf_sd, 4)
        acc_poly[(C, degree)] = mean_poly
        acc_rbf[(C, sigma)] = mean_rbf
        sd_poly[(C, degree)] = poly_sd
        sd_rbf[(C, sigma)] = rbf_sd
best_poly = [0, 0]
best_rbf = [0, 0]
best_ac1 = 0
best_ac2 = 0
for i in acc_poly.keys():
    if acc_poly[i] > best_ac1:
        best_poly[0] = i[0]
        best_poly[1] = i[1]
        best_ac1 = acc_poly[i]
    print('C:', i[0], 'degree:', i[1], "accuracy:", acc_poly[i], "standard deviation:", sd_poly[i])
for i in acc_rbf.keys():
    if acc_rbf[i] > best_ac2:
        best_rbf[0] = i[0]
        best_rbf[1] = i[1]
        best_ac2 = acc_rbf[i]
    print('C:', i[0], 'sigma:', i[1], "accuracy:", acc_rbf[i], "standard deviation:", sd_rbf[i])
print("Our Best Choice")
print(best_poly)
print(best_rbf)

final_input_data = []
final_output_data = []
for line in crude_data:
    X, Y = seperate_data(line)
    final_input_data.append(X)
    final_output_data.append(Y)
clf1 = svm.SVC(C=best_poly[0], degree=best_poly[1], kernel='poly')
clf1.fit(final_input_data, final_output_data)
clf2 = svm.SVC(C=best_rbf[0], gamma=best_rbf[1], kernel='rbf')
clf2.fit(final_input_data, final_output_data)
x_max = x_min = final_input_data[0][0]
y_max = y_min = final_input_data[0][1]
for i in final_input_data:
    x_max = max(x_max, i[0])
    y_max = max(y_max, i[1])
    x_min = min(x_min, i[0])
    y_min = min(y_min, i[1])

h = 0.001
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
x1 = []
y1 = []
for i in range(len(final_input_data)):
    x1.append(final_input_data[i][0])
    y1.append(final_input_data[i][1])
Y = final_output_data
# Plot1
plt.figure(1)
plt.title('Best Choice SVM with Polynomial kernel')
Z = clf1.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.imshow(Z, interpolation='nearest',
           extent=(x_min - 0.4, x_max + 0.4, y_min - 0.4, y_max + 0.4), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=3,
                       linetypes='--')
plt.scatter(x1, y1, s=20, c=Y, cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([-1, 1, -1, 1])

# RBF Classifier

plt.figure(2)
plt.title('Best Choice SVM with RBF Kernel')
Z = clf2.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.imshow(Z, interpolation='nearest',
           extent=(x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours2 = plt.contour(xx, yy, Z, levels=[0], linewidths=3,
                        linetypes='--')
plt.scatter(x1, y1, s=30, c=Y, cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([-1, 1, -1, 1])

plt.show()
