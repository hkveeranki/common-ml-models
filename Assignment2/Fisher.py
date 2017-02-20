"""
This file contains code for Fisher Linear Discriminant
    """
import numpy as np
import copy


class Fisher:
    """This class deals with Fisher Descriminant"""

    def __init__(self):
        """Empty Constructor"""
        self.classsifier = None

    def run(self, data1, data2):
        """ Runs the Fishers Linear Discriminant algorithm
            Produces the classifier
        """
        mean_1 = self.__make_mean(data1)
        mean_2 = self.__make_mean(data2)
        sw1 = self.__makecovariance(data1, mean_1)
        sw2 = self.__makecovariance(data2, mean_2)
        sw = copy.deepcopy(sw1)
        for i in range(len(sw)):
            for j in range(len(sw[i])):
                sw[i][j] += sw2[i][j]
        swArray = np.array(sw)
        mu1 = np.array(mean_1)
        mu1 = np.reshape(mu1, (len(mean_1), 1))
        mu2 = np.array(mean_2)
        mu2 = np.reshape(mu2, (len(mean_2), 1))
        swinv = np.linalg.inv(swArray)
        delmu = mu1 - mu2
        res = np.dot(swinv, delmu)
        res /= np.linalg.norm(res)
        self.classsifier = res.tolist()
        return res

    def __make_mean(self, matrix):
        """ returns the mean vector for the matrix"""
        means = []
        for j in range(len(matrix[0])):
            sum = 0.0
            for i in range(len(matrix)):
                sum += matrix[i][j]
            means.append(sum / len(matrix))
        return means

    def __makecovariance(self, data1, mean1):
        """ returns the covariance matrix """
        covariance = []
        for i in range(len(data1[0])):
            cov = []
            for j in range(len(data1[0])):
                sum = 0
                for k in range(len(data1)):
                    sum += (data1[k][i] - mean1[i]) * (data1[k][j] - mean1[j])
                cov.append(sum)
            covariance.append(cov)
        return covariance
