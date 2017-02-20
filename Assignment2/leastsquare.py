"""
This file contains code for least square approach
"""
import numpy as np
import copy


class LeastSquare:
    """This class implements least square approach"""

    def __init__(self):
        """Default Constructor"""
        self.classsifier = None

    def run(self, data):
        """ Runs the Least square approach algorithm
            Produces the classifier
        """
        new_data, y = self.__process_data(data)
        yarray = np.array(y)
        yarray = np.reshape(yarray, (len(y), 1))
        A = np.array(new_data)
        mat1 = np.dot(A.transpose(), A)
        mat2 = np.dot(A.transpose(), yarray)
        mat1 = np.linalg.inv(mat1)
        res = np.dot(mat1, mat2)
        # res /= np.linalg.norm(res)
        return res

    def __process_data(self, data):
        """ Formats the data"""
        A = copy.deepcopy(data)
        y = []
        for i in range(len(A)):
            p = A[i]
            y.append(p[-1])
            p[-1] = 1
        return A, y
