import unittest
from unittest import TestCase

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

from exercies import SKLEARN_DATA


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


class TestBookExamples(TestCase):

    def setUp(self) -> None:
        self.mnist = None
        self.initial()

    def initial(self):
        mnist = fetch_openml('mnist_784', version=1, cache=True, data_home=SKLEARN_DATA)
        mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings
        sort_by_target(mnist)
        self.mnist = mnist
        X = mnist["data"]
        Y = mnist["target"]
        self.x_train, self.x_test, self.y_train, self.y_test = X[:60000], X[60000:], Y[60000:], Y[:60000]

    def test_print_36000(self):
        image = self.mnist["data"][36000].reshape(28, 28)
        plt.imshow(image, cmap=matplotlib.cm.binary, interpolation="nearest")
        plt.axis("off")
        plt.show()


if __name__ == '__main__':
    unittest.main()
