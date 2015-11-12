"""Tests for skflow."""

from sklearn import datasets
from sklearn.metrics import accuracy_score, mean_squared_error

import skflow

from tensorflow.python.platform import googletest

class SkFlowTest(googletest.TestCase):

  def testIris(self):
    iris = datasets.load_iris()
    classifier = skflow.TensorFlowClassifier(n_classes=3)
    classifier.fit(iris.data, iris.target)
    score = accuracy_score(classifier.predict(iris.data), iris.target)
    self.assertGreater(score, 0.5, "Failed with score = {0}".format(score))

  def testBoston(self):
    boston = datasets.load_boston()
    regressor = skflow.TensorFlowRegressor(n_classes=0,
                                           batch_size=520,
                                           steps=1000,
                                           learning_rate=0.001)
    regressor.fit(boston.data, boston.target)
    score = mean_squared_error(boston.target, regressor.predict(boston.data))
    self.assertLess(score, 150, "Failed with score = {0}".format(score))


if __name__ == "__main__":
  googletest.main()

