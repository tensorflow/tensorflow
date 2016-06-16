# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Non-linear estimator tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators._sklearn import accuracy_score
from tensorflow.contrib.learn.python.learn.estimators._sklearn import mean_squared_error


class NonLinearTest(tf.test.TestCase):
  """Non-linear estimator tests."""

  def setUp(self):
    random.seed(42)
    tf.set_random_seed(42)

  def testIrisDNN(self):
    iris = tf.contrib.learn.datasets.load_iris()
    classifier = tf.contrib.learn.TensorFlowDNNClassifier(
        hidden_units=[10, 20, 10], n_classes=3)
    classifier.fit(iris.data, iris.target)
    score = accuracy_score(iris.target, classifier.predict(iris.data))
    self.assertGreater(score, 0.9, "Failed with score = {0}".format(score))
    weights = classifier.weights_
    self.assertEqual(weights[0].shape, (4, 10))
    self.assertEqual(weights[1].shape, (10, 20))
    self.assertEqual(weights[2].shape, (20, 10))
    self.assertEqual(weights[3].shape, (10, 3))
    biases = classifier.bias_
    self.assertEqual(len(biases), 5)

  def testBostonDNN(self):
    boston = tf.contrib.learn.datasets.load_boston()
    regressor = tf.contrib.learn.TensorFlowDNNRegressor(
        hidden_units=[10, 20, 10], n_classes=0,
        batch_size=boston.data.shape[0], steps=300, learning_rate=0.01)
    regressor.fit(boston.data, boston.target)
    score = mean_squared_error(boston.target, regressor.predict(boston.data))
    self.assertLess(score, 110, "Failed with score = {0}".format(score))
    weights = regressor.weights_
    self.assertEqual(weights[0].shape, (13, 10))
    self.assertEqual(weights[1].shape, (10, 20))
    self.assertEqual(weights[2].shape, (20, 10))
    self.assertEqual(weights[3].shape, (10, 1))
    biases = regressor.bias_
    self.assertEqual(len(biases), 5)

  def testDNNDropout0(self):
    # Dropout prob == 0.
    iris = tf.contrib.learn.datasets.load_iris()
    classifier = tf.contrib.learn.TensorFlowDNNClassifier(
        hidden_units=[10, 20, 10], n_classes=3, dropout=0.0)
    classifier.fit(iris.data, iris.target)
    score = accuracy_score(iris.target, classifier.predict(iris.data))
    self.assertGreater(score, 0.9, "Failed with score = {0}".format(score))

  def testDNNDropout0_1(self):
    # Dropping only a little.
    iris = tf.contrib.learn.datasets.load_iris()
    classifier = tf.contrib.learn.TensorFlowDNNClassifier(
        hidden_units=[10, 20, 10], n_classes=3, dropout=0.1)
    classifier.fit(iris.data, iris.target)
    score = accuracy_score(iris.target, classifier.predict(iris.data))
    # If the quality is lower - dropout is not working.
    self.assertGreater(score, 0.9, "Failed with score = {0}".format(score))

  def testDNNDropout0_9(self):
    # Dropping out most of it.
    iris = tf.contrib.learn.datasets.load_iris()
    classifier = tf.contrib.learn.TensorFlowDNNClassifier(
        hidden_units=[10, 20, 10], n_classes=3, dropout=0.9)
    classifier.fit(iris.data, iris.target)
    score = accuracy_score(iris.target, classifier.predict(iris.data))
    self.assertGreater(score, 0.3, "Failed with score = {0}".format(score))
    # If the quality is higher - dropout is not working.
    self.assertLess(score, 0.6, "Failed with score = {0}".format(score))

  # TODO(ipolosukhin): Restore or remove this.
  # def testDNNAutoencoder(self):
  #   import numpy as np
  #   iris = datasets.load_iris()
  #   autoencoder = learn.TensorFlowDNNAutoencoder(hidden_units=[10, 20])
  #   transformed = autoencoder.fit_transform(iris.data[1:2])
  #   expected = np.array([[
  #       -3.57627869e-07, 1.17000043e+00, 1.01902664e+00, 1.19209290e-07,
  #       0.00000000e+00, 1.19209290e-07, -5.96046448e-08, -2.38418579e-07,
  #       9.74681854e-01, 1.19209290e-07]])
  #   self.assertAllClose(transformed, expected)


if __name__ == "__main__":
  tf.test.main()
