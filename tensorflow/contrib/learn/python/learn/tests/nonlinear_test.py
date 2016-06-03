# pylint: disable=g-bad-file-header
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

  def testIrisDNN(self):
    random.seed(42)
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
    random.seed(42)
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
    random.seed(42)
    # Dropout prob == 0.
    iris = tf.contrib.learn.datasets.load_iris()
    classifier = tf.contrib.learn.TensorFlowDNNClassifier(
        hidden_units=[10, 20, 10], n_classes=3, dropout=0.0)
    classifier.fit(iris.data, iris.target)
    score = accuracy_score(iris.target, classifier.predict(iris.data))
    self.assertGreater(score, 0.9, "Failed with score = {0}".format(score))

  def testDNNDropout0_1(self):
    random.seed(42)
    # Dropping only a little.
    tf.set_random_seed(42)
    iris = tf.contrib.learn.datasets.load_iris()
    classifier = tf.contrib.learn.TensorFlowDNNClassifier(
        hidden_units=[10, 20, 10], n_classes=3, dropout=0.1)
    classifier.fit(iris.data, iris.target)
    score = accuracy_score(iris.target, classifier.predict(iris.data))
    # If the quality is lower - dropout is not working.
    self.assertGreater(score, 0.9, "Failed with score = {0}".format(score))

  def testDNNDropout0_9(self):
    random.seed(42)
    # Dropping out most of it.
    iris = tf.contrib.learn.datasets.load_iris()
    classifier = tf.contrib.learn.TensorFlowDNNClassifier(
        hidden_units=[10, 20, 10], n_classes=3, dropout=0.9)
    classifier.fit(iris.data, iris.target)
    score = accuracy_score(iris.target, classifier.predict(iris.data))
    self.assertGreater(score, 0.3, "Failed with score = {0}".format(score))
    # If the quality is higher - dropout is not working.
    self.assertLess(score, 0.6, "Failed with score = {0}".format(score))

  def testRNN(self):
    random.seed(42)
    data = np.array(
        list([[2, 1, 2, 2, 3], [2, 2, 3, 4, 5], [3, 3, 1, 2, 1], [2, 4, 5, 4, 1]
             ]),
        dtype=np.float32)
    # labels for classification
    labels = np.array(list([1, 0, 1, 0]), dtype=np.float32)
    # targets for regression
    targets = np.array(list([10, 16, 10, 16]), dtype=np.float32)
    test_data = np.array(list([[1, 3, 3, 2, 1], [2, 3, 4, 5, 6]]),
                         dtype=np.float32)

    def _input_fn(X):
      # pylint: disable=invalid-name
      return tf.split(1, 5, X)

    # Classification
    classifier = tf.contrib.learn.TensorFlowRNNClassifier(rnn_size=2,
                                                          cell_type="lstm",
                                                          n_classes=2,
                                                          input_op_fn=_input_fn)
    classifier.fit(data, labels)
    # pylint: disable=pointless-statement
    classifier.weights_
    classifier.bias_
    # pylint: enable=pointless-statement
    predictions = classifier.predict(test_data)
    self.assertAllClose(predictions, np.array([1, 0]))

    classifier = tf.contrib.learn.TensorFlowRNNClassifier(rnn_size=2,
                                                          cell_type="rnn",
                                                          n_classes=2,
                                                          input_op_fn=_input_fn,
                                                          num_layers=2)
    classifier.fit(data, labels)
    classifier = tf.contrib.learn.TensorFlowRNNClassifier(
        rnn_size=2, cell_type="invalid_cell_type", n_classes=2,
        input_op_fn=_input_fn, num_layers=2)
    with self.assertRaises(ValueError):
      classifier.fit(data, labels)

    # Regression
    regressor = tf.contrib.learn.TensorFlowRNNRegressor(rnn_size=2,
                                                        cell_type="gru",
                                                        input_op_fn=_input_fn)
    regressor.fit(data, targets)
    # pylint: disable=pointless-statement
    regressor.weights_
    regressor.bias_
    # pylint: enable=pointless-statement
    predictions = regressor.predict(test_data)

  def testBidirectionalRNN(self):
    random.seed(42)
    data = np.array(
        list([[2, 1, 2, 2, 3], [2, 2, 3, 4, 5], [3, 3, 1, 2, 1], [2, 4, 5, 4, 1]
             ]),
        dtype=np.float32)
    labels = np.array(list([1, 0, 1, 0]), dtype=np.float32)

    def _input_fn(X):
      # pylint: disable=invalid-name
      return tf.split(1, 5, X)

    # Classification
    classifier = tf.contrib.learn.TensorFlowRNNClassifier(rnn_size=2,
                                                          cell_type="lstm",
                                                          n_classes=2,
                                                          input_op_fn=_input_fn,
                                                          bidirectional=True)
    classifier.fit(data, labels)
    test_data = np.array(list([[1, 3, 3, 2, 1], [2, 3, 4,
                                                 5, 6]]), dtype=np.float32)
    predictions = classifier.predict(test_data)
    self.assertAllClose(predictions, np.array([1, 0]))

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
