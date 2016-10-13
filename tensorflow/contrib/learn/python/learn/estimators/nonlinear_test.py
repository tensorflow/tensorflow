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

import tensorflow as tf


class NonLinearTest(tf.test.TestCase):
  """Non-linear estimator tests."""

  def setUp(self):
    random.seed(42)
    tf.set_random_seed(42)

  def testIrisDNN(self):
    iris = tf.contrib.learn.datasets.load_iris()
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3,
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))
    classifier.fit(iris.data, iris.target, max_steps=200)
    weights = classifier.weights_
    self.assertEqual(weights[0].shape, (4, 10))
    self.assertEqual(weights[1].shape, (10, 20))
    self.assertEqual(weights[2].shape, (20, 10))
    self.assertEqual(weights[3].shape, (10, 3))
    biases = classifier.bias_
    self.assertEqual(len(biases), 5)

  def testBostonDNN(self):
    boston = tf.contrib.learn.datasets.load_boston()
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=13)]
    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns, hidden_units=[10, 20, 10],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))
    regressor.fit(
        boston.data, boston.target, steps=300, batch_size=boston.data.shape[0])
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
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3,
        dropout=0.0, config=tf.contrib.learn.RunConfig(tf_random_seed=1))
    classifier.fit(iris.data, iris.target, max_steps=200)

  def testDNNDropout0_1(self):
    # Dropping only a little.
    iris = tf.contrib.learn.datasets.load_iris()
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3,
        dropout=0.1, config=tf.contrib.learn.RunConfig(tf_random_seed=1))
    classifier.fit(iris.data, iris.target, max_steps=200)

  def testDNNDropout0_9(self):
    # Dropping out most of it.
    iris = tf.contrib.learn.datasets.load_iris()
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3,
        dropout=0.9, config=tf.contrib.learn.RunConfig(tf_random_seed=1))
    classifier.fit(iris.data, iris.target, max_steps=200)


if __name__ == "__main__":
  tf.test.main()
