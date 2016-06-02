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

"""Tests for estimators.linear."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class LinearClassifierTest(tf.test.TestCase):

  def testTrain(self):
    """Tests that loss goes down with training."""

    def input_fn():
      return {
          'age': tf.constant([1]),
          'language': tf.SparseTensor(values=['english'],
                                      indices=[[0, 0]],
                                      shape=[1, 1])
      }, tf.constant([[1]])

    language = tf.contrib.layers.sparse_column_with_hash_bucket('language', 100)
    age = tf.contrib.layers.real_valued_column('age')

    classifier = tf.contrib.learn.LinearClassifier(
        feature_columns=[age, language])
    classifier.fit(input_fn=input_fn, steps=100)
    loss1 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    classifier.fit(input_fn=input_fn, steps=200)
    loss2 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss2, loss1)
    self.assertLess(loss2, 0.01)

  def testTrainOptimizerWithL1Reg(self):
    """Tests l1 regularized model has higher loss."""

    def input_fn():
      return {
          'language': tf.SparseTensor(values=['hindi'],
                                      indices=[[0, 0]],
                                      shape=[1, 1])
      }, tf.constant([[1]])

    language = tf.contrib.layers.sparse_column_with_hash_bucket('language', 100)
    classifier_no_reg = tf.contrib.learn.LinearClassifier(
        feature_columns=[language])
    classifier_with_reg = tf.contrib.learn.LinearClassifier(
        feature_columns=[language],
        optimizer=tf.train.FtrlOptimizer(learning_rate=1.0,
                                         l1_regularization_strength=100.))
    loss_no_reg = classifier_no_reg.fit(
        input_fn=input_fn, steps=100).evaluate(
            input_fn=input_fn, steps=1)['loss']
    loss_with_reg = classifier_with_reg.fit(
        input_fn=input_fn, steps=100).evaluate(
            input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss_no_reg, loss_with_reg)

  def testTrainWithMissingFeature(self):
    """Tests that training works with missing features."""

    def input_fn():
      return {
          'language': tf.SparseTensor(values=['Swahili', 'turkish'],
                                      indices=[[0, 0], [2, 0]],
                                      shape=[3, 1])
      }, tf.constant([[1], [1], [1]], dtype=tf.int32)

    language = tf.contrib.layers.sparse_column_with_hash_bucket('language', 100)
    classifier = tf.contrib.learn.LinearClassifier(feature_columns=[language])
    classifier.fit(input_fn=input_fn, steps=100)
    loss = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss, 0.01)

  def testEval(self):
    """Tests that eval produces correct metrics.
    """

    def input_fn():
      return {
          'age': tf.constant([[1], [2]]),
          'language': tf.SparseTensor(values=['greek', 'chinise'],
                                      indices=[[0, 0], [1, 0]],
                                      shape=[2, 1]),
      }, tf.constant([[1], [0]])

    language = tf.contrib.layers.sparse_column_with_hash_bucket('language', 100)
    age = tf.contrib.layers.real_valued_column('age')
    classifier = tf.contrib.learn.LinearClassifier(
        feature_columns=[age, language])

    # Evaluate on trained mdoel
    classifier.fit(input_fn=input_fn, steps=100)
    classifier.evaluate(input_fn=input_fn, steps=2)

    # TODO(ispir): Enable accuracy check after resolving the randomness issue.
    # self.assertLess(evaluated_values['loss/mean'], 0.3)
    # self.assertGreater(evaluated_values['accuracy/mean'], .95)


class LinearRegressorTest(tf.test.TestCase):

  def testRegression(self):
    """Tests that loss goes down with training."""

    def input_fn():
      return {
          'age': tf.constant([1]),
          'language': tf.SparseTensor(values=['english'],
                                      indices=[[0, 0]],
                                      shape=[1, 1])
      }, tf.constant([[10.]])

    language = tf.contrib.layers.sparse_column_with_hash_bucket('language', 100)
    age = tf.contrib.layers.real_valued_column('age')

    classifier = tf.contrib.learn.LinearRegressor(
        feature_columns=[age, language])
    classifier.fit(input_fn=input_fn, steps=100)
    loss1 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    classifier.fit(input_fn=input_fn, steps=200)
    loss2 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']

    self.assertLess(loss2, loss1)
    self.assertLess(loss2, 0.01)

  def testRecoverWeights(self):
    rng = np.random.RandomState(67)
    n = 1000
    n_weights = 10
    bias = 2
    x = rng.uniform(-1, 1, (n, n_weights))
    weights = 10 * rng.randn(n_weights)
    y = np.dot(x, weights)
    y += rng.randn(len(x)) * 0.05 + rng.normal(bias, 0.01)
    regressor = tf.contrib.learn.LinearRegressor()
    regressor.fit(x, y, batch_size=32, steps=1000)
    # Have to flatten weights since they come in (x, 1) shape.
    self.assertAllClose(weights, regressor.weights_.flatten(), rtol=0.01)
    # TODO(ispir): Disable centered_bias.
    # assert abs(bias - regressor.bias_) < 0.1


def boston_input_fn():
  boston = tf.contrib.learn.datasets.load_boston()
  features = tf.cast(tf.reshape(tf.constant(boston.data), [-1, 13]), tf.float32)
  target = tf.cast(tf.reshape(tf.constant(boston.target), [-1, 1]), tf.float32)
  return features, target


class InferedColumnTest(tf.test.TestCase):

  def testTrain(self):
    est = tf.contrib.learn.LinearRegressor()
    est.fit(input_fn=boston_input_fn, steps=1)
    _ = est.evaluate(input_fn=boston_input_fn, steps=1)


if __name__ == '__main__':
  tf.test.main()
