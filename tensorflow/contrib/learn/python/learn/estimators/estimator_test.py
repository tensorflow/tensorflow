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

"""Tests for Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators._sklearn import mean_squared_error


def boston_input_fn():
  boston = tf.contrib.learn.datasets.load_boston()
  features = tf.cast(
      tf.reshape(
          tf.constant(boston.data), [-1, 13]), tf.float32)
  target = tf.cast(
      tf.reshape(
          tf.constant(boston.target), [-1, 1]), tf.float32)
  return features, target


def iris_input_fn():
  iris = tf.contrib.learn.datasets.load_iris()
  features = tf.cast(
      tf.reshape(
          tf.constant(iris.data), [-1, 4]), tf.float32)
  target = tf.cast(
      tf.reshape(
          tf.constant(iris.target), [-1, 1]), tf.int32)
  return features, target


def boston_eval_fn():
  boston = tf.contrib.learn.datasets.load_boston()
  n_examples = len(boston.target)
  features = tf.cast(
      tf.reshape(
          tf.constant(boston.data), [n_examples, 13]), tf.float32)
  target = tf.cast(
      tf.reshape(
          tf.constant(boston.target), [n_examples, 1]), tf.float32)
  return tf.concat(0, [features, features]), tf.concat(0, [target, target])


def linear_model_fn(features, target, unused_mode):
  return tf.contrib.learn.models.linear_regression_zero_init(features, target)


def logistic_model_fn(features, target, unused_mode):
  return tf.contrib.learn.models.logistic_regression_zero_init(features, target)


class CheckCallsMonitor(tf.contrib.learn.monitors.BaseMonitor):

  def __init__(self):
    self.calls = None
    self.expect_calls = None

  def begin(self, max_steps):
    self.calls = 0
    self.expect_calls = max_steps

  def step_end(self, step, outputs):
    self.calls += 1
    return False

  def end(self):
    assert self.calls == self.expect_calls


class EstimatorTest(tf.test.TestCase):

  def testBostonAll(self):
    boston = tf.contrib.learn.datasets.load_boston()
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn,
                                     classification=False)
    est.fit(x=boston.data, y=boston.target.astype(np.float32), steps=100)
    scores = est.evaluate(
        x=boston.data,
        y=boston.target.astype(np.float32))
    predictions = est.predict(x=boston.data)
    other_score = mean_squared_error(predictions, boston.target)
    self.assertAllClose(other_score, scores['mean_squared_error'])

  def testIrisAll(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Estimator(model_fn=logistic_model_fn,
                                     classification=True)
    est.train(input_fn=iris_input_fn, steps=100)
    _ = est.evaluate(input_fn=iris_input_fn, steps=1)
    predictions = est.predict(x=iris.data)
    self.assertEqual(predictions.shape[0], iris.target.shape[0])

  def testTrainInputFn(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn,
                                     classification=False)
    est.train(input_fn=boston_input_fn, steps=1)
    _ = est.evaluate(input_fn=boston_eval_fn, steps=1)

  def testPredict(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn,
                                     classification=False)
    boston = tf.contrib.learn.datasets.load_boston()
    est.train(input_fn=boston_input_fn, steps=1)
    output = est.predict(boston.data)
    self.assertEqual(output.shape[0], boston.target.shape[0])

  def testPredictFn(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn,
                                     classification=False)
    boston = tf.contrib.learn.datasets.load_boston()
    est.train(input_fn=boston_input_fn, steps=1)
    output = est.predict(input_fn=boston_input_fn)
    self.assertEqual(output.shape[0], boston.target.shape[0])

  def testWrongInput(self):
    def other_input_fn():
      return {'other': tf.constant([0, 0, 0])}, tf.constant([0, 0, 0])
    output_dir = tempfile.mkdtemp()
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn,
                                     classification=False, model_dir=output_dir)
    est.train(input_fn=boston_input_fn, steps=1)
    with self.assertRaises(ValueError):
      est.train(input_fn=other_input_fn, steps=1)

  def testMonitors(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn,
                                     classification=False)
    est.train(input_fn=boston_input_fn, steps=21,
              monitors=[CheckCallsMonitor()])


if __name__ == '__main__':
  tf.test.main()
