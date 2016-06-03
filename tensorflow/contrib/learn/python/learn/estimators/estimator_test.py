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

from tensorflow.contrib.learn.python.learn.estimators import _sklearn


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
          tf.constant(iris.target), [-1]), tf.int32)
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


def linear_model_params_fn(features, target, mode, params):
  assert mode in ('train', 'eval', 'infer')
  prediction, loss = (
      tf.contrib.learn.models.linear_regression_zero_init(features, target)
  )
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=params['learning_rate'])
  return prediction, loss, train_op


def linear_model_fn(features, target, mode):
  assert mode in ('train', 'eval', 'infer')
  prediction, loss = (
      tf.contrib.learn.models.linear_regression_zero_init(features, target)
  )
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=0.1)
  return prediction, loss, train_op


def logistic_model_no_mode_fn(features, target):
  target = tf.one_hot(target, 3, 1, 0)
  prediction, loss = (
      tf.contrib.learn.models.logistic_regression_zero_init(features, target)
  )
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=0.1)
  return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op


class CheckCallsMonitor(tf.contrib.learn.monitors.BaseMonitor):

  def __init__(self):
    self.begin_calls = None
    self.end_calls = None
    self.expect_calls = None

  def begin(self, max_steps):
    self.begin_calls = 0
    self.end_calls = 0
    self.expect_calls = max_steps

  def step_begin(self, step):
    self.begin_calls += 1
    return {}

  def step_end(self, step, outputs):
    self.end_calls += 1
    return False

  def end(self):
    assert (self.end_calls == self.expect_calls and
            self.begin_calls == self.expect_calls)


class EstimatorTest(tf.test.TestCase):

  def testUntrained(self):
    boston = tf.contrib.learn.datasets.load_boston()
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    with self.assertRaises(tf.contrib.learn.NotFittedError):
      _ = est.evaluate(
          x=boston.data,
          y=boston.target.astype(np.float32))
    with self.assertRaises(tf.contrib.learn.NotFittedError):
      est.predict(x=boston.data)

  def testContinueTraining(self):
    boston = tf.contrib.learn.datasets.load_boston()
    output_dir = tempfile.mkdtemp()
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn,
                                     model_dir=output_dir)
    est.fit(x=boston.data, y=boston.target.astype(np.float32), steps=50)
    scores = est.evaluate(
        x=boston.data,
        y=boston.target.astype(np.float32),
        metrics={'MSE': tf.contrib.metrics.streaming_mean_squared_error})
    del est
    # Create another estimator object with the same output dir.
    est2 = tf.contrib.learn.Estimator(model_fn=linear_model_fn,
                                      model_dir=output_dir)

    # Check we can evaluate and predict.
    scores2 = est2.evaluate(
        x=boston.data,
        y=boston.target.astype(np.float32),
        metrics={'MSE': tf.contrib.metrics.streaming_mean_squared_error})
    self.assertAllClose(scores2['MSE'],
                        scores['MSE'])
    predictions = est2.predict(x=boston.data)
    other_score = _sklearn.mean_squared_error(predictions, boston.target)
    self.assertAllClose(other_score, scores['MSE'])

    # Check we can keep training.
    est2.fit(x=boston.data, y=boston.target.astype(np.float32), steps=100)
    scores3 = est2.evaluate(
        x=boston.data,
        y=boston.target.astype(np.float32),
        metrics={'MSE': tf.contrib.metrics.streaming_mean_squared_error})
    self.assertLess(scores3['MSE'], scores['MSE'])

  def testEstimatorParams(self):
    boston = tf.contrib.learn.datasets.load_boston()
    est = tf.contrib.learn.Estimator(model_fn=linear_model_params_fn,
                                     params={'learning_rate': 0.01})
    est.fit(x=boston.data, y=boston.target.astype(np.float32), steps=100)

  def testBostonAll(self):
    boston = tf.contrib.learn.datasets.load_boston()
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    est.fit(x=boston.data, y=boston.target.astype(np.float32), steps=100)
    scores = est.evaluate(
        x=boston.data,
        y=boston.target.astype(np.float32),
        metrics={'MSE': tf.contrib.metrics.streaming_mean_squared_error})
    predictions = est.predict(x=boston.data)
    other_score = _sklearn.mean_squared_error(predictions, boston.target)
    self.assertAllClose(other_score, scores['MSE'])

  def testIrisAll(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Estimator(model_fn=logistic_model_no_mode_fn)
    est.fit(iris.data, iris.target, steps=100)
    scores = est.evaluate(
        x=iris.data,
        y=iris.target,
        metrics={('accuracy', 'class'): tf.contrib.metrics.streaming_accuracy})
    predictions = est.predict(x=iris.data)
    predictions_class = est.predict(x=iris.data, outputs=['class'])
    self.assertEqual(predictions['class'].shape[0], iris.target.shape[0])
    self.assertAllClose(predictions['class'], predictions_class['class'])
    self.assertAllClose(predictions['class'], np.argmax(predictions['prob'],
                                                        axis=1))
    other_score = _sklearn.accuracy_score(iris.target, predictions['class'])
    self.assertAllClose(other_score, scores['accuracy'])

  def testIrisInputFn(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Estimator(model_fn=logistic_model_no_mode_fn)
    est.fit(input_fn=iris_input_fn, steps=100)
    _ = est.evaluate(input_fn=iris_input_fn, steps=1)
    predictions = est.predict(x=iris.data)['class']
    self.assertEqual(predictions.shape[0], iris.target.shape[0])

  def testTrainInputFn(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    est.fit(input_fn=boston_input_fn, steps=1)
    _ = est.evaluate(input_fn=boston_eval_fn, steps=1)

  def testPredict(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    boston = tf.contrib.learn.datasets.load_boston()
    est.fit(input_fn=boston_input_fn, steps=1)
    output = est.predict(boston.data)
    self.assertEqual(output.shape[0], boston.target.shape[0])

  def testPredictFn(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    boston = tf.contrib.learn.datasets.load_boston()
    est.fit(input_fn=boston_input_fn, steps=1)
    output = est.predict(input_fn=boston_input_fn)
    self.assertEqual(output.shape[0], boston.target.shape[0])

  def testWrongInput(self):
    def other_input_fn():
      return {'other': tf.constant([0, 0, 0])}, tf.constant([0, 0, 0])
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    est.fit(input_fn=boston_input_fn, steps=1)
    with self.assertRaises(ValueError):
      est.fit(input_fn=other_input_fn, steps=1)

  def testMonitors(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    est.fit(input_fn=boston_input_fn, steps=21, monitors=[CheckCallsMonitor()])


if __name__ == '__main__':
  tf.test.main()
