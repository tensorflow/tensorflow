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
"""Tests for LogisticRegressor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def _iris_data_input_fn():
  # Converts iris data to a logistic regression problem.
  iris = tf.contrib.learn.datasets.load_iris()
  ids = np.where((iris.target == 0) | (iris.target == 1))
  features = tf.constant(iris.data[ids], dtype=tf.float32)
  targets = tf.constant(iris.target[ids], dtype=tf.float32)
  targets = tf.reshape(targets, targets.get_shape().concatenate(1))
  return features, targets


def _logistic_regression_model_fn(features, targets):
  logits = tf.contrib.layers.linear(
      features,
      1,
      weights_initializer=tf.zeros_initializer,
      # Intentionally uses really awful initial values so that
      # AUC/precision/recall/etc will change meaningfully even on a toy dataset.
      biases_initializer=tf.constant_initializer(-10.0))
  predictions = tf.sigmoid(logits)
  loss = tf.contrib.losses.sigmoid_cross_entropy(logits, targets)
  train_op = tf.contrib.layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      optimizer='Adagrad',
      learning_rate=0.1)
  return predictions, loss, train_op


class LogisticRegressorTest(tf.test.TestCase):

  def test_fit_and_evaluate_metrics(self):
    """Tests basic fit and evaluate, and checks the evaluation metrics."""
    regressor = tf.contrib.learn.LogisticRegressor(
        model_fn=_logistic_regression_model_fn)

    # Get some (intentionally horrible) baseline metrics.
    regressor.fit(input_fn=_iris_data_input_fn, steps=1)
    eval_metrics = regressor.evaluate(input_fn=_iris_data_input_fn, steps=1)
    self.assertNear(
        0.0,
        eval_metrics[tf.contrib.learn.LogisticRegressor.PREDICTION_MEAN],
        err=1e-3)
    self.assertNear(
        0.5,
        eval_metrics[tf.contrib.learn.LogisticRegressor.TARGET_MEAN],
        err=1e-6)
    self.assertNear(
        0.5,
        eval_metrics[tf.contrib.learn.LogisticRegressor.ACCURACY_BASELINE],
        err=1e-6)
    self.assertNear(0.5,
                    eval_metrics[tf.contrib.learn.LogisticRegressor.AUC],
                    err=1e-6)
    self.assertNear(
        0.5,
        eval_metrics[tf.contrib.learn.LogisticRegressor.ACCURACY_MEAN % 0.5],
        err=1e-6)
    self.assertNear(
        0.0,
        eval_metrics[tf.contrib.learn.LogisticRegressor.PRECISION_MEAN % 0.5],
        err=1e-6)
    self.assertNear(
        0.0,
        eval_metrics[tf.contrib.learn.LogisticRegressor.RECALL_MEAN % 0.5],
        err=1e-6)

    # Train for more steps and check the metrics again.
    regressor.fit(input_fn=_iris_data_input_fn, steps=100)
    eval_metrics = regressor.evaluate(input_fn=_iris_data_input_fn, steps=1)
    # Mean prediction moves from ~0.0 to ~0.5 as we stop predicting all 0's.
    self.assertNear(
        0.5,
        eval_metrics[tf.contrib.learn.LogisticRegressor.PREDICTION_MEAN],
        err=1e-2)
    # Target mean and baseline both remain the same at 0.5.
    self.assertNear(
        0.5,
        eval_metrics[tf.contrib.learn.LogisticRegressor.TARGET_MEAN],
        err=1e-6)
    self.assertNear(
        0.5,
        eval_metrics[tf.contrib.learn.LogisticRegressor.ACCURACY_BASELINE],
        err=1e-6)
    # AUC improves from 0.5 to 1.0.
    self.assertNear(1.0,
                    eval_metrics[tf.contrib.learn.LogisticRegressor.AUC],
                    err=1e-6)
    # Accuracy improves from 0.5 to >0.9.
    self.assertTrue(
        eval_metrics[tf.contrib.learn.LogisticRegressor.ACCURACY_MEAN % 0.5] >
        0.9)
    # Precision improves from 0.0 to 1.0.
    self.assertNear(
        1.0,
        eval_metrics[tf.contrib.learn.LogisticRegressor.PRECISION_MEAN % 0.5],
        err=1e-6)
    # Recall improves from 0.0 to >0.9.
    self.assertTrue(eval_metrics[tf.contrib.learn.LogisticRegressor.RECALL_MEAN
                                 % 0.5] > 0.9)


if __name__ == '__main__':
  tf.test.main()
