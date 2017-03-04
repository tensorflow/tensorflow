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

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.estimators import logistic_regressor
from tensorflow.contrib.learn.python.learn.estimators import metric_key
from tensorflow.contrib.losses.python.losses import loss_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def _iris_data_input_fn():
  # Converts iris data to a logistic regression problem.
  iris = base.load_iris()
  ids = np.where((iris.target == 0) | (iris.target == 1))
  features = constant_op.constant(iris.data[ids], dtype=dtypes.float32)
  labels = constant_op.constant(iris.target[ids], dtype=dtypes.float32)
  labels = array_ops.reshape(labels, labels.get_shape().concatenate(1))
  return features, labels


def _logistic_regression_model_fn(features, labels, mode):
  _ = mode
  logits = layers.linear(
      features,
      1,
      weights_initializer=init_ops.zeros_initializer(),
      # Intentionally uses really awful initial values so that
      # AUC/precision/recall/etc will change meaningfully even on a toy dataset.
      biases_initializer=init_ops.constant_initializer(-10.0))
  predictions = math_ops.sigmoid(logits)
  loss = loss_ops.sigmoid_cross_entropy(logits, labels)
  train_op = optimizers.optimize_loss(
      loss, variables.get_global_step(), optimizer='Adagrad', learning_rate=0.1)
  return predictions, loss, train_op


class LogisticRegressorTest(test.TestCase):

  def test_fit_and_evaluate_metrics(self):
    """Tests basic fit and evaluate, and checks the evaluation metrics."""
    regressor = logistic_regressor.LogisticRegressor(
        model_fn=_logistic_regression_model_fn)

    # Get some (intentionally horrible) baseline metrics.
    regressor.fit(input_fn=_iris_data_input_fn, steps=1)
    eval_metrics = regressor.evaluate(input_fn=_iris_data_input_fn, steps=1)
    self.assertNear(
        0.0, eval_metrics[metric_key.MetricKey.PREDICTION_MEAN], err=1e-3)
    self.assertNear(
        0.5, eval_metrics[metric_key.MetricKey.LABEL_MEAN], err=1e-6)
    self.assertNear(
        0.5, eval_metrics[metric_key.MetricKey.ACCURACY_BASELINE], err=1e-6)
    self.assertNear(0.5, eval_metrics[metric_key.MetricKey.AUC], err=1e-6)
    self.assertNear(
        0.5, eval_metrics[metric_key.MetricKey.ACCURACY_MEAN % 0.5], err=1e-6)
    self.assertNear(
        0.0, eval_metrics[metric_key.MetricKey.PRECISION_MEAN % 0.5], err=1e-6)
    self.assertNear(
        0.0, eval_metrics[metric_key.MetricKey.RECALL_MEAN % 0.5], err=1e-6)

    # Train for more steps and check the metrics again.
    regressor.fit(input_fn=_iris_data_input_fn, steps=100)
    eval_metrics = regressor.evaluate(input_fn=_iris_data_input_fn, steps=1)
    # Mean prediction moves from ~0.0 to ~0.5 as we stop predicting all 0's.
    self.assertNear(
        0.5, eval_metrics[metric_key.MetricKey.PREDICTION_MEAN], err=1e-2)
    # Label mean and baseline both remain the same at 0.5.
    self.assertNear(
        0.5, eval_metrics[metric_key.MetricKey.LABEL_MEAN], err=1e-6)
    self.assertNear(
        0.5, eval_metrics[metric_key.MetricKey.ACCURACY_BASELINE], err=1e-6)
    # AUC improves from 0.5 to 1.0.
    self.assertNear(1.0, eval_metrics[metric_key.MetricKey.AUC], err=1e-6)
    # Accuracy improves from 0.5 to >0.9.
    self.assertTrue(
        eval_metrics[metric_key.MetricKey.ACCURACY_MEAN % 0.5] > 0.9)
    # Precision improves from 0.0 to 1.0.
    self.assertNear(
        1.0, eval_metrics[metric_key.MetricKey.PRECISION_MEAN % 0.5], err=1e-6)
    # Recall improves from 0.0 to >0.9.
    self.assertTrue(eval_metrics[metric_key.MetricKey.RECALL_MEAN % 0.5] > 0.9)


if __name__ == '__main__':
  test.main()
