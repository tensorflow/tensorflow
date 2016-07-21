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
"""A collection of functions to be used as evaluation metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import losses
from tensorflow.contrib.metrics.python.ops import metric_ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def _accuracy(probabilities, targets):
  predictions = math_ops.argmax(probabilities, 1)
  # undo one-hot
  labels = math_ops.argmax(targets, 1)
  return metric_ops.streaming_accuracy(predictions, labels)


def _r2(probabilities, targets):
  if targets.get_shape().ndims == 1:
    targets = array_ops.expand_dims(targets, -1)
  y_mean = math_ops.reduce_mean(targets, 0)
  squares_total = math_ops.reduce_sum(math_ops.square(targets - y_mean), 0)
  squares_residuals = math_ops.reduce_sum(math_ops.square(
      targets - probabilities), 0)
  score = 1 - math_ops.reduce_sum(squares_residuals / squares_total)
  return metric_ops.streaming_mean(score)


def _sigmoid_entropy(probabilities, targets):
  return metric_ops.streaming_mean(losses.sigmoid_cross_entropy(
      probabilities, targets))


def _softmax_entropy(probabilities, targets):
  return metric_ops.streaming_mean(losses.softmax_cross_entropy(
      probabilities, targets))


def _predictions(probabilities, unused_targets):
  return math_ops.argmax(probabilities, 1)


_EVAL_METRICS = {'sigmoid_entropy': _sigmoid_entropy,
                 'softmax_entropy': _softmax_entropy,
                 'accuracy': _accuracy,
                 'r2': _r2,
                 'predictions': _predictions}


def get_metric(metric_name):
  """Given a metric name, return the corresponding metric function."""
  return _EVAL_METRICS[metric_name]
