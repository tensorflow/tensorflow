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

import numpy as np

from tensorflow.contrib import losses
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.contrib.metrics.python.ops import metric_ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

INFERENCE_PROB_NAME = prediction_key.PredictionKey.PROBABILITIES
INFERENCE_PRED_NAME = prediction_key.PredictionKey.CLASSES

FEATURE_IMPORTANCE_NAME = 'global_feature_importance'


def _top_k_generator(k):
  def _top_k(probabilities, targets):
    targets = math_ops.to_int32(targets)
    if targets.get_shape().ndims > 1:
      targets = array_ops.squeeze(targets, axis=[1])
    return metric_ops.streaming_mean(nn.in_top_k(probabilities, targets, k))
  return _top_k


def _accuracy(predictions, targets, weights=None):
  return metric_ops.streaming_accuracy(predictions, targets, weights=weights)


def _r2(probabilities, targets, weights=None):
  targets = math_ops.to_float(targets)
  y_mean = math_ops.reduce_mean(targets, 0)
  squares_total = math_ops.reduce_sum(math_ops.square(targets - y_mean), 0)
  squares_residuals = math_ops.reduce_sum(
      math_ops.square(targets - probabilities), 0)
  score = 1 - math_ops.reduce_sum(squares_residuals / squares_total)
  return metric_ops.streaming_mean(score, weights=weights)


def _squeeze_and_onehot(targets, depth):
  targets = array_ops.squeeze(targets, axis=[1])
  return array_ops.one_hot(math_ops.to_int32(targets), depth)


def _sigmoid_entropy(probabilities, targets, weights=None):
  return metric_ops.streaming_mean(
      losses.sigmoid_cross_entropy(probabilities,
                                   _squeeze_and_onehot(
                                       targets,
                                       array_ops.shape(probabilities)[1])),
      weights=weights)


def _softmax_entropy(probabilities, targets, weights=None):
  return metric_ops.streaming_mean(
      losses.sparse_softmax_cross_entropy(probabilities,
                                          math_ops.to_int32(targets)),
      weights=weights)


def _predictions(predictions, unused_targets, **unused_kwargs):
  return predictions


def _class_log_loss(probabilities, targets, weights=None):
  return metric_ops.streaming_mean(
      losses.log_loss(probabilities,
                      _squeeze_and_onehot(targets,
                                          array_ops.shape(probabilities)[1])),
      weights=weights)


def _precision(predictions, targets, weights=None):
  return metric_ops.streaming_precision(predictions, targets, weights=weights)


def _precision_at_thresholds(predictions, targets, weights=None):
  return metric_ops.streaming_precision_at_thresholds(
      array_ops.slice(predictions, [0, 1], [-1, 1]),
      targets,
      np.arange(
          0, 1, 0.01, dtype=np.float32),
      weights=weights)


def _recall(predictions, targets, weights=None):
  return metric_ops.streaming_recall(predictions, targets, weights=weights)


def _recall_at_thresholds(predictions, targets, weights=None):
  return metric_ops.streaming_recall_at_thresholds(
      array_ops.slice(predictions, [0, 1], [-1, 1]),
      targets,
      np.arange(
          0, 1, 0.01, dtype=np.float32),
      weights=weights)


def _auc(probs, targets, weights=None):
  return metric_ops.streaming_auc(array_ops.slice(probs, [0, 1], [-1, 1]),
                                  targets, weights=weights)


_EVAL_METRICS = {
    'auc': _auc,
    'sigmoid_entropy': _sigmoid_entropy,
    'softmax_entropy': _softmax_entropy,
    'accuracy': _accuracy,
    'r2': _r2,
    'predictions': _predictions,
    'classification_log_loss': _class_log_loss,
    'precision': _precision,
    'precision_at_thresholds': _precision_at_thresholds,
    'recall': _recall,
    'recall_at_thresholds': _recall_at_thresholds,
    'top_5': _top_k_generator(5)
}

_PREDICTION_KEYS = {
    'auc': INFERENCE_PROB_NAME,
    'sigmoid_entropy': INFERENCE_PROB_NAME,
    'softmax_entropy': INFERENCE_PROB_NAME,
    'accuracy': INFERENCE_PRED_NAME,
    'r2': prediction_key.PredictionKey.SCORES,
    'predictions': INFERENCE_PRED_NAME,
    'classification_log_loss': INFERENCE_PROB_NAME,
    'precision': INFERENCE_PRED_NAME,
    'precision_at_thresholds': INFERENCE_PROB_NAME,
    'recall': INFERENCE_PRED_NAME,
    'recall_at_thresholds': INFERENCE_PROB_NAME,
    'top_5': INFERENCE_PROB_NAME
}


def get_metric(metric_name):
  """Given a metric name, return the corresponding metric function."""
  return _EVAL_METRICS[metric_name]


def get_prediction_key(metric_name):
  return _PREDICTION_KEYS[metric_name]
