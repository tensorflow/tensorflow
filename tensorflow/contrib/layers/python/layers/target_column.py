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
"""TargetColumn abstract a single head in the model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import six

from tensorflow.contrib import losses
from tensorflow.contrib import metrics as metrics_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


def regression_target(label_name=None,
                      weight_column_name=None,
                      target_dimension=1):
  """Creates a _TargetColumn for linear regression.

  Args:
    label_name: String, name of the key in label dict. Can be null if label
        is a tensor (single headed models).
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    target_dimension: dimension of the target for multilabels.

  Returns:
    An instance of _TargetColumn
  """
  return _RegressionTargetColumn(loss_fn=_mean_squared_loss,
                                 label_name=label_name,
                                 weight_column_name=weight_column_name,
                                 target_dimension=target_dimension)

# TODO(zakaria): Add logistic_regression_target


def multi_class_target(n_classes, label_name=None, weight_column_name=None):
  """Creates a _TargetColumn for multi class single label classification.

  The target column uses softmax cross entropy loss.

  Args:
    n_classes: Integer, number of classes, must be >= 2
    label_name: String, name of the key in label dict. Can be null if label
        is a tensor (single headed models).
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.

  Returns:
    An instance of _MultiClassTargetColumn.

  Raises:
    ValueError: if n_classes is < 2
  """
  if n_classes < 2:
    raise ValueError("n_classes must be > 1 for classification.")
  if n_classes == 2:
    loss_fn = _log_loss_with_two_classes
  else:
    loss_fn = _softmax_cross_entropy_loss
  return _MultiClassTargetColumn(loss_fn=loss_fn,
                                 n_classes=n_classes,
                                 label_name=label_name,
                                 weight_column_name=weight_column_name)


def binary_svm_target(label_name=None, weight_column_name=None):
  """Creates a _TargetColumn for binary classification with SVMs.

  The target column uses binary hinge loss.

  Args:
    label_name: String, name of the key in label dict. Can be null if label
      is a tensor (single headed models).
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.

  Returns:
    An instance of _TargetColumn.

  """
  return _BinarySvmTargetColumn(label_name=label_name,
                                weight_column_name=weight_column_name)


class _TargetColumn(object):
  """_TargetColumn is the abstraction for a single head in a model.

    Args:
      loss_fn: a function that returns the loss tensor.
      num_label_columns: Integer, number of label columns.
      label_name: String, name of the key in label dict. Can be null if label
          is a tensor (single headed models).
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.

    Raises:
      ValueError: if loss_fn or n_classes are missing.
  """

  def __init__(self, loss_fn, num_label_columns, label_name,
               weight_column_name):
    if not loss_fn:
      raise ValueError("loss_fn must be provided")
    if num_label_columns is None:  # n_classes can be 0
      raise ValueError("num_label_columns must be provided")

    self._loss_fn = loss_fn
    self._num_label_columns = num_label_columns
    self._label_name = label_name
    self._weight_column_name = weight_column_name

  def logits_to_predictions(self, logits, proba=False):
    # Abstrat, Subclasses must implement.
    raise NotImplementedError()

  def get_eval_ops(self, features, logits, targets, metrics=None):
    """Returns eval op."""
    raise NotImplementedError

  @property
  def label_name(self):
    return self._label_name

  @property
  def weight_column_name(self):
    return self._weight_column_name

  @property
  def num_label_columns(self):
    return self._num_label_columns

  def get_weight_tensor(self, features):
    if not self._weight_column_name:
      return None
    else:
      return array_ops.reshape(
          math_ops.to_float(features[self._weight_column_name]),
          shape=(-1,))

  def loss(self, logits, target, features):
    """Returns loss tensor for this head.

    Args:
      logits: logits, a float tensor.
      target: either a tensor for labels or in multihead case, a dict of string
        to target tensor.
      features: features dict.

    Returns:
      Loss tensor.
    """
    target = target[self.name] if isinstance(target, dict) else target
    loss_unweighted = self._loss_fn(logits, target)

    weight_tensor = self.get_weight_tensor(features)
    if weight_tensor is None:
      return math_ops.reduce_mean(loss_unweighted, name="loss")
    else:
      loss_unweighted = array_ops.reshape(loss_unweighted, shape=(-1,))
      loss_weighted = math_ops.mul(
          loss_unweighted,
          array_ops.reshape(weight_tensor, shape=(-1,)))
      return math_ops.div(
          math_ops.reduce_sum(loss_weighted),
          math_ops.to_float(math_ops.reduce_sum(weight_tensor)),
          name="loss")


class _RegressionTargetColumn(_TargetColumn):
  """_TargetColumn for regression."""

  def __init__(self, loss_fn, label_name, weight_column_name, target_dimension):
    super(_RegressionTargetColumn, self).__init__(
        loss_fn=loss_fn,
        num_label_columns=target_dimension,
        label_name=label_name,
        weight_column_name=weight_column_name)

  def logits_to_predictions(self, logits, proba=False):
    if self.num_label_columns == 1:
      return array_ops.squeeze(logits, squeeze_dims=[1])
    return logits

  def get_eval_ops(self, features, logits, targets, metrics=None):
    loss = self.loss(logits, targets, features)
    result = {"loss": metrics_lib.streaming_mean(loss)}
    if metrics:
      predictions = self.logits_to_predictions(logits, proba=False)
      result.update(_run_metrics(predictions, targets, metrics,
                                 self.get_weight_tensor(features)))
    return result


class _MultiClassTargetColumn(_TargetColumn):
  """_TargetColumn for classification."""

  # TODO(zakaria): support multilabel.
  def __init__(self, loss_fn, n_classes, label_name, weight_column_name):
    if n_classes < 2:
      raise ValueError("n_classes must be >= 2")
    super(_MultiClassTargetColumn, self).__init__(
        loss_fn=loss_fn,
        num_label_columns=1 if n_classes == 2 else n_classes,
        label_name=label_name,
        weight_column_name=weight_column_name)

  def logits_to_predictions(self, logits, proba=False):
    if self.num_label_columns == 1:
      logits = array_ops.concat(1, [array_ops.zeros_like(logits), logits])

    if proba:
      return nn.softmax(logits)
    else:
      return math_ops.argmax(logits, 1)

  def _default_eval_metrics(self):
    if self._num_label_columns == 1:
      return _get_default_binary_metrics_for_eval(thresholds=[.5])
    return {}

  def get_eval_ops(self, features, logits, targets, metrics=None):
    loss = self.loss(logits, targets, features)
    result = {"loss": metrics_lib.streaming_mean(loss)}

    # Adds default metrics.
    if metrics is None:
      # TODO(b/29366811): This currently results in both an "accuracy" and an
      # "accuracy/threshold_0.500000_mean" metric for binary classification.
      metrics = {("accuracy", "classes"): metrics_lib.streaming_accuracy}

    predictions = math_ops.sigmoid(logits)
    targets_float = math_ops.to_float(targets)

    default_metrics = self._default_eval_metrics()
    for metric_name, metric_op in default_metrics.items():
      result[metric_name] = metric_op(predictions, targets_float)

    class_metrics = {}
    proba_metrics = {}
    for name, metric_op in six.iteritems(metrics):
      if isinstance(name, tuple):
        if len(name) != 2:
          raise ValueError("Ignoring metric {}. It returned a tuple with "
                           "len {}, expected 2.".format(name, len(name)))
        else:
          if name[1] not in ["classes", "probabilities"]:
            raise ValueError("Ignoring metric {}. The 2nd element of its "
                             "name should be either 'classes' or "
                             "'probabilities'.".format(name))
          elif name[1] == "classes":
            class_metrics[name[0]] = metric_op
          else:
            proba_metrics[name[0]] = metric_op
      elif isinstance(name, str):
        class_metrics[name] = metric_op
      else:
        raise ValueError("Ignoring metric {}. Its name is not in the correct "
                         "form.".format(name))
    if class_metrics:
      class_predictions = self.logits_to_predictions(logits, proba=False)
      result.update(_run_metrics(class_predictions, targets, class_metrics,
                                 self.get_weight_tensor(features)))
    if proba_metrics:
      predictions = self.logits_to_predictions(logits, proba=True)
      result.update(_run_metrics(predictions, targets, proba_metrics,
                                 self.get_weight_tensor(features)))
    return result


class _BinarySvmTargetColumn(_MultiClassTargetColumn):
  """_TargetColumn for binary classification using SVMs."""

  def __init__(self, label_name, weight_column_name):
    def loss_fn(logits, target):
      check_shape_op = logging_ops.Assert(
          math_ops.less_equal(array_ops.rank(target), 2),
          ["target's shape should be either [batch_size, 1] or [batch_size]"])
      with ops.control_dependencies([check_shape_op]):
        target = array_ops.reshape(
            target, shape=[array_ops.shape(target)[0], 1])
      return losses.hinge_loss(logits, target)

    super(_BinarySvmTargetColumn, self).__init__(
        loss_fn=loss_fn,
        n_classes=2,
        label_name=label_name,
        weight_column_name=weight_column_name)

  def logits_to_predictions(self, logits, proba=False):
    if proba:
      raise ValueError(
          "logits to probabilities is not supported for _BinarySvmTargetColumn")

    logits = array_ops.concat(1, [array_ops.zeros_like(logits), logits])
    return math_ops.argmax(logits, 1)


# TODO(zakaria): use contrib losses.
def _mean_squared_loss(logits, target):
  # To prevent broadcasting inside "-".
  if len(target.get_shape()) == 1:
    target = array_ops.expand_dims(target, dim=[1])

  logits.get_shape().assert_is_compatible_with(target.get_shape())
  return math_ops.square(logits - math_ops.to_float(target))


def _log_loss_with_two_classes(logits, target):
  # sigmoid_cross_entropy_with_logits requires [batch_size, 1] target.
  if len(target.get_shape()) == 1:
    target = array_ops.expand_dims(target, dim=[1])
  loss_vec = nn.sigmoid_cross_entropy_with_logits(logits,
                                                  math_ops.to_float(target))
  return loss_vec


def _softmax_cross_entropy_loss(logits, target):
  # sigmoid_cross_entropy_with_logits requires [batch_size, 1] target.
  # Check that we got int32/int64 for classification.
  if (not target.dtype.is_compatible_with(dtypes.int64) and
      not target.dtype.is_compatible_with(dtypes.int32)):
    raise ValueError("Target's dtype should be int32, int64 or compatible. "
                     "Instead got %s." % target.dtype)
  # sparse_softmax_cross_entropy_with_logits requires [batch_size] target.
  if len(target.get_shape()) == 2:
    target = array_ops.squeeze(target, squeeze_dims=[1])
  loss_vec = nn.sparse_softmax_cross_entropy_with_logits(logits, target)
  return loss_vec


def _run_metrics(predictions, targets, metrics, weights):
  result = {}
  targets = math_ops.cast(targets, predictions.dtype)
  for name, metric in six.iteritems(metrics or {}):
    if "weights" in inspect.getargspec(metric)[0]:
      result[name] = metric(predictions, targets, weights=weights)
    else:
      result[name] = metric(predictions, targets)

  return result


def _get_default_binary_metrics_for_eval(thresholds):
  """Returns a dictionary of basic metrics for logistic regression.

  Args:
    thresholds: List of floating point thresholds to use for accuracy,
      precision, and recall metrics. If None, defaults to [0.5].

  Returns:
    Dictionary mapping metrics string names to metrics functions.
  """
  metrics = {}
  metrics[_MetricKeys.PREDICTION_MEAN] = _predictions_streaming_mean
  metrics[_MetricKeys.TARGET_MEAN] = _targets_streaming_mean
  # Also include the streaming mean of the label as an accuracy baseline, as
  # a reminder to users.
  metrics[_MetricKeys.ACCURACY_BASELINE] = _targets_streaming_mean

  metrics[_MetricKeys.AUC] = metrics_lib.streaming_auc

  for threshold in thresholds:
    metrics[_MetricKeys.ACCURACY_MEAN % threshold] = _streaming_with_threshold(
        metrics_lib.streaming_accuracy, threshold)
    # Precision for positive examples.
    metrics[_MetricKeys.PRECISION_MEAN % threshold] = _streaming_with_threshold(
        metrics_lib.streaming_precision, threshold)
    # Recall for positive examples.
    metrics[_MetricKeys.RECALL_MEAN % threshold] = _streaming_with_threshold(
        metrics_lib.streaming_recall, threshold)

  return metrics


# TODO(zakaria): support weights.
def _targets_streaming_mean(unused_predictions, targets):
  return metrics_lib.streaming_mean(targets)


def _predictions_streaming_mean(predictions, unused_targets):
  return metrics_lib.streaming_mean(predictions)


def _streaming_with_threshold(streaming_metrics_fn, threshold):

  def _streaming_metrics(predictions, targets):
    return streaming_metrics_fn(predictions=math_ops.to_float(
        math_ops.greater_equal(predictions, threshold)),
                                labels=targets)

  return _streaming_metrics


class _MetricKeys(object):
  AUC = "auc"
  PREDICTION_MEAN = "labels/prediction_mean"
  TARGET_MEAN = "labels/actual_target_mean"
  ACCURACY_BASELINE = "accuracy/baseline_target_mean"
  ACCURACY_MEAN = "accuracy/threshold_%f_mean"
  PRECISION_MEAN = "precision/positive_threshold_%f_mean"
  RECALL_MEAN = "recall/positive_threshold_%f_mean"
