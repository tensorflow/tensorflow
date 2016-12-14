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
"""Abstractions for the head(s) of a model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
import six

from tensorflow.contrib import losses as losses_lib
from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import metric_key
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python import summary
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import training


# TODO(zakaria): add functions that creates a head and returns ModelOpFn


def _regression_head(label_name=None,
                     weight_column_name=None,
                     label_dimension=1,
                     enable_centered_bias=False, head_name=None):
  """Creates a _Head for linear regression.

  Args:
    label_name: String, name of the key in label dict. Can be null if label
        is a tensor (single headed models).
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    label_dimension: dimension of the label for multilabels.
    enable_centered_bias: A bool. If True, estimator will learn a centered
      bias variable for each class. Rest of the model structure learns the
      residual after centered bias.
    head_name: name of the head. If provided, predictions, summary and metrics
      keys will be prefixed by the head_name and an underscore.

  Returns:
    An instance of _Head
  """
  return _RegressionHead(label_name=label_name,
                         weight_column_name=weight_column_name,
                         label_dimension=label_dimension,
                         enable_centered_bias=enable_centered_bias,
                         head_name=head_name)

# TODO(zakaria): Add logistic_regression_head


def _multi_class_head(n_classes, label_name=None, weight_column_name=None,
                      enable_centered_bias=False, head_name=None,
                      thresholds=None, metric_class_ids=None):
  """Creates a _Head for multi class single label classification.

  The Head uses softmax cross entropy loss.

  Args:
    n_classes: Integer, number of classes, must be >= 2
    label_name: String, name of the key in label dict. Can be null if label
        is a tensor (single headed models).
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    enable_centered_bias: A bool. If True, estimator will learn a centered
      bias variable for each class. Rest of the model structure learns the
      residual after centered bias.
    head_name: name of the head. If provided, predictions, summary and metrics
      keys will be prefixed by the head_name and an underscore.
    thresholds: thresholds for eval metrics, defaults to [.5]
    metric_class_ids: List of class IDs for which we should report per-class
      metrics. Must all be in the range `[0, n_classes)`. Invalid if
      `n_classes` is 2.

  Returns:
    An instance of _MultiClassHead.

  Raises:
    ValueError: if `n_classes` is < 2, or `metric_class_ids` is provided when
      `n_classes` is 2.
  """
  if (n_classes is None) or (n_classes < 2):
    raise ValueError(
        "n_classes must be > 1 for classification: %s." % n_classes)

  if n_classes == 2:
    if metric_class_ids:
      raise ValueError("metric_class_ids invalid for n_classes==2.")
    return _BinaryLogisticHead(label_name=label_name,
                               weight_column_name=weight_column_name,
                               enable_centered_bias=enable_centered_bias,
                               head_name=head_name,
                               thresholds=thresholds)

  return _MultiClassHead(n_classes=n_classes,
                         label_name=label_name,
                         weight_column_name=weight_column_name,
                         enable_centered_bias=enable_centered_bias,
                         head_name=head_name,
                         thresholds=thresholds,
                         metric_class_ids=metric_class_ids)


def _binary_svm_head(label_name=None, weight_column_name=None,
                     enable_centered_bias=False, head_name=None,
                     thresholds=None,):
  """Creates a `_Head` for binary classification with SVMs.

  The head uses binary hinge loss.

  Args:
    label_name: String, name of the key in label dict. Can be null if label
      is a tensor (single headed models).
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    enable_centered_bias: A bool. If True, estimator will learn a centered
      bias variable for each class. Rest of the model structure learns the
      residual after centered bias.
    head_name: name of the head. If provided, predictions, summary and metrics
      keys will be prefixed by the head_name and an underscore.
    thresholds: thresholds for eval metrics, defaults to [.5]

  Returns:
    An instance of `_Head`.

  """
  return _BinarySvmHead(label_name=label_name,
                        weight_column_name=weight_column_name,
                        enable_centered_bias=enable_centered_bias,
                        head_name=head_name,
                        thresholds=thresholds)


def _multi_label_head(n_classes, label_name=None, weight_column_name=None,
                      enable_centered_bias=False, head_name=None,
                      thresholds=None, metric_class_ids=None):
  """Creates a _Head for multi label classification.

  The Head uses softmax cross entropy loss.

  Args:
    n_classes: Integer, number of classes, must be >= 2
    label_name: String, name of the key in label dict. Can be null if label
        is a tensor (single headed models).
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    enable_centered_bias: A bool. If True, estimator will learn a centered
      bias variable for each class. Rest of the model structure learns the
      residual after centered bias.
    head_name: name of the head. If provided, predictions, summary and metrics
      keys will be prefixed by the head_name and an underscore.
    thresholds: thresholds for eval metrics, defaults to [.5]
    metric_class_ids: List of class IDs for which we should report per-class
      metrics. Must all be in the range `[0, n_classes)`.

  Returns:
    An instance of _MultiClassHead.

  Raises:
    ValueError: if n_classes is < 2
  """
  if n_classes < 2:
    raise ValueError("n_classes must be > 1 for classification.")
  return _MultiLabelHead(n_classes=n_classes,
                         label_name=label_name,
                         weight_column_name=weight_column_name,
                         enable_centered_bias=enable_centered_bias,
                         head_name=head_name,
                         thresholds=thresholds,
                         metric_class_ids=metric_class_ids)


def _multi_head(heads, loss_weights=None):
  """Creates a MultiHead stemming from same logits/hidden layer.

  Args:
    heads: list of _Head objects.
    loss_weights: optional list of weights to be used to combine losses from
        each head. All losses are weighted equally if not provided.

  Returns:
    A _Head instance that combines multiple heads.

  Raises:
    ValueError: if heads and loss_weights have different size.
  """
  if loss_weights:
    if len(loss_weights) != len(heads):
      raise ValueError("heads and loss_weights must have same size")

  def _weighted_loss_combiner(losses):
    if loss_weights:
      if len(losses) != len(loss_weights):
        raise ValueError("losses and loss_weights must have same size")
      weighted_losses = []
      for loss, weight in zip(losses, loss_weights):
        weighted_losses.append(math_ops.multiply(loss, weight))
      return math_ops.add_n(weighted_losses)
    else:
      return math_ops.add_n(losses)

  return _MultiHead(heads, loss_combiner=_weighted_loss_combiner)


# TODO(zakaria): Make the classes public once we are ready for users to subclass
#   them.
class _Head(object):
  """Interface for the head/top of a model.

  Given logits or output of a hidden layer, a Head knows how to compute
  predictions, loss, default metric and export signature.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, head_name):
    self.head_name = head_name

  @abc.abstractproperty
  def logits_dimension(self):
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def head_ops(self, features, labels, mode, train_op_fn, logits=None,
               logits_input=None, scope=None):
    """Returns ops for a model_fn.

    Args:
      features: input dict.
      labels: labels dict or tensor.
      mode: estimator's ModeKeys
      train_op_fn: function that takes a scalar loss and returns an op to
          optimize with the loss.
      logits: logits to be used for the head.
      logits_input: tensor to build logits from.
      scope: Optional scope for variable_scope.

    Returns:
      `ModelFnOps`.

    Raises:
      ValueError: if mode is not recognized.
    """
    raise NotImplementedError("Calling an abstract method.")

  def _create_output_alternatives(self, predictions):
    """Creates output alternative for the Head.

    Args:
      predictions: a dict of {tensor_name: Tensor}, where 'tensor_name' is a
        symbolic name for an output Tensor possibly but not necessarily taken
        from `PredictionKey`, and 'Tensor' is the corresponding output Tensor
        itself.

    Returns:
      `dict` of {submodel_name: (problem_type, {tensor_name: Tensor})}, where
      'submodel_name' is a submodel identifier that should be consistent across
      the pipeline (here likely taken from the head_name),
      'problem_type' is a `ProblemType`,
      'tensor_name' is a symbolic name for an output Tensor possibly but not
       necessarily taken from `PredictionKey`, and
      'Tensor' is the corresponding output Tensor itself.
    """
    return {self.head_name: (self._problem_type, predictions)}


# TODO(zakaria): use contrib losses.
def _mean_squared_loss(logits, labels):
  with ops.name_scope(None, "mean_squared_loss", (logits, labels)) as name:
    # To prevent broadcasting inside "-".
    if len(labels.get_shape()) == 1:
      labels = array_ops.expand_dims(labels, dim=(1,))
    # TODO(zakaria): make sure it does not recreate the broadcast bug.
    if len(logits.get_shape()) == 1:
      logits = array_ops.expand_dims(logits, dim=(1,))
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    return math_ops.square(logits - math_ops.to_float(labels), name=name)


class _RegressionHead(_Head):
  """_Head for regression."""

  def __init__(self, label_name, weight_column_name, label_dimension,
               enable_centered_bias, head_name, loss_fn=_mean_squared_loss):
    """Base type for all single heads.

    Args:
      label_name: String, name of the key in label dict. Can be null if label
          is a tensor (single headed models).
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      label_dimension: Integer, number of label columns.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      head_name: name of the head. If provided, predictions, summary and metrics
        keys will be prefixed by the head_name and an underscore.
      loss_fn: Loss function.
    """
    super(_RegressionHead, self).__init__(head_name=head_name)

    self._loss_fn = loss_fn
    self._logits_dimension = label_dimension
    self._label_name = label_name
    self._weight_column_name = weight_column_name
    self._enable_centered_bias = enable_centered_bias
    self._problem_type = constants.ProblemType.LINEAR_REGRESSION

  @property
  def logits_dimension(self):
    return self._logits_dimension

  def head_ops(self, features, labels, mode, train_op_fn, logits=None,
               logits_input=None, scope=None):
    """See `_Head`."""
    _check_mode_valid(mode)
    _check_logits_input_not_supported(logits, logits_input)

    centered_bias = None
    if self._enable_centered_bias:
      centered_bias = _centered_bias(self._logits_dimension, self.head_name)
      logits = nn.bias_add(logits, centered_bias)

    predictions = self._logits_to_predictions(logits)
    loss = None
    train_op = None
    eval_metric_ops = None
    if (mode != model_fn.ModeKeys.INFER) and (labels is not None):
      labels_tensor = _to_labels_tensor(labels, self._label_name)
      loss = _training_loss(
          features, labels_tensor, logits,
          loss_fn=self._loss_fn,
          weight_column_name=self._weight_column_name,
          head_name=self.head_name)
      if (mode == model_fn.ModeKeys.TRAIN) and (train_op_fn is not None):
        train_op = _train_op(
            loss, labels_tensor, train_op_fn, centered_bias,
            self.logits_dimension, self._loss_fn)
      eval_metric_ops = _eval_metric_ops(
          self._default_metrics(), features, labels, predictions)

    return model_fn.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        signature_fn=self._signature_fn(),
        output_alternatives=self._create_output_alternatives(predictions))

  def _logits_to_predictions(self, logits):
    """Returns a dict of predictions.

    Args:
      logits: logits `Tensor` after applying possible centered bias.

    Returns:
      Dict of prediction `Tensor` keyed by `PredictionKey`.
    """
    key = prediction_key.PredictionKey.SCORES
    with ops.name_scope(None, "predictions", (logits,)):
      if self.logits_dimension == 1:
        logits = array_ops.squeeze(logits, squeeze_dims=(1,), name=key)
      return {key: logits}

  def _signature_fn(self):
    """Returns the signature_fn to be used in exporting."""
    def _regression_signature_fn(examples, features, predictions):
      # pylint: disable=missing-docstring
      del features
      if isinstance(predictions, dict):
        score = predictions[prediction_key.PredictionKey.SCORES]
      else:
        score = predictions

      default_signature = exporter.regression_signature(
          input_tensor=examples, output_tensor=score)
      # TODO(zakaria): add validation
      return default_signature, {}
    return _regression_signature_fn

  def _default_metrics(self):
    """Returns a dict of `MetricSpec` keyed by `MetricKey`."""
    return {_summary_key(self.head_name, metric_key.MetricKey.LOSS):
            _weighted_average_loss_metric_spec(
                self._loss_fn, prediction_key.PredictionKey.SCORES,
                self._label_name, self._weight_column_name)}


def _log_loss_with_two_classes(logits, labels):
  with ops.name_scope(
      None, "log_loss_with_two_classes", (logits, labels)) as name:
    # sigmoid_cross_entropy_with_logits requires [batch_size, 1] labels.
    if len(labels.get_shape()) == 1:
      labels = array_ops.expand_dims(labels, dim=(1,))
    return nn.sigmoid_cross_entropy_with_logits(
        logits, math_ops.to_float(labels), name=name)


def _one_class_to_two_class_logits(logits):
  return array_ops.concat_v2((array_ops.zeros_like(logits), logits), 1)


class _BinaryLogisticHead(_Head):
  """_Head for binary logistic classifciation."""

  def __init__(self, label_name, weight_column_name, enable_centered_bias,
               head_name, loss_fn=_log_loss_with_two_classes, thresholds=None):
    """Base type for all single heads.

    Args:
      label_name: String, name of the key in label dict. Can be null if label
          is a tensor (single headed models).
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      head_name: name of the head. If provided, predictions, summary and metrics
        keys will be prefixed by the head_name and an underscore.
      loss_fn: Loss function.
      thresholds: thresholds for eval.

    Raises:
      ValueError: if n_classes is invalid.
    """
    super(_BinaryLogisticHead, self).__init__(head_name=head_name)
    self._thresholds = thresholds if thresholds else (.5,)
    self._label_name = label_name
    self._weight_column_name = weight_column_name
    self._loss_fn = loss_fn
    self._enable_centered_bias = enable_centered_bias

  @property
  def logits_dimension(self):
    return 1

  def head_ops(self, features, labels, mode, train_op_fn, logits=None,
               logits_input=None, scope=None):
    """See `_Head`."""
    _check_mode_valid(mode)
    _check_logits_input_not_supported(logits, logits_input)

    centered_bias = None
    if self._enable_centered_bias:
      centered_bias = _centered_bias(1, self.head_name)
      logits = nn.bias_add(logits, centered_bias)

    predictions = self._logits_to_predictions(logits)
    loss = None
    train_op = None
    eval_metric_ops = None
    if (mode != model_fn.ModeKeys.INFER) and (labels is not None):
      labels_tensor = _to_labels_tensor(labels, self._label_name)
      loss = _training_loss(
          features, labels_tensor, logits,
          loss_fn=self._loss_fn,
          weight_column_name=self._weight_column_name,
          head_name=self.head_name)
      if (mode == model_fn.ModeKeys.TRAIN) and (train_op_fn is not None):
        train_op = _train_op(
            loss, labels_tensor, train_op_fn, centered_bias,
            self.logits_dimension, self._loss_fn)
      eval_metric_ops = _eval_metric_ops(
          self._default_metrics(), features, labels, predictions)

    return model_fn.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        signature_fn=self._signature_fn())

  def _logits_to_predictions(self, logits):
    """Returns a dict of predictions.

    Args:
      logits: logits `Output` after applying possible centered bias.

    Returns:
      Dict of prediction `Output` keyed by `PredictionKey`.
    """
    with ops.name_scope(None, "predictions", (logits,)):
      two_class_logits = _one_class_to_two_class_logits(logits)
      return {
          prediction_key.PredictionKey.LOGITS: logits,
          prediction_key.PredictionKey.LOGISTIC: math_ops.sigmoid(
              logits, name=prediction_key.PredictionKey.LOGISTIC),
          prediction_key.PredictionKey.PROBABILITIES: nn.softmax(
              two_class_logits,
              name=prediction_key.PredictionKey.PROBABILITIES),
          prediction_key.PredictionKey.CLASSES: math_ops.argmax(
              two_class_logits, 1, name=prediction_key.PredictionKey.CLASSES)
      }

  def _signature_fn(self):
    """Returns the signature_fn to be used in exporting."""
    def _classification_signature_fn(examples, features, predictions):
      """Servo signature function."""
      del features
      if isinstance(predictions, dict):
        default_signature = exporter.classification_signature(
            input_tensor=examples,
            classes_tensor=predictions[prediction_key.PredictionKey.CLASSES],
            scores_tensor=predictions[
                prediction_key.PredictionKey.PROBABILITIES])
      else:
        default_signature = exporter.classification_signature(
            input_tensor=examples,
            scores_tensor=predictions)

      # TODO(zakaria): add validation
      return default_signature, {}
    return _classification_signature_fn

  def _default_metrics(self):
    """Returns a dict of `MetricSpec` objects keyed by name."""
    metrics = {_summary_key(self.head_name, metric_key.MetricKey.LOSS):
               _weighted_average_loss_metric_spec(
                   self._loss_fn, prediction_key.PredictionKey.LOGITS,
                   self._label_name, self._weight_column_name)}

    # TODO(b/29366811): This currently results in both an "accuracy" and an
    # "accuracy/threshold_0.500000_mean" metric for binary classification.
    metrics[_summary_key(self.head_name, metric_key.MetricKey.ACCURACY)] = (
        metric_spec.MetricSpec(metrics_lib.streaming_accuracy,
                               prediction_key.PredictionKey.CLASSES,
                               self._label_name, self._weight_column_name))
    def _add_binary_metric(key, metric_fn):
      metrics[_summary_key(self.head_name, key)] = metric_spec.MetricSpec(
          metric_fn, prediction_key.PredictionKey.LOGISTIC, self._label_name,
          self._weight_column_name)
    _add_binary_metric(
        metric_key.MetricKey.PREDICTION_MEAN, _predictions_streaming_mean)
    _add_binary_metric(
        metric_key.MetricKey.LABEL_MEAN, _indicator_labels_streaming_mean)

    # Also include the streaming mean of the label as an accuracy baseline, as
    # a reminder to users.
    _add_binary_metric(
        metric_key.MetricKey.ACCURACY_BASELINE,
        _indicator_labels_streaming_mean)

    _add_binary_metric(metric_key.MetricKey.AUC, _streaming_auc)

    for threshold in self._thresholds:
      _add_binary_metric(metric_key.MetricKey.ACCURACY_MEAN % threshold,
                         _accuracy_at_threshold(threshold))
      # Precision for positive examples.
      _add_binary_metric(metric_key.MetricKey.PRECISION_MEAN % threshold,
                         _streaming_at_threshold(
                             metrics_lib.streaming_precision_at_thresholds,
                             threshold),)
      # Recall for positive examples.
      _add_binary_metric(metric_key.MetricKey.RECALL_MEAN % threshold,
                         _streaming_at_threshold(
                             metrics_lib.streaming_recall_at_thresholds,
                             threshold))
    return metrics


def _softmax_cross_entropy_loss(logits, labels):
  with ops.name_scope(
      None, "softmax_cross_entropy_loss", (logits, labels,)) as name:
    # Check that we got integer for classification.
    if not labels.dtype.is_integer:
      raise ValueError("Labels dtype should be integer "
                       "Instead got %s." % labels.dtype)
    # sparse_softmax_cross_entropy_with_logits requires [batch_size] labels.
    if len(labels.get_shape()) == 2:
      labels = array_ops.squeeze(labels, squeeze_dims=(1,))
    return nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name=name)


class _MultiClassHead(_Head):
  """_Head for classification."""

  def __init__(self, n_classes, label_name,
               weight_column_name, enable_centered_bias, head_name,
               loss_fn=_softmax_cross_entropy_loss, thresholds=None,
               metric_class_ids=None):
    """_Head for classification.

    Args:
      n_classes: Number of classes, must be greater than 2 (for 2 classes, use
          `_BinaryLogisticHead`).
      label_name: String, name of the key in label dict. Can be null if label
          is a tensor (single headed models).
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      head_name: name of the head. If provided, predictions, summary and metrics
        keys will be prefixed by the head_name and an underscore.
      loss_fn: Loss function.
      thresholds: thresholds for eval.
      metric_class_ids: List of class IDs for which we should report per-class
        metrics. Must all be in the range `[0, n_classes)`.

    Raises:
      ValueError: if `n_classes` or `metric_class_ids` is invalid.
    """
    super(_MultiClassHead, self).__init__(head_name=head_name)

    if (n_classes is None) or (n_classes <= 2):
      raise ValueError("n_classes must be > 2: %s." % n_classes)
    self._thresholds = thresholds if thresholds else (.5,)
    self._logits_dimension = n_classes
    self._label_name = label_name
    self._weight_column_name = weight_column_name
    self._loss_fn = loss_fn
    self._enable_centered_bias = enable_centered_bias
    self._problem_type = constants.ProblemType.CLASSIFICATION
    self._metric_class_ids = tuple(
        [] if metric_class_ids is None else metric_class_ids)
    for class_id in self._metric_class_ids:
      if (class_id < 0) or (class_id >= n_classes):
        raise ValueError("Class ID %s not in [0, %s)." % (class_id, n_classes))

  @property
  def logits_dimension(self):
    return self._logits_dimension

  def head_ops(self, features, labels, mode, train_op_fn, logits=None,
               logits_input=None, scope=None):
    """See `_Head`."""
    _check_mode_valid(mode)
    _check_logits_input_not_supported(logits, logits_input)

    centered_bias = None
    if self._enable_centered_bias:
      centered_bias = _centered_bias(self._logits_dimension, self.head_name)
      logits = nn.bias_add(logits, centered_bias)

    predictions = self._logits_to_predictions(logits)
    loss = None
    train_op = None
    eval_metric_ops = None
    if (mode != model_fn.ModeKeys.INFER) and (labels is not None):
      labels_tensor = _to_labels_tensor(labels, self._label_name)
      loss = _training_loss(
          features, labels_tensor, logits,
          loss_fn=self._loss_fn,
          weight_column_name=self._weight_column_name,
          head_name=self.head_name)
      if (mode == model_fn.ModeKeys.TRAIN) and (train_op_fn is not None):
        train_op = _train_op(
            loss, labels_tensor, train_op_fn, centered_bias,
            self._logits_dimension, self._loss_fn)
      eval_metric_ops = _eval_metric_ops(
          self._default_metrics(), features, labels, predictions)

    return model_fn.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        signature_fn=self._signature_fn(),
        output_alternatives=self._create_output_alternatives(predictions))

  def _logits_to_predictions(self, logits):
    """Returns a dict of predictions.

    Args:
      logits: logits `Tensor` after applying possible centered bias.

    Returns:
      Dict of prediction `Tensor` keyed by `PredictionKey`.
    """
    with ops.name_scope(None, "predictions", (logits,)):
      return {
          prediction_key.PredictionKey.LOGITS: logits,
          prediction_key.PredictionKey.PROBABILITIES: nn.softmax(
              logits, name=prediction_key.PredictionKey.PROBABILITIES),
          prediction_key.PredictionKey.CLASSES: math_ops.argmax(
              logits, 1, name=prediction_key.PredictionKey.CLASSES)
      }

  def _signature_fn(self):
    """Returns the signature_fn to be used in exporting."""
    def _classification_signature_fn(examples, features, predictions):
      """Servo signature function."""
      del features
      if isinstance(predictions, dict):
        default_signature = exporter.classification_signature(
            input_tensor=examples,
            classes_tensor=predictions[prediction_key.PredictionKey.CLASSES],
            scores_tensor=predictions[
                prediction_key.PredictionKey.PROBABILITIES])
      else:
        default_signature = exporter.classification_signature(
            input_tensor=examples,
            scores_tensor=predictions)

      # TODO(zakaria): add validation
      return default_signature, {}
    return _classification_signature_fn

  def _metric_spec(self, metric_fn, prediction_name):
    return metric_spec.MetricSpec(
        metric_fn, prediction_name, self._label_name, self._weight_column_name)

  def _default_metrics(self):
    """Returns a dict of `MetricSpec` objects keyed by name."""
    def _streaming_auc_with_class_id_label(predictions, labels, weights=None):
      indicator_labels = _class_id_labels_to_indicator(
          labels, num_classes=self.logits_dimension)
      return _streaming_auc(predictions, indicator_labels, weights)

    loss_key = _summary_key(self.head_name, metric_key.MetricKey.LOSS)
    accuracy_key = _summary_key(self.head_name, metric_key.MetricKey.ACCURACY)
    auc_key = _summary_key(self.head_name, metric_key.MetricKey.AUC)
    metrics = {
        loss_key: _weighted_average_loss_metric_spec(
            self._loss_fn,
            prediction_key.PredictionKey.LOGITS,
            self._label_name,
            self._weight_column_name),
        # TODO(b/29366811): This currently results in both an "accuracy" and an
        # "accuracy/threshold_0.500000_mean" metric for binary classification.
        accuracy_key: self._metric_spec(
            metrics_lib.streaming_accuracy,
            prediction_key.PredictionKey.CLASSES),
        auc_key: self._metric_spec(
            _streaming_auc_with_class_id_label,
            prediction_key.PredictionKey.PROBABILITIES)
    }

    def _class_predictions_streaming_mean(
        predictions, labels, weights=None, class_id=None):
      del labels
      return metrics_lib.streaming_mean(
          array_ops.where(
              math_ops.equal(
                  math_ops.to_int32(class_id),
                  math_ops.to_int32(predictions)),
              array_ops.ones_like(predictions),
              array_ops.zeros_like(predictions)),
          weights=weights)

    def _class_labels_streaming_mean(
        predictions, labels, weights=None, class_id=None):
      del predictions
      assert class_id is not None
      return metrics_lib.streaming_mean(
          array_ops.where(
              math_ops.equal(
                  math_ops.to_int32(class_id),
                  math_ops.to_int32(labels)),
              array_ops.ones_like(labels),
              array_ops.zeros_like(labels)),
          weights=weights)

    def _class_streaming_auc(predictions, labels, weights=None, class_id=None):
      assert class_id is not None
      indicator_labels = _class_id_labels_to_indicator(
          labels, num_classes=self.logits_dimension)
      return _streaming_auc(
          predictions, indicator_labels, weights=weights, class_id=class_id)

    for class_id in self._metric_class_ids:

      # TODO(ptucker): Add per-class accuracy, precision, recall.

      prediction_mean_key = _summary_key(
          self.head_name,
          metric_key.MetricKey.CLASS_PREDICTION_MEAN % class_id)
      label_mean_key = _summary_key(
          self.head_name, metric_key.MetricKey.CLASS_LABEL_MEAN % class_id)
      probability_mean_key = _summary_key(
          self.head_name,
          metric_key.MetricKey.CLASS_PROBABILITY_MEAN % class_id)
      logits_mean_key = _summary_key(
          self.head_name,
          metric_key.MetricKey.CLASS_LOGITS_MEAN % class_id)
      auc_key = _summary_key(
          self.head_name, metric_key.MetricKey.CLASS_AUC % class_id)

      metrics[prediction_mean_key] = self._metric_spec(
          functools.partial(
              _class_predictions_streaming_mean, class_id=class_id),
          prediction_key.PredictionKey.CLASSES)
      metrics[label_mean_key] = self._metric_spec(
          functools.partial(_class_labels_streaming_mean, class_id=class_id),
          prediction_key.PredictionKey.PROBABILITIES)
      metrics[probability_mean_key] = self._metric_spec(
          functools.partial(_predictions_streaming_mean, class_id=class_id),
          prediction_key.PredictionKey.PROBABILITIES)
      metrics[logits_mean_key] = self._metric_spec(
          functools.partial(_predictions_streaming_mean, class_id=class_id),
          prediction_key.PredictionKey.LOGITS)
      metrics[auc_key] = self._metric_spec(
          functools.partial(_class_streaming_auc, class_id=class_id),
          prediction_key.PredictionKey.LOGITS)

    return metrics


def _to_labels_tensor(labels, label_name):
  labels = labels[label_name] if isinstance(labels, dict) else labels
  if isinstance(labels, sparse_tensor.SparseTensor):
    raise ValueError("SparseTensor is not supported as labels.")
  return labels


def _assert_labels_rank(labels):
  return control_flow_ops.Assert(
      math_ops.less_equal(array_ops.rank(labels), 2),
      ("labels shape should be either [batch_size, 1] or [batch_size]",))


class _BinarySvmHead(_BinaryLogisticHead):
  """_Head for binary classification using SVMs."""

  def __init__(self, label_name, weight_column_name, enable_centered_bias,
               head_name, thresholds):
    def _loss_fn(logits, labels):
      with ops.name_scope(None, "hinge_loss", (logits, labels)) as name:
        with ops.control_dependencies((_assert_labels_rank(labels),)):
          labels = array_ops.reshape(labels, shape=(-1, 1))
        return losses_lib.hinge_loss(logits, labels, scope=name)

    super(_BinarySvmHead, self).__init__(
        label_name=label_name,
        weight_column_name=weight_column_name,
        enable_centered_bias=enable_centered_bias,
        head_name=head_name,
        loss_fn=_loss_fn,
        thresholds=thresholds)

  def _logits_to_predictions(self, logits):
    """See `_MultiClassHead`."""
    with ops.name_scope(None, "predictions", (logits,)):
      return {
          prediction_key.PredictionKey.LOGITS: logits,
          prediction_key.PredictionKey.CLASSES: math_ops.argmax(
              _one_class_to_two_class_logits(logits), 1,
              name=prediction_key.PredictionKey.CLASSES)
      }

  def _default_metrics(self):
    """See `_MultiClassHead`."""
    metrics = {_summary_key(self.head_name, metric_key.MetricKey.LOSS):
               _weighted_average_loss_metric_spec(
                   self._loss_fn, prediction_key.PredictionKey.LOGITS,
                   self._label_name, self._weight_column_name)}
    metrics[_summary_key(self.head_name, metric_key.MetricKey.ACCURACY)] = (
        metric_spec.MetricSpec(
            metrics_lib.streaming_accuracy,
            prediction_key.PredictionKey.CLASSES,
            self._label_name, self._weight_column_name))
    # TODO(sibyl-vie3Poto): add more metrics relevant for svms.
    return metrics


class _MultiLabelHead(_MultiClassHead):
  """_Head for multlabel classification."""

  # TODO(zakaria): add signature and metric for multilabel.
  def __init__(self, n_classes, label_name,
               weight_column_name, enable_centered_bias, head_name,
               thresholds, metric_class_ids=None):

    super(_MultiLabelHead, self).__init__(
        n_classes=n_classes,
        label_name=label_name,
        weight_column_name=weight_column_name,
        enable_centered_bias=enable_centered_bias,
        head_name=head_name,
        loss_fn=_sigmoid_cross_entropy_loss,
        thresholds=thresholds,
        metric_class_ids=metric_class_ids)

  def _logits_to_predictions(self, logits):
    """See `_MultiClassHead`."""
    with ops.name_scope(None, "predictions", (logits,)):
      return {
          prediction_key.PredictionKey.LOGITS: logits,
          prediction_key.PredictionKey.PROBABILITIES: math_ops.sigmoid(
              logits, name=prediction_key.PredictionKey.PROBABILITIES),
          prediction_key.PredictionKey.CLASSES: math_ops.to_int64(
              math_ops.greater(logits, 0),
              name=prediction_key.PredictionKey.CLASSES)
      }

  def _metric_spec(self, metric_fn, prediction_name):
    return metric_spec.MetricSpec(
        metric_fn, prediction_name, self._label_name, self._weight_column_name)

  def _default_metrics(self):
    """Returns a dict of `MetricSpec` objects keyed by name."""
    loss_key = _summary_key(self.head_name, metric_key.MetricKey.LOSS)
    accuracy_key = _summary_key(
        self.head_name, metric_key.MetricKey.ACCURACY)
    auc_key = _summary_key(self.head_name, metric_key.MetricKey.AUC)

    metrics = {
        loss_key: _weighted_average_loss_metric_spec(
            self._loss_fn,
            prediction_key.PredictionKey.LOGITS,
            self._label_name,
            self._weight_column_name),
        # TODO(b/29366811): This currently results in both an "accuracy" and an
        # "accuracy/threshold_0.500000_mean" metric for binary classification.
        accuracy_key: self._metric_spec(
            metrics_lib.streaming_accuracy,
            prediction_key.PredictionKey.CLASSES),
        auc_key: self._metric_spec(
            _streaming_auc, prediction_key.PredictionKey.PROBABILITIES),
    }

    for class_id in self._metric_class_ids:

      # TODO(ptucker): Add per-class accuracy, precision, recall.

      prediction_mean_key = _summary_key(
          self.head_name,
          metric_key.MetricKey.CLASS_PREDICTION_MEAN % class_id)
      label_mean_key = _summary_key(
          self.head_name, metric_key.MetricKey.CLASS_LABEL_MEAN % class_id)
      probability_mean_key = _summary_key(
          self.head_name,
          metric_key.MetricKey.CLASS_PROBABILITY_MEAN % class_id)
      logits_mean_key = _summary_key(
          self.head_name, metric_key.MetricKey.CLASS_LOGITS_MEAN % class_id)
      auc_key = _summary_key(
          self.head_name, metric_key.MetricKey.CLASS_AUC % class_id)

      metrics[prediction_mean_key] = self._metric_spec(
          functools.partial(_predictions_streaming_mean, class_id=class_id),
          prediction_key.PredictionKey.CLASSES)
      metrics[label_mean_key] = self._metric_spec(
          functools.partial(
              _indicator_labels_streaming_mean, class_id=class_id),
          prediction_key.PredictionKey.CLASSES)
      metrics[probability_mean_key] = self._metric_spec(
          functools.partial(_predictions_streaming_mean, class_id=class_id),
          prediction_key.PredictionKey.PROBABILITIES)
      metrics[logits_mean_key] = self._metric_spec(
          functools.partial(_predictions_streaming_mean, class_id=class_id),
          prediction_key.PredictionKey.LOGITS)
      metrics[auc_key] = self._metric_spec(
          functools.partial(_streaming_auc, class_id=class_id),
          prediction_key.PredictionKey.LOGITS)

    return metrics


class _MultiHead(_Head):
  """_Head to combine multiple _Head objects.

  All heads stem from the same logits/logit_input tensor.

  For training, combines losses of each heads according a function provided by
  user.
  For eval, adds a /head_name suffix to the keys in eval metrics.
  For inference, updates keys prediction dict to a 2-tuple,
    (head_name, prediction_key)
  """

  def __init__(self, heads, loss_combiner):
    """_Head to combine multiple _Head objects.

    Args:
      heads: list of _Head objects.
      loss_combiner: function that takes a list of loss tensors for the heads
        and returns the final loss tensor for the multi head.

    Raises:
      ValueError: if any head does not have a name.
    """
    # TODO(zakaria): Keep _Head a pure interface.
    super(_MultiHead, self).__init__(head_name=None)
    self._logits_dimension = 0
    for head in heads:
      if not head.head_name:
        raise ValueError("Head must have a name.")
      self._logits_dimension += head.logits_dimension

    self._heads = heads
    self._loss_combiner = loss_combiner

  @property
  def logits_dimension(self):
    return self._logits_dimension

  def head_ops(self, features, target, mode, train_op_fn, logits=None,
               logits_input=None, scope=None):
    """See _Head.head_ops.

    Args:
      features: input dict.
      target: labels dict.
      mode: estimator's ModeKeys
      train_op_fn: function that takes a scalar loss and returns an op to
          optimize with the loss.
      logits: Concatenated logits of (x, 1) shape where x is the sum of
          logits_dimension of all the heads, i.e., same as logits_dimension of
          this class. This function will split the logits tensor and pass logits
          of proper size to each head.
      logits_input: tensor to build logits from.
      scope: Optional scope for variable_scope.

    Returns:
      `ModelFnOps`.

    Raises:
      ValueError: if mode is not recognized or both logits and logits_input is
          provided.
    """
    def _noop(unused_loss):
      return control_flow_ops.no_op()

    if logits is not None and logits_input is not None:
      raise ValueError("only one of logits and logits_input must be provided.")

    all_model_fn_ops = []
    if logits is not None:
      all_logits = self._split_logits(logits)
      for head, logits in zip(self._heads, all_logits):
        all_model_fn_ops.append(head.head_ops(features, target, mode, _noop,
                                              logits=logits, scope=scope))
    else:
      # Uses logits_input
      for head in self._heads:
        all_model_fn_ops.append(head.head_ops(features, target, mode, _noop,
                                              logits_input=logits_input,
                                              scope=scope))

    if mode == model_fn.ModeKeys.TRAIN:
      return self._combine_train(all_model_fn_ops, train_op_fn)
    if mode == model_fn.ModeKeys.INFER:
      return self._combine_infer(all_model_fn_ops)
    if mode == model_fn.ModeKeys.EVAL:
      return self._combine_eval(all_model_fn_ops)
    raise ValueError("mode=%s unrecognized" % str(mode))

  def _split_logits(self, logits):
    """Splits logits for heads.

    Args:
      logits: the logits tensor.

    Returns:
      A list of logits for the individual heads.
    """
    all_logits = []
    begin = 0
    for head in self._heads:
      current_logits_size = head.logits_dimension
      current_logits = array_ops.slice(logits, [0, begin],
                                       [-1, current_logits_size])
      all_logits.append(current_logits)
      begin += current_logits_size
    return all_logits

  def _combine_train(self, all_model_fn_ops, train_op_fn):
    """Combines list of ModelFnOps for training.

    Args:
      all_model_fn_ops: list of ModelFnOps for the individual heads.
      train_op_fn: Function to create train op. See head_ops documentaion for
          more details.

    Returns:
      ModelFnOps that combines all the heads.
    """
    losses = []
    additional_train_ops = []
    for m in all_model_fn_ops:
      losses.append(m.loss)
      additional_train_ops.append(m.train_op)
    loss = self._loss_combiner(losses)

    train_op = train_op_fn(loss)
    train_op = control_flow_ops.group(train_op, *additional_train_ops)
    return model_fn.ModelFnOps(model_fn.ModeKeys.TRAIN,
                               None, loss, train_op, None, None)

  def _combine_infer(self, all_model_fn_ops):
    """Combines list of ModelFnOps for inference.

    Args:
      all_model_fn_ops: list of ModelFnOps for the individual heads.

    Returns:
      ModelFnOps that combines all the heads.
    """
    predictions = {}
    output_alternatives = {}
    for head, m in zip(self._heads, all_model_fn_ops):
      head_name = head.head_name
      output_alternatives[head_name] = m.output_alternatives[head_name]
      for k, v in m.predictions.items():
        predictions[(head_name, k)] = v

    return model_fn.ModelFnOps(model_fn.ModeKeys.INFER, predictions, None,
                               None, None,
                               # signature_fn is for session bundle, not
                               # applicable for savedmodel.
                               None,
                               output_alternatives)

  def _combine_eval(self, all_model_fn_ops):
    """Combines list of ModelFnOps for eval.

    Args:
      all_model_fn_ops: list of ModelFnOps for the individual heads.

    Returns:
      ModelFnOps that combines all the heads.
    """
    predictions = {}
    metrics = {}
    losses = []
    for head, m in zip(self._heads, all_model_fn_ops):
      losses.append(m.loss)
      head_name = head.head_name
      for k, v in m.predictions.items():
        predictions[(head_name, k)] = v
      for k, v in m.eval_metric_ops.items():
        # metrics["%s/%s" % (k, head_name)] = v
        metrics[k] = v
    loss = self._loss_combiner(losses)

    return model_fn.ModelFnOps(model_fn.ModeKeys.EVAL, predictions, loss,
                               None, metrics, None)


def _weighted_loss(loss, weight):
  """Returns cumulative weighted loss as 1d `Tensor`."""
  with ops.name_scope(None, "weighted_loss", (loss, weight)) as name:
    return math_ops.mul(array_ops.reshape(loss, shape=(-1,)),
                        array_ops.reshape(weight, shape=(-1,)),
                        name=name)


def _weight_tensor(features, weight_column_name):
  """Returns weights as 1d `Tensor`."""
  if not weight_column_name:
    return None
  with ops.name_scope(
      None, "weight_tensor", tuple(six.itervalues(features))) as name:
    return array_ops.reshape(
        math_ops.to_float(features[weight_column_name]),
        shape=(-1,),
        name=name)


def _loss(loss_unweighted, weight, name):
  """Returns a tuple of (loss, weighted_average_loss)."""
  with ops.name_scope(name, values=(loss_unweighted, weight)) as name_scope:
    if weight is None:
      loss = math_ops.reduce_mean(loss_unweighted, name=name_scope)
      return loss, loss
    loss_weighted = _weighted_loss(loss_unweighted, weight)
    weighted_average_loss = math_ops.div(
        math_ops.reduce_sum(loss_weighted),
        math_ops.to_float(math_ops.reduce_sum(weight)),
        name="weighted_average_loss")
    loss = math_ops.reduce_mean(loss_weighted, name=name_scope)
    return loss, weighted_average_loss


def _check_logits_input_not_supported(logits, logits_input):
  if logits_input is not None or logits is None:
    raise NotImplementedError("logits_input is not supported yet, "
                              "must pass logits")


def _check_mode_valid(mode):
  """Raises ValueError if the given mode is invalid."""
  if (mode != model_fn.ModeKeys.TRAIN and
      mode != model_fn.ModeKeys.INFER and
      mode != model_fn.ModeKeys.EVAL):
    raise ValueError("mode=%s unrecognized." % str(mode))


def _centered_bias(logits_dimension, head_name=None):
  """Returns `logits`, optionally with centered bias applied.

  Args:
    logits_dimension: Last dimension of `logits`. Must be >= 1.
    head_name: Optional name of the head.

  Returns:
    Centered bias `Variable`.

  Raises:
    ValueError: if `logits_dimension` is invalid.
  """
  if (logits_dimension is None) or (logits_dimension < 1):
    raise ValueError("Invalid logits_dimension %s." % logits_dimension)
  centered_bias = variable_scope.get_variable(
      name="centered_bias_weight",
      shape=(logits_dimension,),
      initializer=init_ops.zeros_initializer(),
      trainable=True)
  for dim in range(logits_dimension):
    if head_name:
      summary.scalar("centered_bias/bias_%d/%s" % (dim, head_name),
                     centered_bias[dim])
    else:
      summary.scalar("centered_bias/bias_%d" % dim, centered_bias[dim])
  return centered_bias


def _centered_bias_step(centered_bias, logits_dimension, labels, loss_fn):
  """Creates and returns training op for centered bias."""
  if (logits_dimension is None) or (logits_dimension < 1):
    raise ValueError("Invalid logits_dimension %s." % logits_dimension)
  with ops.name_scope(None, "centered_bias_step", (labels,)) as name:
    batch_size = array_ops.shape(labels)[0]
    logits = array_ops.reshape(
        array_ops.tile(centered_bias, (batch_size,)),
        (batch_size, logits_dimension))
    with ops.name_scope(None, "centered_bias", (labels, logits)):
      centered_bias_loss = math_ops.reduce_mean(
          loss_fn(logits, labels), name="training_loss")
  # Learn central bias by an optimizer. 0.1 is a convervative lr for a
  # single variable.
  return training.AdagradOptimizer(0.1).minimize(
      centered_bias_loss, var_list=(centered_bias,), name=name)


def _summary_key(head_name, val):
  return "%s/%s" % (val, head_name) if head_name else val


def _training_loss(
    features, labels, logits, loss_fn, weight_column_name=None, head_name=None):
  """Returns training loss tensor.

  Training loss is different from the loss reported on the tensorboard as we
  should respect the example weights when computing the gradient.

    L = sum_{i} w_{i} * l_{i} / B

  where B is the number of examples in the batch, l_{i}, w_{i} are individual
  losses, and example weight.

  Args:
    features: Features `dict`.
    labels: Either a `Tensor` for labels or in multihead case, a `dict` of
      string to `Tensor`.
    logits: logits, a float `Tensor`. Shape is `(batch_size, logits_dimension)`.
    loss_fn: Function taking `logits` and `labels`, and returning the raw
      unweighted loss.
    weight_column_name: Key for weights `Tensor` in `features`, if applicable.
    head_name: Head name, used for summary.

  Returns:
    A loss `Output`.
  """
  with ops.name_scope(
      None, "training_loss",
      tuple(six.itervalues(features)) + (labels, logits)) as name:
    loss, weighted_average_loss = _loss(
        loss_fn(logits, labels),
        _weight_tensor(features, weight_column_name),
        name=name)
    # The tag must be same as the tag for eval loss, so the losses will show up
    # in the same graph in tensorboard.
    logging_ops.scalar_summary(_summary_key(head_name, "loss"),
                               weighted_average_loss)
    return loss


def _train_op(
    loss, labels, train_op_fn, centered_bias=None, logits_dimension=None,
    loss_fn=None):
  """Returns op for the training step."""
  if centered_bias is not None:
    centered_bias_step = _centered_bias_step(
        centered_bias, logits_dimension, labels, loss_fn)
  else:
    centered_bias_step = None
  with ops.name_scope(None, "train_op", (loss, labels)):
    train_op = train_op_fn(loss)
    if centered_bias_step is not None:
      train_op = control_flow_ops.group(train_op, centered_bias_step)
    return train_op


def _eval_metric_ops(metrics, features, labels, predictions):
  with ops.name_scope(
      None, "metrics",
      (tuple(six.itervalues(features)) +
       (labels,) +
       tuple(six.itervalues(predictions)))):
    # pylint: disable=protected-access
    return estimator._make_metrics_ops(metrics, features, labels, predictions)
    # pylint: enable=protected-access


def _sigmoid_cross_entropy_loss(logits, labels):
  with ops.name_scope(
      None, "sigmoid_cross_entropy_loss", (logits, labels)) as name:
    # sigmoid_cross_entropy_with_logits requires [batch_size, n_classes] labels.
    return nn.sigmoid_cross_entropy_with_logits(
        logits, math_ops.to_float(labels), name=name)


def _float_weights_or_none(weights):
  if weights is None:
    return None
  with ops.name_scope(None, "float_weights", (weights,)) as name:
    return math_ops.to_float(weights, name=name)


def _weighted_average_loss_metric_spec(loss_fn, pred_key,
                                       label_key, weight_key):
  def _streaming_weighted_average_loss(predictions, labels, weights=None):
    loss_unweighted = loss_fn(predictions, labels)
    if weights is not None:
      weights = math_ops.to_float(weights)
    _, weighted_average_loss = _loss(loss_unweighted,
                                     weights,
                                     name="eval_loss")
    return metrics_lib.streaming_mean(weighted_average_loss)
  return metric_spec.MetricSpec(
      _streaming_weighted_average_loss, pred_key, label_key, weight_key)


def _indicator_labels_streaming_mean(
    predictions, labels, weights=None, class_id=None):
  del predictions
  if class_id is not None:
    labels = labels[:, class_id]
  return metrics_lib.streaming_mean(labels, weights=weights)


def _predictions_streaming_mean(
    predictions, labels, weights=None, class_id=None):
  del labels
  if class_id is not None:
    predictions = predictions[:, class_id]
  return metrics_lib.streaming_mean(predictions, weights=weights)


# TODO(ptucker): Add support for SparseTensor labels.
def _class_id_labels_to_indicator(labels, num_classes):
  if (num_classes is None) or (num_classes < 2):
    raise ValueError("Invalid num_classes %s." % num_classes)
  with ops.control_dependencies((_assert_labels_rank(labels),)):
    labels = array_ops.reshape(labels, (-1,))
  return array_ops.one_hot(labels, depth=num_classes, axis=-1)


def _streaming_auc(predictions, labels, weights=None, class_id=None):
  if class_id is not None:
    predictions = predictions[:, class_id]
    labels = labels[:, class_id]
  return metrics_lib.streaming_auc(
      predictions, math_ops.cast(labels, dtypes.bool),
      weights=_float_weights_or_none(weights))


def _assert_class_id(class_id, num_classes=None):
  """Average label value for class `class_id`."""
  if (class_id is None) or (class_id < 0):
    raise ValueError("Invalid class_id %s." % class_id)
  if num_classes is not None:
    if num_classes < 2:
      raise ValueError("Invalid num_classes %s." % num_classes)
    if class_id >= num_classes:
      raise ValueError("Invalid class_id %s." % class_id)


def _accuracy_at_threshold(threshold):

  def _accuracy_metric(predictions, labels, weights=None):
    threshold_predictions = math_ops.to_float(
        math_ops.greater_equal(predictions, threshold))
    return metrics_lib.streaming_accuracy(predictions=threshold_predictions,
                                          labels=labels,
                                          weights=weights)

  return _accuracy_metric


def _streaming_at_threshold(streaming_metrics_fn, threshold):

  def _streaming_metrics(predictions, labels, weights=None):
    precision_tensor, update_op = streaming_metrics_fn(
        predictions, labels=labels, thresholds=(threshold,),
        weights=_float_weights_or_none(weights))
    return array_ops.squeeze(precision_tensor), array_ops.squeeze(update_op)

  return _streaming_metrics
