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
import six

from tensorflow.contrib import framework as framework_lib
from tensorflow.contrib import layers as layers_lib
# TODO(ptucker): Use tf.losses and tf.metrics.
from tensorflow.contrib import losses as losses_lib
from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.contrib.learn.python.learn.estimators.metric_key import MetricKey as mkey
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.summary import summary
from tensorflow.python.training import training

# TODO(zakaria): add functions that creates a head and returns ModelOpFn


def _regression_head(label_name=None,
                     weight_column_name=None,
                     label_dimension=1,
                     enable_centered_bias=False,
                     head_name=None):
  """Creates a _Head for linear regression.

  Args:
    label_name: String, name of the key in label dict. Can be null if label
        is a tensor (single headed models).
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    label_dimension: Number of regression labels per example. This is the size
      of the last dimension of the labels `Tensor` (typically, this has shape
      `[batch_size, label_dimension]`).
    enable_centered_bias: A bool. If True, estimator will learn a centered
      bias variable for each class. Rest of the model structure learns the
      residual after centered bias.
    head_name: name of the head. If provided, predictions, summary and metrics
      keys will be suffixed by `"/" + head_name` and the default variable scope
      will be `head_name`.

  Returns:
    An instance of _Head
  """
  return _RegressionHead(
      label_name=label_name,
      weight_column_name=weight_column_name,
      label_dimension=label_dimension,
      enable_centered_bias=enable_centered_bias,
      head_name=head_name,
      loss_fn=_mean_squared_loss,
      link_fn=array_ops.identity)


def _poisson_regression_head(label_name=None,
                             weight_column_name=None,
                             label_dimension=1,
                             enable_centered_bias=False,
                             head_name=None):
  """Creates a _Head for linear regression.

  Args:
    label_name: String, name of the key in label dict. Can be null if label
        is a tensor (single headed models).
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    label_dimension: Number of regression labels per example. This is the size
      of the last dimension of the labels `Tensor` (typically, this has shape
      `[batch_size, label_dimension]`).
    enable_centered_bias: A bool. If True, estimator will learn a centered
      bias variable for each class. Rest of the model structure learns the
      residual after centered bias.
    head_name: name of the head. If provided, predictions, summary and metrics
      keys will be suffixed by `"/" + head_name` and the default variable scope
      will be `head_name`.

  Returns:
    An instance of _Head
  """
  return _RegressionHead(
      label_name=label_name,
      weight_column_name=weight_column_name,
      label_dimension=label_dimension,
      enable_centered_bias=enable_centered_bias,
      head_name=head_name,
      loss_fn=_poisson_loss,
      link_fn=math_ops.exp)

# TODO(zakaria): Add logistic_regression_head


def _multi_class_head(n_classes,
                      label_name=None,
                      weight_column_name=None,
                      enable_centered_bias=False,
                      head_name=None,
                      thresholds=None,
                      metric_class_ids=None):
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
      keys will be suffixed by `"/" + head_name` and the default variable scope
      will be `head_name`.
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
    raise ValueError("n_classes must be > 1 for classification: %s." %
                     n_classes)

  if n_classes == 2:
    if metric_class_ids:
      raise ValueError("metric_class_ids invalid for n_classes==2.")
    return _BinaryLogisticHead(
        label_name=label_name,
        weight_column_name=weight_column_name,
        enable_centered_bias=enable_centered_bias,
        head_name=head_name,
        thresholds=thresholds)

  return _MultiClassHead(
      n_classes=n_classes,
      label_name=label_name,
      weight_column_name=weight_column_name,
      enable_centered_bias=enable_centered_bias,
      head_name=head_name,
      thresholds=thresholds,
      metric_class_ids=metric_class_ids)


def _binary_svm_head(
    label_name=None,
    weight_column_name=None,
    enable_centered_bias=False,
    head_name=None,
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
      keys will be suffixed by `"/" + head_name` and the default variable scope
      will be `head_name`.
    thresholds: thresholds for eval metrics, defaults to [.5]

  Returns:
    An instance of `_Head`.

  """
  return _BinarySvmHead(
      label_name=label_name,
      weight_column_name=weight_column_name,
      enable_centered_bias=enable_centered_bias,
      head_name=head_name,
      thresholds=thresholds)


def _multi_label_head(n_classes,
                      label_name=None,
                      weight_column_name=None,
                      enable_centered_bias=False,
                      head_name=None,
                      thresholds=None,
                      metric_class_ids=None):
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
      keys will be suffixed by `"/" + head_name` and the default variable scope
      will be `head_name`.
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
  return _MultiLabelHead(
      n_classes=n_classes,
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
#   them. See b/34751732
class _Head(object):
  """Interface for the head/top of a model.

  Given logits or output of a hidden layer, a Head knows how to compute
  predictions, loss, default metric and export signature.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def logits_dimension(self):
    """Size of the last dimension of the logits `Tensor`.

    Typically, logits is of shape `[batch_size, logits_dimension]`.

    Returns:
      Number of logits values per example.
    """
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def create_model_fn_ops(self,
                          features,
                          mode,
                          labels=None,
                          train_op_fn=None,
                          logits=None,
                          logits_input=None,
                          scope=None):
    """Returns ops for a model_fn.

    Exactly one of `logits` and `logits_input` must be provided.

    All args must be passed via name.

    Args:
      features: Input `dict` of `Tensor` objects.
      mode: Estimator's `ModeKeys`.
      labels: Labels `Tensor`, or `dict` of same.
      train_op_fn: Function that takes a scalar loss and returns an op to
          optimize with the loss.
      logits: logits `Tensor`, or `dict` of same, to be used for the head.
      logits_input: `Tensor` from which to build logits.
      scope: Optional scope for `variable_scope`.

    Returns:
      `ModelFnOps`.

    Raises:
      ValueError: if `mode` is not recognized, or neither or both of `logits`
          and `logits_input` is provided.
    """
    raise NotImplementedError("Calling an abstract method.")


class _SingleHead(_Head):
  """Interface for a single head/top of a model."""
  __metaclass__ = abc.ABCMeta

  def __init__(
      self, problem_type, logits_dimension, label_name=None,
      weight_column_name=None, head_name=None):
    if problem_type is None:
      raise ValueError("Invalid problem_type %s." % problem_type)
    if logits_dimension is None or logits_dimension < 1:
      raise ValueError("Invalid logits_dimension %s." % logits_dimension)
    self._problem_type = problem_type
    self._logits_dimension = logits_dimension
    self._label_name = label_name
    self._weight_column_name = weight_column_name
    self._head_name = head_name

  @property
  def logits_dimension(self):
    return self._logits_dimension

  @property
  def label_name(self):
    return self._label_name

  @property
  def weight_column_name(self):
    return self._weight_column_name

  @property
  def head_name(self):
    return self._head_name

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
    return {self._head_name: (self._problem_type, predictions)}


# TODO(zakaria): use contrib losses.
def _mean_squared_loss(logits, labels):
  with ops.name_scope(None, "mean_squared_loss", (logits, labels)) as name:
    logits = ops.convert_to_tensor(logits)
    labels = ops.convert_to_tensor(labels)
    # To prevent broadcasting inside "-".
    if len(labels.get_shape()) == 1:
      labels = array_ops.expand_dims(labels, dim=(1,))
    # TODO(zakaria): make sure it does not recreate the broadcast bug.
    if len(logits.get_shape()) == 1:
      logits = array_ops.expand_dims(logits, dim=(1,))
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    return math_ops.square(logits - math_ops.to_float(labels), name=name)


def _poisson_loss(logits, labels):
  """Computes poisson loss from logits."""
  with ops.name_scope(None, "_poisson_loss", (logits, labels)) as name:
    logits = ops.convert_to_tensor(logits)
    labels = ops.convert_to_tensor(labels)
    # To prevent broadcasting inside "-".
    if len(labels.get_shape()) == 1:
      labels = array_ops.expand_dims(labels, dim=(1,))
    # TODO(zakaria): make sure it does not recreate the broadcast bug.
    if len(logits.get_shape()) == 1:
      logits = array_ops.expand_dims(logits, dim=(1,))
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    return nn.log_poisson_loss(labels, logits,
                               compute_full_loss=True, name=name)


def _logits(logits_input, logits, logits_dimension):
  """Validate logits args, and create `logits` if necessary.

  Exactly one of `logits_input` and `logits` must be provided.

  Args:
    logits_input: `Tensor` input to `logits`.
    logits: `Tensor` output.
    logits_dimension: Integer, last dimension of `logits`. This is used to
      create `logits` from `logits_input` if `logits` is `None`; otherwise, it's
      used to validate `logits`.

  Returns:
    `logits` `Tensor`.

  Raises:
    ValueError: if neither or both of `logits` and `logits_input` are supplied.
  """
  if (logits_dimension is None) or (logits_dimension < 1):
    raise ValueError("Invalid logits_dimension %s." % logits_dimension)

  # If not provided, create logits.
  if logits is None:
    if logits_input is None:
      raise ValueError("Neither logits nor logits_input supplied.")
    return layers_lib.linear(logits_input, logits_dimension, scope="logits")

  if logits_input is not None:
    raise ValueError("Both logits and logits_input supplied.")

  logits = ops.convert_to_tensor(logits, name="logits")
  logits_dims = logits.get_shape().dims
  if logits_dims is not None:
    logits_dims[-1].assert_is_compatible_with(logits_dimension)

  return logits


def _create_model_fn_ops(features,
                         mode,
                         transform_labels_fn,
                         loss_fn,
                         logits_to_predictions_fn,
                         metrics_fn,
                         create_output_alternatives_fn,
                         default_variable_scope_name,
                         labels=None,
                         train_op_fn=None,
                         logits=None,
                         logits_input=None,
                         logits_dimension=None,
                         head_name=None,
                         weight_column_name=None,
                         enable_centered_bias=False):
  """Returns a `ModelFnOps` object."""
  _check_mode_valid(mode)

  with variable_scope.variable_scope(
      None,
      default_name=head_name or default_variable_scope_name,
      values=(tuple(six.itervalues(features)) +
              (labels, logits, logits_input))):
    if (mode != model_fn.ModeKeys.INFER) and (labels is not None):
      labels = transform_labels_fn(labels)
    else:
      labels = None

    logits = _logits(logits_input, logits, logits_dimension)
    centered_bias = None
    if enable_centered_bias:
      centered_bias = _centered_bias(logits_dimension, head_name)
      logits = nn.bias_add(logits, centered_bias)

    predictions = logits_to_predictions_fn(logits)
    loss = None
    train_op = None
    eval_metric_ops = None
    if (mode != model_fn.ModeKeys.INFER) and (labels is not None):
      weight_tensor = _weight_tensor(features, weight_column_name)
      loss, weighted_average_loss = _loss(
          loss_fn(logits, labels), weight_tensor)
      logging_ops.scalar_summary(
          _summary_key(head_name, mkey.LOSS), weighted_average_loss)

      if (mode == model_fn.ModeKeys.TRAIN) and (train_op_fn is not None):
        train_op = _train_op(loss, labels, train_op_fn, centered_bias,
                             logits_dimension, loss_fn)
      eval_metric_ops = metrics_fn(
          weighted_average_loss, predictions, labels, weight_tensor)
    return model_fn.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        output_alternatives=create_output_alternatives_fn(predictions))


class _RegressionHead(_SingleHead):
  """_Head for regression with a generalized linear model."""

  def __init__(self,
               label_dimension,
               loss_fn,
               link_fn,
               label_name=None,
               weight_column_name=None,
               enable_centered_bias=False,
               head_name=None):
    """Head for regression.

    Args:
      label_dimension: Number of regression labels per example. This is the
        size of the last dimension of the labels `Tensor` (typically, this has
        shape `[batch_size, label_dimension]`).
      loss_fn: Loss function, takes logits and labels and returns loss.
      link_fn: Link function, takes a logits tensor and returns the output.
      label_name: String, name of the key in label dict. Can be null if label
          is a tensor (single headed models).
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      head_name: name of the head. Predictions, summary and metrics keys are
        suffixed by `"/" + head_name` and the default variable scope is
        `head_name`.
    """
    super(_RegressionHead, self).__init__(
        problem_type=constants.ProblemType.LINEAR_REGRESSION,
        logits_dimension=label_dimension,
        label_name=label_name,
        weight_column_name=weight_column_name,
        head_name=head_name)

    self._loss_fn = loss_fn
    self._link_fn = link_fn
    self._enable_centered_bias = enable_centered_bias

  def create_model_fn_ops(self,
                          features,
                          mode,
                          labels=None,
                          train_op_fn=None,
                          logits=None,
                          logits_input=None,
                          scope=None):
    """See `_Head`."""
    return _create_model_fn_ops(
        features=features,
        mode=mode,
        transform_labels_fn=self._transform_labels,
        loss_fn=self._loss_fn,
        logits_to_predictions_fn=self._logits_to_predictions,
        metrics_fn=self._metrics,
        create_output_alternatives_fn=self._create_output_alternatives,
        default_variable_scope_name="regression_head",
        labels=labels,
        train_op_fn=train_op_fn,
        logits=logits,
        logits_input=logits_input,
        logits_dimension=self.logits_dimension,
        head_name=self.head_name,
        weight_column_name=self.weight_column_name,
        enable_centered_bias=self._enable_centered_bias)

  def _transform_labels(self, labels):
    """Applies transformations to labels tensor."""
    labels_tensor = _to_labels_tensor(labels, self._label_name)
    _check_no_sparse_tensor(labels_tensor)
    return labels_tensor

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
      return {key: self._link_fn(logits)}

  def _metrics(self, eval_loss, predictions, labels, weights):
    """Returns a dict of metrics keyed by name."""
    del predictions, labels, weights  # Unused by this head.
    with ops.name_scope("metrics", values=[eval_loss]):
      return {
          _summary_key(self.head_name, mkey.LOSS):
              metrics_lib.streaming_mean(eval_loss)}


def _log_loss_with_two_classes(logits, labels):
  with ops.name_scope(None, "log_loss_with_two_classes",
                      (logits, labels)) as name:
    logits = ops.convert_to_tensor(logits)
    labels = math_ops.to_float(labels)
    # TODO(ptucker): This will break for dynamic shapes.
    # sigmoid_cross_entropy_with_logits requires [batch_size, 1] labels.
    if len(labels.get_shape()) == 1:
      labels = array_ops.expand_dims(labels, dim=(1,))
    return nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits, name=name)


def _one_class_to_two_class_logits(logits):
  return array_ops.concat((array_ops.zeros_like(logits), logits), 1)


class _BinaryLogisticHead(_SingleHead):
  """_Head for binary logistic classifciation."""

  def __init__(self,
               label_name=None,
               weight_column_name=None,
               enable_centered_bias=False,
               head_name=None,
               loss_fn=_log_loss_with_two_classes,
               thresholds=None):
    """Base type for all single heads.

    Args:
      label_name: String, name of the key in label dict. Can be `None` if label
          is a tensor (single headed models).
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      head_name: name of the head. Predictions, summary, metrics keys are
        suffixed by `"/" + head_name` and the default variable scope is
        `head_name`.
      loss_fn: Loss function.
      thresholds: thresholds for eval.

    Raises:
      ValueError: if n_classes is invalid.
    """
    super(_BinaryLogisticHead, self).__init__(
        problem_type=constants.ProblemType.LOGISTIC_REGRESSION,
        logits_dimension=1,
        label_name=label_name,
        weight_column_name=weight_column_name,
        head_name=head_name)
    self._thresholds = thresholds if thresholds else (.5,)
    self._loss_fn = loss_fn
    self._enable_centered_bias = enable_centered_bias

  def create_model_fn_ops(self,
                          features,
                          mode,
                          labels=None,
                          train_op_fn=None,
                          logits=None,
                          logits_input=None,
                          scope=None):
    """See `_Head`."""
    return _create_model_fn_ops(
        features=features,
        mode=mode,
        transform_labels_fn=self._transform_labels,
        loss_fn=self._loss_fn,
        logits_to_predictions_fn=self._logits_to_predictions,
        metrics_fn=self._metrics,
        create_output_alternatives_fn=self._create_output_alternatives,
        default_variable_scope_name="binary_logistic_head",
        labels=labels,
        train_op_fn=train_op_fn,
        logits=logits,
        logits_input=logits_input,
        logits_dimension=self.logits_dimension,
        head_name=self.head_name,
        weight_column_name=self.weight_column_name,
        enable_centered_bias=self._enable_centered_bias)

  def _transform_labels(self, labels):
    """Applies transformations to labels tensor."""
    labels_tensor = _to_labels_tensor(labels, self._label_name)
    _check_no_sparse_tensor(labels_tensor)
    return labels_tensor

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
          prediction_key.PredictionKey.LOGITS:
              logits,
          prediction_key.PredictionKey.LOGISTIC:
              math_ops.sigmoid(
                  logits, name=prediction_key.PredictionKey.LOGISTIC),
          prediction_key.PredictionKey.PROBABILITIES:
              nn.softmax(
                  two_class_logits,
                  name=prediction_key.PredictionKey.PROBABILITIES),
          prediction_key.PredictionKey.CLASSES:
              math_ops.argmax(
                  two_class_logits,
                  1,
                  name=prediction_key.PredictionKey.CLASSES)
      }

  def _metrics(self, eval_loss, predictions, labels, weights):
    """Returns a dict of metrics keyed by name."""
    with ops.name_scope("metrics", values=(
        [eval_loss, labels, weights] + list(six.itervalues(predictions)))):
      classes = predictions[prediction_key.PredictionKey.CLASSES]
      logistic = predictions[prediction_key.PredictionKey.LOGISTIC]

      metrics = {_summary_key(self.head_name, mkey.LOSS):
                 metrics_lib.streaming_mean(eval_loss)}
      # TODO(b/29366811): This currently results in both an "accuracy" and an
      # "accuracy/threshold_0.500000_mean" metric for binary classification.
      metrics[_summary_key(self.head_name, mkey.ACCURACY)] = (
          metrics_lib.streaming_accuracy(classes, labels, weights))
      metrics[_summary_key(self.head_name, mkey.PREDICTION_MEAN)] = (
          _predictions_streaming_mean(logistic, weights))
      metrics[_summary_key(self.head_name, mkey.LABEL_MEAN)] = (
          _indicator_labels_streaming_mean(labels, weights))

      # Also include the streaming mean of the label as an accuracy baseline, as
      # a reminder to users.
      metrics[_summary_key(self.head_name, mkey.ACCURACY_BASELINE)] = (
          _indicator_labels_streaming_mean(labels, weights))
      metrics[_summary_key(self.head_name, mkey.AUC)] = (
          _streaming_auc(logistic, labels, weights))

      for threshold in self._thresholds:
        metrics[_summary_key(
            self.head_name, mkey.ACCURACY_MEAN % threshold)] = (
                _streaming_accuracy_at_threshold(logistic, labels, weights,
                                                 threshold))
        # Precision for positive examples.
        metrics[_summary_key(
            self.head_name, mkey.PRECISION_MEAN % threshold)] = (
                _streaming_precision_at_threshold(logistic, labels, weights,
                                                  threshold))
        # Recall for positive examples.
        metrics[_summary_key(
            self.head_name, mkey.RECALL_MEAN % threshold)] = (
                _streaming_recall_at_threshold(logistic, labels, weights,
                                               threshold))

    return metrics


def _softmax_cross_entropy_loss(logits, labels):
  with ops.name_scope(
      None, "softmax_cross_entropy_loss", (logits, labels,)) as name:
    labels = ops.convert_to_tensor(labels)
    # Check that we got integer for classification.
    if not labels.dtype.is_integer:
      raise ValueError("Labels dtype should be integer "
                       "Instead got %s." % labels.dtype)
    # TODO(ptucker): This will break for dynamic shapes.
    # sparse_softmax_cross_entropy_with_logits requires [batch_size] labels.
    if len(labels.get_shape()) == 2:
      labels = array_ops.squeeze(labels, squeeze_dims=(1,))
    return nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name=name)


class _MultiClassHead(_SingleHead):
  """_Head for classification."""

  def __init__(self,
               n_classes,
               label_name=None,
               weight_column_name=None,
               enable_centered_bias=False,
               head_name=None,
               loss_fn=_softmax_cross_entropy_loss,
               thresholds=None,
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
      head_name: name of the head. If provided, predictions, summary, metrics
        keys will be suffixed by `"/" + head_name` and the default variable
        scope will be `head_name`.
      loss_fn: Loss function.
      thresholds: thresholds for eval.
      metric_class_ids: List of class IDs for which we should report per-class
        metrics. Must all be in the range `[0, n_classes)`.

    Raises:
      ValueError: if `n_classes` or `metric_class_ids` is invalid.
    """
    super(_MultiClassHead, self).__init__(
        problem_type=constants.ProblemType.CLASSIFICATION,
        logits_dimension=n_classes,
        label_name=label_name,
        weight_column_name=weight_column_name,
        head_name=head_name)

    if (n_classes is None) or (n_classes <= 2):
      raise ValueError("n_classes must be > 2: %s." % n_classes)
    self._thresholds = thresholds if thresholds else (.5,)
    self._loss_fn = loss_fn
    self._enable_centered_bias = enable_centered_bias
    self._metric_class_ids = tuple([] if metric_class_ids is None else
                                   metric_class_ids)
    for class_id in self._metric_class_ids:
      if (class_id < 0) or (class_id >= n_classes):
        raise ValueError("Class ID %s not in [0, %s)." % (class_id, n_classes))

  def create_model_fn_ops(self,
                          features,
                          mode,
                          labels=None,
                          train_op_fn=None,
                          logits=None,
                          logits_input=None,
                          scope=None):
    """See `_Head`."""
    return _create_model_fn_ops(
        features=features,
        mode=mode,
        transform_labels_fn=self._transform_labels,
        loss_fn=self._loss_fn,
        logits_to_predictions_fn=self._logits_to_predictions,
        metrics_fn=self._metrics,
        create_output_alternatives_fn=self._create_output_alternatives,
        default_variable_scope_name="multi_class_head",
        labels=labels,
        train_op_fn=train_op_fn,
        logits=logits,
        logits_input=logits_input,
        logits_dimension=self.logits_dimension,
        head_name=self.head_name,
        weight_column_name=self.weight_column_name,
        enable_centered_bias=self._enable_centered_bias)

  def _transform_labels(self, labels):
    """Applies transformations to labels tensor."""
    labels_tensor = _to_labels_tensor(labels, self._label_name)
    _check_no_sparse_tensor(labels_tensor)
    return labels_tensor

  def _logits_to_predictions(self, logits):
    """Returns a dict of predictions.

    Args:
      logits: logits `Tensor` after applying possible centered bias.

    Returns:
      Dict of prediction `Tensor` keyed by `PredictionKey`.
    """
    with ops.name_scope(None, "predictions", (logits,)):
      return {
          prediction_key.PredictionKey.LOGITS:
              logits,
          prediction_key.PredictionKey.PROBABILITIES:
              nn.softmax(
                  logits, name=prediction_key.PredictionKey.PROBABILITIES),
          prediction_key.PredictionKey.CLASSES:
              math_ops.argmax(
                  logits, 1, name=prediction_key.PredictionKey.CLASSES)
      }

  def _metrics(self, eval_loss, predictions, labels, weights):
    """Returns a dict of metrics keyed by name."""
    with ops.name_scope("metrics", values=(
        [eval_loss, labels, weights] + list(six.itervalues(predictions)))):
      classes = predictions[prediction_key.PredictionKey.CLASSES]
      probabilities = predictions[prediction_key.PredictionKey.PROBABILITIES]
      logits = predictions[prediction_key.PredictionKey.LOGITS]

      metrics = {_summary_key(self.head_name, mkey.LOSS):
                 metrics_lib.streaming_mean(eval_loss)}
      # TODO(b/29366811): This currently results in both an "accuracy" and an
      # "accuracy/threshold_0.500000_mean" metric for binary classification.
      metrics[_summary_key(self.head_name, mkey.ACCURACY)] = (
          metrics_lib.streaming_accuracy(classes, labels, weights))
      metrics[_summary_key(self.head_name, mkey.AUC)] = (
          _streaming_auc_with_class_id_label(
              probabilities, labels, weights, self.logits_dimension))

      for class_id in self._metric_class_ids:
        # TODO(ptucker): Add per-class accuracy, precision, recall.
        metrics[_summary_key(
            self.head_name, mkey.CLASS_PREDICTION_MEAN % class_id)] = (
                _class_predictions_streaming_mean(classes, weights, class_id))
        metrics[_summary_key(
            self.head_name, mkey.CLASS_LABEL_MEAN % class_id)] = (
                _class_labels_streaming_mean(labels, weights, class_id))
        metrics[_summary_key(
            self.head_name, mkey.CLASS_PROBABILITY_MEAN % class_id)] = (
                _predictions_streaming_mean(probabilities, weights, class_id))
        metrics[_summary_key(
            self.head_name, mkey.CLASS_LOGITS_MEAN % class_id)] = (
                _predictions_streaming_mean(logits, weights, class_id))
        metrics[_summary_key(self.head_name, mkey.CLASS_AUC % class_id)] = (
            _class_streaming_auc(logits, labels, weights, class_id,
                                 self.logits_dimension))

    return metrics


def _to_labels_tensor(labels, label_name):
  """Returns label as a tensor.

  Args:
    labels: Label `Tensor` or `SparseTensor` or a dict containig labels.
    label_name: Label name if labels is a dict.

  Returns:
    Label `Tensor` or `SparseTensor`.
  """
  labels = labels[label_name] if isinstance(labels, dict) else labels
  return framework_lib.convert_to_tensor_or_sparse_tensor(labels)


def _check_no_sparse_tensor(x):
  """Raises ValueError if the given tensor is `SparseTensor`."""
  if isinstance(x, sparse_tensor.SparseTensor):
    raise ValueError("SparseTensor is not supported.")


def _sparse_labels_to_indicator(labels, num_classes):
  """If labels is `SparseTensor`, converts it to indicator `Tensor`.

  Args:
    labels: Label `Tensor` or `SparseTensor`.
    num_classes: Number of classes.

  Returns:
    Dense label `Tensor`.

  Raises:
    ValueError: If labels is `SparseTensot` and `num_classes` < 2.
  """
  if isinstance(labels, sparse_tensor.SparseTensor):
    if num_classes < 2:
      raise ValueError("Must set num_classes >= 2 when passing labels as a "
                       "SparseTensor.")
    return math_ops.to_int64(
        sparse_ops.sparse_to_indicator(labels, num_classes))
  return labels


def _assert_labels_rank(labels):
  return control_flow_ops.Assert(
      math_ops.less_equal(array_ops.rank(labels), 2),
      ("labels shape should be either [batch_size, 1] or [batch_size]",))


class _BinarySvmHead(_SingleHead):
  """_Head for binary classification using SVMs."""

  def __init__(self, label_name, weight_column_name, enable_centered_bias,
               head_name, thresholds):

    def _loss_fn(logits, labels):
      with ops.name_scope(None, "hinge_loss", (logits, labels)) as name:
        with ops.control_dependencies((_assert_labels_rank(labels),)):
          labels = array_ops.reshape(labels, shape=(-1, 1))
        return losses_lib.hinge_loss(logits, labels, scope=name)

    super(_BinarySvmHead, self).__init__(
        problem_type=constants.ProblemType.LOGISTIC_REGRESSION,
        logits_dimension=1,
        label_name=label_name,
        weight_column_name=weight_column_name,
        head_name=head_name)
    self._thresholds = thresholds if thresholds else (.5,)
    self._loss_fn = _loss_fn
    self._enable_centered_bias = enable_centered_bias

  def create_model_fn_ops(self,
                          features,
                          mode,
                          labels=None,
                          train_op_fn=None,
                          logits=None,
                          logits_input=None,
                          scope=None):
    """See `_Head`."""
    return _create_model_fn_ops(
        features=features,
        mode=mode,
        transform_labels_fn=self._transform_labels,
        loss_fn=self._loss_fn,
        logits_to_predictions_fn=self._logits_to_predictions,
        metrics_fn=self._metrics,
        create_output_alternatives_fn=self._create_output_alternatives,
        default_variable_scope_name="binary_svm_head",
        labels=labels,
        train_op_fn=train_op_fn,
        logits=logits,
        logits_input=logits_input,
        logits_dimension=self.logits_dimension,
        head_name=self.head_name,
        weight_column_name=self.weight_column_name,
        enable_centered_bias=self._enable_centered_bias)

  def _transform_labels(self, labels):
    """Applies transformations to labels tensor."""
    labels_tensor = _to_labels_tensor(labels, self._label_name)
    _check_no_sparse_tensor(labels_tensor)
    return labels_tensor

  def _logits_to_predictions(self, logits):
    """See `_MultiClassHead`."""
    with ops.name_scope(None, "predictions", (logits,)):
      return {
          prediction_key.PredictionKey.LOGITS:
              logits,
          prediction_key.PredictionKey.CLASSES:
              math_ops.argmax(
                  _one_class_to_two_class_logits(logits),
                  1,
                  name=prediction_key.PredictionKey.CLASSES)
      }

  def _metrics(self, eval_loss, predictions, labels, weights):
    """See `_MultiClassHead`."""
    with ops.name_scope("metrics", values=(
        [eval_loss, labels, weights] + list(six.itervalues(predictions)))):
      metrics = {_summary_key(self.head_name, mkey.LOSS):
                 metrics_lib.streaming_mean(eval_loss)}

      # TODO(b/29366811): This currently results in both an "accuracy" and an
      # "accuracy/threshold_0.500000_mean" metric for binary classification.
      classes = predictions[prediction_key.PredictionKey.CLASSES]
      metrics[_summary_key(self.head_name, mkey.ACCURACY)] = (
          metrics_lib.streaming_accuracy(classes, labels, weights))
      # TODO(sibyl-vie3Poto): add more metrics relevant for svms.

    return metrics


class _MultiLabelHead(_SingleHead):
  """_Head for multlabel classification."""

  # TODO(zakaria): add signature and metric for multilabel.
  def __init__(self,
               n_classes,
               label_name,
               weight_column_name,
               enable_centered_bias,
               head_name,
               thresholds,
               metric_class_ids=None):

    super(_MultiLabelHead, self).__init__(
        problem_type=constants.ProblemType.CLASSIFICATION,
        logits_dimension=n_classes,
        label_name=label_name,
        weight_column_name=weight_column_name,
        head_name=head_name)

    self._thresholds = thresholds if thresholds else (.5,)
    self._loss_fn = _sigmoid_cross_entropy_loss
    self._enable_centered_bias = enable_centered_bias
    self._metric_class_ids = tuple([] if metric_class_ids is None else
                                   metric_class_ids)
    for class_id in self._metric_class_ids:
      if (class_id < 0) or (class_id >= n_classes):
        raise ValueError("Class ID %s not in [0, %s)." % (class_id, n_classes))

  def create_model_fn_ops(self,
                          features,
                          mode,
                          labels=None,
                          train_op_fn=None,
                          logits=None,
                          logits_input=None,
                          scope=None):
    """See `_Head`."""
    return _create_model_fn_ops(
        features=features,
        mode=mode,
        transform_labels_fn=self._transform_labels,
        loss_fn=self._loss_fn,
        logits_to_predictions_fn=self._logits_to_predictions,
        metrics_fn=self._metrics,
        create_output_alternatives_fn=self._create_output_alternatives,
        default_variable_scope_name="multi_label_head",
        labels=labels,
        train_op_fn=train_op_fn,
        logits=logits,
        logits_input=logits_input,
        logits_dimension=self.logits_dimension,
        head_name=self.head_name,
        weight_column_name=self.weight_column_name,
        enable_centered_bias=self._enable_centered_bias)

  def _transform_labels(self, labels):
    """Applies transformations to labels tensor."""
    labels_tensor = _to_labels_tensor(labels, self._label_name)
    labels_tensor = _sparse_labels_to_indicator(labels_tensor,
                                                self._logits_dimension)
    return labels_tensor

  def _logits_to_predictions(self, logits):
    """See `_MultiClassHead`."""
    with ops.name_scope(None, "predictions", (logits,)):
      return {
          prediction_key.PredictionKey.LOGITS:
              logits,
          prediction_key.PredictionKey.PROBABILITIES:
              math_ops.sigmoid(
                  logits, name=prediction_key.PredictionKey.PROBABILITIES),
          prediction_key.PredictionKey.CLASSES:
              math_ops.to_int64(
                  math_ops.greater(logits, 0),
                  name=prediction_key.PredictionKey.CLASSES)
      }

  def _metrics(self, eval_loss, predictions, labels, weights):
    """Returns a dict of metrics keyed by name."""
    with ops.name_scope("metrics", values=(
        [eval_loss, labels, weights] + list(six.itervalues(predictions)))):
      classes = predictions[prediction_key.PredictionKey.CLASSES]
      probabilities = predictions[prediction_key.PredictionKey.PROBABILITIES]
      logits = predictions[prediction_key.PredictionKey.LOGITS]

      metrics = {_summary_key(self.head_name, mkey.LOSS):
                 metrics_lib.streaming_mean(eval_loss)}
      # TODO(b/29366811): This currently results in both an "accuracy" and an
      # "accuracy/threshold_0.500000_mean" metric for binary classification.
      metrics[_summary_key(self.head_name, mkey.ACCURACY)] = (
          metrics_lib.streaming_accuracy(classes, labels, weights))
      metrics[_summary_key(self.head_name, mkey.AUC)] = _streaming_auc(
          probabilities, labels, weights)

      for class_id in self._metric_class_ids:
        # TODO(ptucker): Add per-class accuracy, precision, recall.
        metrics[_summary_key(
            self.head_name, mkey.CLASS_PREDICTION_MEAN % class_id)] = (
                _predictions_streaming_mean(classes, weights, class_id))
        metrics[_summary_key(
            self.head_name, mkey.CLASS_LABEL_MEAN % class_id)] = (
                _indicator_labels_streaming_mean(labels, weights, class_id))
        metrics[_summary_key(
            self.head_name, mkey.CLASS_PROBABILITY_MEAN % class_id)] = (
                _predictions_streaming_mean(probabilities, weights, class_id))
        metrics[_summary_key(
            self.head_name, mkey.CLASS_LOGITS_MEAN % class_id)] = (
                _predictions_streaming_mean(logits, weights, class_id))
        metrics[_summary_key(self.head_name, mkey.CLASS_AUC % class_id)] = (
            _streaming_auc(logits, labels, weights, class_id))

    return metrics


def _noop(unused_loss):
  return control_flow_ops.no_op()


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
    self._logits_dimension = 0
    for head in heads:
      # TODO(ptucker): Change this, and add head_name to MultiHead, to support
      # nested MultiHeads.
      if not isinstance(head, _SingleHead):
        raise ValueError("Members of MultiHead must be SingleHead.")
      if not head.head_name:
        raise ValueError("Members of MultiHead must have names.")
      self._logits_dimension += head.logits_dimension

    self._heads = heads
    self._loss_combiner = loss_combiner

  @property
  def logits_dimension(self):
    return self._logits_dimension

  def create_model_fn_ops(self,
                          features,
                          mode,
                          labels=None,
                          train_op_fn=None,
                          logits=None,
                          logits_input=None,
                          scope=None):
    """See `_Head.create_model_fn_ops`.

    Args:
      features: Input `dict` of `Tensor` objects.
      mode: Estimator's `ModeKeys`.
      labels: Labels `Tensor`, or `dict` of same.
      train_op_fn: Function that takes a scalar loss and returns an op to
          optimize with the loss.
      logits: Concatenated logits of (x, 1) shape where x is the sum of
          `logits_dimension` of all the heads, i.e., same as `logits_dimension`
          of this class. This function will split the logits tensor and pass
          logits of proper size to each head.
      logits_input: tensor to build logits from.
      scope: Optional scope for variable_scope. If provided, will be passed to
        all heads. Most users will want to set this to `None`, so each head
        constructs a separate variable_scope according to its `head_name`.

    Returns:
      `ModelFnOps`.

    Raises:
      ValueError: if `mode` is not recognized, or neither or both of `logits`
          and `logits_input` is provided.
    """
    _check_mode_valid(mode)
    all_model_fn_ops = []
    if logits is None:
      # Use logits_input.
      for head in self._heads:
        # TODO(ptucker): Do we need to let each head create its own logits?
        all_model_fn_ops.append(
            head.create_model_fn_ops(
                features=features,
                mode=mode,
                labels=labels,
                train_op_fn=_noop,
                logits_input=logits_input,
                scope=scope))
    else:
      # Split logits for each head.
      for head, head_logits in zip(self._heads, self._split_logits(logits)):
        all_model_fn_ops.append(
            head.create_model_fn_ops(
                features=features,
                mode=mode,
                labels=labels,
                train_op_fn=_noop,
                logits=head_logits,
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
      train_op_fn: Function to create train op. See `create_model_fn_ops`
          documentaion for more details.

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
    return model_fn.ModelFnOps(
        mode=model_fn.ModeKeys.TRAIN,
        loss=loss,
        train_op=train_op)

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

    return model_fn.ModelFnOps(
        mode=model_fn.ModeKeys.INFER,
        predictions=predictions,
        output_alternatives=output_alternatives)

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

    return model_fn.ModelFnOps(
        mode=model_fn.ModeKeys.EVAL,
        predictions=predictions,
        loss=loss,
        eval_metric_ops=metrics)


def _weighted_loss(loss, weight):
  """Returns cumulative weighted loss as 1d `Tensor`."""
  with ops.name_scope(None, "weighted_loss", (loss, weight)) as name:
    return math_ops.multiply(
        array_ops.reshape(loss, shape=(-1,)),
        array_ops.reshape(weight, shape=(-1,)),
        name=name)


def _weight_tensor(features, weight_column_name):
  """Returns weights as 1d `Tensor`."""
  if not weight_column_name:
    return None
  with ops.name_scope(None, "weight_tensor",
                      tuple(six.itervalues(features))):
    return math_ops.to_float(features[weight_column_name])


def _loss(loss_unweighted, weight, name="loss"):
  """Returns a tuple of (loss, weighted_average_loss).

  loss is used for gradient descent while weighted_average_loss is used for
  summaries to be backward compatible.

  loss is different from the loss reported on the tensorboard as we
  should respect the example weights when computing the gradient.

    L = sum_{i} w_{i} * l_{i} / B

  where B is the number of examples in the batch, l_{i}, w_{i} are individual
  losses, and example weight.

  Args:
    loss_unweighted: Unweighted loss
    weight: Weight tensor
    name: Optional name

  Returns:
    A tuple of (loss, weighted_average_loss)
  """
  with ops.name_scope(name, values=(loss_unweighted, weight)) as name_scope:
    if weight is None:
      loss = math_ops.reduce_mean(loss_unweighted, name=name_scope)
      return loss, loss
    loss_weighted = _weighted_loss(loss_unweighted, weight)
    # TODO(ptucker): This might be wrong if weights are broadcast to loss shape.
    # We should use tf.losses here.
    weighted_average_loss = math_ops.div(
        math_ops.reduce_sum(loss_weighted),
        math_ops.to_float(math_ops.reduce_sum(weight)),
        name="weighted_average_loss")
    loss = math_ops.reduce_mean(loss_weighted, name=name_scope)
    return loss, weighted_average_loss


def _check_mode_valid(mode):
  """Raises ValueError if the given mode is invalid."""
  if (mode != model_fn.ModeKeys.TRAIN and mode != model_fn.ModeKeys.INFER and
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
  # Do not create a variable with variable_scope.get_variable, because that may
  # create a PartitionedVariable, which does not support indexing, so
  # summary.scalar will not work.
  centered_bias = variables.Variable(
      name="centered_bias_weight",
      initial_value=array_ops.zeros(shape=(logits_dimension,)),
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


def _train_op(loss,
              labels,
              train_op_fn,
              centered_bias=None,
              logits_dimension=None,
              loss_fn=None):
  """Returns op for the training step."""
  if centered_bias is not None:
    centered_bias_step = _centered_bias_step(centered_bias, logits_dimension,
                                             labels, loss_fn)
  else:
    centered_bias_step = None
  with ops.name_scope(None, "train_op", (loss, labels)):
    train_op = train_op_fn(loss)
    if centered_bias_step is not None:
      train_op = control_flow_ops.group(train_op, centered_bias_step)
    return train_op


def _sigmoid_cross_entropy_loss(logits, labels):
  with ops.name_scope(None, "sigmoid_cross_entropy_loss",
                      (logits, labels)) as name:
    # sigmoid_cross_entropy_with_logits requires [batch_size, n_classes] labels.
    return nn.sigmoid_cross_entropy_with_logits(
        labels=math_ops.to_float(labels), logits=logits, name=name)


def _float_weights_or_none(weights):
  if weights is None:
    return None
  with ops.name_scope(None, "float_weights", (weights,)) as name:
    return math_ops.to_float(weights, name=name)


def _indicator_labels_streaming_mean(labels, weights=None, class_id=None):
  labels = ops.convert_to_tensor(labels)
  if class_id is not None:
    labels = labels[:, class_id]
  return metrics_lib.streaming_mean(labels, weights=weights)


def _predictions_streaming_mean(predictions,
                                weights=None,
                                class_id=None):
  predictions = ops.convert_to_tensor(predictions)
  if weights is not None:
    weights = ops.convert_to_tensor(weights)

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


def _class_predictions_streaming_mean(predictions, weights, class_id):
  return metrics_lib.streaming_mean(
      array_ops.where(
          math_ops.equal(
              math_ops.to_int32(class_id), math_ops.to_int32(predictions)),
          array_ops.ones_like(predictions),
          array_ops.zeros_like(predictions)),
      weights=weights)


def _class_labels_streaming_mean(labels, weights, class_id):
  return metrics_lib.streaming_mean(
      array_ops.where(
          math_ops.equal(
              math_ops.to_int32(class_id), math_ops.to_int32(labels)),
          array_ops.ones_like(labels), array_ops.zeros_like(labels)),
      weights=weights)


def _class_streaming_auc(predictions, labels, weights, class_id,
                         num_classes):
  indicator_labels = _class_id_labels_to_indicator(
      labels, num_classes=num_classes)
  return _streaming_auc(predictions, indicator_labels, weights, class_id)


def _streaming_auc_with_class_id_label(predictions, labels, weights,
                                       num_classes):
  indicator_labels = _class_id_labels_to_indicator(
      labels, num_classes=num_classes)
  return _streaming_auc(predictions, indicator_labels, weights)


def _streaming_auc(predictions, labels, weights=None, class_id=None):
  predictions = ops.convert_to_tensor(predictions)
  labels = ops.convert_to_tensor(labels)
  if class_id is not None:
    predictions = predictions[:, class_id]
    labels = labels[:, class_id]
  return metrics_lib.streaming_auc(
      predictions,
      math_ops.cast(labels, dtypes.bool),
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


def _streaming_accuracy_at_threshold(predictions, labels, weights, threshold):
  threshold_predictions = math_ops.to_float(
      math_ops.greater_equal(predictions, threshold))
  return metrics_lib.streaming_accuracy(
      predictions=threshold_predictions, labels=labels, weights=weights)


def _streaming_precision_at_threshold(predictions, labels, weights, threshold):
  precision_tensor, update_op = metrics_lib.streaming_precision_at_thresholds(
      predictions, labels=labels, thresholds=(threshold,),
      weights=_float_weights_or_none(weights))
  return array_ops.squeeze(precision_tensor), array_ops.squeeze(update_op)


def _streaming_recall_at_threshold(predictions, labels, weights, threshold):
  precision_tensor, update_op = metrics_lib.streaming_recall_at_thresholds(
      predictions, labels=labels, thresholds=(threshold,),
      weights=_float_weights_or_none(weights))
  return array_ops.squeeze(precision_tensor), array_ops.squeeze(update_op)
