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

from tensorflow.contrib import losses
from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.training import training


# TODO(zakaria): add functions that creates a head and returns ModelOpFn


def _regression_head(label_name=None,
                     weight_column_name=None,
                     target_dimension=1,
                     enable_centered_bias=False, head_name=None):
  """Creates a _Head for linear regression.

  Args:
    label_name: String, name of the key in label dict. Can be null if label
        is a tensor (single headed models).
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    target_dimension: dimension of the target for multilabels.
    enable_centered_bias: A bool. If True, estimator will learn a centered
      bias variable for each class. Rest of the model structure learns the
      residual after centered bias.
    head_name: name of the head. If provided, predictions, summary and metrics
      keys will be prefixed by the head_name and an underscore.

  Returns:
    An instance of _Head
  """
  return _RegressionHead(train_loss_fn=_mean_squared_loss,
                         eval_loss_fn=_mean_squared_loss,
                         label_name=label_name,
                         weight_column_name=weight_column_name,
                         target_dimension=target_dimension,
                         enable_centered_bias=enable_centered_bias,
                         head_name=head_name)

# TODO(zakaria): Add logistic_regression_head


def _multi_class_head(n_classes, label_name=None, weight_column_name=None,
                      enable_centered_bias=False, head_name=None,
                      thresholds=None):
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

  Returns:
    An instance of _MultiClassHead.

  Raises:
    ValueError: if n_classes is < 2
  """
  if n_classes < 2:
    raise ValueError("n_classes must be > 1 for classification.")

  if n_classes == 2:
    loss_fn = _log_loss_with_two_classes
  else:
    loss_fn = _softmax_cross_entropy_loss
  return _MultiClassHead(train_loss_fn=loss_fn,
                         eval_loss_fn=loss_fn,
                         n_classes=n_classes,
                         label_name=label_name,
                         weight_column_name=weight_column_name,
                         enable_centered_bias=enable_centered_bias,
                         head_name=head_name,
                         thresholds=thresholds)


def _binary_svm_head(label_name=None, weight_column_name=None,
                     enable_centered_bias=False, head_name=None,
                     thresholds=None,):
  """Creates a _TargetColumn for binary classification with SVMs.

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
    An instance of _TargetColumn.

  """
  return _BinarySvmHead(label_name=label_name,
                        weight_column_name=weight_column_name,
                        enable_centered_bias=enable_centered_bias,
                        head_name=head_name,
                        thresholds=thresholds)


def _multi_label_head(n_classes, label_name=None, weight_column_name=None,
                      enable_centered_bias=False, head_name=None,
                      thresholds=None):
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
                         thresholds=thresholds)


# TODO(zakaria): Make the classes public once we are ready for users to subclass
#   them.
class _Head(object):
  """Interface for the head/top of a model.

  Given logits or output of a hidden layer, a Head knows how to compute
  predictions, loss, default metric and export signature.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def logits_dimension(self):
    raise NotImplementedError("Calling an abstract method.")

  def head_ops(self, features, target, mode, train_op_fn, logits=None,
               logits_input=None):
    """Returns ops for a model_fn.

    Args:
      features: input dict.
      target: target dict or tensor.
      mode: estimator's ModeKeys
      train_op_fn: function that takes a scalar loss and returns an op to
          optimize with the loss.
      logits: logits to be used for the head.
      logits_input: tensor to build logits from.

    Returns:
      `estimator.ModelFnOps`

    Raises:
      ValueError: if mode is not recognized.
    """
    _check_logits_input_not_supported(logits, logits_input)
    if mode == estimator.ModeKeys.TRAIN:
      loss, additional_train_op = self._training_loss(features, target,
                                                      logits, logits_input)

      train_op = train_op_fn(loss)

      if additional_train_op:
        if train_op:
          train_op = control_flow_ops.group(train_op, *additional_train_op)
        else:
          train_op = control_flow_ops.group(*additional_train_op)

      return estimator.ModelFnOps(
          mode=estimator.ModeKeys.TRAIN,
          loss=loss,
          training_op=train_op,
          default_metrics=self._default_metric(),
          signature_fn=self._create_signature_fn())

    if mode == estimator.ModeKeys.INFER:
      return estimator.ModelFnOps(
          mode=estimator.ModeKeys.INFER,
          predictions=self._infer_op(logits, logits_input),
          default_metrics=self._default_metric(),
          signature_fn=self._create_signature_fn())

    if mode == estimator.ModeKeys.EVAL:
      predictions, loss = self._eval_op(features, target, logits, logits_input)
      return estimator.ModelFnOps(
          mode=estimator.ModeKeys.EVAL,
          predictions=predictions,
          loss=loss,
          default_metrics=self._default_metric(),
          signature_fn=self._create_signature_fn())

    raise ValueError("mode=%s unrecognized." % str(mode))

  @abc.abstractmethod
  def _training_loss(self, features, target, logits=None, logits_input=None,
                     name="training_loss"):
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def _infer_op(self, logits=None, logits_input=None, name="infer_op"):
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def _eval_op(self, features, target, logits=None, logits_input=None,
               name="eval_op"):
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def _default_metric(self):
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def _create_signature_fn(self):
    """Creates signature function for the Head.
    """
    raise NotImplementedError("Calling an abstract method.")


class _RegressionHead(_Head):
  """_Head for regression."""

  def __init__(self, train_loss_fn, eval_loss_fn, label_name,
               weight_column_name, target_dimension, enable_centered_bias,
               head_name):
    """Base type for all single heads.

    Args:
      train_loss_fn: loss_fn for training.
      eval_loss_fn: loss_fn for eval.
      label_name: String, name of the key in label dict. Can be null if label
          is a tensor (single headed models).
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      target_dimension: Integer, number of label columns.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      head_name: name of the head. If provided, predictions, summary and metrics
        keys will be prefixed by the head_name and an underscore.
    """
    self._train_loss_fn = train_loss_fn
    self._eval_loss_fn = eval_loss_fn
    self._logits_dimension = target_dimension
    self._label_name = label_name
    self._weight_column_name = weight_column_name
    self._head_name = head_name
    self._enable_centered_bias = enable_centered_bias
    self._centered_bias_weight_collection = _head_prefixed(head_name,
                                                           "centered_bias")

  @property
  def logits_dimension(self):
    return self._logits_dimension

  def _training_loss(self, features, target, logits=None,
                     logits_input=None, name="training_loss"):
    """Returns training loss tensor for this head.

    Training loss is different from the loss reported on the tensorboard as we
    should respect the example weights when computing the gradient.

      L = sum_{i} w_{i} * l_{i} / B

    where B is the number of examples in the batch, l_{i}, w_{i} are individual
    losses, and example weight.

    Args:
      features: features dict.
      target: either a tensor for labels or in multihead case, a dict of string
        to target tensor.
      logits: logits, a float tensor.
      logits_input: Output of last hidden layer.
      name: Op name.

    Returns:
      A tuple of training Loss and additional_train_op (possibly None)
    """
    target = _check_target(target, self._label_name)

    centered_bias_step = None
    if self._enable_centered_bias:
      logits = nn.bias_add(logits, _centered_bias(
          self.logits_dimension,
          self._centered_bias_weight_collection))
      centered_bias_step = [_centered_bias_step(
          self.logits_dimension,
          self._centered_bias_weight_collection,
          target,
          self._train_loss_fn)]

    loss_unweighted = self._train_loss_fn(logits, target)
    loss, weighted_average_loss = _loss(
        loss_unweighted,
        _weight_tensor(features, self._weight_column_name),
        name=name)
    logging_ops.scalar_summary(_head_prefixed(self._head_name, "loss"),
                               weighted_average_loss)
    return loss, centered_bias_step

  def _eval_op(self, features, target, logits=None, logits_input=None,
               name="eval_op"):
    target = _check_target(target, self._label_name)
    if self._enable_centered_bias:
      logits = nn.bias_add(logits, _centered_bias(
          self.logits_dimension,
          self._centered_bias_weight_collection))
    loss_unweighted = self._eval_loss_fn(logits, target)
    loss, _ = _loss(loss_unweighted,
                    _weight_tensor(features, self._weight_column_name),
                    name=name)

    predictions = self._logits_to_prediction(logits)

    return predictions, loss

  def _infer_op(self, logits=None, logits_input=None):
    if self._enable_centered_bias:
      logits = nn.bias_add(logits, _centered_bias(
          self.logits_dimension,
          self._centered_bias_weight_collection))
    return self._logits_to_prediction(logits)

  def _logits_to_prediction(self, logits=None):
    predictions = {}
    if self.logits_dimension == 1:
      predictions[PredictionKey.SCORES] = array_ops.squeeze(
          logits, squeeze_dims=[1])
    else:
      predictions[PredictionKey.SCORES] = logits
    return predictions

  # pylint: disable=undefined-variable
  def _create_signature_fn(self):
    def _regression_signature_fn(examples, unused_features, predictions):
      if isinstance(predictions, dict):
        score = predictions[PredictionKey.SCORES]
      else:
        score = predictions

      default_signature = exporter.regression_signature(
          input_tensor=examples, output_tensor=score)
      # TODO(zakaria): add validation
      return default_signature, {}
    return _regression_signature_fn

  def _default_metric(self):
    return {_head_prefixed(self._head_name, MetricKey.LOSS):
            _weighted_average_loss_metric_spec(self._eval_loss_fn,
                                               PredictionKey.SCORES,
                                               self._label_name,
                                               self._weight_column_name)}


class _MultiClassHead(_Head):
  """_Head for classification."""

  def __init__(self, train_loss_fn, eval_loss_fn, n_classes, label_name,
               weight_column_name, enable_centered_bias, head_name,
               thresholds=None):
    """Base type for all single heads.

    Args:
      train_loss_fn: loss_fn for training.
      eval_loss_fn: loss_fn for eval.
      n_classes: number of classes.
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
      thresholds: thresholds for eval.

    Raises:
      ValueError: if n_classes is invalid.
    """
    if n_classes < 2:
      raise ValueError("n_classes must be >= 2")
    self._thresholds = thresholds if thresholds else [.5]

    self._train_loss_fn = train_loss_fn
    self._eval_loss_fn = eval_loss_fn
    self._logits_dimension = 1 if n_classes == 2 else n_classes
    self._label_name = label_name
    self._weight_column_name = weight_column_name
    self._head_name = head_name
    self._enable_centered_bias = enable_centered_bias
    self._centered_bias_weight_collection = _head_prefixed(head_name,
                                                           "centered_bias")

  @property
  def logits_dimension(self):
    return self._logits_dimension

  def _training_loss(self, features, target, logits=None,
                     logits_input=None, name="training_loss"):
    """Returns training loss tensor for this head.

    Training loss is different from the loss reported on the tensorboard as we
    should respect the example weights when computing the gradient.

      L = sum_{i} w_{i} * l_{i} / B

    where B is the number of examples in the batch, l_{i}, w_{i} are individual
    losses, and example weight.

    Args:
      features: features dict.
      target: either a tensor for labels or in multihead case, a dict of string
        to target tensor.
      logits: logits, a float tensor.
      logits_input: Output of last hidden layer.
      name: Op name.

    Returns:
      A tuple of training Loss and additional_train_op (possibly None)
    """
    target = _check_target(target, self._label_name)

    centered_bias_step = None
    if self._enable_centered_bias:
      logits = nn.bias_add(logits, _centered_bias(
          self.logits_dimension,
          self._centered_bias_weight_collection))
      centered_bias_step = [_centered_bias_step(
          self.logits_dimension,
          self._centered_bias_weight_collection,
          target,
          self._train_loss_fn)]

    loss_unweighted = self._train_loss_fn(logits, target)
    loss, weighted_average_loss = _loss(
        loss_unweighted,
        _weight_tensor(features, self._weight_column_name),
        name=name)
    logging_ops.scalar_summary(_head_prefixed(self._head_name, "loss"),
                               weighted_average_loss)
    return loss, centered_bias_step

  def _eval_op(self, features, target, logits=None, logits_input=None,
               name="eval_op"):
    target = _check_target(target, self._label_name)
    if self._enable_centered_bias:
      logits = nn.bias_add(logits, _centered_bias(
          self.logits_dimension,
          self._centered_bias_weight_collection))
    loss_unweighted = self._eval_loss_fn(logits, target)
    loss, _ = _loss(loss_unweighted,
                    _weight_tensor(features, self._weight_column_name),
                    name=name)

    predictions = self._logits_to_prediction(logits)

    return predictions, loss

  def _infer_op(self, logits=None, logits_input=None):
    if self._enable_centered_bias:
      logits = nn.bias_add(logits, _centered_bias(
          self.logits_dimension,
          self._centered_bias_weight_collection))
    return self._logits_to_prediction(logits)

  def _logits_to_prediction(self, logits=None):
    predictions = {PredictionKey.LOGITS: logits}
    if self.logits_dimension == 1:
      predictions[PredictionKey.LOGISTIC] = math_ops.sigmoid(logits)
      logits = array_ops.concat(1, [array_ops.zeros_like(logits), logits])
    predictions[PredictionKey.PROBABILITIES] = nn.softmax(logits)
    predictions[PredictionKey.CLASSES] = math_ops.argmax(logits, 1)

    return predictions

  def _create_signature_fn(self):
    """See superclass."""
    def _classification_signature_fn(examples, unused_features, predictions):
      """Servo signature function."""
      if isinstance(predictions, dict):
        default_signature = exporter.classification_signature(
            input_tensor=examples,
            classes_tensor=predictions[PredictionKey.CLASSES],
            scores_tensor=predictions[PredictionKey.PROBABILITIES])
      else:
        default_signature = exporter.classification_signature(
            input_tensor=examples,
            scores_tensor=predictions)

      # TODO(zakaria): add validation
      return default_signature, {}
    return _classification_signature_fn

  def _default_metric(self):
    metrics = {_head_prefixed(self._head_name, MetricKey.LOSS):
               _weighted_average_loss_metric_spec(self._eval_loss_fn,
                                                  PredictionKey.LOGITS,
                                                  self._label_name,
                                                  self._weight_column_name)}

    # TODO(b/29366811): This currently results in both an "accuracy" and an
    # "accuracy/threshold_0.500000_mean" metric for binary classification.
    metrics[_head_prefixed(self._head_name, MetricKey.ACCURACY)] = (
        metric_spec.MetricSpec(metrics_lib.streaming_accuracy,
                               PredictionKey.CLASSES, self._label_name,
                               self._weight_column_name))
    if self.logits_dimension == 1:
      def _add_binary_metric(metric_key, metric_fn):
        metrics[_head_prefixed(self._head_name, metric_key)] = (
            metric_spec.MetricSpec(metric_fn,
                                   PredictionKey.LOGISTIC,
                                   self._label_name,
                                   self._weight_column_name))
      _add_binary_metric(MetricKey.PREDICTION_MEAN, _predictions_streaming_mean)
      _add_binary_metric(MetricKey.TARGET_MEAN, _target_streaming_mean)

      # Also include the streaming mean of the label as an accuracy baseline, as
      # a reminder to users.
      _add_binary_metric(MetricKey.ACCURACY_BASELINE, _target_streaming_mean)

      _add_binary_metric(MetricKey.AUC, _streaming_auc)

      for threshold in self._thresholds:
        _add_binary_metric(MetricKey.ACCURACY_MEAN % threshold,
                           _accuracy_at_threshold(threshold))
        # Precision for positive examples.
        _add_binary_metric(MetricKey.PRECISION_MEAN % threshold,
                           _streaming_at_threshold(
                               metrics_lib.streaming_precision_at_thresholds,
                               threshold),)
        # Recall for positive examples.
        _add_binary_metric(MetricKey.RECALL_MEAN % threshold,
                           _streaming_at_threshold(
                               metrics_lib.streaming_recall_at_thresholds,
                               threshold))
    return metrics


def _check_target(target, label_name):
  target = target[label_name] if isinstance(target, dict) else target
  if isinstance(target, ops.SparseTensor):
    raise ValueError("SparseTensor is not supported as a target/label.")
  return target


class _BinarySvmHead(_MultiClassHead):
  """_Head for binary classification using SVMs."""

  def __init__(self, label_name, weight_column_name, enable_centered_bias,
               head_name, thresholds):
    def loss_fn(logits, target):
      check_shape_op = control_flow_ops.Assert(
          math_ops.less_equal(array_ops.rank(target), 2),
          ["target's shape should be either [batch_size, 1] or [batch_size]"])
      with ops.control_dependencies([check_shape_op]):
        target = array_ops.reshape(
            target, shape=[array_ops.shape(target)[0], 1])
      return losses.hinge_loss(logits, target)

    super(_BinarySvmHead, self).__init__(
        train_loss_fn=loss_fn,
        eval_loss_fn=loss_fn,
        n_classes=2,
        label_name=label_name,
        weight_column_name=weight_column_name,
        enable_centered_bias=enable_centered_bias,
        head_name=head_name,
        thresholds=thresholds)

  def _logits_to_prediction(self, logits=None):
    predictions = {}
    predictions[PredictionKey.LOGITS] = logits
    logits = array_ops.concat(1, [array_ops.zeros_like(logits), logits])
    predictions[PredictionKey.CLASSES] = math_ops.argmax(logits, 1)

    return predictions

  def _default_metric(self):
    metrics = {_head_prefixed(self._head_name, MetricKey.LOSS):
               _weighted_average_loss_metric_spec(self._eval_loss_fn,
                                                  PredictionKey.LOGITS,
                                                  self._label_name,
                                                  self._weight_column_name)}
    metrics[_head_prefixed(self._head_name, MetricKey.ACCURACY)] = (
        metric_spec.MetricSpec(metrics_lib.streaming_accuracy,
                               PredictionKey.CLASSES, self._label_name,
                               self._weight_column_name))
    # TODO(sibyl-vie3Poto): add more metrics relevant for svms.
    return metrics


class _MultiLabelHead(_MultiClassHead):
  """_Head for multlabel classification."""

  # TODO(zakaria): add signature and metric for multilabel.
  def __init__(self, n_classes, label_name,
               weight_column_name, enable_centered_bias, head_name,
               thresholds):

    super(_MultiLabelHead, self).__init__(
        train_loss_fn=_sigmoid_cross_entropy_loss,
        eval_loss_fn=_sigmoid_cross_entropy_loss,
        n_classes=n_classes,
        label_name=label_name,
        weight_column_name=weight_column_name,
        enable_centered_bias=enable_centered_bias,
        head_name=head_name,
        thresholds=thresholds)

  def _logits_to_prediction(self, logits=None):
    predictions = {PredictionKey.LOGITS: logits}
    if self.logits_dimension == 1:
      predictions[PredictionKey.LOGISTIC] = math_ops.sigmoid(logits)
      logits = array_ops.concat(1, [array_ops.zeros_like(logits), logits])
    predictions[PredictionKey.PROBABILITIES] = math_ops.sigmoid(logits)
    predictions[PredictionKey.CLASSES] = math_ops.to_int64(
        math_ops.greater(logits, 0))
    return predictions


def _weighted_loss(loss, weight):
  """Returns cumulative weighted loss."""
  unweighted_loss = array_ops.reshape(loss, shape=(-1,))
  weighted_loss = math_ops.mul(unweighted_loss,
                               array_ops.reshape(
                                   weight, shape=(-1,)))
  return weighted_loss


def _weight_tensor(features, weight_column_name):
  if not weight_column_name:
    return None
  else:
    return array_ops.reshape(
        math_ops.to_float(features[weight_column_name]),
        shape=(-1,))


def _loss(loss_unweighted, weight, name):
  """Returns loss."""
  if weight is None:
    loss = math_ops.reduce_mean(loss_unweighted, name=name)
    return loss, loss
  else:
    loss_weighted = _weighted_loss(loss_unweighted, weight)
    weighted_average_loss = math_ops.div(
        math_ops.reduce_sum(loss_weighted),
        math_ops.to_float(math_ops.reduce_sum(weight)),
        name="weighted_average_loss")
    loss = math_ops.reduce_mean(loss_weighted, name=name)
    return loss, weighted_average_loss


def _check_logits_input_not_supported(logits, logits_input):
  if logits_input is not None or logits is None:
    raise NotImplementedError("logits_input is not supported yet, "
                              "must pass logits")


def _centered_bias(logits_dimension, weight_collection):
  """Creates and returns centered bias."""
  centered_bias = variables.Variable(
      array_ops.zeros([logits_dimension]),
      collections=[weight_collection, ops.GraphKeys.VARIABLES],
      name="centered_bias_weight")
  logging_ops.scalar_summary(
      ["centered_bias_%d" % cb for cb in range(logits_dimension)],
      array_ops.reshape(centered_bias, [-1]))
  return centered_bias


def _centered_bias_step(logits_dimension, weight_collection, target,
                        train_loss_fn):
  """Creates and returns training op for centered bias."""
  centered_bias = ops.get_collection(weight_collection)
  batch_size = array_ops.shape(target)[0]
  logits = array_ops.reshape(
      array_ops.tile(centered_bias[0], [batch_size]),
      [batch_size, logits_dimension])
  with ops.name_scope(None, "centered_bias", (target, logits)):
    centered_bias_loss = math_ops.reduce_mean(
        train_loss_fn(logits, target), name="training_loss")
  # Learn central bias by an optimizer. 0.1 is a convervative lr for a
  # single variable.
  return training.AdagradOptimizer(0.1).minimize(
      centered_bias_loss, var_list=centered_bias)


def _head_prefixed(head_name, val):
  return "%s_%s" % (head_name, val) if head_name else val


# TODO(zakaria): use contrib losses.
def _mean_squared_loss(logits, target):
  # To prevent broadcasting inside "-".
  if len(target.get_shape()) == 1:
    target = array_ops.expand_dims(target, dim=[1])
  # TODO(zakaria): make sure it does not recreate the broadcast bug.
  if len(logits.get_shape()) == 1:
    logits = array_ops.expand_dims(logits, dim=[1])
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
  # Check that we got integer for classification.
  if not target.dtype.is_integer:
    raise ValueError("Target's dtype should be integer "
                     "Instead got %s." % target.dtype)
  # sparse_softmax_cross_entropy_with_logits requires [batch_size] target.
  if len(target.get_shape()) == 2:
    target = array_ops.squeeze(target, squeeze_dims=[1])
  loss_vec = nn.sparse_softmax_cross_entropy_with_logits(logits, target)
  return loss_vec


def _sigmoid_cross_entropy_loss(logits, target):
  # sigmoid_cross_entropy_with_logits requires [batch_size, n_classes] target.
  return nn.sigmoid_cross_entropy_with_logits(logits, math_ops.to_float(target))


def _float_weights_or_none(weights):
  if weights is None:
    return None
  return math_ops.to_float(weights)


def _weighted_average_loss_metric_spec(loss_fn, predictoin_key,
                                       label_key, weight_key):
  def _streaming_weighted_average_loss(predictions, target, weights=None):
    loss_unweighted = loss_fn(predictions, target)
    if weights is not None:
      weights = math_ops.to_float(weights)
    _, weighted_average_loss = _loss(loss_unweighted,
                                     weights,
                                     name="eval_loss")
    return metrics_lib.streaming_mean(weighted_average_loss)
  return metric_spec.MetricSpec(_streaming_weighted_average_loss,
                                predictoin_key, label_key, weight_key)


def _target_streaming_mean(unused_predictions, target, weights=None):
  return metrics_lib.streaming_mean(target, weights=weights)


def _predictions_streaming_mean(predictions, unused_target, weights=None):
  return metrics_lib.streaming_mean(predictions, weights=weights)


def _streaming_auc(predictions, target, weights=None):
  return metrics_lib.streaming_auc(predictions, target,
                                   weights=_float_weights_or_none(weights))


def _accuracy_at_threshold(threshold):

  def _accuracy_metric(predictions, target, weights=None):
    threshold_predictions = math_ops.to_float(
        math_ops.greater_equal(predictions, threshold))
    return metrics_lib.streaming_accuracy(predictions=threshold_predictions,
                                          labels=target,
                                          weights=weights)

  return _accuracy_metric


def _streaming_at_threshold(streaming_metrics_fn, threshold):

  def _streaming_metrics(predictions, target, weights=None):
    precision_tensor, update_op = streaming_metrics_fn(
        predictions, labels=target, thresholds=[threshold],
        weights=_float_weights_or_none(weights))
    return array_ops.squeeze(precision_tensor), update_op

  return _streaming_metrics


class PredictionKey(object):
  CLASSES = "classes"
  PROBABILITIES = "probabilities"
  LOGITS = "logits"
  LOGISTIC = "logistic"
  SCORES = "scores"


class MetricKey(object):
  LOSS = "loss"
  AUC = "auc"
  PREDICTION_MEAN = "labels/prediction_mean"
  TARGET_MEAN = "labels/actual_target_mean"
  ACCURACY = "accuracy"
  ACCURACY_BASELINE = "accuracy/baseline_target_mean"
  ACCURACY_MEAN = "accuracy/threshold_%f_mean"
  PRECISION_MEAN = "precision/positive_threshold_%f_mean"
  RECALL_MEAN = "recall/positive_threshold_%f_mean"
