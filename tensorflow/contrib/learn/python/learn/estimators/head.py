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
"""Abstractions for the head(s) of a model (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.contrib import framework as framework_lib
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.contrib.learn.python.learn.estimators.metric_key import MetricKey as mkey
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import losses as losses_lib
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import training
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.deprecation import deprecated


class Head(object):
  """Interface for the head/top of a model.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Given logits (or output of a hidden layer), a Head knows how to compute
  predictions, loss, default metric and export signature. It is meant to,

  1) Simplify writing model_fn and to make model_fn more configurable
  2) Support wide range of machine learning models. Since most heads can work
      with logits, they can support DNN, RNN, Wide, Wide&Deep,
      Global objectives, Gradient boosted trees and many other types
      of machine learning models.
  2) To allow users to seamlessly switch between 1 to n heads for multi
  objective learning (See _MultiHead implementation for more details)

  Common usage:
  Here is simplified model_fn to build a multiclass DNN model.
    ```python
    def _my_dnn_model_fn(features, labels, mode, params, config=None):
      # Optionally your callers can pass head to model_fn as a param.
      head = tf.contrib.learn.multi_class_head(...)
      input = tf.contrib.layers.input_from_feature_columns(features, ...)
      last_hidden_layer_out = tf.contrib.layers.stack(
          input, tf.contrib.layers.fully_connected, [1000, 500])
      logits = tf.contrib.layers.fully_connected(
          last_hidden_layer_out, head.logits_dimension, activation_fn=None)

      def _train_op_fn(loss):
        return optimizer.minimize(loss)

      return head.create_model_fn_ops(
          features=features,
          labels=labels,
          mode=mode,
          train_op_fn=_train_op_fn,
          logits=logits,
          scope=...)
    ```

  Most heads also support logits_input which is typically the output of the last
  hidden layer. Some heads (like heads responsible for candidate sampling or
  hierarchical softmax) intrinsically will not support logits and you have
  to pass logits_input. Here is a common usage,
    ```python
    return head.create_model_fn_ops(
        features=features,
        labels=labels,
        mode=mode,
        train_op_fn=_train_op_fn,
        logits_input=last_hidden_layer_out,
        scope=...)
    ```python

  There are cases where computing and applying gradients can not be meaningfully
  captured with train_op_fn we support (for example, with sync optimizer). In
  such case, you can take the responsibility on your own. Here is a common
  use case,
    ```python
    model_fn_ops = head.create_model_fn_ops(
        features=features,
        labels=labels,
        mode=mode,
        train_op_fn=tf.contrib.learn.no_op_train_fn,
        logits=logits,
        scope=...)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      optimizer = ...
      sync = tf.train.SyncReplicasOptimizer(opt=optimizer, ...)
      update_op = tf.contrib.layers.optimize_loss(optimizer=sync,
                                                  loss=model_fn_ops.loss, ...)
      hooks = [sync.make_session_run_hook(is_chief)]
      ... update train_op and hooks in ModelFnOps and return
    ```
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def logits_dimension(self):
    """Size of the last dimension of the logits `Tensor`.

    Typically, logits is of shape `[batch_size, logits_dimension]`.

    Returns:
      The expected size of the `logits` tensor.
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
    """Returns `ModelFnOps` that a model_fn can return.

    Please note that,
    + Exactly one of `logits` and `logits_input` must be provided.
    + All args must be passed via name.

    Args:
      features: Input `dict` of `Tensor` objects.
      mode: Estimator's `ModeKeys`.
      labels: Labels `Tensor`, or `dict` of same.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns an op
          to optimize the model with the loss. This is used in TRAIN mode and
          must not be None. None is allowed in other modes. If you want to
          optimize loss yourself you can pass `no_op_train_fn` and then use
          ModeFnOps.loss to compute and apply gradients.
      logits: logits `Tensor` to be used by the head.
      logits_input: `Tensor` from which to build logits, often needed when you
        don't want to compute the logits. Typically this is the activation of
        the last hidden layer in a DNN. Some heads (like the ones responsible
        for candidate sampling) intrinsically avoid computing full logits and
        only accepts logits_input.
      scope: Optional scope for `variable_scope`.

    Returns:
      An instance of `ModelFnOps`.

    Raises:
      ValueError: If `mode` is not recognized.
      ValueError: If neither or both of `logits` and `logits_input` is provided.
    """
    raise NotImplementedError("Calling an abstract method.")


@deprecated(None, "Please switch to tf.contrib.estimator.*_head.")
def regression_head(label_name=None,
                    weight_column_name=None,
                    label_dimension=1,
                    enable_centered_bias=False,
                    head_name=None,
                    link_fn=None):
  """Creates a `Head` for linear regression.

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
    link_fn: link function to convert logits to predictions. If provided,
      this link function will be used instead of identity.

  Returns:
    An instance of `Head` for linear regression.
  """
  return _RegressionHead(
      label_name=label_name,
      weight_column_name=weight_column_name,
      label_dimension=label_dimension,
      enable_centered_bias=enable_centered_bias,
      head_name=head_name,
      loss_fn=_mean_squared_loss,
      link_fn=(link_fn if link_fn is not None else array_ops.identity))


@deprecated(None, "Please switch to tf.contrib.estimator.*_head.")
def poisson_regression_head(label_name=None,
                            weight_column_name=None,
                            label_dimension=1,
                            enable_centered_bias=False,
                            head_name=None):
  """Creates a `Head` for poisson regression.

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
    An instance of `Head` for poisson regression.
  """
  return _RegressionHead(
      label_name=label_name,
      weight_column_name=weight_column_name,
      label_dimension=label_dimension,
      enable_centered_bias=enable_centered_bias,
      head_name=head_name,
      loss_fn=_poisson_loss,
      link_fn=math_ops.exp)

# TODO(zakaria): Consider adding a _RegressionHead for logistic_regression


@deprecated(None, "Please switch to tf.contrib.estimator.*_head.")
def multi_class_head(n_classes,
                     label_name=None,
                     weight_column_name=None,
                     enable_centered_bias=False,
                     head_name=None,
                     thresholds=None,
                     metric_class_ids=None,
                     loss_fn=None,
                     label_keys=None):
  """Creates a `Head` for multi class single label classification.

  The Head uses softmax cross entropy loss.

  This head expects to be fed integer labels specifying the class index. But
  if `label_keys` is specified, then labels must be strings from this
  vocabulary, and the predicted classes will be strings from the same
  vocabulary.

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
    loss_fn: Optional function that takes (`labels`, `logits`, `weights`) as
      parameter and returns a weighted scalar loss. `weights` should be
      optional. See `tf.losses`
    label_keys: Optional list of strings with size `[n_classes]` defining the
      label vocabulary. Only supported for `n_classes` > 2.

  Returns:
    An instance of `Head` for multi class classification.

  Raises:
    ValueError: if `n_classes` is < 2.
    ValueError: If `metric_class_ids` is provided when `n_classes` is 2.
    ValueError: If `len(label_keys) != n_classes`.
  """
  if (n_classes is None) or (n_classes < 2):
    raise ValueError("n_classes must be > 1 for classification: %s." %
                     n_classes)
  if loss_fn:
    _verify_loss_fn_args(loss_fn)

  loss_fn = _wrap_custom_loss_fn(loss_fn) if loss_fn else None
  if n_classes == 2:
    if metric_class_ids:
      raise ValueError("metric_class_ids invalid for n_classes==2.")
    if label_keys:
      raise ValueError("label_keys is not supported for n_classes=2.")
    return _BinaryLogisticHead(
        label_name=label_name,
        weight_column_name=weight_column_name,
        enable_centered_bias=enable_centered_bias,
        head_name=head_name,
        thresholds=thresholds,
        loss_fn=loss_fn)

  return _MultiClassHead(
      n_classes=n_classes,
      label_name=label_name,
      weight_column_name=weight_column_name,
      enable_centered_bias=enable_centered_bias,
      head_name=head_name,
      thresholds=thresholds,
      metric_class_ids=metric_class_ids,
      loss_fn=loss_fn,
      label_keys=label_keys)


@deprecated(None, "Please switch to tf.contrib.estimator.*_head.")
def binary_svm_head(
    label_name=None,
    weight_column_name=None,
    enable_centered_bias=False,
    head_name=None,
    thresholds=None,):
  """Creates a `Head` for binary classification with SVMs.

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
    An instance of `Head` for binary classification with SVM.
  """
  return _BinarySvmHead(
      label_name=label_name,
      weight_column_name=weight_column_name,
      enable_centered_bias=enable_centered_bias,
      head_name=head_name,
      thresholds=thresholds)


@deprecated(None, "Please switch to tf.contrib.estimator.*_head.")
def multi_label_head(n_classes,
                     label_name=None,
                     weight_column_name=None,
                     enable_centered_bias=False,
                     head_name=None,
                     thresholds=None,
                     metric_class_ids=None,
                     loss_fn=None):
  """Creates a Head for multi label classification.

  Multi-label classification handles the case where each example may have zero
  or more associated labels, from a discrete set.  This is distinct from
  `multi_class_head` which has exactly one label from a discrete set.

  This head by default uses sigmoid cross entropy loss, which expects as input
  a multi-hot tensor of shape `(batch_size, num_classes)`.

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
    loss_fn: Optional function that takes (`labels`, `logits`, `weights`) as
      parameter and returns a weighted scalar loss. `weights` should be
      optional. See `tf.losses`

  Returns:
    An instance of `Head` for multi label classification.

  Raises:
    ValueError: If n_classes is < 2
    ValueError: If loss_fn does not have expected signature.
  """
  if n_classes < 2:
    raise ValueError("n_classes must be > 1 for classification.")
  if loss_fn:
    _verify_loss_fn_args(loss_fn)

  return _MultiLabelHead(
      n_classes=n_classes,
      label_name=label_name,
      weight_column_name=weight_column_name,
      enable_centered_bias=enable_centered_bias,
      head_name=head_name,
      thresholds=thresholds,
      metric_class_ids=metric_class_ids,
      loss_fn=_wrap_custom_loss_fn(loss_fn) if loss_fn else None)


@deprecated(None, "Please switch to tf.contrib.estimator.*_head.")
def loss_only_head(loss_fn, head_name=None):
  """Creates a Head that contains only loss terms.

  Loss only head holds additional loss terms to be added to other heads and
  usually represents additional regularization terms in the objective function.

  Args:
    loss_fn: a function that takes no argument and returns a list of
        scalar tensors.
    head_name: a name for the head.

  Returns:
    An instance of `Head` to hold the additional losses.
  """
  return _LossOnlyHead(loss_fn, head_name=head_name)


@deprecated(None, "Please switch to tf.contrib.estimator.*_head.")
def multi_head(heads, loss_weights=None):
  """Creates a MultiHead stemming from same logits/hidden layer.

  Args:
    heads: list of Head objects.
    loss_weights: optional list of weights to be used to merge losses from
        each head. All losses are weighted equally if not provided.

  Returns:
    A instance of `Head` that merges multiple heads.

  Raises:
    ValueError: if heads and loss_weights have different size.
  """
  if loss_weights:
    if len(loss_weights) != len(heads):
      raise ValueError("heads and loss_weights must have same size")

  def _weighted_loss_merger(losses):
    if loss_weights:
      if len(losses) != len(loss_weights):
        raise ValueError("losses and loss_weights must have same size")
      weighted_losses = []
      for loss, weight in zip(losses, loss_weights):
        weighted_losses.append(math_ops.multiply(loss, weight))
      return math_ops.add_n(weighted_losses)
    else:
      return math_ops.add_n(losses)

  return _MultiHead(heads, loss_merger=_weighted_loss_merger)


@deprecated(None, "Use 'lambda _: tf.no_op()'.")
def no_op_train_fn(loss):
  del loss
  return control_flow_ops.no_op()


class _SingleHead(Head):
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
def _mean_squared_loss(labels, logits, weights=None):
  with ops.name_scope(None, "mean_squared_loss", (logits, labels)) as name:
    logits = ops.convert_to_tensor(logits)
    labels = ops.convert_to_tensor(labels)
    # To prevent broadcasting inside "-".
    if len(labels.get_shape()) == 1:
      labels = array_ops.expand_dims(labels, axis=1)
    # TODO(zakaria): make sure it does not recreate the broadcast bug.
    if len(logits.get_shape()) == 1:
      logits = array_ops.expand_dims(logits, axis=1)
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    loss = math_ops.square(logits - math_ops.to_float(labels), name=name)
    return _compute_weighted_loss(loss, weights)


def _poisson_loss(labels, logits, weights=None):
  """Computes poisson loss from logits."""
  with ops.name_scope(None, "_poisson_loss", (logits, labels)) as name:
    logits = ops.convert_to_tensor(logits)
    labels = ops.convert_to_tensor(labels)
    # To prevent broadcasting inside "-".
    if len(labels.get_shape()) == 1:
      labels = array_ops.expand_dims(labels, axis=1)
    # TODO(zakaria): make sure it does not recreate the broadcast bug.
    if len(logits.get_shape()) == 1:
      logits = array_ops.expand_dims(logits, axis=1)
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    loss = nn.log_poisson_loss(labels, logits, compute_full_loss=True,
                               name=name)
    return _compute_weighted_loss(loss, weights)


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
                         loss_fn,
                         logits_to_predictions_fn,
                         metrics_fn,
                         create_output_alternatives_fn,
                         labels=None,
                         train_op_fn=None,
                         logits=None,
                         logits_dimension=None,
                         head_name=None,
                         weight_column_name=None,
                         enable_centered_bias=False):
  """Returns a `ModelFnOps` object."""
  _check_mode_valid(mode)

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
    loss, weighted_average_loss = loss_fn(labels, logits, weight_tensor)
    # The name_scope escapism is needed to maintain the same summary tag
    # after switching away from the now unsupported API.
    with ops.name_scope(""):
      summary_loss = array_ops.identity(weighted_average_loss)
      summary.scalar(_summary_key(head_name, mkey.LOSS), summary_loss)

    if mode == model_fn.ModeKeys.TRAIN:
      if train_op_fn is None:
        raise ValueError("train_op_fn can not be None in TRAIN mode")
      batch_size = array_ops.shape(logits)[0]
      train_op = _train_op(loss, labels, train_op_fn, centered_bias,
                           batch_size, loss_fn, weight_tensor)
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
  """`Head` for regression with a generalized linear model."""

  def __init__(self,
               label_dimension,
               loss_fn,
               link_fn,
               logits_dimension=None,
               label_name=None,
               weight_column_name=None,
               enable_centered_bias=False,
               head_name=None):
    """`Head` for regression.

    Args:
      label_dimension: Number of regression labels per example. This is the
        size of the last dimension of the labels `Tensor` (typically, this has
        shape `[batch_size, label_dimension]`).
      loss_fn: Loss function, takes logits and labels and returns loss.
      link_fn: Link function, takes a logits tensor and returns the output.
      logits_dimension: Number of logits per example. This is the
        size of the last dimension of the logits `Tensor` (typically, this has
        shape `[batch_size, label_dimension]`).
        Default value: `label_dimension`.
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
        logits_dimension=(logits_dimension if logits_dimension is not None
                          else label_dimension),
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
    """See `Head`."""
    with variable_scope.variable_scope(
        scope,
        default_name=self.head_name or "regression_head",
        values=(tuple(six.itervalues(features)) +
                (labels, logits, logits_input))):
      labels = self._transform_labels(mode=mode, labels=labels)
      logits = _logits(logits_input, logits, self.logits_dimension)
      return _create_model_fn_ops(
          features=features,
          mode=mode,
          loss_fn=self._loss_fn,
          logits_to_predictions_fn=self._logits_to_predictions,
          metrics_fn=self._metrics,
          create_output_alternatives_fn=self._create_output_alternatives,
          labels=labels,
          train_op_fn=train_op_fn,
          logits=logits,
          logits_dimension=self.logits_dimension,
          head_name=self.head_name,
          weight_column_name=self.weight_column_name,
          enable_centered_bias=self._enable_centered_bias)

  def _transform_labels(self, mode, labels):
    """Applies transformations to labels tensor."""
    if (mode == model_fn.ModeKeys.INFER) or (labels is None):
      return None
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
        logits = array_ops.squeeze(logits, axis=(1,), name=key)
      return {key: self._link_fn(logits)}

  def _metrics(self, eval_loss, predictions, labels, weights):
    """Returns a dict of metrics keyed by name."""
    del predictions, labels, weights  # Unused by this head.
    with ops.name_scope("metrics", values=[eval_loss]):
      return {
          _summary_key(self.head_name, mkey.LOSS):
              metrics_lib.mean(eval_loss)}


def _log_loss_with_two_classes(labels, logits, weights=None):
  with ops.name_scope(None, "log_loss_with_two_classes",
                      (logits, labels)) as name:
    logits = ops.convert_to_tensor(logits)
    labels = math_ops.to_float(labels)
    # TODO(ptucker): This will break for dynamic shapes.
    # sigmoid_cross_entropy_with_logits requires [batch_size, 1] labels.
    if len(labels.get_shape()) == 1:
      labels = array_ops.expand_dims(labels, axis=1)
    loss = nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits,
                                                name=name)
    return _compute_weighted_loss(loss, weights)


def _one_class_to_two_class_logits(logits):
  return array_ops.concat((array_ops.zeros_like(logits), logits), 1)


class _BinaryLogisticHead(_SingleHead):
  """`Head` for binary classification with logistic regression."""

  def __init__(self,
               label_name=None,
               weight_column_name=None,
               enable_centered_bias=False,
               head_name=None,
               loss_fn=None,
               thresholds=None):
    """`Head` for binary classification with logistic regression.

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
    self._loss_fn = loss_fn if loss_fn else _log_loss_with_two_classes
    self._enable_centered_bias = enable_centered_bias

  def create_model_fn_ops(self,
                          features,
                          mode,
                          labels=None,
                          train_op_fn=None,
                          logits=None,
                          logits_input=None,
                          scope=None):
    """See `Head`."""
    with variable_scope.variable_scope(
        scope,
        default_name=self.head_name or "binary_logistic_head",
        values=(tuple(six.itervalues(features)) +
                (labels, logits, logits_input))):
      labels = self._transform_labels(mode=mode, labels=labels)
      logits = _logits(logits_input, logits, self.logits_dimension)
      return _create_model_fn_ops(
          features=features,
          mode=mode,
          loss_fn=self._loss_fn,
          logits_to_predictions_fn=self._logits_to_predictions,
          metrics_fn=self._metrics,
          create_output_alternatives_fn=_classification_output_alternatives(
              self.head_name, self._problem_type),
          labels=labels,
          train_op_fn=train_op_fn,
          logits=logits,
          logits_dimension=self.logits_dimension,
          head_name=self.head_name,
          weight_column_name=self.weight_column_name,
          enable_centered_bias=self._enable_centered_bias)

  def _transform_labels(self, mode, labels):
    """Applies transformations to labels tensor."""
    if (mode == model_fn.ModeKeys.INFER) or (labels is None):
      return None
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
                 metrics_lib.mean(eval_loss)}
      # TODO(b/29366811): This currently results in both an "accuracy" and an
      # "accuracy/threshold_0.500000_mean" metric for binary classification.
      metrics[_summary_key(self.head_name, mkey.ACCURACY)] = (
          metrics_lib.accuracy(labels, classes, weights))
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
      metrics[_summary_key(self.head_name, mkey.AUC_PR)] = (
          _streaming_auc(logistic, labels, weights, curve="PR"))

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


def _softmax_cross_entropy_loss(labels, logits, weights=None):
  with ops.name_scope(
      None, "softmax_cross_entropy_loss", (logits, labels,)) as name:
    labels = ops.convert_to_tensor(labels)
    # Check that we got integer for classification.
    if not labels.dtype.is_integer:
      raise ValueError("Labels dtype should be integer "
                       "Instead got %s." % labels.dtype)

    # sparse_softmax_cross_entropy_with_logits requires [batch_size] labels.
    is_squeezed_labels = False
    # TODO(ptucker): This will break for dynamic shapes.
    if len(labels.get_shape()) == 2:
      labels = array_ops.squeeze(labels, axis=(1,))
      is_squeezed_labels = True

    loss = nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name=name)

    # Restore squeezed dimension, if necessary, so loss matches weights shape.
    if is_squeezed_labels:
      loss = array_ops.expand_dims(loss, axis=(1,))

    return _compute_weighted_loss(loss, weights)


class _MultiClassHead(_SingleHead):
  """'Head' for multi class classification."""

  def __init__(self,
               n_classes,
               label_name=None,
               weight_column_name=None,
               enable_centered_bias=False,
               head_name=None,
               loss_fn=None,
               thresholds=None,
               metric_class_ids=None,
               label_keys=None):
    """'Head' for multi class classification.

    This head expects to be fed integer labels specifying the class index. But
    if `label_keys` is specified, then labels must be strings from this
    vocabulary, and the predicted classes will be strings from the same
    vocabulary.

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
      loss_fn: Loss function. Defaults to softmax cross entropy loss.
      thresholds: thresholds for eval.
      metric_class_ids: List of class IDs for which we should report per-class
        metrics. Must all be in the range `[0, n_classes)`.
      label_keys: Optional list of strings with size `[n_classes]` defining the
        label vocabulary.

    Raises:
      ValueError: if `n_classes`, `metric_class_ids` or `label_keys` is invalid.
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
    self._loss_fn = loss_fn if loss_fn else _softmax_cross_entropy_loss
    self._enable_centered_bias = enable_centered_bias
    self._metric_class_ids = tuple([] if metric_class_ids is None else
                                   metric_class_ids)
    for class_id in self._metric_class_ids:
      if (class_id < 0) or (class_id >= n_classes):
        raise ValueError("Class ID %s not in [0, %s)." % (class_id, n_classes))
    if label_keys and len(label_keys) != n_classes:
      raise ValueError("Length of label_keys must equal n_classes.")
    self._label_keys = label_keys

  def create_model_fn_ops(self,
                          features,
                          mode,
                          labels=None,
                          train_op_fn=None,
                          logits=None,
                          logits_input=None,
                          scope=None):
    """See `Head`."""
    with variable_scope.variable_scope(
        scope,
        default_name=self.head_name or "multi_class_head",
        values=(tuple(six.itervalues(features)) +
                (labels, logits, logits_input))):
      labels = self._transform_labels(mode=mode, labels=labels)
      logits = _logits(logits_input, logits, self.logits_dimension)
      return _create_model_fn_ops(
          features=features,
          mode=mode,
          loss_fn=self._wrapped_loss_fn,
          logits_to_predictions_fn=self._logits_to_predictions,
          metrics_fn=self._metrics,
          create_output_alternatives_fn=_classification_output_alternatives(
              self.head_name, self._problem_type, self._label_keys),
          labels=labels,
          train_op_fn=train_op_fn,
          logits=logits,
          logits_dimension=self.logits_dimension,
          head_name=self.head_name,
          weight_column_name=self.weight_column_name,
          enable_centered_bias=self._enable_centered_bias)

  def _transform_labels(self, mode, labels):
    """Returns a dict that contains both the original labels and label IDs."""
    if (mode == model_fn.ModeKeys.INFER) or (labels is None):
      return None
    labels_tensor = _to_labels_tensor(labels, self._label_name)
    _check_no_sparse_tensor(labels_tensor)
    if self._label_keys:
      table = lookup_ops.index_table_from_tensor(
          self._label_keys, name="label_id_lookup")
      return {
          "labels": labels_tensor,
          "label_ids": table.lookup(labels_tensor),
      }
    return {
        "labels": labels_tensor,
        "label_ids": labels_tensor,
    }

  def _labels(self, labels_dict):
    """Returns labels `Tensor` of the same type as classes."""
    return labels_dict["labels"]

  def _label_ids(self, labels_dict):
    """Returns integer label ID `Tensor`."""
    return labels_dict["label_ids"]

  def _wrapped_loss_fn(self, labels, logits, weights=None):
    return self._loss_fn(self._label_ids(labels), logits, weights=weights)

  def _logits_to_predictions(self, logits):
    """Returns a dict of predictions.

    Args:
      logits: logits `Tensor` after applying possible centered bias.

    Returns:
      Dict of prediction `Tensor` keyed by `PredictionKey`.
    """
    with ops.name_scope(None, "predictions", (logits,)):
      class_ids = math_ops.argmax(
          logits, 1, name=prediction_key.PredictionKey.CLASSES)
      if self._label_keys:
        table = lookup_ops.index_to_string_table_from_tensor(
            self._label_keys, name="class_string_lookup")
        classes = table.lookup(class_ids)
      else:
        classes = class_ids
      return {
          prediction_key.PredictionKey.LOGITS: logits,
          prediction_key.PredictionKey.PROBABILITIES:
              nn.softmax(
                  logits, name=prediction_key.PredictionKey.PROBABILITIES),
          prediction_key.PredictionKey.CLASSES: classes
      }

  def _metrics(self, eval_loss, predictions, labels, weights):
    """Returns a dict of metrics keyed by name."""
    with ops.name_scope(
        "metrics",
        values=((eval_loss, self._labels(labels), self._label_ids(labels),
                 weights) + tuple(six.itervalues(predictions)))):
      logits = predictions[prediction_key.PredictionKey.LOGITS]
      probabilities = predictions[prediction_key.PredictionKey.PROBABILITIES]
      classes = predictions[prediction_key.PredictionKey.CLASSES]

      metrics = {_summary_key(self.head_name, mkey.LOSS):
                 metrics_lib.mean(eval_loss)}
      # TODO(b/29366811): This currently results in both an "accuracy" and an
      # "accuracy/threshold_0.500000_mean" metric for binary classification.
      metrics[_summary_key(self.head_name, mkey.ACCURACY)] = (
          metrics_lib.accuracy(self._labels(labels), classes, weights))

      if not self._label_keys:
        # Classes are IDs. Add some metrics.
        for class_id in self._metric_class_ids:
          metrics[_summary_key(
              self.head_name, mkey.CLASS_PREDICTION_MEAN % class_id)] = (
                  _class_predictions_streaming_mean(classes, weights, class_id))
          # TODO(ptucker): Add per-class accuracy, precision, recall.
          metrics[_summary_key(
              self.head_name, mkey.CLASS_LABEL_MEAN % class_id)] = (
                  _class_labels_streaming_mean(
                      self._label_ids(labels), weights, class_id))
          metrics[_summary_key(
              self.head_name, mkey.CLASS_PROBABILITY_MEAN % class_id)] = (
                  _predictions_streaming_mean(probabilities, weights, class_id))
          metrics[_summary_key(
              self.head_name, mkey.CLASS_LOGITS_MEAN % class_id)] = (
                  _predictions_streaming_mean(logits, weights, class_id))

    return metrics


def _to_labels_tensor(labels, label_name):
  """Returns label as a tensor.

  Args:
    labels: Label `Tensor` or `SparseTensor` or a dict containing labels.
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
    ValueError: If labels is `SparseTensor` and `num_classes` < 2.
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
  """`Head` for binary classification using SVM."""

  def __init__(self, label_name, weight_column_name, enable_centered_bias,
               head_name, thresholds):

    def _loss_fn(labels, logits, weights=None):
      with ops.name_scope(None, "hinge_loss", (logits, labels)) as name:
        with ops.control_dependencies((_assert_labels_rank(labels),)):
          labels = array_ops.reshape(labels, shape=(-1, 1))
        loss = losses_lib.hinge_loss(labels=labels, logits=logits, scope=name,
                                     reduction=losses_lib.Reduction.NONE)
        return _compute_weighted_loss(loss, weights)

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
    """See `Head`."""
    with variable_scope.variable_scope(
        scope,
        default_name=self.head_name or "binary_svm_head",
        values=(tuple(six.itervalues(features)) +
                (labels, logits, logits_input))):
      labels = self._transform_labels(mode=mode, labels=labels)
      logits = _logits(logits_input, logits, self.logits_dimension)
      return _create_model_fn_ops(
          features=features,
          mode=mode,
          loss_fn=self._loss_fn,
          logits_to_predictions_fn=self._logits_to_predictions,
          metrics_fn=self._metrics,
          # TODO(zakaria): Handle labels for export.
          create_output_alternatives_fn=self._create_output_alternatives,
          labels=labels,
          train_op_fn=train_op_fn,
          logits=logits,
          logits_dimension=self.logits_dimension,
          head_name=self.head_name,
          weight_column_name=self.weight_column_name,
          enable_centered_bias=self._enable_centered_bias)

  def _transform_labels(self, mode, labels):
    """Applies transformations to labels tensor."""
    if (mode == model_fn.ModeKeys.INFER) or (labels is None):
      return None
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
                 metrics_lib.mean(eval_loss)}

      # TODO(b/29366811): This currently results in both an "accuracy" and an
      # "accuracy/threshold_0.500000_mean" metric for binary classification.
      classes = predictions[prediction_key.PredictionKey.CLASSES]
      metrics[_summary_key(self.head_name, mkey.ACCURACY)] = (
          metrics_lib.accuracy(labels, classes, weights))
      # TODO(sibyl-vie3Poto): add more metrics relevant for svms.

    return metrics


class _MultiLabelHead(_SingleHead):
  """`Head` for multi-label classification."""

  # TODO(zakaria): add signature and metric for multilabel.
  def __init__(self,
               n_classes,
               label_name,
               weight_column_name,
               enable_centered_bias,
               head_name,
               thresholds,
               metric_class_ids=None,
               loss_fn=None):

    super(_MultiLabelHead, self).__init__(
        problem_type=constants.ProblemType.CLASSIFICATION,
        logits_dimension=n_classes,
        label_name=label_name,
        weight_column_name=weight_column_name,
        head_name=head_name)

    self._thresholds = thresholds if thresholds else (.5,)
    self._loss_fn = loss_fn if loss_fn else _sigmoid_cross_entropy_loss
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
    """See `Head`."""
    with variable_scope.variable_scope(
        scope,
        default_name=self.head_name or "multi_label_head",
        values=(tuple(six.itervalues(features)) +
                (labels, logits, logits_input))):
      labels = self._transform_labels(mode=mode, labels=labels)
      logits = _logits(logits_input, logits, self.logits_dimension)
      return _create_model_fn_ops(
          features=features,
          mode=mode,
          loss_fn=self._loss_fn,
          logits_to_predictions_fn=self._logits_to_predictions,
          metrics_fn=self._metrics,
          create_output_alternatives_fn=_classification_output_alternatives(
              self.head_name, self._problem_type),
          labels=labels,
          train_op_fn=train_op_fn,
          logits=logits,
          logits_dimension=self.logits_dimension,
          head_name=self.head_name,
          weight_column_name=self.weight_column_name,
          enable_centered_bias=self._enable_centered_bias)

  def _transform_labels(self, mode, labels):
    """Applies transformations to labels tensor."""
    if (mode == model_fn.ModeKeys.INFER) or (labels is None):
      return None
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
                 metrics_lib.mean(eval_loss)}
      # TODO(b/29366811): This currently results in both an "accuracy" and an
      # "accuracy/threshold_0.500000_mean" metric for binary classification.
      metrics[_summary_key(self.head_name, mkey.ACCURACY)] = (
          metrics_lib.accuracy(labels, classes, weights))
      metrics[_summary_key(self.head_name, mkey.AUC)] = _streaming_auc(
          probabilities, labels, weights)
      metrics[_summary_key(self.head_name, mkey.AUC_PR)] = _streaming_auc(
          probabilities, labels, weights, curve="PR")

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
            _streaming_auc(probabilities, labels, weights, class_id))
        metrics[_summary_key(self.head_name, mkey.CLASS_AUC_PR % class_id)] = (
            _streaming_auc(probabilities, labels, weights, class_id,
                           curve="PR"))

    return metrics


class _LossOnlyHead(Head):
  """`Head` implementation for additional loss terms.

  This class only holds loss terms unrelated to any other heads (labels),
  e.g. regularization.

  Common usage:
  This is oftem combine with other heads in a multi head setup.
    ```python
    head = multi_head([
        head1, head2, loss_only_head('regularizer', regularizer)])
    ```
  """

  def __init__(self, loss_fn, head_name=None):
    self._loss_fn = loss_fn
    self.head_name = head_name or "loss_only_head"

  @property
  def logits_dimension(self):
    return 0

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
      features: Not been used.
      mode: Estimator's `ModeKeys`.
      labels: Labels `Tensor`, or `dict` of same.
      train_op_fn: Function that takes a scalar loss and returns an op to
          optimize with the loss.
      logits: Not been used.
      logits_input: Not been used.
      scope: Optional scope for variable_scope. If provided, will be passed to
          all heads. Most users will want to set this to `None`, so each head
          constructs a separate variable_scope according to its `head_name`.

    Returns:
      A `ModelFnOps` object.

    Raises:
      ValueError: if `mode` is not recognition.
    """
    _check_mode_valid(mode)
    loss = None
    train_op = None
    if mode != model_fn.ModeKeys.INFER:
      with variable_scope.variable_scope(scope, default_name=self.head_name):
        loss = self._loss_fn()
        if isinstance(loss, list):
          loss = math_ops.add_n(loss)
        # The name_scope escapism is needed to maintain the same summary tag
        # after switching away from the now unsupported API.
        with ops.name_scope(""):
          summary_loss = array_ops.identity(loss)
          summary.scalar(_summary_key(self.head_name, mkey.LOSS),
                         summary_loss)
        if mode == model_fn.ModeKeys.TRAIN:
          if train_op_fn is None:
            raise ValueError("train_op_fn can not be None in TRAIN mode")
          with ops.name_scope(None, "train_op", (loss,)):
            train_op = train_op_fn(loss)

    return model_fn.ModelFnOps(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions={},
        eval_metric_ops={})


class _MultiHead(Head):
  """`Head` implementation for multi objective learning.

  This class is responsible for using and merging the output of multiple
  `Head` objects.

  All heads stem from the same logits/logit_input tensor.

  Common usage:
  For simple use cases you can pass the activation of hidden layer like
  this from your model_fn,
    ```python
    last_hidden_layer_activation = ... Build your model.
    multi_head = ...
    return multi_head.create_model_fn_ops(
        ..., logits_input=last_hidden_layer_activation, ...)
    ```

  Or you can create a logits tensor of
  [batch_size, multi_head.logits_dimension] shape. _MultiHead will split the
  logits for you.
    return multi_head.create_model_fn_ops(..., logits=logits, ...)

  For more complex use cases like a multi-task/multi-tower model or when logits
  for each head has to be created separately, you can pass a dict of logits
  where the keys match the name of the single heads.
    ```python
    logits = {"head1": logits1, "head2": logits2}
    return multi_head.create_model_fn_ops(..., logits=logits, ...)
    ```

  Here is what this class does,
  + For training, merges losses of each heads according a function provided by
      user, calls user provided train_op_fn with this final loss.
  + For eval, merges metrics by adding head_name suffix to the keys in eval
      metrics.
  + For inference, updates keys in prediction dict to a 2-tuple,
      (head_name, prediction_key)
  """

  def __init__(self, heads, loss_merger):
    """_Head to merges multiple _Head objects.

    Args:
      heads: list of _Head objects.
      loss_merger: function that takes a list of loss tensors for the heads
        and returns the final loss tensor for the multi head.

    Raises:
      ValueError: if any head does not have a name.
    """
    self._logits_dimension = 0
    for head in heads:
      if not head.head_name:
        raise ValueError("Members of MultiHead must have names.")
      self._logits_dimension += head.logits_dimension

    self._heads = heads
    self._loss_merger = loss_merger

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
      logits: Concatenated logits for all heads or a dict of head name to logits
          tensor. If concatenated logits, it should have (batchsize, x) shape
          where x is the sum of `logits_dimension` of all the heads,
          i.e., same as `logits_dimension` of this class. create_model_fn_ops
          will split the logits tensor and pass logits of proper size to each
          head. This is useful if we want to be agnostic about whether you
          creating a single versus multihead. logits can also be a dict for
          convenience where you are creating the head specific logits explicitly
          and don't want to concatenate them yourself.
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
        all_model_fn_ops.append(
            head.create_model_fn_ops(
                features=features,
                mode=mode,
                labels=labels,
                train_op_fn=no_op_train_fn,
                logits_input=logits_input,
                scope=scope))
    else:
      head_logits_pairs = []
      if isinstance(logits, dict):
        head_logits_pairs = []
        for head in self._heads:
          if isinstance(head, _LossOnlyHead):
            head_logits_pairs.append((head, None))
          else:
            head_logits_pairs.append((head, logits[head.head_name]))
      else:
        # Split logits for each head.
        head_logits_pairs = zip(self._heads, self._split_logits(logits))

      for head, head_logits in head_logits_pairs:
        all_model_fn_ops.append(
            head.create_model_fn_ops(
                features=features,
                mode=mode,
                labels=labels,
                train_op_fn=no_op_train_fn,
                logits=head_logits,
                scope=scope))

    if mode == model_fn.ModeKeys.TRAIN:
      if train_op_fn is None:
        raise ValueError("train_op_fn can not be None in TRAIN mode.")
      return self._merge_train(all_model_fn_ops, train_op_fn)
    if mode == model_fn.ModeKeys.INFER:
      return self._merge_infer(all_model_fn_ops)
    if mode == model_fn.ModeKeys.EVAL:
      return self._merge_eval(all_model_fn_ops)
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

  def _merge_train(self, all_model_fn_ops, train_op_fn):
    """Merges list of ModelFnOps for training.

    Args:
      all_model_fn_ops: list of ModelFnOps for the individual heads.
      train_op_fn: Function to create train op. See `create_model_fn_ops`
          documentation for more details.

    Returns:
      ModelFnOps that merges all heads for TRAIN.
    """
    losses = []
    metrics = {}
    additional_train_ops = []
    for m in all_model_fn_ops:
      losses.append(m.loss)
      if m.eval_metric_ops is not None:
        for k, v in six.iteritems(m.eval_metric_ops):
          # metrics["%s/%s" % (k, head_name)] = v
          metrics[k] = v
      additional_train_ops.append(m.train_op)
    loss = self._loss_merger(losses)

    train_op = train_op_fn(loss)
    train_op = control_flow_ops.group(train_op, *additional_train_ops)
    return model_fn.ModelFnOps(
        mode=model_fn.ModeKeys.TRAIN,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

  def _merge_infer(self, all_model_fn_ops):
    """Merges list of ModelFnOps for inference.

    Args:
      all_model_fn_ops: list of ModelFnOps for the individual heads.

    Returns:
      ModelFnOps that Merges all the heads for INFER.
    """
    predictions = {}
    output_alternatives = {}
    for head, m in zip(self._heads, all_model_fn_ops):
      if isinstance(head, _LossOnlyHead):
        continue
      head_name = head.head_name
      output_alternatives[head_name] = m.output_alternatives[head_name]
      for k, v in m.predictions.items():
        predictions[(head_name, k)] = v

    return model_fn.ModelFnOps(
        mode=model_fn.ModeKeys.INFER,
        predictions=predictions,
        output_alternatives=output_alternatives)

  def _merge_eval(self, all_model_fn_ops):
    """Merges list of ModelFnOps for eval.

    Args:
      all_model_fn_ops: list of ModelFnOps for the individual heads.

    Returns:
      ModelFnOps that merges all the heads for EVAL.
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
    loss = self._loss_merger(losses)

    return model_fn.ModelFnOps(
        mode=model_fn.ModeKeys.EVAL,
        predictions=predictions,
        loss=loss,
        eval_metric_ops=metrics)


def _weight_tensor(features, weight_column_name):
  """Returns weights as `Tensor` of rank 0, or at least 2."""
  if not weight_column_name:
    return None
  if weight_column_name not in features:
    raise ValueError("Weights {} missing from features.".format(
        weight_column_name))
  with ops.name_scope(None, "weight_tensor", tuple(six.itervalues(features))):
    weight_tensor = math_ops.to_float(features[weight_column_name])
    shape = weight_tensor.get_shape()
    rank = shape.ndims
    # We don't bother with expanding dims of non-staticly shaped tensors or
    # scalars, and >1d is already in a good format.
    if rank == 1:
      logging.warning("Weights {} has shape {}, expanding to make it 2d.".
                      format(weight_column_name, shape))
      return (
          sparse_ops.sparse_reshape(weight_tensor, (-1, 1))
          if isinstance(weight_tensor, sparse_tensor.SparseTensor) else
          array_ops.reshape(weight_tensor, (-1, 1)))
    return weight_tensor


# TODO(zakaria): This function is needed for backward compatibility and should
#   be removed when we migrate to core.
def _compute_weighted_loss(loss_unweighted, weight, name="loss"):
  """Returns a tuple of (loss_train, loss_report).

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
    A tuple of losses. First one for training and the second one for reporting.
  """
  with ops.name_scope(name, values=(loss_unweighted, weight)) as name_scope:
    if weight is None:
      loss = math_ops.reduce_mean(loss_unweighted, name=name_scope)
      return loss, loss
    weight = weights_broadcast_ops.broadcast_weights(weight, loss_unweighted)
    with ops.name_scope(None, "weighted_loss",
                        (loss_unweighted, weight)) as name:
      weighted_loss = math_ops.multiply(loss_unweighted, weight, name=name)
    weighted_loss_mean = math_ops.reduce_mean(weighted_loss, name=name_scope)
    weighted_loss_normalized = math_ops.div(
        math_ops.reduce_sum(weighted_loss),
        math_ops.to_float(math_ops.reduce_sum(weight)),
        name="weighted_average_loss")

    return weighted_loss_mean, weighted_loss_normalized


def _wrap_custom_loss_fn(loss_fn):
  def _wrapper(labels, logits, weights=None):
    if weights is None:
      loss = loss_fn(labels, logits)
    else:
      loss = loss_fn(labels, logits, weights)
    return loss, loss
  return _wrapper


def _check_mode_valid(mode):
  """Raises ValueError if the given mode is invalid."""
  if (mode != model_fn.ModeKeys.TRAIN and mode != model_fn.ModeKeys.INFER and
      mode != model_fn.ModeKeys.EVAL):
    raise ValueError("mode=%s unrecognized." % str(mode))


def _get_arguments(func):
  """Returns a spec of given func."""
  _, func = tf_decorator.unwrap(func)
  if hasattr(func, "__code__"):
    # Regular function.
    return tf_inspect.getargspec(func)
  elif hasattr(func, "func"):
    # Partial function.
    return _get_arguments(func.func)
  elif hasattr(func, "__call__"):
    # Callable object.
    return _get_arguments(func.__call__)


def _verify_loss_fn_args(loss_fn):
  args = _get_arguments(loss_fn).args
  for arg_name in ["labels", "logits", "weights"]:
    if arg_name not in args:
      raise ValueError("Argument %s not found in loss_fn." % arg_name)


def _centered_bias(logits_dimension, head_name=None):
  """Returns centered_bias `Variable`.

  Args:
    logits_dimension: Last dimension of `logits`. Must be >= 1.
    head_name: Optional name of the head.

  Returns:
    `Variable` with shape `[logits_dimension]`.

  Raises:
    ValueError: if `logits_dimension` is invalid.
  """
  if (logits_dimension is None) or (logits_dimension < 1):
    raise ValueError("Invalid logits_dimension %s." % logits_dimension)
  # Do not create a variable with variable_scope.get_variable, because that may
  # create a PartitionedVariable, which does not support indexing, so
  # summary.scalar will not work.
  centered_bias = variable_scope.variable(
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


def _centered_bias_step(centered_bias, batch_size, labels, loss_fn, weights):
  """Creates and returns training op for centered bias."""
  with ops.name_scope(None, "centered_bias_step", (labels,)) as name:
    logits_dimension = array_ops.shape(centered_bias)[0]
    logits = array_ops.reshape(
        array_ops.tile(centered_bias, (batch_size,)),
        (batch_size, logits_dimension))
    with ops.name_scope(None, "centered_bias", (labels, logits)):
      centered_bias_loss = math_ops.reduce_mean(
          loss_fn(labels, logits, weights), name="training_loss")
  # Learn central bias by an optimizer. 0.1 is a convervative lr for a
  # single variable.
  return training.AdagradOptimizer(0.1).minimize(
      centered_bias_loss, var_list=(centered_bias,), name=name)


def _summary_key(head_name, val):
  return "%s/%s" % (val, head_name) if head_name else val


def _train_op(loss, labels, train_op_fn, centered_bias, batch_size, loss_fn,
              weights):
  """Returns op for the training step."""
  if centered_bias is not None:
    centered_bias_step = _centered_bias_step(
        centered_bias=centered_bias,
        batch_size=batch_size,
        labels=labels,
        loss_fn=loss_fn,
        weights=weights)
  else:
    centered_bias_step = None
  with ops.name_scope(None, "train_op", (loss, labels)):
    train_op = train_op_fn(loss)
    if centered_bias_step is not None:
      train_op = control_flow_ops.group(train_op, centered_bias_step)
    return train_op


def _sigmoid_cross_entropy_loss(labels, logits, weights=None):
  with ops.name_scope(None, "sigmoid_cross_entropy_loss",
                      (logits, labels)) as name:
    # sigmoid_cross_entropy_with_logits requires [batch_size, n_classes] labels.
    loss = nn.sigmoid_cross_entropy_with_logits(
        labels=math_ops.to_float(labels), logits=logits, name=name)
    return _compute_weighted_loss(loss, weights)


def _float_weights_or_none(weights):
  if weights is None:
    return None
  with ops.name_scope(None, "float_weights", (weights,)) as name:
    return math_ops.to_float(weights, name=name)


def _indicator_labels_streaming_mean(labels, weights=None, class_id=None):
  labels = math_ops.to_float(labels)
  weights = _float_weights_or_none(weights)
  if weights is not None:
    weights = weights_broadcast_ops.broadcast_weights(weights, labels)
  if class_id is not None:
    if weights is not None:
      weights = weights[:, class_id]
    labels = labels[:, class_id]
  return metrics_lib.mean(labels, weights)


def _predictions_streaming_mean(predictions,
                                weights=None,
                                class_id=None):
  predictions = math_ops.to_float(predictions)
  weights = _float_weights_or_none(weights)
  if weights is not None:
    weights = weights_broadcast_ops.broadcast_weights(weights, predictions)
  if class_id is not None:
    if weights is not None:
      weights = weights[:, class_id]
    predictions = predictions[:, class_id]
  return metrics_lib.mean(predictions, weights)


# TODO(ptucker): Add support for SparseTensor labels.
def _class_id_labels_to_indicator(labels, num_classes):
  if (num_classes is None) or (num_classes < 2):
    raise ValueError("Invalid num_classes %s." % num_classes)
  with ops.control_dependencies((_assert_labels_rank(labels),)):
    labels = array_ops.reshape(labels, (-1,))
  return array_ops.one_hot(labels, depth=num_classes, axis=-1)


def _class_predictions_streaming_mean(predictions, weights, class_id):
  return metrics_lib.mean(
      array_ops.where(
          math_ops.equal(
              math_ops.to_int32(class_id), math_ops.to_int32(predictions)),
          array_ops.ones_like(predictions),
          array_ops.zeros_like(predictions)),
      weights=weights)


def _class_labels_streaming_mean(labels, weights, class_id):
  return metrics_lib.mean(
      array_ops.where(
          math_ops.equal(
              math_ops.to_int32(class_id), math_ops.to_int32(labels)),
          array_ops.ones_like(labels), array_ops.zeros_like(labels)),
      weights=weights)


def _streaming_auc(predictions, labels, weights=None, class_id=None,
                   curve="ROC"):
  # pylint: disable=missing-docstring
  predictions = math_ops.to_float(predictions)
  if labels.dtype.base_dtype != dtypes.bool:
    logging.warning("Casting %s labels to bool.", labels.dtype)
    labels = math_ops.cast(labels, dtypes.bool)
  weights = _float_weights_or_none(weights)
  if weights is not None:
    weights = weights_broadcast_ops.broadcast_weights(weights, predictions)
  if class_id is not None:
    if weights is not None:
      weights = weights[:, class_id]
    predictions = predictions[:, class_id]
    labels = labels[:, class_id]
  return metrics_lib.auc(labels, predictions, weights, curve=curve)


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
  return metrics_lib.accuracy(labels, threshold_predictions, weights)


def _streaming_precision_at_threshold(predictions, labels, weights, threshold):
  precision_tensor, update_op = metrics_lib.precision_at_thresholds(
      labels, predictions, (threshold,), _float_weights_or_none(weights))
  return array_ops.squeeze(precision_tensor), array_ops.squeeze(update_op)


def _streaming_recall_at_threshold(predictions, labels, weights, threshold):
  precision_tensor, update_op = metrics_lib.recall_at_thresholds(
      labels, predictions, (threshold,), _float_weights_or_none(weights))
  return array_ops.squeeze(precision_tensor), array_ops.squeeze(update_op)


def _classification_output_alternatives(head_name, problem_type,
                                        label_keys=None):
  """Creates a func to generate output alternatives for classification.

  Servo expects classes to be a string tensor, and have the same dimensions
  as the probabilities tensor. It should contain the labels of the corresponding
  entries in probabilities. This function creates a new classes tensor that
  satisfies these conditions and can be exported.

  Args:
    head_name: Name of the head.
    problem_type: `ProblemType`
    label_keys: Optional label keys

  Returns:
    A function to generate output alternatives.
  """
  def _create_output_alternatives(predictions):
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

    Raises:
      ValueError: if predictions does not have PredictionKey.PROBABILITIES key.
    """
    probabilities = predictions.get(prediction_key.PredictionKey.PROBABILITIES)
    if probabilities is None:
      raise ValueError("%s missing in predictions" %
                       prediction_key.PredictionKey.PROBABILITIES)

    with ops.name_scope(None, "_classification_output_alternatives",
                        (probabilities,)):
      batch_size = array_ops.shape(probabilities)[0]
      if label_keys:
        classes = array_ops.tile(
            input=array_ops.expand_dims(input=label_keys, axis=0),
            multiples=[batch_size, 1],
            name="classes_tensor")
      else:
        n = array_ops.shape(probabilities)[1]
        classes = array_ops.tile(
            input=array_ops.expand_dims(input=math_ops.range(n), axis=0),
            multiples=[batch_size, 1])
        classes = string_ops.as_string(classes, name="classes_tensor")

    exported_predictions = {
        prediction_key.PredictionKey.PROBABILITIES: probabilities,
        prediction_key.PredictionKey.CLASSES: classes}
    return {head_name: (problem_type, exported_predictions)}

  return _create_output_alternatives

# Aliases
# TODO(zakaria): Remove these aliases, See b/34751732
_regression_head = regression_head
_poisson_regression_head = poisson_regression_head
_multi_class_head = multi_class_head
_binary_svm_head = binary_svm_head
_multi_label_head = multi_label_head
_multi_head = multi_head
_Head = Head
