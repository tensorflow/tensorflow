# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Abstractions for the head(s) of a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import six

from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator import util
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.export import export_output
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import nn
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.summary import summary
from tensorflow.python.training import training_util

_DEFAULT_SERVING_KEY = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

# The above default is defined by TF Serving, but these next three are just
# a local convention without any special meaning.
_CLASSIFY_SERVING_KEY = 'classification'
_REGRESS_SERVING_KEY = 'regression'
_PREDICT_SERVING_KEY = 'predict'


# A LossSpec contains
# * a scalar `Tensor` representing reduced weighted training loss
# * a `Tensor` representing the unreduced unweighted loss
# * a `Tensor` representing the example weights
# * possibly processed labels (e.g. vocabulary lookup, shape manipulation, etc)
LossSpec = collections.namedtuple(
    'LossSpec', ['training_loss', 'unreduced_loss', 'weights',
                 'processed_labels'])


def _summary_key(head_name, val):
  return '%s/%s' % (val, head_name) if head_name else val


class _Head(object):
  """Interface for the head/top of a model.

  Given logits (or output of a hidden layer), a Head knows how to compute
  predictions, loss, train_op, metrics and export outputs. It is meant to:

  1. Simplify writing model_fn and to make model_fn more configurable
  2. Support wide range of machine learning models. Since most heads can work
     with logits, they can support DNN, RNN, Wide, Wide&Deep,
     Global objectives, Gradient boosted trees and many other types
     of machine learning models.

  Common usage:
  Here is simplified model_fn to build a DNN regression model.
    ```python
    def _my_dnn_model_fn(features, labels, mode, params, config=None):
      # Optionally your callers can pass head to model_fn as a param.
      head = tf.contrib.estimator.regression_head(...)
      inputs = tf.feature_column.input_layer(features, ...)
      hidden_layer0 = tf.layers.dense(
          inputs, units=1000, activation=tf.nn.relu)
      hidden_layer1 = tf.layers.dense(
          hidden_layer0, units=500, activation=tf.nn.relu)
      logits = tf.layers.dense(
          hidden_layer1, units=head.logits_dimension, activation=None)

      return head.create_estimator_spec(
          features=features,
          labels=labels,
          mode=mode,
          logits=logits,
          optimizer=optimizer)
    ```

  There are cases where computing and applying gradients can not be meaningfully
  captured with optimizer or train_op_fn we support (for example, with sync
  optimizer). In such case, you can take the responsibility on your own. Here is
  a common use case,
    ```python
    estimator_spec = head.create_estimator_spec(
        features=features,
        labels=labels,
        mode=mode,
        logits=logits,
        train_op_fn=lambda _: tf.no_op())
    if mode == model_fn.ModeKeys.TRAIN:
      optimizer = ...
      sync = tf.train.SyncReplicasOptimizer(opt=optimizer, ...)
      update_op = sync.minimize(
          estimator_spec.loss, global_step=tf.get_global_step())
      hooks = [sync.make_session_run_hook(is_chief)]
      ... update train_op and hooks in EstimatorSpec and return
    ```
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def name(self):
    """The name of this head.

    Returns:
      A string.
    """
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractproperty
  def logits_dimension(self):
    """Size of the last dimension of the logits `Tensor`.

    Typically, logits is of shape `[batch_size, logits_dimension]`.

    Returns:
      The expected size of the `logits` tensor.
    """
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def create_loss(self, features, mode, logits, labels):
    """Returns a loss Tensor from provided logits.

    This function is designed to be used by framework developers.  Almost all
    users should use create_estimator_spec(), which calls this internally.
    `mode` and `features` are most likely not used, but some Head
    implementations may require them.

    Args:
      features: Input `dict` of `Tensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` to be used for loss construction.
      labels: Labels `Tensor`, or `dict` of same.

    Returns:
      A LossSpec that contains
      * the scalar `Tensor` representing reduced weighted training loss
      * the `Tensor` representing the unreduced unweighted loss
      * the `Tensor` representing the example weights
      * possibly processed labels (e.g. vocabulary lookup, shape manipulation,
        etc.)

      To be extendable in the future.
    """
    raise NotImplementedError('Calling an abstract method.')

  # TODO(b/65403806): By default, collect regularization_losses from
  # GraphKeys.REGULARIZATION_LOSSES collection.
  @abc.abstractmethod
  def create_estimator_spec(
      self, features, mode, logits, labels=None, optimizer=None,
      train_op_fn=None, regularization_losses=None):
    """Returns `EstimatorSpec` that a model_fn can return.

    Please note that,
    + All args must be passed via name.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` to be used by the head.
      labels: Labels `Tensor`, or `dict` of same.
      optimizer: `Optimizer` instance to optimize the loss in TRAIN mode.
        Namely, sets `train_op = optimizer.minimize(loss, global_step)`, which
        updates variables and increments `global_step`.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns an op
        to optimize the model with the loss in TRAIN mode. Used if `optimizer`
        is `None`. Exactly one of `train_op_fn` and `optimizer` must be set in
        TRAIN mode. None is allowed in other modes. If you want to optimize loss
        yourself you can pass `lambda _: tf.no_op()` and then use
        EstimatorSpec.loss to compute and apply gradients.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses.

    Returns:
      `EstimatorSpec`.
    """
    raise NotImplementedError('Calling an abstract method.')


def _check_dense_labels_match_logits_and_reshape(
    labels, logits, expected_labels_dimension):
  """Checks that labels shape matches logits and reshapes if needed.

  Consider logits of shape [D0, D1, ... DN, logits_dimension]. Then labels
  shape must be [D0, D1, ... DN, expected_labels_dimension].
  If expected_labels_dimension=1, labels could be [D0, D1, ... DN] and this
  method reshapes them to [D0, D1, ... DN, 1].

  Args:
    labels: labels Tensor.
    logits: logits Tensor.
    expected_labels_dimension: Integer.
  Returns:
    Validated and reshaped labels Tensor.
  Raises:
    ValueError: If labels is a SparseTensor.
    ValueError: If labels shape is statically defined and fails validation.
    OpError: If labels shape is not statically defined and fails validation.
  """
  if labels is None:
    raise ValueError(
        'You must provide a labels Tensor. Given: None. '
        'Suggested troubleshooting steps: Check that your data contain '
        'your label feature. Check that your input_fn properly parses and '
        'returns labels.')
  with ops.name_scope(None, 'labels', (labels, logits)) as scope:
    labels = sparse_tensor.convert_to_tensor_or_sparse_tensor(labels)
    if isinstance(labels, sparse_tensor.SparseTensor):
      raise ValueError(
          'SparseTensor labels are not supported. '
          'labels must be a Tensor of shape [D0, D1, ..., DN, %s], '
          'e.g. [batch_size, %s]. '
          'Suggested Fix (1): Check the label feature in your data. '
          'Each example must contain %s value(s). If not, your choice of label '
          'was probably incorrect. '
          'Suggested Fix (2): In your input_fn, use '
          'tf.sparse_tensor_to_dense() to turn labels into a Tensor.'
          '' % (expected_labels_dimension, expected_labels_dimension,
                expected_labels_dimension))
    if (labels.shape.ndims is not None and logits.shape.ndims is not None and
        labels.shape.ndims == logits.shape.ndims - 1):
      labels = array_ops.expand_dims(labels, -1)
    labels_shape = array_ops.shape(labels)
    logits_shape = array_ops.shape(logits)
    err_msg = (
        'labels shape must be [D0, D1, ... DN, {}]. '
        'Suggested Fix: check your n_classes argument to the estimator '
        'and/or the shape of your label.'.format(expected_labels_dimension))
    assert_rank = check_ops.assert_rank_at_least(labels, 2, message=err_msg)
    with ops.control_dependencies([assert_rank]):
      static_shape = labels.shape
      if static_shape.ndims is not None:
        dim1 = static_shape[-1]
        if (dim1 is not None) and (dim1 != expected_labels_dimension):
          raise ValueError(
              'Mismatched label shape. '
              'Expected labels dimension=%s.  Received %s. '
              'Suggested Fix:'
              'If your classifier expects one-hot encoding label,'
              'check your n_classes argument to the estimator'
              'and/or the shape of your label.'
              'Otherwise, check the shape of your label.' %
              (expected_labels_dimension, dim1))
      expected_labels_shape = array_ops.concat(
          [logits_shape[:-1], [expected_labels_dimension]], axis=0)
      assert_dimension = check_ops.assert_equal(
          expected_labels_shape, labels_shape, message=err_msg,
          data=['expected_labels_shape: ', expected_labels_shape,
                'labels_shape: ', labels_shape])
      with ops.control_dependencies([assert_dimension]):
        return array_ops.identity(labels, name=scope)


def _get_weights_and_check_match_logits(
    features, weight_column, logits, allow_per_logit_weights=False):
  """Fetches weights from features and checks that the shape matches logits.

  Consider logits of shape [D0, D1, ... DN, logits_dimension]. Weights shape
  can be either:
  * [D0, D1, ... DN, logits_dimension] if `allow_per_logit_weights=True`.
  * [D0, D1, ... DN, 1]
  * [D0, D1, ... DN]: In this case, weights is reshaped into
    [D0, D1, ... DN, 1] to work with weight broadcasting rules.

  Args:
    features: The features dict that contains weights.
    weight_column: The weight column. If not given, this method returns 1.
    logits: logits Tensor.
    allow_per_logit_weights: Boolean. Whether we allow weights along the logits
      dimension, namely shape `[D0, D1, ... DN, logits_dimension]`.
  Returns:
    Validated and reshaped weights Tensor.
  Raises:
    ValueError: If the weights `Tensor` cannot be cast into float.
  """
  if allow_per_logit_weights:
    err_msg = (
        'weights shape must be [D0, D1, ... DN], [D0, D1, ... DN, 1] or '
        '[D0, D1, ... DN, logits_dimension]')
  else:
    err_msg = (
        'weights shape must be [D0, D1, ... DN] or [D0, D1, ... DN, 1]')
  with ops.name_scope(
      None, 'weights',
      values=tuple(six.itervalues(features)) + (logits,)) as scope:
    # Fetch the weights.
    if weight_column is None:
      return 1.
    if isinstance(weight_column, six.string_types):
      weight_column = feature_column_lib.numeric_column(
          key=weight_column, shape=(1,))
    if not isinstance(weight_column, feature_column_lib._NumericColumn):  # pylint: disable=protected-access
      raise TypeError('Weight column must be either a string or _NumericColumn.'
                      ' Given type: {}.'.format(type(weight_column)))
    weights = weight_column._get_dense_tensor(  # pylint: disable=protected-access
        feature_column_lib._LazyBuilder(features))  # pylint: disable=protected-access
    if not (weights.dtype.is_floating or weights.dtype.is_integer):
      raise ValueError('Weight column should be castable to float. '
                       'Given dtype: {}'.format(weights.dtype))
    weights = math_ops.to_float(weights, name='weights')

    # Validate the weights shape.
    weights_shape = array_ops.shape(weights, name='weights_shape')
    logits_shape = array_ops.shape(logits, name='logits_shape')
    if (weights.shape.ndims is not None and logits.shape.ndims is not None and
        weights.shape.ndims == logits.shape.ndims - 1):
      assert_dimension = check_ops.assert_equal(
          logits_shape[:-1], weights_shape, message=err_msg,
          data=['logits_shape: ', logits_shape,
                'weights_shape: ', weights_shape])
      with ops.control_dependencies([assert_dimension]):
        return array_ops.expand_dims(weights, -1, name=scope)
    supported_weights_shape = array_ops.concat([logits_shape[:-1], [1]], axis=0)
    if allow_per_logit_weights:
      condition = math_ops.reduce_any(
          [math_ops.reduce_all(math_ops.equal(logits_shape, weights_shape)),
           math_ops.reduce_all(math_ops.equal(
               supported_weights_shape, weights_shape))])
      assert_dimension = control_flow_ops.Assert(
          condition=condition,
          data=[err_msg, 'logits_shape: ', logits_shape,
                'weights_shape: ', weights_shape])
    else:
      assert_dimension = check_ops.assert_equal(
          supported_weights_shape, weights_shape, message=err_msg,
          data=['logits_shape: ', logits_shape,
                'weights_shape: ', weights_shape])
    with ops.control_dependencies([assert_dimension]):
      return array_ops.identity(weights, name=scope)


def _check_logits_final_dim(logits, expected_logits_dimension):
  """Checks that logits shape is [D0, D1, ... DN, logits_dimension]."""
  with ops.name_scope(None, 'logits', (logits,)) as scope:
    logits = math_ops.to_float(logits)
    logits_shape = array_ops.shape(logits)
    assert_rank = check_ops.assert_rank_at_least(
        logits, 2, data=[logits_shape],
        message='logits shape must be [D0, D1, ... DN, logits_dimension]')
    with ops.control_dependencies([assert_rank]):
      static_shape = logits.shape
      if static_shape.ndims is not None and static_shape[-1] is not None:
        if static_shape[-1] != expected_logits_dimension:
          raise ValueError(
              'logits shape must be [D0, D1, ... DN, logits_dimension], '
              'got %s.' % (static_shape,))
        return logits
      assert_dimension = check_ops.assert_equal(
          expected_logits_dimension, logits_shape[-1], data=[logits_shape],
          message='logits shape must be [D0, D1, ... DN, logits_dimension]')
      with ops.control_dependencies([assert_dimension]):
        return array_ops.identity(logits, name=scope)


def _validate_loss_fn_args(loss_fn):
  """Validates loss_fn arguments.

  Required arguments: labels, logits.
  Optional arguments: features.

  Args:
    loss_fn: The loss function.
  Raises:
    ValueError: If the signature is unexpected.
  """
  loss_fn_args = util.fn_args(loss_fn)
  for required_arg in ['labels', 'logits']:
    if required_arg not in loss_fn_args:
      raise ValueError(
          'loss_fn must contain argument: {}. '
          'Given arguments: {}'.format(required_arg, loss_fn_args))
  invalid_args = list(set(loss_fn_args) - set(['labels', 'logits', 'features']))
  if invalid_args:
    raise ValueError('loss_fn has unexpected args: {}'.format(invalid_args))


def _call_loss_fn(loss_fn, labels, logits, features, expected_loss_dim=1):
  """Calls loss_fn and checks the returned shape.

  Args:
    loss_fn: The loss function.
    labels: Processed labels Tensor.
    logits: Logits Tensor of shape [D0, D1, ... DN, logits_dimension].
    features: Features dict.
    expected_loss_dim: The expected last dimension of loss Tensor.
  Returns:
    Loss Tensor with shape [D0, D1, ... DN, expected_loss_dim].
  """
  loss_fn_args = util.fn_args(loss_fn)
  kwargs = {}
  if 'features' in loss_fn_args:
    kwargs['features'] = features
  with ops.name_scope(
      None, 'call_loss_fn',
      values=[labels, logits] + list(six.itervalues(features))):
    unweighted_loss = loss_fn(labels=labels, logits=logits, **kwargs)
    logits_shape = array_ops.shape(logits, name='logits_shape')
    expected_loss_shape = array_ops.concat(
        [logits_shape[:-1], [expected_loss_dim]], axis=0,
        name='expected_loss_shape')
    loss_shape = array_ops.shape(unweighted_loss, name='loss_shape')
    check_loss_shape_op = control_flow_ops.Assert(
        math_ops.reduce_all(math_ops.equal(loss_shape, expected_loss_shape)),
        data=[
            'loss_fn must return Tensor of shape '
            '[D0, D1, ... DN, {}]. '.format(expected_loss_dim),
            'logits_shape: ', logits_shape, 'loss_shape: ', loss_shape],
        name='check_loss_shape')
    with ops.control_dependencies([check_loss_shape_op]):
      return array_ops.identity(unweighted_loss)


def _indicator_labels_mean(labels, weights=None, name=None):
  with ops.name_scope(name, 'labels_mean', (labels, weights)) as scope:
    labels = math_ops.to_float(labels, name='labels')
    if weights is not None:
      weights = weights_broadcast_ops.broadcast_weights(weights, labels)
    return metrics_lib.mean(labels, weights=weights, name=scope)


def _classification_output(scores, n_classes, label_vocabulary=None):
  batch_size = array_ops.shape(scores)[0]
  if label_vocabulary:
    export_class_list = label_vocabulary
  else:
    export_class_list = string_ops.as_string(math_ops.range(n_classes))
  export_output_classes = array_ops.tile(
      input=array_ops.expand_dims(input=export_class_list, axis=0),
      multiples=[batch_size, 1])
  return export_output.ClassificationOutput(
      scores=scores,
      # `ClassificationOutput` requires string classes.
      classes=export_output_classes)


def _accuracy_baseline(labels_mean):
  """Return accuracy baseline based on labels mean.

  This is the best the model could do by always predicting one class.

  Args:
    labels_mean: Tuple of value and update op.

  Returns:
    Tuple of value and update op.
  """
  with ops.name_scope(None, 'accuracy_baseline', labels_mean):
    value, update_op = labels_mean
    return (
        math_ops.maximum(value, 1. - value, name='value'),
        math_ops.maximum(update_op, 1 - update_op, name='update_op'))


def _predictions_mean(predictions, weights=None, name=None):
  with ops.name_scope(
      name, 'predictions_mean', (predictions, weights)) as scope:
    predictions = math_ops.to_float(predictions, name='predictions')
    if weights is not None:
      weights = weights_broadcast_ops.broadcast_weights(weights, predictions)
    return metrics_lib.mean(predictions, weights=weights, name=scope)


def _auc(labels, predictions, weights=None, curve='ROC', name=None):
  with ops.name_scope(name, 'auc', (predictions, labels, weights)) as scope:
    predictions = math_ops.to_float(predictions, name='predictions')
    if weights is not None:
      weights = weights_broadcast_ops.broadcast_weights(weights, predictions)
    return metrics_lib.auc(
        labels=labels, predictions=predictions, weights=weights, curve=curve,
        name=scope)


def _accuracy_at_threshold(labels, predictions, weights, threshold, name=None):
  with ops.name_scope(
      name, 'accuracy_at_%s' % threshold,
      (predictions, labels, weights, threshold)) as scope:
    threshold_predictions = math_ops.to_float(
        math_ops.greater_equal(predictions, threshold))
    return metrics_lib.accuracy(
        labels=labels, predictions=threshold_predictions, weights=weights,
        name=scope)


def _precision_at_threshold(labels, predictions, weights, threshold, name=None):
  with ops.name_scope(
      name, 'precision_at_%s' % threshold,
      (predictions, labels, weights, threshold)) as scope:
    precision_tensor, update_op = metrics_lib.precision_at_thresholds(
        labels=labels, predictions=predictions, thresholds=(threshold,),
        weights=weights, name=scope)
    return array_ops.squeeze(precision_tensor), array_ops.squeeze(update_op)


def _recall_at_threshold(labels, predictions, weights, threshold, name=None):
  with ops.name_scope(
      name, 'recall_at_%s' % threshold,
      (predictions, labels, weights, threshold)) as scope:
    precision_tensor, update_op = metrics_lib.recall_at_thresholds(
        labels=labels, predictions=predictions, thresholds=(threshold,),
        weights=weights, name=scope)
    return array_ops.squeeze(precision_tensor), array_ops.squeeze(update_op)


def _multi_class_head_with_softmax_cross_entropy_loss(
    n_classes,
    weight_column=None,
    label_vocabulary=None,
    loss_reduction=losses.Reduction.SUM,
    loss_fn=None,
    name=None):
  """Creates a '_Head' for multi class classification.

  The head expects `logits` with shape `[D0, D1, ... DN, n_classes]`.
  In many applications, the shape is `[batch_size, n_classes]`.

  `labels` must be a dense `Tensor` with shape matching `logits`, namely
  `[D0, D1, ... DN, 1]`. If `label_vocabulary` given, `labels` must be a string
  `Tensor` with values from the vocabulary. If `label_vocabulary` is not given,
  `labels` must be an integer `Tensor` with values specifying the class index.

  If `weight_column` is specified, weights must be of shape
  `[D0, D1, ... DN]`, or `[D0, D1, ... DN, 1]`.

  The loss is the weighted sum over the input dimensions. Namely, if the input
  labels have shape `[batch_size, 1]`, the loss is the weighted sum over
  `batch_size`.

  Also supports custom `loss_fn`. `loss_fn` takes `(labels, logits)` or
  `(labels, logits, features)` as arguments and returns unreduced loss with
  shape `[D0, D1, ... DN, 1]`. `loss_fn` must support integer `labels` with
  shape `[D0, D1, ... DN, 1]`. Namely, the head applies `label_vocabulary` to
  the input labels before passing them to `loss_fn`.

  Args:
    n_classes: Number of classes, must be greater than 2 (for 2 classes, use
      `_BinaryLogisticHeadWithSigmoidCrossEntropyLoss`).
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    label_vocabulary: A list or tuple of strings representing possible label
      values. If it is not given, that means labels are already encoded as an
      integer within [0, n_classes). If given, labels must be of string type and
      have any value in `label_vocabulary`. Note that errors will be raised if
      `label_vocabulary` is not provided but labels are strings.
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch. Defaults to `SUM`.
    loss_fn: Optional loss function.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + name`. Also used as `name_scope` when creating ops.

  Returns:
    An instance of `_Head` for multi class classification.

  Raises:
    ValueError: If `n_classes`, `label_vocabulary` or `loss_reduction` is
      invalid.
  """
  if label_vocabulary is not None and not isinstance(label_vocabulary,
                                                     (list, tuple)):
    raise ValueError(
        'label_vocabulary should be a list or a tuple. Given type: {}'.format(
            type(label_vocabulary)))
  if (loss_reduction not in losses.Reduction.all() or
      loss_reduction == losses.Reduction.NONE):
    raise ValueError('Invalid loss_reduction: {}'.format(loss_reduction))
  if loss_fn:
    _validate_loss_fn_args(loss_fn)
  return _MultiClassHeadWithSoftmaxCrossEntropyLoss(
      n_classes=n_classes,
      weight_column=weight_column,
      label_vocabulary=label_vocabulary,
      loss_reduction=loss_reduction,
      loss_fn=loss_fn,
      name=name)


class _MultiClassHeadWithSoftmaxCrossEntropyLoss(_Head):
  """See `_multi_class_head_with_softmax_cross_entropy_loss`."""

  def __init__(self,
               n_classes,
               weight_column=None,
               label_vocabulary=None,
               loss_reduction=losses.Reduction.SUM,
               loss_fn=None,
               name=None):
    if (n_classes is None) or (n_classes <= 2):
      raise ValueError('n_classes must be > 2: %s.' % n_classes)
    self._n_classes = n_classes
    self._weight_column = weight_column
    self._label_vocabulary = label_vocabulary
    self._loss_reduction = loss_reduction
    self._loss_fn = loss_fn
    self._name = name

  @property
  def name(self):
    return self._name

  @property
  def logits_dimension(self):
    return self._n_classes

  def _eval_metric_ops(
      self, labels, class_ids, weights, unreduced_loss, regularization_loss):
    """Returns the Eval metric ops."""
    with ops.name_scope(
        None, 'metrics',
        (labels, class_ids, weights, unreduced_loss, regularization_loss)):
      keys = metric_keys.MetricKeys
      metric_ops = {
          # Estimator already adds a metric for loss.
          # TODO(xiejw): Any other metrics?
          _summary_key(self._name, keys.LOSS_MEAN):
              metrics_lib.mean(
                  values=unreduced_loss,
                  weights=weights,
                  name=keys.LOSS_MEAN),
          _summary_key(self._name, keys.ACCURACY):
              metrics_lib.accuracy(
                  labels=labels,
                  predictions=class_ids,
                  weights=weights,
                  name=keys.ACCURACY),
      }
      if regularization_loss is not None:
        metric_ops[_summary_key(self._name, keys.LOSS_REGULARIZATION)] = (
            metrics_lib.mean(
                values=regularization_loss,
                name=keys.LOSS_REGULARIZATION))
    return metric_ops

  def _label_ids(self, labels):
    """Converts labels to integer id space."""
    if self._label_vocabulary is None:
      if not labels.dtype.is_integer:
        raise ValueError('Labels dtype should be integer. Instead got {}.'.
                         format(labels.dtype))
      label_ids = labels
    else:
      if labels.dtype != dtypes.string:
        raise ValueError('Labels dtype should be string if there is a '
                         'vocabulary. Instead got {}'.format(labels.dtype))
      label_ids = lookup_ops.index_table_from_tensor(
          vocabulary_list=tuple(self._label_vocabulary),
          name='class_id_lookup').lookup(labels)
    return _assert_range(label_ids, self._n_classes)

  def create_loss(self, features, mode, logits, labels):
    """See `Head`."""
    del mode  # Unused for this head.
    logits = ops.convert_to_tensor(logits)
    labels = _check_dense_labels_match_logits_and_reshape(
        labels=labels, logits=logits, expected_labels_dimension=1)
    label_ids = self._label_ids(labels)
    if self._loss_fn:
      unweighted_loss = _call_loss_fn(
          loss_fn=self._loss_fn, labels=label_ids, logits=logits,
          features=features, expected_loss_dim=1)
    else:
      unweighted_loss = losses.sparse_softmax_cross_entropy(
          labels=label_ids, logits=logits, reduction=losses.Reduction.NONE)
      # Restore the squeezed dim, so unweighted_loss matches the weights shape.
      unweighted_loss = array_ops.expand_dims(unweighted_loss, axis=-1)
    weights = _get_weights_and_check_match_logits(
        features=features, weight_column=self._weight_column, logits=logits)
    training_loss = losses.compute_weighted_loss(
        unweighted_loss, weights=weights, reduction=self._loss_reduction)
    return LossSpec(
        training_loss=training_loss,
        unreduced_loss=unweighted_loss,
        weights=weights,
        processed_labels=label_ids)

  def create_estimator_spec(
      self, features, mode, logits, labels=None, optimizer=None,
      train_op_fn=None, regularization_losses=None):
    """Returns an `EstimatorSpec`.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` with shape `[D0, D1, ... DN, logits_dimension]`.
        For many applications, the shape is `[batch_size, logits_dimension]`.
      labels: Labels integer or string `Tensor` with shape matching `logits`,
        namely `[D0, D1, ... DN, 1]` or `[D0, D1, ... DN]`. `labels` is
        required argument when `mode` equals `TRAIN` or `EVAL`.
      optimizer: `Optimizer` instance to optimize the loss in TRAIN mode.
        Namely, sets `train_op = optimizer.minimize(loss, global_step)`, which
        updates variables and increments `global_step`.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns
        `train_op`. Used if `optimizer` is `None`.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses. These losses are
        usually expressed as a batch average, so for best results users need to
        set `loss_reduction=SUM_OVER_BATCH_SIZE` or
        `loss_reduction=SUM_OVER_NONZERO_WEIGHTS` when creating the head to
        avoid scaling errors.
    Returns:
      `EstimatorSpec`.
    Raises:
      ValueError: If both `train_op_fn` and `optimizer` are `None` in TRAIN
        mode, or if both are set.
    """
    with ops.name_scope(self._name, 'head'):
      logits = _check_logits_final_dim(logits, self.logits_dimension)

      # Predict.
      pred_keys = prediction_keys.PredictionKeys
      with ops.name_scope(None, 'predictions', (logits,)):
        # class_ids's shape is [D0, D1, ... DN].
        class_ids = math_ops.argmax(logits, axis=-1, name=pred_keys.CLASS_IDS)
        class_ids = array_ops.expand_dims(class_ids, axis=-1)
        if self._label_vocabulary:
          table = lookup_ops.index_to_string_table_from_tensor(
              vocabulary_list=self._label_vocabulary,
              name='class_string_lookup')
          classes = table.lookup(class_ids)
        else:
          classes = string_ops.as_string(class_ids, name='str_classes')

        probabilities = nn.softmax(logits, name=pred_keys.PROBABILITIES)
        predictions = {
            pred_keys.LOGITS: logits,
            pred_keys.PROBABILITIES: probabilities,
            # Expand to [batch_size, 1]
            pred_keys.CLASS_IDS: class_ids,
            pred_keys.CLASSES: classes,
        }
      if mode == model_fn.ModeKeys.PREDICT:
        classifier_output = _classification_output(
            scores=probabilities, n_classes=self._n_classes,
            label_vocabulary=self._label_vocabulary)
        return model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                _DEFAULT_SERVING_KEY: classifier_output,
                _CLASSIFY_SERVING_KEY: classifier_output,
                _PREDICT_SERVING_KEY: export_output.PredictOutput(predictions)
            })

      training_loss, unreduced_loss, weights, label_ids = self.create_loss(
          features=features, mode=mode, logits=logits, labels=labels)
      if regularization_losses:
        regularization_loss = math_ops.add_n(regularization_losses)
        regularized_training_loss = math_ops.add_n(
            [training_loss, regularization_loss])
      else:
        regularization_loss = None
        regularized_training_loss = training_loss
      # Eval.
      if mode == model_fn.ModeKeys.EVAL:
        return model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions=predictions,
            loss=regularized_training_loss,
            eval_metric_ops=self._eval_metric_ops(
                labels=label_ids,
                class_ids=class_ids,
                weights=weights,
                unreduced_loss=unreduced_loss,
                regularization_loss=regularization_loss))

      # Train.
      if optimizer is not None:
        if train_op_fn is not None:
          raise ValueError('train_op_fn and optimizer cannot both be set.')
        train_op = optimizer.minimize(
            regularized_training_loss,
            global_step=training_util.get_global_step())
      elif train_op_fn is not None:
        train_op = train_op_fn(regularized_training_loss)
      else:
        raise ValueError('train_op_fn and optimizer cannot both be None.')
      # Only summarize mean_loss for SUM reduction to preserve backwards
      # compatibility. Otherwise skip it to avoid unnecessary computation.
      if self._loss_reduction == losses.Reduction.SUM:
        example_weight_sum = math_ops.reduce_sum(
            weights * array_ops.ones_like(unreduced_loss))
        mean_loss = training_loss / example_weight_sum
      else:
        mean_loss = None
    with ops.name_scope(''):
      keys = metric_keys.MetricKeys
      summary.scalar(
          _summary_key(self._name, keys.LOSS),
          regularized_training_loss)
      if mean_loss is not None:
        summary.scalar(
            _summary_key(self._name, keys.LOSS_MEAN),
            mean_loss)
      if regularization_loss is not None:
        summary.scalar(
            _summary_key(self._name, keys.LOSS_REGULARIZATION),
            regularization_loss)
    return model_fn.EstimatorSpec(
        mode=model_fn.ModeKeys.TRAIN,
        predictions=predictions,
        loss=regularized_training_loss,
        train_op=train_op)


def _binary_logistic_head_with_sigmoid_cross_entropy_loss(
    weight_column=None,
    thresholds=None,
    label_vocabulary=None,
    loss_reduction=losses.Reduction.SUM,
    loss_fn=None,
    name=None):
  """Creates a `_Head` for single label binary classification.

  This head uses `sigmoid_cross_entropy_with_logits` loss.

  The head expects `logits` with shape `[D0, D1, ... DN, 1]`.
  In many applications, the shape is `[batch_size, 1]`.

  `labels` must be a dense `Tensor` with shape matching `logits`, namely
  `[D0, D1, ... DN, 1]`. If `label_vocabulary` given, `labels` must be a string
  `Tensor` with values from the vocabulary. If `label_vocabulary` is not given,
  `labels` must be float `Tensor` with values in the interval `[0, 1]`.

  If `weight_column` is specified, weights must be of shape
  `[D0, D1, ... DN]`, or `[D0, D1, ... DN, 1]`.

  The loss is the weighted sum over the input dimensions. Namely, if the input
  labels have shape `[batch_size, 1]`, the loss is the weighted sum over
  `batch_size`.

  Also supports custom `loss_fn`. `loss_fn` takes `(labels, logits)` or
  `(labels, logits, features)` as arguments and returns unreduced loss with
  shape `[D0, D1, ... DN, 1]`. `loss_fn` must support float `labels` with
  shape `[D0, D1, ... DN, 1]`. Namely, the head applies `label_vocabulary` to
  the input labels before passing them to `loss_fn`.

  Args:
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    thresholds: Iterable of floats in the range `(0, 1)`. For binary
      classification metrics such as precision and recall, an eval metric is
      generated for each threshold value. This threshold is applied to the
      logistic values to determine the binary classification (i.e., above the
      threshold is `true`, below is `false`.
    label_vocabulary: A list or tuple of strings representing possible label
      values. If it is not given, that means labels are already encoded within
      [0, 1]. If given, labels must be string type and have any value in
      `label_vocabulary`. Note that errors will be raised if `label_vocabulary`
      is not provided but labels are strings.
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch. Defaults to `SUM`.
    loss_fn: Optional loss function.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + name`. Also used as `name_scope` when creating ops.

  Returns:
    An instance of `_Head` for binary classification.

  Raises:
    ValueError: If `thresholds` contains a value outside of `(0, 1)`.
    ValueError: If `loss_reduction` is invalid.
    TypeError: if `label_vocabulary` has invalid type.
  """
  thresholds = tuple(thresholds) if thresholds else tuple()
  if label_vocabulary is not None and not isinstance(label_vocabulary,
                                                     (list, tuple)):
    raise TypeError(
        'label_vocabulary should be a list or tuple. Given type: {}'.format(
            type(label_vocabulary)))

  for threshold in thresholds:
    if (threshold <= 0.0) or (threshold >= 1.0):
      raise ValueError('thresholds not in (0, 1): {}.'.format((thresholds,)))
  if (loss_reduction not in losses.Reduction.all() or
      loss_reduction == losses.Reduction.NONE):
    raise ValueError('Invalid loss_reduction: {}'.format(loss_reduction))
  if loss_fn:
    _validate_loss_fn_args(loss_fn)
  return _BinaryLogisticHeadWithSigmoidCrossEntropyLoss(
      weight_column=weight_column,
      thresholds=thresholds,
      label_vocabulary=label_vocabulary,
      loss_reduction=loss_reduction,
      loss_fn=loss_fn,
      name=name)


class _BinaryLogisticHeadWithSigmoidCrossEntropyLoss(_Head):
  """See `_binary_logistic_head_with_sigmoid_cross_entropy_loss`."""

  def __init__(self,
               weight_column=None,
               thresholds=None,
               label_vocabulary=None,
               loss_reduction=losses.Reduction.SUM,
               loss_fn=None,
               name=None):
    self._weight_column = weight_column
    self._thresholds = thresholds
    self._label_vocabulary = label_vocabulary
    self._loss_reduction = loss_reduction
    self._loss_fn = loss_fn
    self._name = name

  @property
  def name(self):
    return self._name

  @property
  def logits_dimension(self):
    return 1

  def _eval_metric_ops(self, labels, logits, logistic, class_ids, weights,
                       unreduced_loss, regularization_loss):
    with ops.name_scope(None, 'metrics',
                        (labels, logits, logistic, class_ids, weights,
                         unreduced_loss, regularization_loss)):
      keys = metric_keys.MetricKeys
      labels_mean = _indicator_labels_mean(
          labels=labels, weights=weights, name=keys.LABEL_MEAN)
      metric_ops = {
          # Estimator already adds a metric for loss.
          _summary_key(self._name, keys.LOSS_MEAN):
              metrics_lib.mean(
                  values=unreduced_loss,
                  weights=weights,
                  name=keys.LOSS_MEAN),
          _summary_key(self._name, keys.ACCURACY):
              metrics_lib.accuracy(
                  labels=labels,
                  predictions=class_ids,
                  weights=weights,
                  name=keys.ACCURACY),
          _summary_key(self._name, keys.PRECISION):
              metrics_lib.precision(
                  labels=labels,
                  predictions=class_ids,
                  weights=weights,
                  name=keys.PRECISION),
          _summary_key(self._name, keys.RECALL):
              metrics_lib.recall(
                  labels=labels,
                  predictions=class_ids,
                  weights=weights,
                  name=keys.RECALL),
          _summary_key(self._name, keys.PREDICTION_MEAN):
              _predictions_mean(
                  predictions=logistic,
                  weights=weights,
                  name=keys.PREDICTION_MEAN),
          _summary_key(self._name, keys.LABEL_MEAN):
              labels_mean,
          _summary_key(self._name, keys.ACCURACY_BASELINE):
              _accuracy_baseline(labels_mean),
          _summary_key(self._name, keys.AUC):
              _auc(
                  labels=labels,
                  predictions=logistic,
                  weights=weights,
                  name=keys.AUC),
          _summary_key(self._name, keys.AUC_PR):
              _auc(
                  labels=labels,
                  predictions=logistic,
                  weights=weights,
                  curve='PR',
                  name=keys.AUC_PR)
      }
      if regularization_loss is not None:
        metric_ops[_summary_key(self._name, keys.LOSS_REGULARIZATION)] = (
            metrics_lib.mean(
                values=regularization_loss,
                name=keys.LOSS_REGULARIZATION))
      for threshold in self._thresholds:
        accuracy_key = keys.ACCURACY_AT_THRESHOLD % threshold
        metric_ops[_summary_key(self._name,
                                accuracy_key)] = _accuracy_at_threshold(
                                    labels=labels,
                                    predictions=logistic,
                                    weights=weights,
                                    threshold=threshold,
                                    name=accuracy_key)
        # Precision for positive examples.
        precision_key = keys.PRECISION_AT_THRESHOLD % threshold
        metric_ops[_summary_key(self._name,
                                precision_key)] = _precision_at_threshold(
                                    labels=labels,
                                    predictions=logistic,
                                    weights=weights,
                                    threshold=threshold,
                                    name=precision_key)
        # Recall for positive examples.
        recall_key = keys.RECALL_AT_THRESHOLD % threshold
        metric_ops[_summary_key(self._name,
                                recall_key)] = _recall_at_threshold(
                                    labels=labels,
                                    predictions=logistic,
                                    weights=weights,
                                    threshold=threshold,
                                    name=recall_key)
      return metric_ops

  def create_loss(self, features, mode, logits, labels):
    """See `Head`."""
    del mode  # Unused for this head.
    logits = ops.convert_to_tensor(logits)
    labels = _check_dense_labels_match_logits_and_reshape(
        labels=labels, logits=logits, expected_labels_dimension=1)
    if self._label_vocabulary is not None:
      labels = lookup_ops.index_table_from_tensor(
          vocabulary_list=tuple(self._label_vocabulary),
          name='class_id_lookup').lookup(labels)
    labels = math_ops.to_float(labels)
    labels = _assert_range(labels, 2)
    if self._loss_fn:
      unweighted_loss = _call_loss_fn(
          loss_fn=self._loss_fn, labels=labels, logits=logits,
          features=features, expected_loss_dim=1)
    else:
      unweighted_loss = nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)
    weights = _get_weights_and_check_match_logits(
        features=features, weight_column=self._weight_column, logits=logits)
    training_loss = losses.compute_weighted_loss(
        unweighted_loss, weights=weights, reduction=self._loss_reduction)
    return LossSpec(
        training_loss=training_loss,
        unreduced_loss=unweighted_loss,
        weights=weights,
        processed_labels=labels)

  def create_estimator_spec(
      self, features, mode, logits, labels=None, optimizer=None,
      train_op_fn=None, regularization_losses=None):
    """Returns an `EstimatorSpec`.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` with shape `[D0, D1, ... DN, 1]`. For many
        applications, the shape is `[batch_size, 1]`.
      labels: Labels integer or string `Tensor` with shape matching `logits`,
        namely `[D0, D1, ... DN, 1]` or `[D0, D1, ... DN]`. `labels` is required
        argument when `mode` equals `TRAIN` or `EVAL`.
      optimizer: `Optimizer` instance to optimize the loss in TRAIN mode.
        Namely, sets `train_op = optimizer.minimize(loss, global_step)`, which
        updates variables and increments `global_step`.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns
        `train_op`. Used if `optimizer` is `None`.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses. These losses are
        usually expressed as a batch average, so for best results users need to
        set `loss_reduction=SUM_OVER_BATCH_SIZE` or
        `loss_reduction=SUM_OVER_NONZERO_WEIGHTS` when creating the head to
        avoid scaling errors.
    Returns:
      `EstimatorSpec`.
    Raises:
      ValueError: If both `train_op_fn` and `optimizer` are `None` in TRAIN
        mode, or if both are set.
    """
    # Predict.
    with ops.name_scope(self._name, 'head'):
      with ops.name_scope(None, 'predictions', (logits,)):
        pred_keys = prediction_keys.PredictionKeys
        logits = _check_logits_final_dim(logits, self.logits_dimension)
        logistic = math_ops.sigmoid(logits, name=pred_keys.LOGISTIC)
        two_class_logits = array_ops.concat(
            (array_ops.zeros_like(logits), logits),
            axis=-1, name='two_class_logits')
        probabilities = nn.softmax(
            two_class_logits, name=pred_keys.PROBABILITIES)
        class_ids = math_ops.argmax(
            two_class_logits, axis=-1, name=pred_keys.CLASS_IDS)
        class_ids = array_ops.expand_dims(class_ids, axis=-1)
        if self._label_vocabulary:
          table = lookup_ops.index_to_string_table_from_tensor(
              vocabulary_list=self._label_vocabulary,
              name='class_string_lookup')
          classes = table.lookup(class_ids)
        else:
          classes = string_ops.as_string(class_ids, name='str_classes')
        predictions = {
            pred_keys.LOGITS: logits,
            pred_keys.LOGISTIC: logistic,
            pred_keys.PROBABILITIES: probabilities,
            pred_keys.CLASS_IDS: class_ids,
            pred_keys.CLASSES: classes,
        }
      if mode == model_fn.ModeKeys.PREDICT:
        classifier_output = _classification_output(
            scores=probabilities, n_classes=2,
            label_vocabulary=self._label_vocabulary)
        return model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                _DEFAULT_SERVING_KEY: classifier_output,
                _CLASSIFY_SERVING_KEY: classifier_output,
                _REGRESS_SERVING_KEY: export_output.RegressionOutput(
                    value=logistic),
                _PREDICT_SERVING_KEY: export_output.PredictOutput(predictions)
            })

      (training_loss, unreduced_loss, weights, processed_labels) = (
          self.create_loss(
              features=features, mode=mode, logits=logits, labels=labels))
      if regularization_losses:
        regularization_loss = math_ops.add_n(regularization_losses)
        regularized_training_loss = math_ops.add_n(
            [training_loss, regularization_loss])
      else:
        regularization_loss = None
        regularized_training_loss = training_loss

      # Eval.
      if mode == model_fn.ModeKeys.EVAL:
        return model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions=predictions,
            loss=regularized_training_loss,
            eval_metric_ops=self._eval_metric_ops(
                labels=processed_labels,
                logits=logits,
                logistic=logistic,
                class_ids=class_ids,
                weights=weights,
                unreduced_loss=unreduced_loss,
                regularization_loss=regularization_loss))

      # Train.
      if optimizer is not None:
        if train_op_fn is not None:
          raise ValueError('train_op_fn and optimizer cannot both be set.')
        train_op = optimizer.minimize(
            regularized_training_loss,
            global_step=training_util.get_global_step())
      elif train_op_fn is not None:
        train_op = train_op_fn(regularized_training_loss)
      else:
        raise ValueError('train_op_fn and optimizer cannot both be None.')
      # Only summarize mean_loss for SUM reduction to preserve backwards
      # compatibility. Otherwise skip it to avoid unnecessary computation.
      if self._loss_reduction == losses.Reduction.SUM:
        example_weight_sum = math_ops.reduce_sum(
            weights * array_ops.ones_like(unreduced_loss))
        mean_loss = training_loss / example_weight_sum
      else:
        mean_loss = None
    with ops.name_scope(''):
      keys = metric_keys.MetricKeys
      summary.scalar(
          _summary_key(self._name, keys.LOSS),
          regularized_training_loss)
      if mean_loss is not None:
        summary.scalar(
            _summary_key(self._name, keys.LOSS_MEAN), mean_loss)
      if regularization_loss is not None:
        summary.scalar(
            _summary_key(self._name, keys.LOSS_REGULARIZATION),
            regularization_loss)
    return model_fn.EstimatorSpec(
        mode=model_fn.ModeKeys.TRAIN,
        predictions=predictions,
        loss=regularized_training_loss,
        train_op=train_op)


def _regression_head_with_mean_squared_error_loss(
    weight_column=None,
    label_dimension=1,
    loss_reduction=losses.Reduction.SUM,
    loss_fn=None,
    inverse_link_fn=None,
    name=None):
  """Creates a `_Head` for regression using the `mean_squared_error` loss.

  The loss is the weighted sum over all input dimensions. Namely, if the input
  labels have shape `[batch_size, label_dimension]`, the loss is the weighted
  sum over both `batch_size` and `label_dimension`.

  The head expects `logits` with shape `[D0, D1, ... DN, label_dimension]`.
  In many applications, the shape is `[batch_size, label_dimension]`.

  The `labels` shape must match `logits`, namely
  `[D0, D1, ... DN, label_dimension]`. If `label_dimension=1`, shape
  `[D0, D1, ... DN]` is also supported.

  If `weight_column` is specified, weights must be of shape
  `[D0, D1, ... DN]`, `[D0, D1, ... DN, 1]` or
  `[D0, D1, ... DN, label_dimension]`.

  Supports custom `loss_fn`. `loss_fn` takes `(labels, logits)` or
  `(labels, logits, features)` as arguments and returns unreduced loss with
  shape `[D0, D1, ... DN, label_dimension]`.

  Also supports custom `inverse_link_fn`, also known as 'mean function'.
  `inverse_link_fn` takes `logits` as argument and returns predicted values.
  This function is the inverse of the link function defined in
  https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function
  Namely, for poisson regression, set `inverse_link_fn=tf.exp`.

  Args:
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    label_dimension: Number of regression labels per example. This is the size
      of the last dimension of the labels `Tensor` (typically, this has shape
      `[batch_size, label_dimension]`).
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch. Defaults to `SUM`.
    loss_fn: Optional loss function. Defaults to `mean_squared_error`.
    inverse_link_fn: Optional inverse link function, also known as 'mean
      function'. Defaults to identity.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + name`. Also used as `name_scope` when creating ops.

  Returns:
    An instance of `_Head` for linear regression.

  Raises:
    ValueError: If `label_dimension` or `loss_reduction` is invalid.
  """
  if (loss_reduction not in losses.Reduction.all() or
      loss_reduction == losses.Reduction.NONE):
    raise ValueError('Invalid loss_reduction: {}'.format(loss_reduction))
  if loss_fn:
    _validate_loss_fn_args(loss_fn)
  return _RegressionHeadWithMeanSquaredErrorLoss(
      weight_column=weight_column,
      label_dimension=label_dimension,
      loss_reduction=loss_reduction,
      loss_fn=loss_fn,
      inverse_link_fn=inverse_link_fn,
      name=name)


class _RegressionHeadWithMeanSquaredErrorLoss(_Head):
  """`Head` for regression using the mean squared loss."""

  def __init__(
      self,
      label_dimension,
      weight_column=None,
      loss_reduction=losses.Reduction.SUM,
      loss_fn=None,
      inverse_link_fn=None,
      name=None):
    """`Head` for regression."""
    if label_dimension < 1:
      raise ValueError('Invalid label_dimension %s.' % label_dimension)
    self._logits_dimension = label_dimension
    self._weight_column = weight_column
    self._loss_reduction = loss_reduction
    self._loss_fn = loss_fn
    self._inverse_link_fn = inverse_link_fn
    self._name = name

  @property
  def name(self):
    return self._name

  @property
  def logits_dimension(self):
    return self._logits_dimension

  def create_loss(self, features, mode, logits, labels):
    """See `Head`."""
    del mode  # Unused for this head.
    logits = ops.convert_to_tensor(logits)
    labels = _check_dense_labels_match_logits_and_reshape(
        labels=labels, logits=logits,
        expected_labels_dimension=self._logits_dimension)
    labels = math_ops.to_float(labels)
    if self._loss_fn:
      unweighted_loss = _call_loss_fn(
          loss_fn=self._loss_fn, labels=labels, logits=logits,
          features=features, expected_loss_dim=self._logits_dimension)
    else:
      unweighted_loss = losses.mean_squared_error(
          labels=labels, predictions=logits, reduction=losses.Reduction.NONE)
    weights = _get_weights_and_check_match_logits(
        features=features, weight_column=self._weight_column, logits=logits,
        allow_per_logit_weights=True)
    training_loss = losses.compute_weighted_loss(
        unweighted_loss, weights=weights, reduction=self._loss_reduction)
    return LossSpec(
        training_loss=training_loss,
        unreduced_loss=unweighted_loss,
        weights=weights,
        processed_labels=labels)

  def create_estimator_spec(
      self, features, mode, logits, labels=None, optimizer=None,
      train_op_fn=None, regularization_losses=None):
    """Returns an `EstimatorSpec`.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` with shape `[D0, D1, ... DN, logits_dimension]`.
        For many applications, the shape is `[batch_size, logits_dimension]`.
      labels: Labels `Tensor` with shape matching `logits`, namely
        `[D0, D1, ... DN, logits_dimension]`. When `logits_dimension=1`, shape
        `[D0, D1, ... DN]` is also supported. `labels` is required argument when
        `mode` equals `TRAIN` or `EVAL`.
      optimizer: `Optimizer` instance to optimize the loss in TRAIN mode.
        Namely, sets `train_op = optimizer.minimize(loss, global_step)`, which
        updates variables and increments `global_step`.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns
        `train_op`. Used if `optimizer` is `None`.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses. These losses are
        usually expressed as a batch average, so for best results users need to
        set `loss_reduction=SUM_OVER_BATCH_SIZE` or
        `loss_reduction=SUM_OVER_NONZERO_WEIGHTS` when creating the head to
        avoid scaling errors.
    Returns:
      `EstimatorSpec`.
    Raises:
      ValueError: If both `train_op_fn` and `optimizer` are `None` in TRAIN
        mode, or if both are set.
    """
    # Predict.
    with ops.name_scope(self._name, 'head'):
      logits = _check_logits_final_dim(logits, self._logits_dimension)
      if self._inverse_link_fn:
        predicted_value = self._inverse_link_fn(logits)
        predictions = {
            prediction_keys.PredictionKeys.PREDICTIONS: predicted_value,
            prediction_keys.PredictionKeys.LOGITS: logits,
        }
      else:
        predicted_value = logits
        predictions = {
            prediction_keys.PredictionKeys.PREDICTIONS: predicted_value}
      if mode == model_fn.ModeKeys.PREDICT:
        regression_output = export_output.RegressionOutput(
            value=predicted_value)
        return model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                _DEFAULT_SERVING_KEY: regression_output,
                _REGRESS_SERVING_KEY: regression_output,
                _PREDICT_SERVING_KEY: export_output.PredictOutput(predictions)
            })

      training_loss, unreduced_loss, weights, _ = self.create_loss(
          features=features, mode=mode, logits=logits, labels=labels)
      if regularization_losses:
        regularization_loss = math_ops.add_n(regularization_losses)
        regularized_training_loss = math_ops.add_n(
            [training_loss, regularization_loss])
      else:
        regularization_loss = None
        regularized_training_loss = training_loss

      # Eval.
      if mode == model_fn.ModeKeys.EVAL:
        keys = metric_keys.MetricKeys
        # Estimator already adds a metric for loss.
        eval_metric_ops = {
            _summary_key(self._name, keys.LOSS_MEAN):
                metrics_lib.mean(
                    values=unreduced_loss,
                    weights=weights)
        }
        if regularization_loss is not None:
          regularization_loss_key = _summary_key(
              self._name, keys.LOSS_REGULARIZATION)
          eval_metric_ops[regularization_loss_key] = metrics_lib.mean(
              values=regularization_loss,
              name=keys.LOSS_REGULARIZATION)
        return model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions=predictions,
            loss=regularized_training_loss,
            eval_metric_ops=eval_metric_ops)

      # Train.
      if optimizer is not None:
        if train_op_fn is not None:
          raise ValueError('train_op_fn and optimizer cannot both be set.')
        train_op = optimizer.minimize(
            regularized_training_loss,
            global_step=training_util.get_global_step())
      elif train_op_fn is not None:
        train_op = train_op_fn(regularized_training_loss)
      else:
        raise ValueError('train_op_fn and optimizer cannot both be None.')
      # Only summarize mean_loss for SUM reduction to preserve backwards
      # compatibility. Otherwise skip it to avoid unnecessary computation.
      if self._loss_reduction == losses.Reduction.SUM:
        example_weight_sum = math_ops.reduce_sum(
            weights * array_ops.ones_like(unreduced_loss))
        mean_loss = training_loss / example_weight_sum
      else:
        mean_loss = None
    with ops.name_scope(''):
      keys = metric_keys.MetricKeys
      summary.scalar(
          _summary_key(self._name, keys.LOSS),
          regularized_training_loss)
      if mean_loss is not None:
        summary.scalar(
            _summary_key(self._name, keys.LOSS_MEAN), mean_loss)
      if regularization_loss is not None:
        summary.scalar(
            _summary_key(self._name, keys.LOSS_REGULARIZATION),
            regularization_loss)
    return model_fn.EstimatorSpec(
        mode=model_fn.ModeKeys.TRAIN,
        predictions=predictions,
        loss=regularized_training_loss,
        train_op=train_op)


def _assert_range(labels, n_classes, message=None):
  with ops.name_scope(None, 'assert_range', (labels,)):
    assert_less = check_ops.assert_less(
        labels,
        ops.convert_to_tensor(n_classes, dtype=labels.dtype),
        message=message or 'Label IDs must < n_classes')
    assert_greater = check_ops.assert_non_negative(
        labels, message=message or 'Label IDs must >= 0')
    with ops.control_dependencies((assert_less, assert_greater)):
      return array_ops.identity(labels)


# TODO(b/69000400): Delete this method.
def _weights(features, weight_column):
  """Fetches weights from features."""
  with ops.name_scope(None, 'weights', values=features.values()):
    if weight_column is None:
      return 1.
    if isinstance(weight_column, six.string_types):
      weight_column = feature_column_lib.numeric_column(
          key=weight_column, shape=(1,))
    if not isinstance(weight_column, feature_column_lib._NumericColumn):  # pylint: disable=protected-access
      raise TypeError('Weight column must be either a string or _NumericColumn.'
                      ' Given type: {}.'.format(type(weight_column)))
    weights = weight_column._get_dense_tensor(  # pylint: disable=protected-access
        feature_column_lib._LazyBuilder(features))  # pylint: disable=protected-access
    if not (weights.dtype.is_floating or weights.dtype.is_integer):
      raise ValueError('Weight column should be castable to float. '
                       'Given dtype: {}'.format(weights.dtype))
    return math_ops.to_float(weights, name='weights')
