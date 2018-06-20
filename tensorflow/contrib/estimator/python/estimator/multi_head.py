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

import six

from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.export import export_output as export_output_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.summary import summary
from tensorflow.python.training import training_util


_DEFAULT_SERVING_KEY = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY


def multi_head(heads, head_weights=None):
  """Creates a `_Head` for multi-objective learning.

  This class merges the output of multiple `_Head` objects.
  Specifically:
  * For training, sums losses of each head, calls `train_op_fn` with this
    final loss.
  * For eval, merges metrics by adding `head.name` suffix to the keys in eval
    metrics, such as `precision/head1`, `precision/head2`.
  * For prediction, merges predictions and updates keys in prediction dict to a
    2-tuple, `(head.name, prediction_key)`. Merges `export_outputs` such that
    by default the first head is served.

  Usage:

  ```python
  # In `input_fn` specify labels as a dict keyed by head name:
  def input_fn():
    features = ...
    labels1 = ...
    labels2 = ...
    return features, {'head1': labels1, 'head2': labels2}

  # In `model_fn`, specify logits as a dict keyed by head name:
  def model_fn(features, labels, mode):
    # Create simple heads and specify head name.
    head1 = multi_class_head(n_classes=3, name='head1')
    head2 = binary_classification_head(name='head2')
    # Create multi-head from two simple heads.
    head = multi_head([head1, head2])
    # Create logits for each head, and combine them into a dict.
    logits1, logits2 = logit_fn()
    logits = {'head1': logits1, 'head2': logits2}
    # Return the merged EstimatorSpec
    return head.create_estimator_spec(..., logits=logits, ...)

  # Create an estimator with this model_fn.
  estimator = tf.estimator.Estimator(model_fn=model_fn)
  estimator.train(input_fn=input_fn, steps=100)
  ```

  Also supports `logits` as a `Tensor` of shape
  `[D0, D1, ... DN, logits_dimension]`. It will split the `Tensor` along the
  last dimension and distribute it appropriately among the heads. E.g.:

  ```python
  def model_fn(features, labels, mode):
    # Create simple heads and specify head name.
    head1 = multi_class_head(n_classes=3, name='head1')
    head2 = binary_classification_head(name='head2')
    # Create multi-head from two simple heads.
    head = multi_head([head1, head2])
    # Create logits for the multihead.
    logits = logit_fn(logits_dimension=head.logits_dimension)
    # Return the merged EstimatorSpec
    return head.create_estimator_spec(..., logits=logits, ...)
  ```

  Args:
    heads: List or tuple of `_Head` instances. All heads must have `name`
      specified. The first head in the list is the default used at serving time.
    head_weights: Optional list of weights, same length as `heads`. Used when
      merging losses to calculate the weighted sum of losses from each head. If
      `None`, all losses are weighted equally.

  Returns:
    A instance of `_Head` that merges multiple heads.

  Raises:
    ValueError: If `heads` is empty.
    ValueError: If any of the `heads` does not have `name` specified.
    ValueError: If `heads` and `head_weights` have different size.
  """
  if head_weights:
    if len(head_weights) != len(heads):
      raise ValueError(
          'heads and head_weights must have the same size. '
          'Given len(heads): {}. Given len(head_weights): {}.'.format(
              len(heads), len(head_weights)))
  if not heads:
    raise ValueError('Must specify heads. Given: {}'.format(heads))
  for head in heads:
    if not head.name:
      raise ValueError(
          'All given heads must have name specified. '
          'Given: {}'.format(head))

  return _MultiHead(
      heads=tuple(heads),
      head_weights=tuple(head_weights) if head_weights else tuple())


def _no_op_train_fn(loss):
  del loss
  return control_flow_ops.no_op()


def _merge_losses(losses, head_weights=None):
  """Merges the given losses into one tensor."""
  losses = tuple(losses)
  with ops.name_scope(
      'merge_losses', values=losses + (head_weights or tuple())):
    if head_weights:
      weighted_losses = []
      for loss, weight in zip(losses, head_weights):
        weighted_losses.append(math_ops.multiply(loss, weight))
    else:
      weighted_losses = losses
    return math_ops.add_n(weighted_losses)


def _default_export_output(export_outputs, head_name):
  """Extracts the default export output from the given export_outputs dict."""
  if len(export_outputs) == 1:
    return next(six.itervalues(export_outputs))
  for k, v in six.iteritems(export_outputs):
    if k == _DEFAULT_SERVING_KEY:
      return v
  raise ValueError(
      '{} did not specify default export_outputs. '
      'Given: {} '
      'Suggested fix: Use one of the heads in tf.contrib.estimator, or include '
      'key {} in export_outputs.'.format(
          head_name, export_outputs, _DEFAULT_SERVING_KEY))


class _MultiHead(head_lib._Head):  # pylint:disable=protected-access
  """`_Head` for multi objective learning."""

  def __init__(self, heads, head_weights):
    self._logits_dimension = 0
    for head in heads:
      self._logits_dimension += head.logits_dimension

    self._heads = heads
    self._head_weights = head_weights

  @property
  def name(self):
    return '_'.join([h.name for h in self._heads])

  @property
  def logits_dimension(self):
    return self._logits_dimension

  def create_loss(self, features, mode, logits, labels):
    """See `Head`."""
    if isinstance(logits, dict):
      logits_dict = logits
    else:
      logits_dict = self._split_logits(logits)
    training_losses = []
    labels_by_head = {}
    unreduced_losses_by_head = {}
    example_weights_by_head = {}
    for i, head in enumerate(self._heads):
      (training_loss, unreduced_loss,
       weights, processed_labels) = head.create_loss(
           features, mode, logits_dict[head.name], labels[head.name])
      training_losses.append(training_loss)
      labels_by_head[head.name] = processed_labels
      if self._head_weights:
        head_weight = self._head_weights[i]
        unreduced_losses_by_head[head.name] = math_ops.multiply(
            unreduced_loss, head_weight)
        example_weights_by_head[head.name] = math_ops.multiply(
            weights, head_weight)
      else:
        unreduced_losses_by_head[head.name] = unreduced_loss
        example_weights_by_head[head.name] = weights

    training_losses = tuple(training_losses)
    with ops.name_scope(
        'merge_losses',
        values=training_losses + (self._head_weights or tuple())):
      if self._head_weights:
        head_weighted_training_losses = []
        for training_loss, head_weight in zip(
            training_losses, self._head_weights):
          head_weighted_training_losses.append(
              math_ops.multiply(training_loss, head_weight))
        merged_training_loss = math_ops.add_n(head_weighted_training_losses)
      else:
        merged_training_loss = math_ops.add_n(training_losses)

    return head_lib.LossSpec(
        training_loss=merged_training_loss,
        unreduced_loss=unreduced_losses_by_head,
        weights=example_weights_by_head,
        processed_labels=labels_by_head)

  # TODO(b/65403806): Support regularization_losses arg.
  def create_estimator_spec(
      self, features, mode, logits, labels=None, optimizer=None,
      train_op_fn=None):
    """See `_Head`."""
    if isinstance(logits, dict):
      logits_dict = logits
    else:
      logits_dict = self._split_logits(logits)
    if labels and not isinstance(labels, dict):
      raise ValueError('labels must be a dict. Given: {}'.format(labels))

    all_estimator_spec = []
    for head in self._heads:
      head_name = head.name
      all_estimator_spec.append(
          head.create_estimator_spec(
              features=features,
              mode=mode,
              logits=logits_dict[head_name],
              labels=labels[head_name] if labels else None,
              train_op_fn=_no_op_train_fn))

    if mode == model_fn.ModeKeys.TRAIN:
      spec = self._merge_train(
          all_estimator_spec=all_estimator_spec,
          optimizer=optimizer,
          train_op_fn=train_op_fn)
      with ops.name_scope(''):
        summary.scalar(metric_keys.MetricKeys.LOSS, spec.loss)
      return spec
    if mode == model_fn.ModeKeys.PREDICT:
      return self._merge_predict(all_estimator_spec)
    if mode == model_fn.ModeKeys.EVAL:
      return self._merge_eval(all_estimator_spec)
    raise ValueError('mode={} unrecognized'.format(mode))

  def _split_logits(self, logits):
    """Splits logits along the last dimension and returns a dict."""
    logits_dict = {}
    with ops.name_scope(None, 'split_logits', values=[logits]):
      logits = ops.convert_to_tensor(logits)
      batch_shape = array_ops.shape(logits)[:-1]
      zeros_like_batch_shape = array_ops.zeros_like(batch_shape)
      minus_ones_like_batch_shape = -1 * array_ops.ones_like(batch_shape)
      begin_idx = 0
      for head in self._heads:
        begin_tensor = array_ops.concat(
            [zeros_like_batch_shape, [begin_idx]], axis=0)
        size_tensor = array_ops.concat(
            [minus_ones_like_batch_shape, [head.logits_dimension]], axis=0)
        logits_dict[head.name] = array_ops.slice(
            logits, begin=begin_tensor, size=size_tensor)
        begin_idx += head.logits_dimension
    return logits_dict

  def _merge_train(self, all_estimator_spec, optimizer, train_op_fn):
    """Merges list of `EstimatorSpec` for training.

    Args:
      all_estimator_spec: list of `EstimatorSpec` for the individual heads.
      optimizer: `Optimizer` instance to create train op. See
        `create_estimator_spec` documentation for more details.
      train_op_fn: Function to create train op. Used if `optimizer` is `None`.

    Returns:
      `EstimatorSpec` that merges all heads for TRAIN.

    Raises:
      ValueError: If both `train_op_fn` and `optimizer` are `None` in TRAIN
        mode.
    """
    losses = []
    metrics = {}
    for spec in all_estimator_spec:
      losses.append(spec.loss)
      # Metric keys already contain head.name.
      metrics.update(spec.eval_metric_ops or {})
    loss = _merge_losses(losses, self._head_weights)
    if optimizer is not None:
      if train_op_fn is not None:
        raise ValueError('train_op_fn and optimizer cannot both be set.')
      train_op = optimizer.minimize(
          loss, global_step=training_util.get_global_step())
    elif train_op_fn is not None:
      train_op = train_op_fn(loss)
    else:
      raise ValueError('train_op_fn and optimizer cannot both be None.')

    return model_fn.EstimatorSpec(
        mode=model_fn.ModeKeys.TRAIN,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

  def _merge_predict(self, all_estimator_spec):
    """Merges list of `EstimatorSpec` for prediction.

    Args:
      all_estimator_spec: list of `EstimatorSpec` for the individual heads.

    Returns:
      `EstimatorSpec` that merges all heads for PREDICT.
    """
    predictions = {}
    export_outputs = {
        _DEFAULT_SERVING_KEY: _default_export_output(
            all_estimator_spec[0].export_outputs,
            self._heads[0].name),
    }
    merged_predict_outputs = {}
    for head, spec in zip(self._heads, all_estimator_spec):
      head_name = head.name
      for k, v in six.iteritems(spec.export_outputs):
        if k == _DEFAULT_SERVING_KEY:
          key = head_name
        else:
          key = '%s/%s' % (head_name, k)
        export_outputs[key] = v
        if (k == head_lib._PREDICT_SERVING_KEY and  # pylint:disable=protected-access
            isinstance(v, export_output_lib.PredictOutput)):
          for kp, vp in six.iteritems(v.outputs):
            key = '%s/%s' % (head_name, kp)
            merged_predict_outputs[key] = vp
      for k, v in six.iteritems(spec.predictions):
        predictions[(head_name, k)] = v
    export_outputs[head_lib._PREDICT_SERVING_KEY] = (  # pylint:disable=protected-access
        export_output_lib.PredictOutput(merged_predict_outputs))

    return model_fn.EstimatorSpec(
        mode=model_fn.ModeKeys.PREDICT,
        predictions=predictions,
        export_outputs=export_outputs)

  def _merge_eval(self, all_estimator_spec):
    """Merges list of `EstimatorSpec` for eval.

    Args:
      all_estimator_spec: list of `EstimatorSpec` for the individual heads.

    Returns:
      `EstimatorSpec` that merges all heads for EVAL.
    """
    predictions = {}
    metrics = {}
    losses = []
    with ops.name_scope('merge_eval'):
      for head, spec in zip(self._heads, all_estimator_spec):
        losses.append(spec.loss)
        head_name = head.name
        # Loss metric is not added by default.
        loss_name = head_lib._summary_key(  # pylint:disable=protected-access
            head_name, metric_keys.MetricKeys.LOSS)
        metrics[loss_name] = metrics_lib.mean(spec.loss, name=loss_name)
        # Metric keys already contain head.name.
        metrics.update(spec.eval_metric_ops or {})
        for k, v in six.iteritems(spec.predictions):
          predictions[(head_name, k)] = v
      loss = _merge_losses(losses, self._head_weights)

    return model_fn.EstimatorSpec(
        mode=model_fn.ModeKeys.EVAL,
        predictions=predictions,
        loss=loss,
        eval_metric_ops=metrics)
