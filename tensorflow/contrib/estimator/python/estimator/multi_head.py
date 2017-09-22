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
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.saved_model import signature_constants


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
    logits1 = ...
    logits2 = ...
    logits = {'head1': logits1, 'head2': logits2}
    # Return the merged EstimatorSpec
    return head.create_estimator_spec(..., logits=logits, ...)

  # Create an estimator with this model_fn.
  estimator = tf.estimator.Estimator(model_fn=model_fn)
  estimator.train(input_fn=input_fn, steps=100)
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
    # TODO(roumposg): Implement it.
    raise NotImplementedError('create_loss not yet implemented for MultiHead.')

  def create_estimator_spec(
      self, features, mode, logits, labels=None, train_op_fn=None):
    """See `_Head`."""
    if not isinstance(logits, dict):
      raise ValueError('logits must be a dict. Given: {}'.format(logits))
    if labels and not isinstance(labels, dict):
      raise ValueError('labels must be a dict. Given: {}'.format(labels))

    all_estimator_spec = []
    for head in self._heads:
      head_name = head.name
      all_estimator_spec.append(
          head.create_estimator_spec(
              features=features,
              mode=mode,
              logits=logits[head_name],
              labels=labels[head_name] if labels else None,
              train_op_fn=_no_op_train_fn))

    if mode == model_fn.ModeKeys.TRAIN:
      if train_op_fn is None:
        raise ValueError('train_op_fn can not be None in TRAIN mode.')
      return self._merge_train(all_estimator_spec, train_op_fn)
    if mode == model_fn.ModeKeys.PREDICT:
      return self._merge_predict(all_estimator_spec)
    if mode == model_fn.ModeKeys.EVAL:
      return self._merge_eval(all_estimator_spec)
    raise ValueError('mode={} unrecognized'.format(mode))

  def _merge_train(self, all_estimator_spec, train_op_fn):
    """Merges list of `EstimatorSpec` for training.

    Args:
      all_estimator_spec: list of `EstimatorSpec` for the individual heads.
      train_op_fn: Function to create train op. See `create_estimator_spec`
        documentation for more details.

    Returns:
      `EstimatorSpec` that merges all heads for TRAIN.
    """
    losses = []
    metrics = {}
    for spec in all_estimator_spec:
      losses.append(spec.loss)
      # Metric keys already contain head.name.
      metrics.update(spec.eval_metric_ops or {})
    loss = _merge_losses(losses, self._head_weights)

    return model_fn.EstimatorSpec(
        mode=model_fn.ModeKeys.TRAIN,
        loss=loss,
        train_op=train_op_fn(loss),
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
    for head, spec in zip(self._heads, all_estimator_spec):
      head_name = head.name
      for k, v in six.iteritems(spec.export_outputs):
        key = '%s/%s' % (k, head_name) if k else head_name
        export_outputs[key] = v
      for k, v in six.iteritems(spec.predictions):
        predictions[(head_name, k)] = v

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
    for head, spec in zip(self._heads, all_estimator_spec):
      losses.append(spec.loss)
      head_name = head.name
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
