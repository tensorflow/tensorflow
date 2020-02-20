#  Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utilites for `Model.compile`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import six

from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.keras import losses as losses_mod
from tensorflow.python.keras import metrics as metrics_mod
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util as tf_losses_utils
from tensorflow.python.util import nest


class LossesContainer(object):
  """A container class for losses passed to `Model.compile`."""

  def __init__(self, losses, loss_weights=None, output_names=None):
    # Keep user-supplied values untouched for recompiling and serialization.
    self._user_losses = losses
    self._user_loss_weights = loss_weights

    self._losses = losses
    self._loss_weights = loss_weights
    self._output_names = output_names
    self._per_output_metrics = None  # Per-output losses become metrics.
    self._loss_metric = metrics_mod.Mean(name='loss')  # Total loss.
    self._built = False

  @property
  def metrics(self):
    """Per-output loss metrics."""
    if not self._built:
      return []
    per_output_metrics = [
        metric_obj for metric_obj in nest.flatten(self._per_output_metrics)
        if metric_obj is not None
    ]
    return [self._loss_metric] + per_output_metrics

  def _build(self, y_pred):
    """One-time setup of loss objects."""

    if self._output_names is None:
      # In Subclass API,  output names like 'output_1' are used for
      # `Metric` names.
      self._output_names = create_pseudo_output_names(y_pred)

    # Accept a dict of losses keyed by output_name when outputs are a flat
    # list.
    self._losses = map_to_output_names(y_pred, self._output_names, self._losses)
    self._loss_weights = map_to_output_names(y_pred, self._output_names,
                                             self._loss_weights)

    # Broadcast single config values to apply to each output.
    if not nest.is_sequence(self._losses):
      self._losses = nest.map_structure(lambda output: self._losses, y_pred)
    if not nest.is_sequence(self._loss_weights):
      self._loss_weights = nest.map_structure(lambda output: self._loss_weights,
                                              y_pred)

    self._losses = nest.map_structure(self._get_loss_object, self._losses)

    # Now that structures have been checked, it is safe to flatten.
    self._losses = nest.flatten(self._losses)
    self._loss_weights = nest.flatten(self._loss_weights)

    # Create per-output loss metrics, but only for multi-output Models.
    if len(self._output_names) == 1:
      self._per_output_metrics = [None]
    else:
      self._per_output_metrics = []
      for loss_obj, output_name in zip(self._losses, self._output_names):
        if loss_obj is None:
          self._per_output_metrics.append(None)
        else:
          self._per_output_metrics.append(
              metrics_mod.Mean(output_name + '_loss'))

    self._built = True

  def __call__(self,
               y_true,
               y_pred,
               sample_weight=None,
               regularization_losses=None):
    """Computes the overall loss.

    Arguments:
      y_true: An arbitrary structure of Tensors representing the ground truth.
      y_pred: An arbitrary structure of Tensors representing a Model's outputs.
      sample_weight: An arbitrary structure of Tensors representing the
        per-sample loss weights. If one Tensor is passed, it is used for all
        losses. If multiple Tensors are passed, the structure should match
        `y_pred`.
      regularization_losses: Additional losses to be added to the total loss.

    Returns:
      Tuple of `(total_loss, per_output_loss_list)`
    """
    y_true = map_to_output_names(y_pred, self._output_names, y_true)
    sample_weight = map_to_output_names(y_pred, self._output_names,
                                        sample_weight)

    if not self._built:
      self._build(y_pred)

    y_true = nest.flatten(y_true) if y_true is not None else []
    y_pred = nest.flatten(y_pred)

    # TODO(omalleyt): Remove ambiguity here.
    # This is currently needed to support passing only 1 loss and 1 target
    # to a Functional Model with multiple outputs. However, this is
    # ambiguous, especially with subclass, and we should reconsider how we
    # support this.
    if len(y_true) == 1 and len(y_pred) > 1:
      y_true = y_true * len(y_pred)

    sample_weight = nest.flatten(sample_weight)
    # Allows passing one sample-weight array for all outputs.
    if len(sample_weight) == 1 and len(y_pred) > 1:
      sample_weight = sample_weight * len(y_pred)

    loss_values = []  # Used for gradient calculation.
    loss_metric_values = []  # Used for loss metric calculation.
    zip_args = (y_true, y_pred, sample_weight, self._losses, self._loss_weights,
                self._per_output_metrics)
    for y_t, y_p, sw, loss_obj, loss_weight, metric_obj in zip(*zip_args):
      if loss_obj is None:  # Ok to have no loss for an output.
        continue

      y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
      sw = apply_mask(y_p, sw)

      loss_value = loss_obj(y_t, y_p, sample_weight=sw)

      loss_metric_value = loss_value
      # Correct for the `Mean` loss metrics counting each replica as a batch.
      if loss_obj.reduction == losses_utils.ReductionV2.SUM:
        loss_metric_value *= ds_context.get_strategy().num_replicas_in_sync
      if metric_obj is not None:
        metric_obj.update_state(loss_metric_value)

      if loss_weight is not None:
        loss_value *= loss_weight
        loss_metric_value *= loss_weight

      if (loss_obj.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE or
          loss_obj.reduction == losses_utils.ReductionV2.AUTO):
        loss_value = losses_utils.scale_loss_for_distribution(loss_value)

      loss_values.append(loss_value)
      loss_metric_values.append(loss_metric_value)

    if regularization_losses:
      reg_loss = math_ops.add_n(regularization_losses)
      loss_metric_values.append(reg_loss)
      loss_values.append(losses_utils.scale_loss_for_distribution(reg_loss))

    if loss_values:
      total_loss_metric_value = math_ops.add_n(loss_metric_values)
      self._loss_metric.update_state(total_loss_metric_value)

      total_loss = math_ops.add_n(loss_values)
      return total_loss
    else:
      # Ok for a model to have no compiled loss.
      return array_ops.zeros(shape=())

  def _get_loss_object(self, loss):
    """Returns a `Loss` object.

    Converts the user-supplied loss to a `Loss` object. Also allows
    `SUM_OVER_BATCH_SIZE` reduction to be used for this loss.

    Arguments:
      loss: A string, function, or `Loss` object.

    Returns:
      A `Loss` object.
    """
    if loss is None:
      return None  # Ok to have no loss for an output.

    loss = losses_mod.get(loss)
    if not isinstance(loss, losses_mod.Loss):
      loss_name = loss.__name__
      loss = losses_mod.LossFunctionWrapper(loss, name=loss_name)
    loss._allow_sum_over_batch_size = True  # pylint: disable=protected-access
    return loss


class MetricsContainer(object):
  """A container class for metrics passed to `Model.compile`."""

  def __init__(self, metrics=None, weighted_metrics=None, output_names=None):
    # Keep user-supplied values untouched for recompiling and serialization.
    self._user_metrics = metrics
    self._user_weighted_metrics = weighted_metrics

    self._metrics = metrics
    self._weighted_metrics = weighted_metrics
    self._output_names = output_names
    self._built = False

  @property
  def metrics(self):
    """Metrics created by this container."""
    if not self._built:
      return []
    return self._metrics_in_order

  def _build(self, y_pred, y_true):
    """One-time setup of metric objects."""

    if self._output_names is None:
      # Subclass output names like 'output_1' are used for `Metric` names.
      self._output_names = create_pseudo_output_names(y_pred)

    # If a single metric or flat list of metrics, apply to all outputs.
    self._metrics = self._maybe_broadcast(self._metrics, y_pred)
    self._weighted_metrics = self._maybe_broadcast(self._weighted_metrics,
                                                   y_pred)

    # Accept a dict of metrics keyed by output_name when outputs are a flat
    # list.
    self._metrics = map_to_output_names(y_pred, self._output_names,
                                        self._metrics)
    self._weighted_metrics = map_to_output_names(y_pred, self._output_names,
                                                 self._weighted_metrics)

    # Standardize on tuple since `tf.data` turns lists into `Tensor`s.
    # pylint: disable=protected-access
    y_pred = nest._list_to_tuple(y_pred)
    y_true = nest._list_to_tuple(y_true)
    self._metrics = nest._list_to_tuple(self._metrics)
    self._weighted_metrics = nest._list_to_tuple(self._weighted_metrics)
    # pylint: enable=protected-access

    # Convert to `Metric` objects, potentially disambiguating based on output
    # properties.
    self._metrics = nest.map_structure_up_to(y_pred, self._get_metric_objects,
                                             self._metrics, y_true, y_pred)
    self._weighted_metrics = nest.map_structure_up_to(y_pred,
                                                      self._get_metric_objects,
                                                      self._weighted_metrics,
                                                      y_true, y_pred)

    self._metrics = nest.flatten_up_to(y_pred, self._metrics, check_types=False)
    self._weighted_metrics = nest.flatten_up_to(
        y_pred, self._weighted_metrics, check_types=False)

    # Assumes metrics, weighted_metrics have been flattened up to outputs.
    self._set_metric_names()

    # Cache the flat order needed when returning metrics, for backwards compat.
    self._metrics_in_order = []
    for output_metrics, output_weighted_metrics in zip(self._metrics,
                                                       self._weighted_metrics):
      for m in nest.flatten(output_metrics):
        if m is not None:
          self._metrics_in_order.append(m)
      for wm in nest.flatten(output_weighted_metrics):
        if wm is not None:
          self._metrics_in_order.append(wm)

    self._built = True

  def _set_metric_names(self):
    """Sets unique metric names."""
    # For multi-output models, prepend the output name to the metric name.
    # For weighted metrics, prepend "weighted_" if the name would be non-unique.
    # pylint: disable=protected-access
    metric_names = set()
    is_multi_output = len(self._output_names) > 1
    zip_args = (self._output_names, self._metrics, self._weighted_metrics)
    for output_name, output_metrics, weighted_output_metrics in zip(*zip_args):
      for m in output_metrics:
        if m is None:
          continue
        if is_multi_output:
          m._name = output_name + '_' + m._name
        if m._name in metric_names:
          raise ValueError('Found two metrics with the same name: {}'.format(
              m._name))
        metric_names.add(m._name)

      for wm in weighted_output_metrics:
        if wm is None:
          continue
        if is_multi_output:
          if output_name + '_' + wm._name in metric_names:
            wm._name = output_name + '_weighted_' + wm._name
          else:
            wm._name = output_name + '_' + wm._name
        elif wm._name in metric_names:
          wm._name = 'weighted_' + wm._name

        if wm._name in metric_names:
          raise ValueError('Found two metrics with the same name: {}'.format(
              wm._name))
        metric_names.add(wm._name)
    # pylint: enable=protected-access

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Updates the state of per-output metrics."""
    y_true = map_to_output_names(y_pred, self._output_names, y_true)
    sample_weight = map_to_output_names(y_pred, self._output_names,
                                        sample_weight)

    flat_y_true = nest.flatten(y_true) if y_true is not None else []
    flat_y_pred = nest.flatten(y_pred)

    if not flat_y_true:
      return  # Handle case where no targets are passed.

    # TODO(omalleyt): Remove ambiguity here (see LossesContainer).
    if len(flat_y_true) == 1 and len(flat_y_pred) > 1:
      y_true = nest.map_structure(lambda _: flat_y_true[0], y_pred)
      flat_y_true = nest.flatten(y_true)

    if not self._built:
      # `_build` needs unflattened outputs and labels.
      self._build(y_pred, y_true)

    y_true = flat_y_true
    y_pred = flat_y_pred

    sample_weight = nest.flatten(sample_weight)
    # Allows passing one sample-weight array for all outputs.
    if len(sample_weight) == 1 and len(y_pred) > 1:
      sample_weight = sample_weight * len(y_pred)

    zip_args = (y_true, y_pred, sample_weight, self._metrics,
                self._weighted_metrics)
    for y_t, y_p, sw, metric_objs, weighted_metric_objs in zip(*zip_args):
      y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
      sw = apply_mask(y_p, sw)

      for metric_obj in metric_objs:
        if metric_obj is None:
          continue
        metric_obj.update_state(y_t, y_p)

      for weighted_metric_obj in weighted_metric_objs:
        if weighted_metric_obj is None:
          continue
        weighted_metric_obj.update_state(y_t, y_p, sample_weight=sw)

  def _get_metric_objects(self, metrics, y_t, y_p):
    """Convert user-supplied metrics to `Metric` objects."""
    metrics = nest.flatten(metrics)
    return [self._get_metric_object(m, y_t, y_p) for m in metrics]

  def _get_metric_object(self, metric, y_t, y_p):
    """Converts user-supplied metric to a `Metric` object.

    Arguments:
      metric: A string, function, or `Metric` object.
      y_t: Sample of label.
      y_p: Sample of output.

    Returns:
      A `Metric` object.
    """
    if metric is None:
      return None  # Ok to have no metric for an output.

    # Convenience feature for selecting b/t binary, categorical,
    # and sparse categorical.
    if metric not in ['accuracy', 'acc', 'crossentropy', 'ce']:
      metric_obj = metrics_mod.get(metric)
    else:
      y_t_rank = len(y_t.shape.as_list())
      y_p_rank = len(y_p.shape.as_list())
      y_t_last_dim = y_t.shape.as_list()[-1]
      y_p_last_dim = y_p.shape.as_list()[-1]

      is_binary = y_p_last_dim == 1
      is_sparse_categorical = (
          y_t_rank < y_p_rank or y_t_last_dim == 1 and y_p_last_dim > 1)

      if metric in ['accuracy', 'acc']:
        if is_binary:
          metric_obj = metrics_mod.binary_accuracy
        elif is_sparse_categorical:
          metric_obj = metrics_mod.sparse_categorical_accuracy
        else:
          metric_obj = metrics_mod.categorical_accuracy
      else:
        if is_binary:
          metric_obj = metrics_mod.binary_crossentropy
        elif is_sparse_categorical:
          metric_obj = metrics_mod.sparse_categorical_crossentropy
        else:
          metric_obj = metrics_mod.categorical_crossentropy

    if not isinstance(metric_obj, metrics_mod.Metric):
      if isinstance(metric, six.string_types):
        metric_name = metric
      elif hasattr(metric, 'name'):
        metric_name = metric.name  # TODO(omalleyt): Is this needed?
      else:
        # function was passed.
        metric_name = metric.__name__

      metric_obj = metrics_mod.MeanMetricWrapper(metric_obj, name=metric_name)

    return metric_obj

  def _maybe_broadcast(self, metrics, y_pred):
    """If a flat list of Metrics is supplied, apply them to all outputs."""

    def _should_broadcast(metrics):
      # e.g. 'mse'.
      if not nest.is_sequence(metrics):
        return True
      # e.g. ['mse'] or ['mse', 'mae'].
      return (isinstance(metrics, (list, tuple)) and
              not any(nest.is_sequence(m) for m in metrics))

    if _should_broadcast(metrics):
      copy_metrics = len(nest.flatten(y_pred)) > 1

      def _maybe_copy(m):
        if copy_metrics and isinstance(m, metrics_mod.Metric):
          return m.__class__.from_config(m.get_config())
        return m

      metrics = nest.flatten(metrics)
      return nest.map_structure(lambda _: [_maybe_copy(m) for m in metrics],
                                y_pred)

    return metrics


def create_pseudo_output_names(outputs):
  """Create pseudo output names for a subclassed Model."""
  return _create_pseudo_names(outputs, prefix='output_')


def create_pseudo_input_names(inputs):
  """Create pseudo input names for a subclassed Model."""
  return _create_pseudo_names(inputs, prefix='input_')


def _create_pseudo_names(tensors, prefix):
  """Creates pseudo {input | output} names for subclassed Models.

  Warning: this function should only be used to define default
  names for `Metics` and `SavedModel`. No other use cases should
  rely on a `Model`'s input or output names.

  Example with dict:

  `{'a': [x1, x2], 'b': x3}` becomes:
  `['a_1', 'a_2', 'b']`

  Example with list:

  `[x, y]` becomes:
  `['output_1', 'output_2']`

  Arguments:
    tensors: `Model`'s outputs or inputs.
    prefix: 'output_' for outputs, 'input_' for inputs.

  Returns:
    Flattened list of pseudo names.
  """

  def one_index(ele):
    # Start with "output_1" instead of "output_0".
    if isinstance(ele, int):
      return ele + 1
    return ele

  flat_paths = list(nest.yield_flat_paths(tensors))
  flat_paths = nest.map_structure(one_index, flat_paths)
  names = []
  for path in flat_paths:
    if not path:
      name = prefix + '1'  # Single output.
    else:
      name = '_'.join(str(p) for p in path)
      if isinstance(path[0], int):
        name = prefix + name
    names.append(name)
  return names


def map_to_output_names(y_pred, output_names, struct):
  """Maps a dict to a list using `output_names` as keys.

  This is a convenience feature only. When a `Model`'s outputs
  are a list, you can specify per-output losses and metrics as
  a dict, where the keys are the output names. If you specify
  per-output losses and metrics via the same structure as the
  `Model`'s outputs (recommended), no mapping is performed.

  For the Functional API, the output names are the names of the
  last layer of each output. For the Subclass API, the output names
  are determined by `create_pseudo_output_names` (For example:
  `['output_1', 'output_2']` for a list of outputs).

  This mapping preserves backwards compatibility for `compile` and
  `fit`.

  Arguments:
    y_pred: Sample outputs of the Model, to determine if this convenience
      feature should be applied (`struct` is returned unmodified if `y_pred`
      isn't a flat list).
    output_names: List. The names of the outputs of the Model.
    struct: The structure to map.

  Returns:
    `struct` mapped to a list in same order as `output_names`.
  """
  outputs_are_flat_list = (
      isinstance(y_pred, (list, tuple)) and
      not any(nest.is_sequence(y_p) for y_p in y_pred))
  single_output = not nest.is_sequence(y_pred)

  if (single_output or outputs_are_flat_list) and isinstance(struct, dict):
    output_names = output_names or create_pseudo_output_names(y_pred)
    struct = copy.copy(struct)
    new_struct = [struct.pop(name, None) for name in output_names]
    if struct:
      raise ValueError('Found unexpected keys that do not correspond '
                       'to any Model output: {}. Expected: {}'.format(
                           struct.keys(), output_names))
    if len(new_struct) == 1:
      return new_struct[0]
    return new_struct
  else:
    return struct


def match_dtype_and_rank(y_t, y_p, sw):
  """Match dtype and rank of predictions."""
  # Rank.
  y_t_rank = len(y_t.shape)
  y_p_rank = len(y_p.shape)
  if y_t_rank == 1 and y_p_rank == 2:
    y_t = array_ops.expand_dims_v2(y_t, axis=-1)
  if sw is not None:
    sw_rank = len(sw.shape)
    if sw_rank == 1 and y_p_rank == 2:
      sw = array_ops.expand_dims_v2(sw, axis=-1)

  # Dtype.
  y_t = math_ops.cast(y_t, y_p.dtype)
  if sw is not None:
    sw = math_ops.cast(sw, y_p.dtype)
  return y_t, y_p, sw


def apply_mask(y_p, sw):
  """Applies any mask on predictions to sample weights."""
  # Handle Keras mask on outputs.
  mask = getattr(y_p, '_keras_mask', None)
  if mask is not None:
    mask = math_ops.cast(mask, y_p.dtype)
    if sw is not None:
      mask, _, sw = (
          tf_losses_utils.squeeze_or_expand_dimensions(mask, sample_weight=sw))
      sw *= mask
    else:
      sw = mask
  return sw
