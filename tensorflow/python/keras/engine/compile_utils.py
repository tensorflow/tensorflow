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

from tensorflow.python.keras import losses as losses_mod
from tensorflow.python.keras import metrics as metrics_mod
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util as tf_losses_utils
from tensorflow.python.util import nest


class LossesContainer(object):
  """A container class for losses passed to `Model.compile`."""

  def __init__(self, losses, loss_weights=None, output_names=None):
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
      self._output_names = create_output_names(y_pred)

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

  def __call__(self, y_true, y_pred, sample_weight=None):
    """Computes the overall loss.

    Arguments:
      y_true: An arbitrary structure of Tensors representing the ground truth.
      y_pred: An arbitrary structure of Tensors representing a Model's outputs.
      sample_weight: An arbitrary structure of Tensors representing the
        per-sample loss weights. If one Tensor is passed, it is used for all
        losses. If multiple Tensors are passed, the structure should match
        `y_pred`.

    Returns:
      Tuple of `(total_loss, per_output_loss_list)`
    """
    if not self._built:
      self._build(y_pred)

    y_true = nest.flatten(y_true)
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

    loss_values = []
    zip_args = (y_true, y_pred, sample_weight, self._losses, self._loss_weights,
                self._per_output_metrics)
    for y_t, y_p, sw, loss_obj, loss_weight, metric_obj in zip(*zip_args):
      if loss_obj is None:  # Ok to have no loss for an output.
        continue

      y_t = math_ops.cast(y_t, y_p.dtype)
      if sw is not None:
        sw = math_ops.cast(sw, y_p.dtype)

      # Handle Keras mask on outputs.
      mask = getattr(y_p, '_keras_mask', None)
      if mask is not None:
        mask = math_ops.cast(mask, y_p.dtype)
        if sw is not None:
          mask, _, sw = (
              tf_losses_utils.squeeze_or_expand_dimensions(
                  mask, sample_weight=sw))
          sw *= mask
        else:
          sw = mask

      loss_value = loss_obj(y_t, y_p, sample_weight=sw)

      if metric_obj is not None:
        metric_obj.update_state(loss_value)

      if loss_weight is not None:
        loss_value *= loss_weight

      if (loss_obj.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE or
          loss_obj.reduction == losses_utils.ReductionV2.AUTO):
        loss_value = losses_utils.scale_loss_for_distribution(loss_value)
      loss_values.append(loss_value)

    if loss_values:
      total_loss = math_ops.add_n(loss_values)
      self._loss_metric.update_state(total_loss)
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
      loss = losses_mod.LossFunctionWrapper(loss, name=loss.__name__)
    loss._allow_sum_over_batch_size = True  # pylint: disable=protected-access
    return loss


class MetricsContainer(object):
  """A container class for metrics passed to `Model.compile`."""

  def __init__(self, metrics=None, weighted_metrics=None, output_names=None):
    self._metrics = metrics
    self._weighted_metrics = weighted_metrics
    self._output_names = output_names
    self._built = False

  @property
  def metrics(self):
    """Metrics created by this container."""
    if not self._built:
      return []
    metrics = [
        metric_obj for metric_obj in nest.flatten(self._metrics)
        if metric_obj is not None
    ]
    weighted_metrics = [
        metric_obj for metric_obj in nest.flatten(self._weighted_metrics)
        if metric_obj is not None
    ]
    return metrics + weighted_metrics

  def _build(self, y_pred, y_true):
    """One-time setup of metric objects."""

    if self._output_names is None:
      # Subclass output names like 'output_1' are used for `Metric` names.
      self._output_names = create_output_names(y_pred)

    # Accept a dict of metrics keyed by output_name when outputs are a flat
    # list.
    self._metrics = map_to_output_names(y_pred, self._output_names,
                                        self._metrics)
    self._weighted_metrics = map_to_output_names(y_pred, self._output_names,
                                                 self._weighted_metrics)

    # If a single metric is supplied, apply to all outputs.
    self._metrics = self._maybe_broadcast(self._metrics, y_pred)
    self._weighted_metrics = self._maybe_broadcast(self._weighted_metrics,
                                                   y_pred)

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
          wm._name = output_name + '_' + wm._name
        if wm._name in metric_names:
          wm._name = 'weighted_' + wm._name
        if wm._name in metric_names:
          raise ValueError('Found two metrics with the same name: {}'.format(
              wm._name))
        metric_names.add(wm._name)
    # pylint: enable=protected-access

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Updates the state of per-output metrics."""
    flat_y_true = nest.flatten(y_true)
    flat_y_pred = nest.flatten(y_pred)

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
      y_t = math_ops.cast(y_t, y_p.dtype)
      if sw is not None:
        sw = math_ops.cast(sw, y_p.dtype)

      # Handle Keras mask on outputs.
      mask = getattr(y_p, '_keras_mask', None)
      if mask is not None:
        mask = math_ops.cast(mask, y_p.dtype)
        if sw is not None:
          mask, _, sw = (
              tf_losses_utils.squeeze_or_expand_dimensions(
                  mask, sample_weight=sw))
          sw *= mask
        else:
          sw = mask

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
    metrics = generic_utils.to_list(metrics)
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
    """If a single Metric is supplied, applies it to all outputs."""

    def _should_broadcast(metrics):
      single_valued_list = (
          isinstance(metrics, list) and len(metrics) == 1 and
          not nest.is_sequence(metrics[0]))
      # I.e. `metrics=['accuracy']` or `metrics='accuracy'`.
      # In this special case we apply the metric to each output.
      return not nest.is_sequence(metrics) or single_valued_list

    def _copy(metric):
      if isinstance(metric, metrics_mod.Metric):
        return metrics_mod.Metric.from_config(metric.get_config())
      return metric

    if _should_broadcast(metrics):
      metric = metrics[0] if isinstance(metrics, list) else metrics
      return nest.map_structure(lambda _: _copy(metric), y_pred)
    return metrics


def create_output_names(y_pred):
  """Creates output names for subclassed Model outputs.

  These names are used for naming `Metric`s.

  Example with dict:

  `{'a': [x1, x2], 'b': x3}` becomes:
  `['a_1', 'a_2', 'b']`

  Example with list:

  `[x, y]` becomes:
  `['output_1', 'output_2']`

  Arguments:
    y_pred: `Model`'s outputs.

  Returns:
    Flattened list of output names.
  """

  def one_index(ele):
    # Start with "output_1" instead of "output_0".
    if isinstance(ele, int):
      return ele + 1
    return ele

  flat_paths = list(nest.yield_flat_paths(y_pred))
  flat_paths = nest.map_structure(one_index, flat_paths)
  output_names = []
  for path in flat_paths:
    if not path:
      output_name = 'output_1'
    else:
      output_name = '_'.join(str(p) for p in path)
      if isinstance(path[0], int):
        output_name = 'output_' + output_name
    output_names.append(output_name)
  return output_names


def map_to_output_names(y_pred, output_names, struct):
  """Maps a dict to a list using `output_names` as keys.

  This is a convenience feature only. When a `Model`'s outputs
  are a list, you can specify per-output losses and metrics as
  a dict, where the keys are the output names. If you specify
  per-output losses and metrics via the same structure as the
  `Model`'s outputs (recommended), no mapping is performed.

  For the Functional API, the output names are the names of the
  last layer of each output. For the Subclass API, the output names
  are determined by `create_output_names` (For example:
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
  if not outputs_are_flat_list:
    # In this case, `y_pred` and `struct` must have the same structure.
    return struct

  if not isinstance(struct, dict):
    return struct

  struct = copy.copy(struct)
  new_struct = [struct.pop(name, None) for name in output_names]
  if struct:
    raise ValueError('Found unexpected keys that do not correspond '
                     'to any Model output: {}. Expected: {}'.format(
                         struct.keys(), output_names))
  return new_struct
