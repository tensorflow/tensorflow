# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.keras import losses as losses_mod
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util as tf_losses_utils
from tensorflow.python.util import nest


class LossesContainer(object):
  """A container class for losses passed to `Model.compile`."""

  def __init__(self, losses, loss_weights=None, output_names=None):

    # Used only with Functional API models to flatten losses passed as a dict
    # into a list using the Model's named outputs.
    if output_names:
      losses = map_to_outputs(output_names, losses)
      loss_weights = map_to_outputs(output_names, loss_weights)

    self._losses = losses
    self._loss_weights = loss_weights
    self._built = False

  def _build(self, y_pred):
    """One-time setup of loss objects."""
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
    metric_loss_values = []  # The loss value passed on to `Mean` metrics.
    zip_args = (y_true, y_pred, sample_weight, self._losses, self._loss_weights)
    for y_t, y_p, sw, loss_obj, loss_weight in zip(*zip_args):
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
      if loss_weight is not None:
        loss_value *= loss_weight
      metric_loss_values.append(loss_value)

      # TODO(omalleyt): Should this be in the `Loss` class?
      if (loss_obj.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE or
          loss_obj.reduction == losses_utils.ReductionV2.AUTO):
        loss_value = losses_utils.scale_loss_for_distribution(loss_value)
      loss_values.append(loss_value)

    # Ok for a model to have no compiled loss.
    total_loss = math_ops.add_n(
        loss_values) if loss_values else array_ops.zeros((1,))

    # TODO(omalleyt): Don't return per-output losses once MetricsContainer
    # handles this.
    return total_loss, metric_loss_values

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

    # TODO(omalleyt): Handle special casing for crossentropy.
    loss = losses_mod.get(loss)
    if not isinstance(loss, losses_mod.Loss):
      loss = losses_mod.LossFunctionWrapper(loss)
    # Allow AUTO and SUM_OVER_BATCH_SIZE reductions.
    # TODO(omalleyt): Can we reconcile CTL and built-in loss reductions?
    loss._allow_sum_over_batch_size = True  # pylint: disable=protected-access
    return loss


def map_to_outputs(output_names, struct):
  """Map losses/metrics to outputs in the Functional API."""
  # Used only for Functional API Models: allows users to specify
  # metrics/losses using a dict with output names as keys.
  if isinstance(struct, dict):
    struct = copy.copy(struct)
    new_struct = [struct.pop(name, None) for name in output_names]
    if struct:
      raise ValueError('Found unexpected keys that do not correspond '
                       'to any Model output: {}. Expected: {}'.format(
                           struct.keys(), output_names))
    return new_struct
  return struct
