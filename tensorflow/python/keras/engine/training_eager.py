# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Keras training and evaluation routines for eager execution.
"""
# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util as tf_losses_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


def _eager_loss_fn(outputs, targets, loss_fn, output_name):
  with backend.name_scope(output_name + '_loss'):
    loss = loss_fn(targets, outputs)
  return loss


def _eager_metrics_fn(model, outputs, targets, sample_weights=None, masks=None):
  """Calculates the metrics for each output of the given model.

  Arguments:
      model: The model on which metrics are being calculated.
      outputs: The outputs of the given model.
      targets: The predictions or targets of the given model.
      sample_weights: Optional list of sample weights for each output.
      masks: Optional list of masks for each output.

  Returns:
      Returns the metric results for each output of the model.
  """
  outputs = nest.flatten(outputs)
  targets = nest.flatten(targets)
  # Invoke all(weighted and unweighted) metrics.
  metric_results = []
  if targets:
    # Insert None values corresponding to the targets that need to be skipped
    # on the model.
    if len(model._targets) != len(targets):
      new_targets = [
          None if t is None else targets.pop(0) for t in model._targets
      ]
      targets = new_targets

    metric_results = model._handle_metrics(
        outputs,
        targets=targets,
        sample_weights=sample_weights,
        masks=masks,
        return_weighted_and_unweighted_metrics=True,
        skip_target_masks=model._prepare_skip_target_masks())

  # Add metric results from the `add_metric` metrics.
  metric_results.extend([
      m.result()
      for m in model.metrics
      if m not in model._compile_metric_functions
  ])
  return metric_results


def _model_loss(model,
                inputs,
                targets,
                output_loss_metrics=None,
                sample_weights=None,
                training=False):
  """Calculates the loss for a given model.

  Arguments:
      model: The model on which metrics are being calculated.
      inputs: Either a dictionary of inputs to the model or a list of input
        arrays.
      targets: List of target arrays.
      output_loss_metrics: List of metrics that are used to aggregated output
        loss values.
      sample_weights: Optional list of sample weight arrays.
      training: Whether the model should be run in inference or training mode.

  Returns:
     Returns the model output, total loss, loss value calculated using the
     specified loss function and masks for each output. The total loss includes
     regularization losses and applies masking and sample weighting
     to the loss value.
  """
  # TODO(psv): Dedup code here with graph mode prepare_total_loss() fn.
  # Used to keep track of the total loss value (stateless).
  # eg., total_loss = loss_weight_1 * output_1_loss_fn(...) +
  #                   loss_weight_2 * output_2_loss_fn(...) +
  #                   layer losses.
  total_loss = 0
  kwargs = {}
  if model._expects_training_arg:
    kwargs['training'] = training
  if len(inputs) == 1 and not isinstance(inputs, dict):
    inputs = inputs[0]

  # Allow mixed `NumPy` and `EagerTensor` input here.
  if any(
      isinstance(input_t, (np.ndarray, float, int))
      for input_t in nest.flatten(inputs)):
    inputs = nest.map_structure(ops.convert_to_tensor, inputs)

  outs = model(inputs, **kwargs)
  outs = nest.flatten(outs)

  if targets:
    targets = training_utils.cast_if_floating_dtype_and_mismatch(targets, outs)
  # TODO(sallymatson/psv): check if we should do same mismatch fix for weights
  if sample_weights:
    sample_weights = [
        training_utils.cast_if_floating_dtype(ops.convert_to_tensor(val))
        if val is not None else None for val in sample_weights
    ]

  masks = [getattr(t, '_keras_mask', None) for t in outs]
  targets = nest.flatten(targets)

  # Used to keep track of individual output losses.
  output_losses = []

  with backend.name_scope('loss'):
    loss_fns = [
        loss_fn for loss_fn in model.loss_functions if loss_fn is not None
    ]
    for i, loss_fn in enumerate(loss_fns):
      weights = sample_weights[i] if sample_weights else None
      mask = masks[i]
      with backend.name_scope(model.output_names[i] + '_loss'):
        if mask is not None:
          mask = math_ops.cast(mask, outs[i].dtype)
          # Update weights with mask.
          if weights is None:
            weights = mask
          else:
            # Update dimensions of weights to match with mask if possible.
            weights = math_ops.cast(weights, outs[i].dtype)
            mask, _, weights = (
                tf_losses_utils.squeeze_or_expand_dimensions(
                    mask, sample_weight=weights))
            weights *= mask

        if hasattr(loss_fn, 'reduction'):
          per_sample_losses = loss_fn.call(targets[i], outs[i])
          weighted_losses = losses_utils.compute_weighted_loss(
              per_sample_losses,
              sample_weight=weights,
              reduction=losses_utils.ReductionV2.NONE)
          loss_reduction = loss_fn.reduction

          # `AUTO` loss reduction defaults to `SUM_OVER_BATCH_SIZE` for all
          # compile use cases.
          if loss_reduction == losses_utils.ReductionV2.AUTO:
            loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE

          # Compute the stateless loss value.
          output_loss = losses_utils.reduce_weighted_loss(
              weighted_losses, reduction=loss_reduction)
        else:
          # Compute the stateless loss value for a custom loss class.
          # Here we assume that the class takes care of loss reduction
          # because if this class returns a vector value we cannot
          # differentiate between use case where a custom optimizer
          # expects a vector loss value vs unreduced per-sample loss value.
          output_loss = loss_fn(targets[i], outs[i], sample_weight=weights)
          loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE

      # If the number of outputs is 1 then we don't append the loss metric
      # associated with each model output. When there are multiple outputs
      # associated with a model, each output's loss is calculated and returned
      # as part of the loss_metrics.
      if len(model.outputs) > 1:
        # Keep track of the stateful output loss result.
        output_losses.append(output_loss_metrics[i](output_loss))

      # Scale output loss for distribution. For custom losses we assume
      # reduction was mean.
      if loss_reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE:
        output_loss = losses_utils.scale_loss_for_distribution(output_loss)
      total_loss += model._loss_weights_list[i] * output_loss

    # Add regularization losses
    custom_losses = model.losses
    if custom_losses:
      total_loss += losses_utils.scale_loss_for_distribution(
          math_ops.add_n(custom_losses))

  return outs, total_loss, output_losses, masks


def _process_single_batch(model,
                          inputs,
                          targets,
                          output_loss_metrics=None,
                          sample_weights=None,
                          training=False):
  """Calculate the loss and gradient for one input batch.

     The model weights are updated if training is set to True.

  Arguments:
      model: Model whose loss has to be calculated.
      inputs: List of input arrays.
      targets: List of target arrays.
      output_loss_metrics: List of metrics that are used to aggregated output
        loss values.
      sample_weights: Optional list of sample weight arrays.
      training: The boolean represents if the weights of the model are updated.
              'fit' methods will set this to True while 'evaluate' methods will
              set this to False.

  Returns:
      output of the model, total loss, the loss and the mask
      associated with each output.

  Raises:
      ValueError: If the model has no loss to optimize.
  """
  with backend.eager_learning_phase_scope(1 if training else 0), \
      training_utils.RespectCompiledTrainableState(model):
    with GradientTape() as tape:
      outs, total_loss, output_losses, masks = (
          _model_loss(
              model,
              inputs,
              targets,
              output_loss_metrics=output_loss_metrics,
              sample_weights=sample_weights,
              training=training))
      if total_loss is None:
        raise ValueError('The model cannot be run '
                         'because it has no loss to optimize.')
      if isinstance(model.optimizer, loss_scale_optimizer.LossScaleOptimizer):
        scaled_total_loss = model.optimizer.get_scaled_loss(total_loss)
      else:
        scaled_total_loss = total_loss
    if training:
      trainable_weights = model.trainable_weights
      if trainable_weights:
        # TODO(tanzheny) b/132690565: Provide mechanism for user to override
        # model.train_on_batch.
        if hasattr(model, '_backwards'):
          model._backwards(tape, scaled_total_loss)
        else:
          grads = tape.gradient(scaled_total_loss, trainable_weights)
          if isinstance(model.optimizer,
                        loss_scale_optimizer.LossScaleOptimizer):
            grads = model.optimizer.get_unscaled_gradients(grads)
          model.optimizer.apply_gradients(zip(grads, trainable_weights))
      else:
        logging.warning('The list of trainable weights is empty. Make sure that'
                        ' you are not setting model.trainable to False before '
                        'compiling the model.')
    return outs, total_loss, output_losses, masks


def train_on_batch(model,
                   inputs,
                   targets,
                   sample_weights=None,
                   output_loss_metrics=None):
  """Calculates the loss and gradient updates for one input batch.

  Arguments:
      model: Model whose loss has to be calculated.
      inputs: Input batch data.
      targets: Target batch data.
      sample_weights: Sample weight batch data.
      output_loss_metrics: List of metrics that are used to aggregated output
        loss values.

  Returns:
      Dict with three items:
        'total_loss': list with a single tensor for overall loss,
        'output_losses': list of tensors for loss corresponding to each of the
          model output. Could be a empty list when model has only one output.
        'metrics': list of tensors for metric specified.
  """
  inputs = training_utils.cast_to_model_input_dtypes(inputs, model)
  outs, total_loss, output_losses, masks = (
      _process_single_batch(
          model,
          inputs,
          targets,
          sample_weights=sample_weights,
          training=True,
          output_loss_metrics=output_loss_metrics))
  if not isinstance(outs, list):
    outs = [outs]
  metrics_results = _eager_metrics_fn(
      model, outs, targets, sample_weights=sample_weights, masks=masks)
  total_loss = nest.flatten(total_loss)
  return {'total_loss': total_loss,
          'output_losses': output_losses,
          'metrics': metrics_results}


def test_on_batch(model,
                  inputs,
                  targets,
                  sample_weights=None,
                  output_loss_metrics=None):
  """Calculates the loss for one input batch.

  Arguments:
      model: Model whose loss has to be calculated.
      inputs: Input batch data.
      targets: Target batch data.
      sample_weights: Sample weight batch data.
      output_loss_metrics: List of metrics that are used to aggregated output
        loss values.

  Returns:
      Dict with three items:
        'total_loss': single tensor for overall loss,
        'output_losses': list of tensors for loss corresponding to each of the
          model output. Could be a empty list when model has only one output.
        'metrics': list of tensors for metric specified.
  """
  inputs = training_utils.cast_to_model_input_dtypes(inputs, model)

  with backend.eager_learning_phase_scope(0):
    outs, total_loss, output_losses, masks = (
        _model_loss(
            model,
            inputs,
            targets,
            sample_weights=sample_weights,
            training=False,
            output_loss_metrics=output_loss_metrics))
  if not isinstance(outs, list):
    outs = [outs]
  metrics_results = _eager_metrics_fn(
      model, outs, targets, sample_weights=sample_weights, masks=masks)
  total_loss = nest.flatten(total_loss)

  return {'total_loss': total_loss,
          'output_losses': output_losses,
          'metrics': metrics_results}
