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

import collections

from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils.losses_utils import squeeze_or_expand_dimensions
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


def _eager_loss_fn(outputs, targets, loss_fn, output_name):
  with backend.name_scope(output_name + '_loss'):
    loss = loss_fn(targets, outputs)
  return loss


def _eager_metrics_fn(model,
                      outputs,
                      targets,
                      sample_weights=None,
                      masks=None,
                      return_stateful_result=True):
  """Calculates the metrics for each output of the given model.

  Arguments:
      model: The model on which metrics are being calculated.
      outputs: The outputs of the given model.
      targets: The predictions or targets of the given model.
      sample_weights: Optional list of sample weights for each output.
      masks: Optional list of masks for each output.
      return_stateful_result: Boolean, indicates whether the stateful
        (aggregated)/stateless metric result should be returned.

  Returns:
      Returns the metric results for each output of the model.
  """
  outputs = nest.flatten(outputs)
  targets = nest.flatten(targets)
  # TODO(psv): Consider supporting skip target indices in eager mode?
  metric_results = model._handle_metrics(
      outputs,
      targets=targets,
      sample_weights=sample_weights,
      masks=masks,
      return_stateful_result=return_stateful_result)
  return [backend.mean(t) for t in metric_results]


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
  total_loss = 0
  kwargs = {}
  if model._expects_training_arg:
    kwargs['training'] = training
  if len(inputs) == 1 and not isinstance(inputs, dict):
    inputs = inputs[0]

  if model._compute_output_and_mask_jointly:
    outs, masks = model._call_and_compute_mask(inputs, **kwargs)
    masks = nest.flatten(masks)
  else:
    outs = model.call(inputs, **kwargs)
    masks = None

  outs = nest.flatten(outs)
  if masks is None:
    masks = [None for _ in outs]
  targets = nest.flatten(targets)

  loss_metrics = []
  aggregated_loss_metrics = []
  with backend.name_scope('loss'):
    for i, loss_fn in enumerate(model.loss_functions):
      if sample_weights:
        weights = sample_weights[i]
      else:
        weights = None
      mask = masks[i]
      with backend.name_scope(model.output_names[i] + '_loss'):
        if mask is not None:
          mask = math_ops.cast(mask, outs[i].dtype)
          # Update weights with mask.
          if weights is None:
            weights = mask
          else:
            # Update dimensions of weights to match with mask if possible.
            mask, _, weights = squeeze_or_expand_dimensions(mask, None, weights)
            weights *= mask
        output_loss = loss_fn(targets[i], outs[i], sample_weight=weights)

      # If the number of outputs is 1 then we don't append the loss metric
      # associated with each model output. When there are multiple outputs
      # associated with a model, each output's loss is calculated and returned
      # as part of the loss_metrics.
      if len(model.outputs) > 1:
        loss_metrics.append(backend.mean(output_loss))

        if output_loss_metrics is not None:
          # Keep track of the stateful loss result.
          aggregated_loss_metrics.append(
              training_utils.call_metric_function(
                  output_loss_metrics[i],
                  targets[i],
                  outs[i],
                  weights=weights,
                  mask=mask))

      loss_weight = model.loss_weights_list[i]
      if total_loss is None:
        total_loss = loss_weight * output_loss
      else:
        total_loss += loss_weight * output_loss

    total_loss = backend.mean(total_loss)
    # Add regularization losses
    custom_losses = model.losses
    if custom_losses:
      total_loss += math_ops.add_n(custom_losses)
    model._clear_losses()

  return outs, total_loss, loss_metrics, aggregated_loss_metrics, masks


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
  with backend.eager_learning_phase_scope(1 if training else 0):
    with GradientTape() as tape:
      outs, loss, loss_metrics, aggregated_loss_metrics, masks\
        = _model_loss(
            model,
            inputs,
            targets,
            output_loss_metrics=output_loss_metrics,
            sample_weights=sample_weights,
            training=training)
      if loss is None:
        raise ValueError('The model cannot be run '
                         'because it has no loss to optimize.')
    if training:
      if not model._collected_trainable_weights:
        logging.warning('The list of trainable weights is empty. Make sure that'
                        ' you are not setting model.trainable to False before '
                        'compiling the model.')
      else:
        grads = tape.gradient(loss, model._collected_trainable_weights)
        model.optimizer.apply_gradients(zip(grads,
                                            model._collected_trainable_weights))
    return outs, loss, loss_metrics, aggregated_loss_metrics, masks


def train_on_batch(model, inputs, targets, sample_weights=None):
  """Calculates the loss and gradient updates for one input batch.

  Arguments:
      model: Model whose loss has to be calculated.
      inputs: Input batch data.
      targets: Target batch data.
      sample_weights: Sample weight batch data.

  Returns:
      total loss and the loss associated with each output.
  """
  if isinstance(inputs, collections.Sequence):
    if len(inputs) and tensor_util.is_tensor(inputs[0]):
      inputs = training_utils.cast_if_floating_dtype(inputs)
      targets = training_utils.cast_if_floating_dtype(targets)
    else:
      inputs = training_utils.cast_if_floating_dtype([
          ops.convert_to_tensor(val) for val in inputs
      ])
      targets = training_utils.cast_if_floating_dtype([
          ops.convert_to_tensor(val) for val in targets
      ])
  if sample_weights:
    sample_weights = [
        training_utils.cast_if_floating_dtype(ops.convert_to_tensor(val))
        if val is not None else None for val in sample_weights
    ]

  outs, loss, loss_metrics, _, masks = _process_single_batch(
      model, inputs, targets, sample_weights=sample_weights, training=True)
  if not isinstance(outs, list):
    outs = [outs]
  metrics_results = _eager_metrics_fn(
      model,
      outs,
      targets,
      sample_weights=sample_weights,
      masks=masks,
      return_stateful_result=True)
  loss = nest.flatten(loss)

  return [
      tensor_util.constant_value(v)
      for v in loss + loss_metrics + metrics_results
  ]


def test_on_batch(model, inputs, targets, sample_weights=None):
  """Calculates the loss for one input batch.

  Arguments:
      model: Model whose loss has to be calculated.
      inputs: Input batch data.
      targets: Target batch data.
      sample_weights: Sample weight batch data.

  Returns:
      total loss, loss and metrics associated with each output.
  """
  if isinstance(inputs, collections.Sequence):
    if len(inputs) and tensor_util.is_tensor(inputs[0]):
      inputs = training_utils.cast_if_floating_dtype(inputs)
      targets = training_utils.cast_if_floating_dtype(targets)
    else:
      inputs = training_utils.cast_if_floating_dtype([
          ops.convert_to_tensor(val) for val in inputs
      ])
      targets = training_utils.cast_if_floating_dtype([
          ops.convert_to_tensor(val) for val in targets
      ])
  if sample_weights:
    sample_weights = [
        training_utils.cast_if_floating_dtype(ops.convert_to_tensor(val))
        if val is not None else None for val in sample_weights
    ]
  outs, loss, loss_metrics, _, masks = _model_loss(
      model, inputs, targets, sample_weights=sample_weights, training=False)
  if not isinstance(outs, list):
    outs = [outs]
  metrics_results = _eager_metrics_fn(
      model,
      outs,
      targets,
      sample_weights=sample_weights,
      masks=masks,
      return_stateful_result=True)
  loss = nest.flatten(loss)

  return [
      tensor_util.constant_value(v)
      for v in loss + loss_metrics + metrics_results
  ]
