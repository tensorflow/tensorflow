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

import copy

import numpy as np

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.platform import tf_logging as logging


def _get_metrics_info(metric, internal_output_shapes=None, loss_func=None):
  if metric == 'accuracy' or metric == 'acc':
    # custom handling of accuracy
    # (because of class mode duality)
    output_shape = internal_output_shapes
    if output_shape[-1] == 1 or loss_func == losses.binary_crossentropy:
      # case: binary accuracy
      acc_fn = metrics_module.binary_accuracy
    elif loss_func == losses.sparse_categorical_crossentropy:
      # case: categorical accuracy with sparse targets
      acc_fn = metrics_module.sparse_categorical_accuracy
    else:
      acc_fn = metrics_module.categorical_accuracy

    metric_name = 'acc'
    return metric_name, acc_fn
  else:
    metric_fn = metrics_module.get(metric)
    metric_name = metric_fn.__name__
    return metric_name, metric_fn


def _eager_loss_fn(outputs, targets, loss_fn, output_name):
  with backend.name_scope(output_name + '_loss'):
    loss = loss_fn(targets, outputs)
  return loss


def _eager_metrics_fn(model, outputs, targets):
  """Calculates the metrics for each output of the given model.

  Arguments:
      model: The model on which metrics are being calculated.
      outputs: The outputs of the given model.
      targets: The predictions or targets of the given model.

  Returns:
      Returns the metric names and metric results for each output of the model.
  """
  metric_names = []
  metric_results = []
  if not isinstance(outputs, list):
    outputs = [outputs]

  if not isinstance(targets, list):
    targets = [targets]

  for i in range(len(model.outputs)):
    output_metrics = model.nested_metrics[i]
    for nested_output_metric in output_metrics:
      metric_name, metric_fn = _get_metrics_info(
          nested_output_metric, backend.int_shape(model.outputs[i]),
          model.loss_functions[i])

      if len(model.output_names) > 1:
        metric_name = model.output_names[i] + '_' + metric_name
        if metric_name not in model.metrics_names:
          model.metrics_names.append(metric_name)

      with backend.name_scope(metric_name):
        metric_result = metric_fn(targets[i], outputs[i])
        metric_names.append(metric_name)
        metric_results.append(backend.mean(metric_result))

  return metric_results


def _model_loss(model, inputs, targets, sample_weights=None, training=False):
  """Calculates the loss for a given model.

  Arguments:
      model: The model on which metrics are being calculated.
      inputs: List of input arrays.
      targets: List of target arrays.
      sample_weights: Optional list of sample weight arrays.
      training: Whether the model should be run in inference or training mode.

  Returns:
     Returns the model output, total loss and loss value calculated using the
     specified loss function. The total loss includes regularization losses and
     applies masking and sample weighting to the loss value.
  """
  total_loss = 0
  if len(inputs) == 1:
    if model._expects_training_arg:
      outs = model.call(inputs[0], training=training)
    else:
      outs = model.call(inputs[0])
  else:
    if model._expects_training_arg:
      outs = model.call(inputs, training=training)
    else:
      outs = model.call(inputs)
  if not isinstance(outs, list):
    outs = [outs]

  if not isinstance(targets, list):
    targets = [targets]

  loss_metrics = []
  with backend.name_scope('loss'):
    for i, loss_fn in enumerate(model.loss_functions):
      if sample_weights:
        weights = sample_weights[i]
      else:
        weights = None

      # TODO(fchollet): support masking; in practice `_keras_mask` is never
      # set in this context currently.
      mask = outs[i]._keras_mask

      weighted_masked_fn = training_utils.weighted_masked_objective(loss_fn)
      with backend.name_scope(model.output_names[i] + '_loss'):
        output_loss = weighted_masked_fn(
            targets[i], outs[i], weights, mask=mask)
      # If the number of outputs is 1 then we don't append the loss metric
      # associated with each model output. When there are multiple outputs
      # associated with a model, each output's loss is calculated and returned
      # as part of the loss_metrics.
      if len(model.outputs) > 1:
        loss_metrics.append(backend.mean(output_loss))

      loss_weight = model.loss_weights_list[i]
      if total_loss is None:
        total_loss = loss_weight * output_loss
      else:
        total_loss += loss_weight * output_loss

    total_loss = backend.mean(total_loss)
    # Add regularization losses
    custom_losses = []
    for layer in model.layers:
      if layer.losses:
        custom_losses += layer.losses

    if custom_losses:
      total_loss += sum(custom_losses)

  return outs, total_loss, loss_metrics


def iterator_fit_loop(model,
                      inputs,
                      class_weight,
                      steps_per_epoch,
                      callback_model,
                      out_labels,
                      epoch_logs,
                      val_inputs=None,
                      val_targets=None,
                      val_sample_weights=None,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      callback_metrics=None,
                      validation_steps=None,
                      do_validation=False,
                      batch_size=None):
  """Fit function for eager execution when input is given as dataset iterator.

  Updates the given epoch logs.

  Arguments:
      model: Instance of the `Model`.
      inputs: Input dataset iterator.
      class_weight: Optional class-weight array to weight the importance of
          samples in `inputs` based on the class they belong to, as conveyed by
          the targets from the `inputs` iterator.
      steps_per_epoch: Total number of steps (batches of samples)
          before declaring one epoch finished and starting the
          next epoch.
      callback_model: Instance of `Model` to callback.
      out_labels: Output labels generated from model metric names.
      epoch_logs: Dictionary of logs from every epoch.
      val_inputs: Input data for validation.
      val_targets: Target data for validation.
      val_sample_weights: Sample weight data for validation.
      epochs: Number of times to iterate over the data
      verbose: Verbosity mode, 0, 1 or 2
      callbacks: List of callbacks to be called during training
      callback_metrics: List of strings, the display names of the metrics
          passed to the callbacks. They should be the
          concatenation of list the display names of the outputs of
           `f` and the list of display names of the outputs of `f_val`.
      validation_steps: Number of steps to run validation for (only if doing
        validation from data tensors). Ignored with default value of `None`.
      do_validation: Boolean value indicating whether we should do validation.
      batch_size: int, val_inputs and val_targets will be evaled batch by
        batch with size batch_size if they are array.

  Raises:
      ValueError: In case of mismatch between given number of inputs and
        expectations of the model.
  """
  assert isinstance(inputs, iterator_ops.EagerIterator)

  # make sure either x,y or x,y,sample_weights is provided
  if (not isinstance(inputs.output_shapes, (list, tuple)) or
      len(inputs.output_shapes) not in (2, 3)):
    raise ValueError('Please provide either inputs and targets'
                     'or inputs, targets, and sample_weights')

  for step_index in range(steps_per_epoch):
    batch_logs = {'batch': step_index, 'size': 1}
    callbacks.on_batch_begin(step_index, batch_logs)

    # Get data from the iterator.
    try:
      next_element = inputs.get_next()
    except errors.OutOfRangeError:
      logging.warning(
          'Your dataset iterator ran out of data; '
          'interrupting training. Make sure that your dataset'
          ' can generate at least `steps_per_epoch * epochs` '
          'batches (in this case, %d batches).' % steps_per_epoch * epochs)
      break

    if len(inputs.output_shapes) == 2:
      x, y = next_element
      sample_weights = None
    else:
      x, y, sample_weights = next_element

    # Validate and standardize data.
    x, y, sample_weights = model._standardize_user_data(
        x, y, sample_weight=sample_weights, class_weight=class_weight)
    x = training_utils.cast_if_floating_dtype(x)
    y = training_utils.cast_if_floating_dtype(y)
    if sample_weights:
      sample_weights = [
          training_utils.cast_if_floating_dtype(
              ops.convert_to_tensor(val, dtype=backend.floatx()))
          if val is not None else None for val in sample_weights
      ]

    if step_index == 0 and not callback_metrics:
      out_labels = model.metrics_names
      if do_validation:
        callback_metrics = copy.copy(out_labels) + [
            'val_' + n for n in out_labels
        ]
      else:
        callback_metrics = copy.copy(out_labels)
      callbacks.set_params({
          'epochs': epochs,
          'steps': steps_per_epoch,
          'verbose': verbose,
          'do_validation': do_validation,
          'metrics': callback_metrics or [],
      })

    # Train model.
    outs, loss, loss_metrics = _process_single_batch(
        model, x, y, sample_weights=sample_weights, training=True)
    if not isinstance(outs, list):
      outs = [outs]

    # Calculate metrics.
    for l, o in zip(out_labels, outs):
      batch_logs[l] = o
    # Required for eager execution
    metrics_results = _eager_metrics_fn(model, outs, y)
    batch_logs['loss'] = tensor_util.constant_value(backend.mean(loss))

    for k, v in zip(model.metrics_names,
                    [backend.mean(loss)] + loss_metrics + metrics_results):
      batch_logs[k] = tensor_util.constant_value(v)
    callbacks.on_batch_end(step_index, batch_logs)
    if callback_model.stop_training:
      break

    if step_index == steps_per_epoch - 1:
      if do_validation:
        val_outs = test_loop(
            model,
            val_inputs,
            val_targets,
            sample_weights=val_sample_weights,
            steps=validation_steps,
            verbose=0,
            batch_size=batch_size)
        if not isinstance(val_outs, list):
          val_outs = [val_outs]
        # Same labels assumed.
        for l, o in zip(out_labels, val_outs):
          epoch_logs['val_' + l] = o


def iterator_test_loop(model, inputs, steps, verbose=0):
  """Test function for eager execution when input is given as dataset iterator.

  Arguments:
      model: Model instance that is being evaluated in Eager mode.
      inputs: Input dataset iterator.
      steps: Total number of steps (batches of samples) before declaring
      predictions finished.
      verbose: Verbosity mode.

  Returns:
      Scalar loss (if the model has a single output and no metrics)
      or list of scalars (if the model has multiple outputs
      and/or metrics). The attribute `model.metrics_names` will give you
      the display labels for the scalar outputs.

  Raises:
      ValueError: In case of mismatch between given number of inputs and
        expectations of the model.
  """
  assert isinstance(inputs, iterator_ops.EagerIterator)
  # make sure either x,y or x,y,sample_weights is provided
  if (not isinstance(inputs.output_shapes, (list, tuple)) or
      len(inputs.output_shapes) < 2 or len(inputs.output_shapes) > 3):
    raise ValueError('Please provide either inputs and targets'
                     'or inputs, targets, and sample_weights')
  outs = []
  num_samples = 0
  if verbose == 1:
    progbar = generic_utils.Progbar(target=steps)
  for step_index in range(steps):
    # Get data from the iterator.
    try:
      next_element = inputs.get_next()
    except errors.OutOfRangeError:
      logging.warning(
          'Your dataset iterator ran out of data interrupting testing. '
          'Make sure that your dataset can generate at least `steps` batches '
          '(in this case, %d batches).', steps)
      break

    if len(inputs.output_shapes) == 2:
      x, y = next_element
      sample_weights = None
    else:
      x, y, sample_weights = next_element

    # Validate and standardize data.
    x, y, sample_weights = model._standardize_user_data(x, y)
    x = training_utils.cast_if_floating_dtype(x)
    y = training_utils.cast_if_floating_dtype(y)

    # Calculate model output, loss values.
    loss_outs, loss, loss_metrics = _model_loss(
        model, x, y, sample_weights=sample_weights, training=False)
    metrics_results = _eager_metrics_fn(model, loss_outs, y)
    batch_outs = []
    for _, v in zip(model.metrics_names,
                    [backend.mean(loss)] + loss_metrics + metrics_results):
      batch_outs.append(tensor_util.constant_value(v))

    # Get current step size.
    if isinstance(x, list):
      step_size = x[0].get_shape().as_list()[0]
    else:
      step_size = x.get_shape().as_list()[0]

    # Accumulate results in output array.
    if not isinstance(batch_outs, list):
      batch_outs = [batch_outs]
    if step_index == 0:
      for _ in enumerate(batch_outs):
        outs.append(0.)
    for i, batch_out in enumerate(batch_outs):
      outs[i] += batch_out * step_size

    # Calculate sample size.
    num_samples += step_size
    if verbose == 1:
      progbar.update(step_index + 1)

  for i in range(len(outs)):
    outs[i] /= num_samples
  if len(outs) == 1:
    return outs[0]
  return outs


def iterator_predict_loop(model, inputs, steps, verbose=0):
  """Predict function for eager execution when input is dataset iterator.

  Arguments:
      model: Instance of `Model`.
      inputs: Input dataset iterator.
      steps: Total number of steps (batches of samples) before declaring
          `_predict_loop` finished.
      verbose: Verbosity mode.

  Returns:
      Array of predictions (if the model has a single output)
      or list of arrays of predictions (if the model has multiple outputs).

  Raises:
      ValueError: In case of mismatch between given number of inputs and
        expectations of the model.
  """
  assert isinstance(inputs, iterator_ops.EagerIterator)
  if not isinstance(inputs.output_shapes,
                    (list, tuple)) or len(inputs.output_shapes) > 2:
    raise ValueError(
        'Please provide data as a list or tuple of 1 or 2 elements '
        ' - input or input and target pair. Received %s. We do not use the '
        '`target` value here.' % inputs.output_shapes)
  outs = []
  if verbose == 1:
    progbar = generic_utils.Progbar(target=steps)
  for step_index in range(steps):
    # Get data from the iterator.
    try:
      next_element = inputs.get_next()
    except errors.OutOfRangeError:
      logging.warning(
          'Your dataset iterator ran out of data; '
          'interrupting prediction. Make sure that your '
          'dataset can generate at least `steps` '
          'batches (in this case, %d batches).', steps)
      break

    # expects a tuple, where first element of tuple represents inputs
    x = next_element[0]

    # Validate and standardize data.
    x, _, _ = model._standardize_user_data(x)
    x = training_utils.cast_if_floating_dtype(x)

    if model._expects_training_arg:
      batch_outs = model.call(x[0] if len(x) == 1 else x, training=False)
    else:
      batch_outs = model.call(x[0] if len(x) == 1 else x)
    if not isinstance(batch_outs, list):
      batch_outs = [batch_outs]

    # We collect the results from every step and then concatenate them once
    # in the end. This is an expensive process. We are doing this because we
    # do not know the number of samples beforehand.
    if step_index == 0:
      for _ in batch_outs:
        outs.append([])
    for i, batch_out in enumerate(batch_outs):
      outs[i].append(backend.get_value(batch_out))

    if verbose == 1:
      progbar.update(step_index + 1)
  for i, out in enumerate(outs):
    outs[i] = np.concatenate(tuple(out), axis=0)
  if len(outs) == 1:
    return outs[0]
  return outs


def _process_single_batch(model,
                          inputs,
                          targets,
                          sample_weights=None,
                          training=False):
  """Calculate the loss and gradient for one input batch.

     The model weights are updated if training is set to True.

  Arguments:
      model: Model whose loss has to be calculated.
      inputs: List of input arrays.
      targets: List of target arrays.
      sample_weights: Optional list of sample weight arrays.
      training: The boolean represents if the weights of the model are updated.
              'fit' methods will set this to True while 'evaluate' methods will
              set this to False.

  Returns:
      output of the model, total loss and the loss associated with each output.

  Raises:
      ValueError: If the model has no loss to optimize.
  """
  with backend.learning_phase_scope(1 if training else 0):
    with GradientTape() as tape:
      outs, loss, loss_metrics = _model_loss(model, inputs, targets,
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
    return outs, loss, loss_metrics


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
  if len(inputs) and tensor_util.is_tensor(inputs[0]):
    inputs = training_utils.cast_if_floating_dtype(inputs)
    targets = training_utils.cast_if_floating_dtype(targets)
  else:
    inputs = [
        ops.convert_to_tensor(val, dtype=backend.floatx()) for val in inputs
    ]
    targets = [
        ops.convert_to_tensor(val, dtype=backend.floatx()) for val in targets
    ]
  if sample_weights:
    sample_weights = [
        ops.convert_to_tensor(val, dtype=backend.floatx())
        if val is not None else None for val in sample_weights
    ]

  outs, loss, _ = _process_single_batch(
      model, inputs, targets, sample_weights=sample_weights, training=True)
  if not isinstance(outs, list):
    outs = [outs]
  metrics_results = _eager_metrics_fn(model, outs, targets)
  if not isinstance(loss, list):
    loss = [loss]
  return loss + metrics_results


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
  if len(inputs) and tensor_util.is_tensor(inputs[0]):
    inputs = training_utils.cast_if_floating_dtype(inputs)
    targets = training_utils.cast_if_floating_dtype(targets)
  else:
    inputs = [
        ops.convert_to_tensor(val, dtype=backend.floatx()) for val in inputs
    ]
    targets = [
        ops.convert_to_tensor(val, dtype=backend.floatx()) for val in targets
    ]
  if sample_weights:
    sample_weights = [
        ops.convert_to_tensor(val, dtype=backend.floatx())
        if val is not None else None for val in sample_weights
    ]
  outs, loss, loss_metrics = _model_loss(
      model, inputs, targets, sample_weights=sample_weights, training=False)
  if not isinstance(outs, list):
    outs = [outs]
  metrics_results = _eager_metrics_fn(model, outs, targets)
  if not isinstance(loss, list):
    loss = [loss]
  return loss + loss_metrics + metrics_results


def fit_loop(model,
             inputs,
             targets,
             sample_weights=None,
             class_weight=None,
             val_inputs=None,
             val_targets=None,
             val_sample_weights=None,
             batch_size=None,
             epochs=1,
             verbose=1,
             callbacks=None,
             shuffle=True,
             callback_metrics=None,
             initial_epoch=0,
             steps_per_epoch=None,
             validation_steps=None):
  """Fit function for eager execution.

  Arguments:
      model: Instance of the model that is being executed in Eager mode.
      inputs: List of input arrays.
      targets: List of target arrays.
      sample_weights: Optional list of sample weight arrays.
      class_weight: Optional class-weight array to weight the importance of
          samples in `inputs` based on the class they belong to, as conveyed by
          `targets`.
      val_inputs: Input data for validation.
      val_targets: Target data for validation.
      val_sample_weights: Sample weight data for validation.
      batch_size: Integer batch size or None if unknown.
      epochs: Number of times to iterate over the data
      verbose: Verbosity mode, 0, 1 or 2
      callbacks: List of callbacks to be called during training
      shuffle: Whether to shuffle the data at the beginning of each epoch
      callback_metrics: List of strings, the display names of the metrics
          passed to the callbacks. They should be the
          concatenation of list the display names of the outputs of
           `f` and the list of display names of the outputs of `f_val`.
      initial_epoch: Epoch at which to start training
          (useful for resuming a previous training run)
      steps_per_epoch: Total number of steps (batches of samples)
          before declaring one epoch finished and starting the
          next epoch. Ignored with the default value of `None`.
      validation_steps: Number of steps to run validation for (only if doing
        validation from data tensors). Ignored with default value of `None`.

  Returns:
      `History` object.

  Raises:
    ValueError: In case of invalid argument values.
  """
  # Convert training inputs to an EagerIterator
  inputs, steps_per_epoch = training_utils.convert_to_iterator(
      x=inputs,
      y=targets,
      sample_weights=sample_weights,
      batch_size=batch_size,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      shuffle=shuffle)
  # Required for eager execution
  with backend.learning_phase_scope(1):
    do_validation = False
    if val_inputs:
      do_validation = True

    num_train_samples = None
    out_labels = None
    if model._is_compiled:
      out_labels = model.metrics_names
      if do_validation:
        callback_metrics = copy.copy(out_labels) + [
            'val_' + n for n in out_labels
        ]
      else:
        callback_metrics = copy.copy(out_labels)

    model.history = cbks.History()
    callbacks = [cbks.BaseLogger()] + (callbacks or []) + [model.history]
    if verbose:
      callbacks += [cbks.ProgbarLogger('steps')]
    callbacks = cbks.CallbackList(callbacks)

    # it's possible to callback a different model than self
    # (used by Sequential models)
    if hasattr(model, 'callback_model') and model.callback_model:
      callback_model = model.callback_model
    else:
      callback_model = model

    callbacks.set_model(callback_model)

    callback_params = {
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps_per_epoch,
        'samples': num_train_samples,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics or [],
    }
    if validation_steps:
      callback_params.update({'validation_steps': validation_steps})
    callbacks.set_params(callback_params)

    for cbk in callbacks:
      if not val_inputs:
        cbk.validation_data = []
      elif isinstance(val_inputs, iterator_ops.EagerIterator):
        cbk.validation_data = val_inputs
      elif val_sample_weights:
        cbk.validation_data = val_inputs + val_targets + val_sample_weights
      else:
        cbk.validation_data = val_inputs + val_targets
    # validation_data must be set before on_train_begin() is called
    # so that TensorboardCallback can validate its input
    callbacks.on_train_begin()
    callback_model.stop_training = False

    for epoch in range(initial_epoch, epochs):
      callbacks.on_epoch_begin(epoch)
      epoch_logs = {}
      iterator_fit_loop(
          model,
          inputs,
          class_weight,
          steps_per_epoch=steps_per_epoch,
          callback_model=callback_model,
          out_labels=out_labels,
          epoch_logs=epoch_logs,
          val_inputs=val_inputs,
          val_targets=val_targets,
          val_sample_weights=val_sample_weights,
          epochs=epochs,
          verbose=verbose,
          callbacks=callbacks,
          callback_metrics=callback_metrics,
          validation_steps=validation_steps,
          do_validation=do_validation,
          batch_size=batch_size)
      callbacks.on_epoch_end(epoch, epoch_logs)
      if callback_model.stop_training:
        break
  callbacks.on_train_end()
  return model.history


def test_loop(model, inputs, targets,
              sample_weights=None,
              batch_size=None,
              verbose=0,
              steps=None):
  """Test function for eager execution.

  Arguments:
      model: Model instance that is being evaluated in Eager mode.
      inputs: List of input arrays.
      targets: List of target arrays.
      sample_weights: Optional list of sample weight arrays.
      batch_size: integer batch size or `None`.
      verbose: verbosity mode.
      steps: Total number of steps (batches of samples)
          before declaring predictions finished.
          Ignored with the default value of `None`.

  Returns:
      Scalar loss (if the model has a single output and no metrics)
      or list of scalars (if the model has multiple outputs
      and/or metrics). The attribute `model.metrics_names` will give you
      the display labels for the scalar outputs.
  """
  inputs, steps = training_utils.convert_to_iterator(
      x=inputs,
      y=targets,
      sample_weights=sample_weights,
      batch_size=batch_size,
      steps_per_epoch=steps)
  with backend.learning_phase_scope(0):
    return iterator_test_loop(model, inputs, steps, verbose=verbose)


def predict_loop(model, inputs,
                 batch_size=32,
                 verbose=0,
                 steps=None):
  """Predict function for eager execution.

  Arguments:
      model: Instance of `Model`.
      inputs: List of input arrays.
      batch_size: integer batch size.
      verbose: verbosity mode.
      steps: Total number of steps (batches of samples)
          before declaring `_predict_loop` finished.
          Ignored with the default value of `None`.

  Returns:
      Array of predictions (if the model has a single output)
      or list of arrays of predictions
      (if the model has multiple outputs).
  """
  with backend.learning_phase_scope(0):
    inputs, steps = training_utils.convert_to_iterator(
        x=inputs, batch_size=batch_size, steps_per_epoch=steps)
    return iterator_predict_loop(model, inputs, steps, verbose=verbose)
