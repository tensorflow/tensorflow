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
"""Keras training and evaluation routines.
"""
# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import callbacks as cbks
from tensorflow.python.keras._impl.keras import losses
from tensorflow.python.keras._impl.keras import metrics as metrics_module
from tensorflow.python.keras._impl.keras.utils.generic_utils import Progbar


def _make_batches(size, batch_size):
  """Returns a list of batch indices (tuples of indices).

  Arguments:
      size: Integer, total size of the data to slice into batches.
      batch_size: Integer, batch size.

  Returns:
      A list of tuples of array indices.
  """
  num_batches = int(np.ceil(size / float(batch_size)))
  return [(i * batch_size, min(size, (i + 1) * batch_size))
          for i in range(0, num_batches)]


def _slice_arrays(arrays, start=None, stop=None):
  """Slice an array or list of arrays.

  This takes an array-like, or a list of
  array-likes, and outputs:
      - arrays[start:stop] if `arrays` is an array-like
      - [x[start:stop] for x in arrays] if `arrays` is a list

  Can also work on list/array of indices: `_slice_arrays(x, indices)`

  Arguments:
      arrays: Single array or list of arrays.
      start: can be an integer index (start index)
          or a list/array of indices
      stop: integer (stop index); should be None if
          `start` was a list.

  Returns:
      A slice of the array(s).

  Raises:
      ValueError: If the value of start is a list and stop is not None.
  """
  if arrays is None:
    return [None]
  if isinstance(start, list) and stop is not None:
    raise ValueError('The stop argument has to be None if the value of start is'
                     'a list.')
  elif isinstance(arrays, list):
    if hasattr(start, '__len__'):
      # hdf5 datasets only support list objects as indices
      if hasattr(start, 'shape'):
        start = start.tolist()
      return [None if x is None else x[start] for x in arrays]
    else:
      return [None if x is None else x[start:stop] for x in arrays]
  else:
    if hasattr(start, '__len__'):
      if hasattr(start, 'shape'):
        start = start.tolist()
      return arrays[start]
    elif hasattr(start, '__getitem__'):
      return arrays[start:stop]
    else:
      return [None]


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
  with K.name_scope(output_name + '_loss'):
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
          nested_output_metric, model._internal_output_shapes[i],
          model.loss_functions[i])

      if len(model.output_names) > 1:
        metric_name = model.output_names[i] + '_' + metric_name
        if metric_name not in model.metrics_names:
          model.metrics_names.append(metric_name)

      with K.name_scope(metric_name):
        metric_result = metric_fn(outputs[i], targets[i])
        metric_names.append(metric_name)
        metric_results.append(K.mean(metric_result))

  return metric_names, metric_results


def _model_loss(model, inputs, targets):
  """Calculates the loss for a given model.

  Arguments:
     model: The model on which metrics are being calculated.
     inputs: The inputs of the given model. This is typically the mini batch of
              data that is fed to the model.
     targets: The predictions or targets of the given model.

  Returns:
     Returns the model output, total loss and loss value calculated using the
     specified loss function. The total loss includes regularization losses and
     applies masking and sample weighting to the loss value.
  """
  total_loss = 0
  outs = model(inputs)
  if not isinstance(outs, list):
    outs = [outs]

  if not isinstance(targets, list):
    targets = [targets]

  loss_metrics = []
  with K.name_scope('loss'):
    for i, loss_fn in enumerate(model.loss_functions):
      # compute the loss
      output_loss = _eager_loss_fn(outs[i], targets[i], loss_fn,
                                   model.output_names[i])
      loss_metrics.append(K.mean(output_loss))

      mask = outs[i]._keras_mask
      # adapted from weighted_loss_fn
      if mask is not None:
        # mask should have the same shape as output_loss
        output_loss *= mask
        #  the loss per batch should be proportional
        #  to the number of unmasked samples.
        output_loss /= K.mean(mask)

      # adapted from weighted_loss_fn
      # apply sample weighting
      if model.sample_weights:
        # reduce score_array to same ndim as weight array
        ndim = K.ndim(output_loss)
        weight_ndim = K.ndim(model.sample_weights)
        output_loss = K.mean(output_loss, axis=list(range(weight_ndim, ndim)))
        output_loss *= model.sample_weights
        output_loss /= K.mean(K.cast(K.not_equal(model.sample_weights, 0),
                                     K.floatx()))
        output_loss = K.mean(output_loss)

      loss_weight = model.loss_weights_list[i]
      if total_loss is None:
        total_loss = loss_weight * output_loss
      else:
        total_loss += loss_weight * output_loss

    total_loss = K.mean(total_loss)
    # Add regularization losses
    custom_losses = []
    for layer in model.layers:
      if layer.losses:
        custom_losses += layer.losses

    if custom_losses:
      total_loss += sum(custom_losses)

  return outs, total_loss, loss_metrics


def _process_single_batch(eager_model_inputs, eager_model_outputs, model,
                          training=True):
  """Calculate the loss and gradient for one input batch.

     The model weights are updated if training is set to True.

  Arguments:
      eager_model_inputs: Input batch data.
      eager_model_outputs: Output batch data.
      model: Model whose loss has to be calculated.
      training: The boolean represents if the weights of the model are updated.
              'fit' methods will set this to True while 'evaluate' methods will
              set this to False.

  Returns:
      output of the model, total loss and the loss associated with each output.

  Raises:
      ValueError: If the model loss is 0 or if the trainable weights list is
                  empty when the trainable parameter is set to True.
  """
  K.set_learning_phase(training)
  with GradientTape() as tape:
    outs, loss, loss_metrics = _model_loss(model, eager_model_inputs,
                                           eager_model_outputs)
    if loss is None:
      raise ValueError('The model cannot be run '
                       'because it has no loss to optimize.')
  if training:
    if not model._collected_trainable_weights:
      raise ValueError('The list of trainable weights is empty. Make sure that '
                       'you are not setting model.trainable to False before '
                       'compiling the model.')
    grads = tape.gradient(loss, model._collected_trainable_weights)
    model.optimizer.apply_gradients(zip(grads,
                                        model._collected_trainable_weights))
  return outs, loss, loss_metrics


def train_on_batch(model, ins):
  """Calculates the loss and gradient updates for one input batch.

  Arguments:
      model: Given model on which loss and gradients are calculated.
      ins: Input and output batch numpy arrays.

  Returns:
      total loss and the loss associated with each output.
  """
  ins_batch_converted = []
  for ib in ins:
    ins_batch_converted.append(ops.convert_to_tensor(ib, dtype=K.floatx()))
  eager_model_inputs = []
  eager_model_outputs = []
  for i in range(len(model.inputs)):
    eager_model_inputs.append(ins_batch_converted[i])
  for i in range(len(model.inputs), len(ins_batch_converted)):
    eager_model_outputs.append(ins_batch_converted[i])
  outs, loss, _ = _process_single_batch(
      eager_model_inputs, eager_model_outputs, model)
  if not isinstance(outs, list):
    outs = [outs]
  _, metrics_results = _eager_metrics_fn(
      model, outs, eager_model_outputs)
  if not isinstance(loss, list):
    loss = [loss]
  return loss + metrics_results


def test_on_batch(model, ins):
  """Calculates the loss for one input batch.

  Arguments:
      model: Given model on which loss is calculated.
      ins: Input and output batch numpy arrays.

  Returns:
      total loss, loss and metrics associated with each output.
  """
  ins_batch_converted = []
  for ib in ins:
    ins_batch_converted.append(ops.convert_to_tensor(ib, dtype=K.floatx()))
  eager_model_inputs = []
  eager_model_outputs = []
  for i in range(len(model.inputs)):
    eager_model_inputs.append(ins_batch_converted[i])
  for i in range(len(model.inputs), len(ins_batch_converted)):
    eager_model_outputs.append(ins_batch_converted[i])
  outs, loss, loss_metrics = _process_single_batch(
      eager_model_inputs, eager_model_outputs, model, training=False)
  if not isinstance(outs, list):
    outs = [outs]
  metric_names, metrics_results = _eager_metrics_fn(
      model, outs, eager_model_outputs)
  model.metrics_names.append(metric_names)
  if not isinstance(loss, list):
    loss = [loss]
  return loss + loss_metrics + metrics_results


def fit_loop(
    model,
    ins,
    out_labels=None,
    batch_size=None,
    epochs=100,
    verbose=1,
    callbacks=None,
    val_ins=None,
    shuffle=True,
    callback_metrics=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None):
  """Abstract fit function for `f(ins)`.

  Assume that f returns a list, labeled by out_labels.

  Arguments:
      model: Instance of the model that is being executed in Eager mode.
      ins: List of tensors to be fed to `f`
      out_labels: List of strings, display names of
          the outputs of `f`
      batch_size: Integer batch size or None if unknown.
      epochs: Number of times to iterate over the data
      verbose: Verbosity mode, 0, 1 or 2
      callbacks: List of callbacks to be called during training
      val_ins: List of tensors to be fed to `val_f`
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
  # Required for Eager mode
  K.set_learning_phase(True)

  do_validation = False
  if val_ins:
    do_validation = True
    if (verbose and ins and hasattr(ins[0], 'shape') and
        hasattr(val_ins[0], 'shape')):
      print('Train on %d samples, validate on %d samples' %
            (ins[0].shape[0], val_ins[0].shape[0]))
  if validation_steps:
    if steps_per_epoch is None:
      raise ValueError('Can only use `validation_steps` when doing step-wise '
                       'training, i.e. `steps_per_epoch` must be set.')
    do_validation = True

  num_train_samples = model._check_num_samples(
      ins, batch_size, steps_per_epoch, 'steps_per_epoch')

  if num_train_samples is not None:
    index_array = np.arange(num_train_samples)

  model.history = cbks.History()
  callbacks = [cbks.BaseLogger()] + (callbacks or []) + [model.history]
  if verbose:
    if steps_per_epoch is not None:
      count_mode = 'steps'
    else:
      count_mode = 'samples'
    callbacks += [cbks.ProgbarLogger(count_mode)]
  callbacks = cbks.CallbackList(callbacks)
  out_labels = out_labels or []

  # it's possible to callback a different model than self
  # (used by Sequential models)
  if hasattr(model, 'callback_model') and model.callback_model:
    callback_model = model.callback_model
  else:
    callback_model = model

  callbacks.set_model(callback_model)

  callbacks.set_params({
      'batch_size': batch_size,
      'epochs': epochs,
      'steps': steps_per_epoch,
      'samples': num_train_samples,
      'verbose': verbose,
      'do_validation': do_validation,
      'metrics': callback_metrics or [],
  })
  callbacks.on_train_begin()
  callback_model.stop_training = False
  for cbk in callbacks:
    cbk.validation_data = val_ins

  for epoch in range(initial_epoch, epochs):
    callbacks.on_epoch_begin(epoch)
    epoch_logs = {}
    if shuffle == 'batch':
      index_array = model._batch_shuffle(index_array, batch_size)
    elif shuffle:
      np.random.shuffle(index_array)

    batches = _make_batches(num_train_samples, batch_size)

    for batch_index, (batch_start, batch_end) in enumerate(batches):
      batch_ids = index_array[batch_start:batch_end]
      try:
        if isinstance(ins[-1], float):
          # Do not slice the training phase flag.
          ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
        else:
          ins_batch = _slice_arrays(ins, batch_ids)
      except TypeError:
        raise TypeError('TypeError while preparing batch. '
                        'If using HDF5 input data, '
                        'pass shuffle="batch".')
      batch_logs = {}
      batch_logs['batch'] = batch_index
      batch_logs['size'] = len(batch_ids)

      callbacks.on_batch_begin(batch_index, batch_logs)

      ins_batch_converted = []
      for ib in ins_batch:
        ins_batch_converted.append(ops.convert_to_tensor(ib, dtype=K.floatx()))
      eager_model_inputs = []
      eager_model_outputs = []
      for i in range(len(model.inputs)):
        eager_model_inputs.append(ins_batch_converted[i])

      for i in range(len(model.inputs), len(ins_batch_converted)):
        eager_model_outputs.append(ins_batch_converted[i])

      outs, loss, loss_metrics = _process_single_batch(eager_model_inputs,
                                                       eager_model_outputs,
                                                       model)

      if not isinstance(outs, list):
        outs = [outs]

      for l, o in zip(out_labels, outs):
        batch_logs[l] = o
      # Required for Eager mode
      metrics_names, metrics_results = _eager_metrics_fn(model, outs,
                                                         eager_model_outputs)
      batch_logs['loss'] = tensor_util.constant_value(K.mean(loss))

      # TODO(anjalisridhar): Move this to compile to avoid duplicate code.
      # In graph mode we set the metric names in compile. However in
      # Eager mode we calculate the metrics for each batch in fit_loop.
      # We could calculate the metric names and functions in compile.
      # This would avoid setting the callback parameters separately.
      # We need to do this for the first iteration alone
      for m in metrics_names:
        if m not in callback_metrics:
          callback_metrics.append(m)

      callbacks.set_params({
          'batch_size': batch_size,
          'epochs': epochs,
          'steps': steps_per_epoch,
          'samples': num_train_samples,
          'verbose': verbose,
          'do_validation': do_validation,
          'metrics': callback_metrics or [],
      })

      for k, v in zip(model.metrics_names,
                      [K.mean(loss)] + loss_metrics + metrics_results):
        batch_logs[k] = tensor_util.constant_value(v)

      callbacks.on_batch_end(batch_index, batch_logs)
      if callback_model.stop_training:
        break

      if batch_index == len(batches) - 1:  # Last batch.
        if do_validation:
          val_outs = test_loop(
              model, val_ins, batch_size=batch_size, verbose=0)
          if not isinstance(val_outs, list):
            val_outs = [val_outs]
          # Same labels assumed.
          for l, o in zip(out_labels, val_outs):
            epoch_logs['val_' + l] = o
    callbacks.on_epoch_end(epoch, epoch_logs)
    if callback_model.stop_training:
      break
  callbacks.on_train_end()
  return model.history


def test_loop(model, ins, batch_size=None, verbose=0, steps=None):
  """Abstract method to loop over some data in batches.

  Arguments:
      model: Model instance that is being evaluated in Eager mode.
      ins: list of tensors to be fed to `f`.
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
  K.set_learning_phase(False)
  num_samples = model._check_num_samples(ins, batch_size, steps, 'steps')
  outs = []
  if verbose == 1:
    progbar = Progbar(target=num_samples)
  batches = _make_batches(num_samples, batch_size)
  index_array = np.arange(num_samples)
  for batch_index, (batch_start, batch_end) in enumerate(batches):
    batch_ids = index_array[batch_start:batch_end]
    if isinstance(ins[-1], float):
      # Do not slice the training phase flag.
      ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
    else:
      ins_batch = _slice_arrays(ins, batch_ids)

    ins_batch_converted = []
    for ib in ins_batch:
      ins_batch_converted.append(ops.convert_to_tensor(ib, dtype=K.floatx()))

    eager_model_inputs = []
    eager_model_outputs = []
    for i in range(len(model.inputs)):
      eager_model_inputs.append(ins_batch_converted[i])

    for i in range(len(model.inputs), len(ins_batch_converted)):
      eager_model_outputs.append(ins_batch_converted[i])

    loss_outs, loss, loss_metrics = _model_loss(model, eager_model_inputs,
                                                eager_model_outputs)
    _, metrics_results = _eager_metrics_fn(model, loss_outs,
                                           eager_model_outputs)
    batch_outs = []
    for _, v in zip(model.metrics_names,
                    [K.mean(loss)] + loss_metrics + metrics_results):
      batch_outs.append(tensor_util.constant_value(v))

    if isinstance(batch_outs, list):
      if batch_index == 0:
        for batch_out in enumerate(batch_outs):
          outs.append(0.)
      for i, batch_out in enumerate(batch_outs):
        outs[i] += batch_out * len(batch_ids)
    else:
      if batch_index == 0:
        outs.append(0.)
      outs[0] += batch_outs * len(batch_ids)

    if verbose == 1:
      progbar.update(batch_end)
  for i in range(len(outs)):
    outs[i] /= num_samples
  if len(outs) == 1:
    return outs[0]
  return outs


def predict_loop(model, ins, batch_size=32, verbose=0, steps=None):
  """Abstract method to loop over some data in batches.

  Arguments:
      model:
      ins: list of tensors to be fed to `f`.
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
  K.set_learning_phase(False)
  num_samples = model._check_num_samples(ins, batch_size, steps, 'steps')
  if verbose == 1:
    if steps is not None:
      progbar = Progbar(target=steps)
    else:
      progbar = Progbar(target=num_samples)

  outs = []
  batches = _make_batches(num_samples, batch_size)
  index_array = np.arange(num_samples)
  for batch_index, (batch_start, batch_end) in enumerate(batches):
    batch_ids = index_array[batch_start:batch_end]
    if ins and isinstance(ins[-1], float):
      # Do not slice the training phase flag.
      ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
    else:
      ins_batch = _slice_arrays(ins, batch_ids)

    ins_batch_converted = []
    for ib in ins_batch:
      ins_batch_converted.append(ops.convert_to_tensor(ib, dtype=K.floatx()))

    eager_model_inputs = []
    for i in range(len(model.inputs)):
      eager_model_inputs.append(ins_batch_converted[i])

    batch_outs = model(eager_model_inputs)

    if not isinstance(batch_outs, list):
      batch_outs = [batch_outs]
    if batch_index == 0:
      # Pre-allocate the results arrays.
      for batch_out in batch_outs:
        dims = batch_out.shape[1:].dims
        dims_list = [d.value for d in dims]
        shape = (num_samples,) + tuple(dims_list)
        outs.append(np.zeros(shape, dtype=batch_out.dtype.as_numpy_dtype))
    for i, batch_out in enumerate(batch_outs):
      outs[i][batch_start:batch_end] = batch_out
    if verbose == 1:
      progbar.update(batch_end)
  if len(outs) == 1:
    return outs[0]
  return outs
