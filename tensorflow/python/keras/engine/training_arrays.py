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
"""Part of the Keras training engine related to plain array data.
"""
# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np

from tensorflow.python.framework import errors
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils.generic_utils import make_batches
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.generic_utils import slice_arrays
from tensorflow.python.platform import tf_logging as logging

try:
  from scipy.sparse import issparse  # pylint: disable=g-import-not-at-top
except ImportError:
  issparse = None


def fit_loop(model,
             inputs,
             targets,
             sample_weights=None,
             batch_size=None,
             epochs=100,
             verbose=1,
             callbacks=None,
             val_inputs=None,
             val_targets=None,
             val_sample_weights=None,
             shuffle=True,
             callback_metrics=None,
             initial_epoch=0,
             steps_per_epoch=None,
             validation_steps=None):
  """Abstract fit function for arrays of data.

  Arguments:
      model: Keras Model instance.
      inputs: List of input arrays.
      targets: List of target arrays.
      sample_weights: Optional list of sample weight arrays.
      batch_size: Integer batch size or None if unknown.
      epochs: Number of times to iterate over the data
      verbose: Verbosity mode, 0, 1 or 2
      callbacks: List of callbacks to be called during training
      val_inputs: List of input arrays.
      val_targets: List of target arrays.
      val_sample_weights: Optional list of sample weight arrays.
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
      validation_steps: Number of steps to run validation for
          (only if doing validation from data tensors).
          Ignored with the default value of `None`.

  Returns:
      `History` object.

  Raises:
      ValueError: in case of invalid arguments.
  """
  model._make_train_function()
  f = model.train_function

  sample_weights = sample_weights or []
  val_sample_weights = val_sample_weights or []
  if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
    ins = inputs + targets + sample_weights + [1]
    if val_inputs:
      val_ins = val_inputs + val_targets + val_sample_weights + [1]
  else:
    ins = inputs + targets + sample_weights
    if val_inputs:
      val_ins = val_inputs + val_targets + val_sample_weights
  if not val_inputs:
    val_ins = []

  do_validation = False
  if val_inputs:
    do_validation = True
    if (steps_per_epoch is None and verbose and inputs and
        hasattr(inputs[0], 'shape') and hasattr(val_inputs[0], 'shape')):
      print('Train on %d samples, validate on %d samples' %
            (inputs[0].shape[0], val_inputs[0].shape[0]))
  if validation_steps:
    do_validation = True
    if steps_per_epoch is None:
      raise ValueError('Can only use `validation_steps` '
                       'when doing step-wise '
                       'training, i.e. `steps_per_epoch` '
                       'must be set.')

  out_labels = model.metrics_names
  if do_validation:
    callback_metrics = copy.copy(out_labels) + [
        'val_' + n for n in out_labels
    ]
  else:
    callback_metrics = copy.copy(out_labels)

  num_train_samples = training_utils.check_num_samples(
      ins, batch_size, steps_per_epoch, 'steps_per_epoch')
  if num_train_samples is not None:
    index_array = np.arange(num_train_samples)

  model.history = cbks.History()
  all_callbacks = [cbks.BaseLogger(
      stateful_metrics=model.stateful_metric_names)]
  if verbose:
    if steps_per_epoch is not None:
      count_mode = 'steps'
    else:
      count_mode = 'samples'
    all_callbacks.append(
        cbks.ProgbarLogger(
            count_mode, stateful_metrics=model.stateful_metric_names))
  all_callbacks += (callbacks or []) + [model.history]
  callbacks = cbks.CallbackList(all_callbacks)
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

  # To prevent a slowdown, we find beforehand the arrays that need conversion.
  feed = model._feed_inputs + model._feed_targets + model._feed_sample_weights
  indices_for_conversion_to_dense = []
  for i in range(len(feed)):
    if issparse is not None and issparse(ins[i]) and not K.is_sparse(feed[i]):
      indices_for_conversion_to_dense.append(i)

  for epoch in range(initial_epoch, epochs):
    # Reset stateful metrics
    for m in model.stateful_metric_functions:
      m.reset_states()
    # Update callbacks
    callbacks.on_epoch_begin(epoch)
    epoch_logs = {}
    if steps_per_epoch is not None:
      for step_index in range(steps_per_epoch):
        batch_logs = {}
        batch_logs['batch'] = step_index
        batch_logs['size'] = 1
        callbacks.on_batch_begin(step_index, batch_logs)
        try:
          outs = f(ins)
        except errors.OutOfRangeError:
          logging.warning('Your dataset iterator ran out of data; '
                          'interrupting training. Make sure that your dataset '
                          'can generate at least `steps_per_epoch * epochs` '
                          'batches (in this case, %d batches).' %
                          steps_per_epoch * epochs)
          break

        if not isinstance(outs, list):
          outs = [outs]
        for l, o in zip(out_labels, outs):
          batch_logs[l] = o

        callbacks.on_batch_end(step_index, batch_logs)
        if callback_model.stop_training:
          break

      if do_validation:
        val_outs = test_loop(
            model,
            val_inputs,
            val_targets,
            sample_weights=val_sample_weights,
            batch_size=batch_size,
            steps=validation_steps,
            verbose=0)
        if not isinstance(val_outs, list):
          val_outs = [val_outs]
        # Same labels assumed.
        for l, o in zip(out_labels, val_outs):
          epoch_logs['val_' + l] = o
    else:
      if shuffle == 'batch':
        index_array = training_utils.batch_shuffle(index_array, batch_size)
      elif shuffle:
        np.random.shuffle(index_array)

      batches = make_batches(num_train_samples, batch_size)

      for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        try:
          if isinstance(ins[-1], int):
            # Do not slice the training phase flag.
            ins_batch = slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
          else:
            ins_batch = slice_arrays(ins, batch_ids)
        except TypeError:
          raise TypeError('TypeError while preparing batch. '
                          'If using HDF5 input data, '
                          'pass shuffle="batch".')
        batch_logs = {}
        batch_logs['batch'] = batch_index
        batch_logs['size'] = len(batch_ids)
        callbacks.on_batch_begin(batch_index, batch_logs)
        for i in indices_for_conversion_to_dense:
          ins_batch[i] = ins_batch[i].toarray()

        outs = f(ins_batch)
        if not isinstance(outs, list):
          outs = [outs]
        for l, o in zip(out_labels, outs):
          batch_logs[l] = o

        callbacks.on_batch_end(batch_index, batch_logs)
        if callback_model.stop_training:
          break

        if batch_index == len(batches) - 1:  # Last batch.
          if do_validation:
            val_outs = test_loop(
                model,
                val_inputs,
                val_targets,
                sample_weights=val_sample_weights,
                batch_size=batch_size,
                verbose=0)
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


def predict_loop(model, inputs, batch_size=32, verbose=0, steps=None):
  """Abstract method to loop over some data in batches.

  Arguments:
      model: Keras Model instance.
      inputs: list of tensors to be fed to `f`.
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
  model._make_predict_function()
  f = model.predict_function

  if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
    ins = inputs + [0]
  else:
    ins = inputs

  num_samples = training_utils.check_num_samples(
      inputs, batch_size, steps, 'steps')
  if verbose == 1:
    if steps is not None:
      progbar = Progbar(target=steps)
    else:
      progbar = Progbar(target=num_samples)

  indices_for_conversion_to_dense = []
  for i in range(len(model._feed_inputs)):
    if (issparse is not None and issparse(inputs[i]) and
        not K.is_sparse(model._feed_inputs[i])):
      indices_for_conversion_to_dense.append(i)

  if steps is not None:
    # Step-based predictions.
    # Since we do not know how many samples
    # we will see, we cannot pre-allocate
    # the returned Numpy arrays.
    # Instead, we store one array per batch seen
    # and concatenate them upon returning.
    unconcatenated_outs = []
    for step in range(steps):
      batch_outs = f(ins)
      if not isinstance(batch_outs, list):
        batch_outs = [batch_outs]
      if step == 0:
        for batch_out in batch_outs:
          unconcatenated_outs.append([])
      for i, batch_out in enumerate(batch_outs):
        unconcatenated_outs[i].append(batch_out)
      if verbose == 1:
        progbar.update(step + 1)
    if len(unconcatenated_outs) == 1:
      return np.concatenate(unconcatenated_outs[0], axis=0)
    return [
        np.concatenate(unconcatenated_outs[i], axis=0)
        for i in range(len(unconcatenated_outs))
    ]
  else:
    # Sample-based predictions.
    outs = []
    batches = make_batches(num_samples, batch_size)
    index_array = np.arange(num_samples)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
      batch_ids = index_array[batch_start:batch_end]
      if ins and isinstance(ins[-1], int):
        # Do not slice the training phase flag.
        ins_batch = slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
      else:
        ins_batch = slice_arrays(ins, batch_ids)
      for i in indices_for_conversion_to_dense:
        ins_batch[i] = ins_batch[i].toarray()

      batch_outs = f(ins_batch)
      if not isinstance(batch_outs, list):
        batch_outs = [batch_outs]
      if batch_index == 0:
        # Pre-allocate the results arrays.
        for batch_out in batch_outs:
          shape = (num_samples,) + batch_out.shape[1:]
          outs.append(np.zeros(shape, dtype=batch_out.dtype))
      for i, batch_out in enumerate(batch_outs):
        outs[i][batch_start:batch_end] = batch_out
      if verbose == 1:
        progbar.update(batch_end)
    if len(outs) == 1:
      return outs[0]
    return outs


def test_loop(model, inputs, targets,
              sample_weights=None,
              batch_size=None,
              verbose=0,
              steps=None):
  """Abstract method to loop over some data in batches.

  Arguments:
      model: Keras Model instance.
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
  model._make_test_function()
  f = model.test_function

  sample_weights = sample_weights or []
  if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
    ins = inputs + targets + sample_weights + [0]
  else:
    ins = inputs + targets + sample_weights

  if hasattr(model, 'metrics'):
    for m in model.stateful_metric_functions:
      m.reset_states()
    stateful_metric_indices = [
        i for i, name in enumerate(model.metrics_names)
        if str(name) in model.stateful_metric_names
    ]
  else:
    stateful_metric_indices = []

  num_samples = training_utils.check_num_samples(
      ins, batch_size, steps, 'steps')
  outs = []
  if verbose == 1:
    if steps is not None:
      progbar = Progbar(target=steps)
    else:
      progbar = Progbar(target=num_samples)

  # To prevent a slowdown, we find beforehand the arrays that need conversion.
  feed = model._feed_inputs + model._feed_targets + model._feed_sample_weights
  indices_for_conversion_to_dense = []
  for i in range(len(feed)):
    if issparse is not None and issparse(ins[i]) and not K.is_sparse(feed[i]):
      indices_for_conversion_to_dense.append(i)

  if steps is not None:
    for step in range(steps):
      batch_outs = f(ins)
      if isinstance(batch_outs, list):
        if step == 0:
          for _ in enumerate(batch_outs):
            outs.append(0.)
        for i, batch_out in enumerate(batch_outs):
          if i in stateful_metric_indices:
            outs[i] = batch_out
          else:
            outs[i] += batch_out
      else:
        if step == 0:
          outs.append(0.)
        outs[0] += batch_outs
      if verbose == 1:
        progbar.update(step + 1)
    for i in range(len(outs)):
      if i not in stateful_metric_indices:
        outs[i] /= steps
  else:
    batches = make_batches(num_samples, batch_size)
    index_array = np.arange(num_samples)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
      batch_ids = index_array[batch_start:batch_end]
      if isinstance(ins[-1], int):
        # Do not slice the training phase flag.
        ins_batch = slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
      else:
        ins_batch = slice_arrays(ins, batch_ids)
      for i in indices_for_conversion_to_dense:
        ins_batch[i] = ins_batch[i].toarray()

      batch_outs = f(ins_batch)

      if isinstance(batch_outs, list):
        if batch_index == 0:
          for batch_out in enumerate(batch_outs):
            outs.append(0.)
        for i, batch_out in enumerate(batch_outs):
          if i in stateful_metric_indices:
            outs[i] = batch_out
          else:
            outs[i] += batch_out * len(batch_ids)
      else:
        if batch_index == 0:
          outs.append(0.)
        outs[0] += batch_outs * len(batch_ids)
      if verbose == 1:
        progbar.update(batch_end)
    for i in range(len(outs)):
      if i not in stateful_metric_indices:
        outs[i] /= num_samples
  if len(outs) == 1:
    return outs[0]
  return outs
