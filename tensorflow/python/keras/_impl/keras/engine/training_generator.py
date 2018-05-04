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
"""Part of the Keras training engine related to Python generators of array data.
"""
# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import callbacks as cbks
from tensorflow.python.keras._impl.keras.utils.data_utils import GeneratorEnqueuer
from tensorflow.python.keras._impl.keras.utils.data_utils import OrderedEnqueuer
from tensorflow.python.keras._impl.keras.utils.data_utils import Sequence
from tensorflow.python.keras._impl.keras.utils.generic_utils import Progbar
from tensorflow.python.platform import tf_logging as logging


def fit_generator(model,
                  generator,
                  steps_per_epoch=None,
                  epochs=1,
                  verbose=1,
                  callbacks=None,
                  validation_data=None,
                  validation_steps=None,
                  class_weight=None,
                  max_queue_size=10,
                  workers=1,
                  use_multiprocessing=False,
                  shuffle=True,
                  initial_epoch=0):
  """See docstring for `Model.fit_generator`."""
  wait_time = 0.01  # in seconds
  epoch = initial_epoch

  do_validation = bool(validation_data)
  model._make_train_function()
  if do_validation:
    model._make_test_function()

  is_sequence = isinstance(generator, Sequence)
  if not is_sequence and use_multiprocessing and workers > 1:
    logging.warning(
        UserWarning('Using a generator with `use_multiprocessing=True`'
                    ' and multiple workers may duplicate your data.'
                    ' Please consider using the`keras.utils.Sequence'
                    ' class.'))
  if steps_per_epoch is None:
    if is_sequence:
      steps_per_epoch = len(generator)
    else:
      raise ValueError('`steps_per_epoch=None` is only valid for a'
                       ' generator based on the `keras.utils.Sequence`'
                       ' class. Please specify `steps_per_epoch` or use'
                       ' the `keras.utils.Sequence` class.')

  # python 2 has 'next', 3 has '__next__'
  # avoid any explicit version checks
  val_gen = (
      hasattr(validation_data, 'next') or
      hasattr(validation_data, '__next__') or
      isinstance(validation_data, Sequence))
  if (val_gen and not isinstance(validation_data, Sequence) and
      not validation_steps):
    raise ValueError('`validation_steps=None` is only valid for a'
                     ' generator based on the `keras.utils.Sequence`'
                     ' class. Please specify `validation_steps` or use'
                     ' the `keras.utils.Sequence` class.')

  # Prepare display labels.
  out_labels = model.metrics_names
  callback_metrics = out_labels + ['val_%s' % n for n in out_labels]

  # prepare callbacks
  model.history = cbks.History()
  callbacks = [cbks.BaseLogger()] + (callbacks or []) + [model.history]
  if verbose:
    callbacks += [cbks.ProgbarLogger(count_mode='steps')]
  callbacks = cbks.CallbackList(callbacks)

  # it's possible to callback a different model than self:
  if hasattr(model, 'callback_model') and model.callback_model:
    callback_model = model.callback_model
  else:
    callback_model = model
  callbacks.set_model(callback_model)
  callbacks.set_params({
      'epochs': epochs,
      'steps': steps_per_epoch,
      'verbose': verbose,
      'do_validation': do_validation,
      'metrics': callback_metrics,
  })
  callbacks.on_train_begin()

  enqueuer = None
  val_enqueuer = None

  try:
    if do_validation and not val_gen:
      # Prepare data for validation
      if len(validation_data) == 2:
        val_x, val_y = validation_data  # pylint: disable=unpacking-non-sequence
        val_sample_weight = None
      elif len(validation_data) == 3:
        val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
      else:
        raise ValueError(
            '`validation_data` should be a tuple '
            '`(val_x, val_y, val_sample_weight)` '
            'or `(val_x, val_y)`. Found: ' + str(validation_data))
      val_x, val_y, val_sample_weights = model._standardize_user_data(
          val_x, val_y, val_sample_weight)
      val_data = val_x + val_y + val_sample_weights
      if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        val_data += [0.]
      for cbk in callbacks:
        cbk.validation_data = val_data

    if workers > 0:
      if is_sequence:
        enqueuer = OrderedEnqueuer(
            generator,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle)
      else:
        enqueuer = GeneratorEnqueuer(
            generator,
            use_multiprocessing=use_multiprocessing,
            wait_time=wait_time)
      enqueuer.start(workers=workers, max_queue_size=max_queue_size)
      output_generator = enqueuer.get()
    else:
      if is_sequence:
        output_generator = iter(generator)
      else:
        output_generator = generator

    callback_model.stop_training = False
    # Construct epoch logs.
    epoch_logs = {}
    while epoch < epochs:
      callbacks.on_epoch_begin(epoch)
      steps_done = 0
      batch_index = 0
      while steps_done < steps_per_epoch:
        generator_output = next(output_generator)

        if not hasattr(generator_output, '__len__'):
          raise ValueError('Output of generator should be '
                           'a tuple `(x, y, sample_weight)` '
                           'or `(x, y)`. Found: ' + str(generator_output))

        if len(generator_output) == 2:
          x, y = generator_output
          sample_weight = None
        elif len(generator_output) == 3:
          x, y, sample_weight = generator_output
        else:
          raise ValueError('Output of generator should be '
                           'a tuple `(x, y, sample_weight)` '
                           'or `(x, y)`. Found: ' + str(generator_output))
        # build batch logs
        batch_logs = {}
        if isinstance(x, list):
          batch_size = x[0].shape[0]
        elif isinstance(x, dict):
          batch_size = list(x.values())[0].shape[0]
        else:
          batch_size = x.shape[0]
        batch_logs['batch'] = batch_index
        batch_logs['size'] = batch_size
        callbacks.on_batch_begin(batch_index, batch_logs)

        outs = model.train_on_batch(
            x, y, sample_weight=sample_weight, class_weight=class_weight)

        if not isinstance(outs, list):
          outs = [outs]
        for l, o in zip(out_labels, outs):
          batch_logs[l] = o

        callbacks.on_batch_end(batch_index, batch_logs)

        batch_index += 1
        steps_done += 1

        # Epoch finished.
        if steps_done >= steps_per_epoch and do_validation:
          if val_gen:
            val_outs = evaluate_generator(
                model,
                validation_data,
                validation_steps,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                max_queue_size=max_queue_size)
          else:
            # No need for try/except because
            # data has already been validated.
            val_outs = model.evaluate(
                val_x,
                val_y,
                batch_size=batch_size,
                sample_weight=val_sample_weights,
                verbose=0)
          if not isinstance(val_outs, list):
            val_outs = [val_outs]
          # Same labels assumed.
          for l, o in zip(out_labels, val_outs):
            epoch_logs['val_' + l] = o

        if callback_model.stop_training:
          break

      callbacks.on_epoch_end(epoch, epoch_logs)
      epoch += 1
      if callback_model.stop_training:
        break

  finally:
    try:
      if enqueuer is not None:
        enqueuer.stop()
    finally:
      if val_enqueuer is not None:
        val_enqueuer.stop()

  callbacks.on_train_end()
  return model.history


def evaluate_generator(model,
                       generator,
                       steps=None,
                       max_queue_size=10,
                       workers=1,
                       use_multiprocessing=False):
  """See docstring for `Model.evaluate_generator`."""
  model._make_test_function()

  steps_done = 0
  wait_time = 0.01
  all_outs = []
  batch_sizes = []
  is_sequence = isinstance(generator, Sequence)
  if not is_sequence and use_multiprocessing and workers > 1:
    logging.warning(
        UserWarning('Using a generator with `use_multiprocessing=True`'
                    ' and multiple workers may duplicate your data.'
                    ' Please consider using the`keras.utils.Sequence'
                    ' class.'))
  if steps is None:
    if is_sequence:
      steps = len(generator)
    else:
      raise ValueError('`steps=None` is only valid for a generator'
                       ' based on the `keras.utils.Sequence` class.'
                       ' Please specify `steps` or use the'
                       ' `keras.utils.Sequence` class.')
  enqueuer = None

  try:
    if workers > 0:
      if is_sequence:
        enqueuer = OrderedEnqueuer(
            generator, use_multiprocessing=use_multiprocessing)
      else:
        enqueuer = GeneratorEnqueuer(
            generator,
            use_multiprocessing=use_multiprocessing,
            wait_time=wait_time)
      enqueuer.start(workers=workers, max_queue_size=max_queue_size)
      output_generator = enqueuer.get()
    else:
      if is_sequence:
        output_generator = iter(generator)
      else:
        output_generator = generator

    while steps_done < steps:
      generator_output = next(output_generator)
      if not hasattr(generator_output, '__len__'):
        raise ValueError('Output of generator should be a tuple '
                         '(x, y, sample_weight) '
                         'or (x, y). Found: ' + str(generator_output))
      if len(generator_output) == 2:
        x, y = generator_output
        sample_weight = None
      elif len(generator_output) == 3:
        x, y, sample_weight = generator_output
      else:
        raise ValueError('Output of generator should be a tuple '
                         '(x, y, sample_weight) '
                         'or (x, y). Found: ' + str(generator_output))
      outs = model.test_on_batch(x, y, sample_weight=sample_weight)

      if isinstance(x, list):
        batch_size = x[0].shape[0]
      elif isinstance(x, dict):
        batch_size = list(x.values())[0].shape[0]
      else:
        batch_size = x.shape[0]
      if batch_size == 0:
        raise ValueError('Received an empty batch. '
                         'Batches should at least contain one item.')
      all_outs.append(outs)

      steps_done += 1
      batch_sizes.append(batch_size)

  finally:
    if enqueuer is not None:
      enqueuer.stop()

  if not isinstance(outs, list):
    return np.average(np.asarray(all_outs), weights=batch_sizes)
  else:
    averages = []
    for i in range(len(outs)):
      averages.append(
          np.average([out[i] for out in all_outs], weights=batch_sizes))
    return averages


def predict_generator(model,
                      generator,
                      steps=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      verbose=0):
  """See docstring for `Model.predict_generator`."""
  model._make_predict_function()

  steps_done = 0
  wait_time = 0.01
  all_outs = []
  is_sequence = isinstance(generator, Sequence)
  if not is_sequence and use_multiprocessing and workers > 1:
    logging.warning(
        UserWarning('Using a generator with `use_multiprocessing=True`'
                    ' and multiple workers may duplicate your data.'
                    ' Please consider using the`keras.utils.Sequence'
                    ' class.'))
  if steps is None:
    if is_sequence:
      steps = len(generator)
    else:
      raise ValueError('`steps=None` is only valid for a generator'
                       ' based on the `keras.utils.Sequence` class.'
                       ' Please specify `steps` or use the'
                       ' `keras.utils.Sequence` class.')
  enqueuer = None

  try:
    if workers > 0:
      if is_sequence:
        enqueuer = OrderedEnqueuer(
            generator, use_multiprocessing=use_multiprocessing)
      else:
        enqueuer = GeneratorEnqueuer(
            generator,
            use_multiprocessing=use_multiprocessing,
            wait_time=wait_time)
      enqueuer.start(workers=workers, max_queue_size=max_queue_size)
      output_generator = enqueuer.get()
    else:
      if is_sequence:
        output_generator = iter(generator)
      else:
        output_generator = generator

    if verbose == 1:
      progbar = Progbar(target=steps)

    while steps_done < steps:
      generator_output = next(output_generator)
      if isinstance(generator_output, tuple):
        # Compatibility with the generators
        # used for training.
        if len(generator_output) == 2:
          x, _ = generator_output
        elif len(generator_output) == 3:
          x, _, _ = generator_output
        else:
          raise ValueError('Output of generator should be '
                           'a tuple `(x, y, sample_weight)` '
                           'or `(x, y)`. Found: ' + str(generator_output))
      else:
        # Assumes a generator that only
        # yields inputs (not targets and sample weights).
        x = generator_output

      outs = model.predict_on_batch(x)
      if not isinstance(outs, list):
        outs = [outs]

      if not all_outs:
        for out in outs:
          all_outs.append([])

      for i, out in enumerate(outs):
        all_outs[i].append(out)
      steps_done += 1
      if verbose == 1:
        progbar.update(steps_done)

  finally:
    if enqueuer is not None:
      enqueuer.stop()

  if len(all_outs) == 1:
    if steps_done == 1:
      return all_outs[0][0]
    else:
      return np.concatenate(all_outs[0])
  if steps_done == 1:
    return [out[0] for out in all_outs]
  else:
    return [np.concatenate(out) for out in all_outs]
