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

import functools
import math

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


def model_iteration(model,
                    data,
                    steps_per_epoch=None,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_data=None,
                    validation_steps=None,
                    validation_freq=1,
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=False,
                    initial_epoch=0,
                    mode=ModeKeys.TRAIN,
                    batch_size=None,
                    steps_name='steps',
                    **kwargs):
  """Loop function for arrays of data with modes TRAIN/TEST/PREDICT.

  Args:
      model: Keras Model instance.
      data: Either a tuple of NumPy/Tensor inputs (i.e. `(x,)` or `(x, y)` or
        `(x, y, sample_weights)`) or a generator or
        `keras.utils.data_utils.Sequence` object or Eager Iterator or Dataset.
      steps_per_epoch: Total number of steps (batches of samples) before
        declaring one epoch finished and starting the next epoch. Ignored with
        the default value of `None`.
      epochs: Number of times to iterate over the data.
      verbose: 0, 1, or 2. Verbosity mode.
        0 = silent, 1 = progress bar, 2 = one line per epoch.
        Note that the progress bar is not particularly useful when
        logged to a file, so verbose=2 is recommended when not running
        interactively (eg, in a production environment).
      callbacks: List of callbacks to be called during training.
      validation_data: Either a tuple of NumPy/Tensor inputs (i.e. `(x,)` or
        `(x, y)` or `(x, y, sample_weights)`) or a generator or
        `keras.utils.data_utils.Sequence` object or Eager Iterator or Dataset.
      validation_steps: Total number of steps (batches of samples) before
        declaring validation finished.
      validation_freq: Only relevant if validation data is provided. Integer or
        `collections.abc.Container` instance (e.g. list, tuple, etc.). If an
        integer, specifies how many training epochs to run before a new
        validation run is performed, e.g. `validation_freq=2` runs
        validation every 2 epochs. If a Container, specifies the epochs on
        which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
        validation at the end of the 1st, 2nd, and 10th epochs.
      class_weight: Dictionary mapping class indices to a weight for the class.
      max_queue_size: Integer. Maximum size for the generator queue. If
        unspecified, `max_queue_size` will default to 10.
      workers: Integer. Maximum number of processes to spin up when using
        process-based threading. If unspecified, `workers` will default to 1. If
        0, will execute the generator on the main thread.
      use_multiprocessing: Boolean. If `True`, use process-based threading. If
        unspecified, `use_multiprocessing` will default to `False`. Note that
        because this implementation relies on multiprocessing, you should not
        pass non-picklable arguments to the generator as they can't be passed
        easily to children processes.
      shuffle: Boolean. Whether to shuffle the order of the batches at the
        beginning of each epoch. Only used with instances of `Sequence`
        (`keras.utils.Sequence`). Has no effect when `steps_per_epoch` is not
        `None`.
      initial_epoch: Epoch at which to start training (useful for resuming a
        previous training run).
      mode: One of ModeKeys.TRAIN/ModeKeys.TEST/ModeKeys.PREDICT.
      batch_size: Integer batch size or None if unknown. Will only be used if
        `data` is in NumPy/Tensor format.
      steps_name: The string name of the steps argument, either `steps`,
        `validation_steps`, or `steps_per_epoch`. Only used for error message
        formatting.
      **kwargs: Additional arguments for backwards compatibility. `steps` is
        accepted as an alias for `steps_per_epoch`.

  Returns:
      - In TRAIN mode: `History` object.
      - In TEST mode: Evaluation metrics.
      - In PREDICT mode: Outputs of the Model called on inputs.

  Raises:
      ValueError: in case of invalid arguments.
  """
  if 'steps' in kwargs:
    steps_per_epoch = kwargs['steps']

  # Determine the number of steps per epoch and whether we should reset the
  # dataset at the end of each epoch.
  reset_dataset_after_each_epoch = False
  original_dataset = None
  is_dataset = isinstance(data, (dataset_ops.DatasetV2, dataset_ops.DatasetV1))
  if is_dataset:
    original_dataset = data
    if steps_per_epoch is None:
      reset_dataset_after_each_epoch = True
      steps_per_epoch = training_utils_v1.infer_steps_for_dataset(
          model, data, steps_per_epoch, epochs=epochs, steps_name=steps_name)

  # Convert to a format that supports `next(generator)`.
  generator, steps_per_epoch = convert_to_generator_like(
      data,
      steps_per_epoch=steps_per_epoch,
      batch_size=batch_size,
      epochs=epochs - initial_epoch,
      shuffle=shuffle)

  do_validation = validation_data is not None
  is_sequence = isinstance(generator, data_utils.Sequence)
  _validate_arguments(is_sequence, is_dataset, use_multiprocessing, workers,
                      steps_per_epoch, validation_data, validation_steps, mode,
                      kwargs)

  batch_function = _make_execution_function(
      model, mode, class_weight=class_weight)

  # Create the queue for the generator.
  enqueuer = None
  if not is_dataset:
    generator, enqueuer = _make_enqueued_generator(
        generator,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size,
        shuffle=shuffle)

  num_samples_or_steps, use_steps = _get_num_samples_or_steps(
      data, steps_per_epoch)

  count_mode = 'steps' if use_steps else 'samples'
  callbacks = cbks.configure_callbacks(
      callbacks,
      model,
      do_validation=do_validation,
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      batch_size=batch_size,
      samples=num_samples_or_steps,
      count_mode=count_mode,
      verbose=verbose,
      mode=mode)

  if mode == ModeKeys.PREDICT:
    aggregator = training_utils_v1.OutputsAggregator(
        True, steps=steps_per_epoch)
  else:
    aggregator = training_utils_v1.MetricsAggregator(
        True, steps=steps_per_epoch)

  should_set_learning_phase = context.executing_eagerly() and model.run_eagerly
  if should_set_learning_phase:
    learning_phase_scope = backend.eager_learning_phase_scope(
        1 if mode == ModeKeys.TRAIN else 0)
    learning_phase_scope.__enter__()

  callbacks.model.stop_training = False
  callbacks._call_begin_hook(mode)

  initial_epoch = model._maybe_load_initial_epoch_from_ckpt(initial_epoch, mode)

  for epoch in range(initial_epoch, epochs):
    if callbacks.model.stop_training:
      break

    # Setup work for each epoch.
    model.reset_metrics()
    epoch_logs = {}
    if mode == ModeKeys.TRAIN:
      callbacks.on_epoch_begin(epoch, epoch_logs)

    if steps_per_epoch is None:
      # Loop over dataset until `OutOfRangeError` is raised.
      target_steps = np.inf
    else:
      # Loop over dataset for the specified number of steps.
      target_steps = steps_per_epoch

    step = 0
    while step < target_steps:
      batch_data = _get_next_batch(generator)
      if batch_data is None:
        if is_dataset:
          # The dataset passed by the user ran out of batches.
          # Now we know the cardinality of the dataset.
          # If steps_per_epoch was specified, then running out of data is
          # unexpected, so we stop training and inform the user.
          if steps_per_epoch:
            callbacks.model.stop_training = True
            logging.warning(
                'Your dataset ran out of data; interrupting training. '
                'Make sure that your dataset can generate at least '
                '`%s * epochs` batches (in this case, %d batches). '
                'You may need to use the repeat() function when '
                'building your dataset.'
                % (steps_name, steps_per_epoch * epochs))
          elif step > 0:
            steps_per_epoch = step
            aggregator.steps = steps_per_epoch
        else:
          # We ran out of batches while the user passed an iterator (legacy).
          callbacks.model.stop_training = True
          logging.warning(
              'Your dataset iterator ran out of data; '
              'interrupting training. Make sure that your iterator '
              'can generate at least `%s * epochs` '
              'batches (in this case, %d batches). You may need to'
              'use the repeat() function when building your '
              'dataset.' % (steps_name, steps_per_epoch * epochs))
        break

      # `batch_size` used for validation data if validation
      # data is NumPy/EagerTensors.
      batch_size = int(nest.flatten(batch_data)[0].shape[0])

      # Callbacks batch begin.
      batch_logs = {'batch': step, 'size': batch_size}
      callbacks._call_batch_hook(mode, 'begin', step, batch_logs)

      is_deferred = not model._is_compiled
      batch_outs = batch_function(*batch_data)
      if not isinstance(batch_outs, list):
        batch_outs = [batch_outs]

      if step == 0:
        aggregator.create(batch_outs)

        if is_deferred:
          # Set callbacks params. We do this here when model is compiled only
          # in the first iteration of this loop (deferred build scenario).
          cbks.set_callback_parameters(
              callbacks,
              model,
              do_validation=do_validation,
              batch_size=batch_size,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              samples=num_samples_or_steps,
              verbose=verbose,
              mode=mode)

      # Aggregate results.
      aggregator.aggregate(batch_outs)

      # Callbacks batch end.
      batch_logs = cbks.make_logs(model, batch_logs, batch_outs, mode)
      callbacks._call_batch_hook(mode, 'end', step, batch_logs)
      step += 1

      if callbacks.model.stop_training:
        break

    aggregator.finalize()
    results = aggregator.results
    epoch_logs = cbks.make_logs(model, epoch_logs, results, mode)
    if len(results) == 1:
      results = results[0]

    # Run the test loop every epoch during training.
    if (do_validation and
        training_utils_v1.should_run_validation(validation_freq, epoch) and
        not callbacks.model.stop_training):
      val_results = model_iteration(
          model,
          validation_data,
          steps_per_epoch=validation_steps,
          batch_size=batch_size,
          class_weight=class_weight,
          workers=workers,
          use_multiprocessing=use_multiprocessing,
          max_queue_size=max_queue_size,
          callbacks=callbacks,
          verbose=verbose,
          mode=ModeKeys.TEST,
          steps_name='validation_steps')

      if not isinstance(val_results, list):
        val_results = [val_results]
      epoch_logs = cbks.make_logs(
          model, epoch_logs, val_results, mode, prefix='val_')

    if mode == ModeKeys.TRAIN:
      # Epochs only apply to `fit`.
      callbacks.on_epoch_end(epoch, epoch_logs)

    # Recreate dataset iterator for the next epoch.
    if reset_dataset_after_each_epoch and epoch < epochs - 1:
      generator = dataset_ops.make_one_shot_iterator(original_dataset)

  model._successful_loop_finish = True
  callbacks._call_end_hook(mode)

  if enqueuer is not None:
    enqueuer.stop()

  if should_set_learning_phase:
    learning_phase_scope.__exit__(None, None, None)

  if mode == ModeKeys.TRAIN:
    return model.history
  return results


# Maintain compatibility with the existing names.
fit_generator = functools.partial(model_iteration, mode=ModeKeys.TRAIN)
evaluate_generator = functools.partial(
    model_iteration, mode=ModeKeys.TEST, shuffle=False)
predict_generator = functools.partial(
    model_iteration, mode=ModeKeys.PREDICT, shuffle=False)


def _get_next_batch(generator):
  """Retrieves the next batch of input data."""
  try:
    generator_output = next(generator)
  except (StopIteration, errors.OutOfRangeError):
    return None

  if not isinstance(generator_output, tuple):
    # Always wrap in a tuple.
    generator_output = (generator_output,)
  if len(generator_output) not in [1, 2, 3]:
    raise ValueError(
        'Output of generator should be a tuple of 1 or 2 or 3 '
        'elements: (input,) or (input, target) or '
        '(input, target, sample_weights). Received {}'.format(generator_output))
  return generator_output


def _validate_arguments(is_sequence, is_dataset, use_multiprocessing, workers,
                        steps_per_epoch, validation_data, validation_steps,
                        mode, kwargs):
  """Raises errors if arguments are invalid.

  Args:
    is_sequence: Boolean, whether data is a `keras.utils.data_utils.Sequence`
      instance.
    is_dataset: Boolean, whether data is a dataset instance.
    use_multiprocessing: Boolean. If `True`, use process-based threading. If
      unspecified, `use_multiprocessing` will default to `False`. Note that
      because this implementation relies on multiprocessing, you should not pass
      non-picklable arguments to the generator as they can't be passed easily to
      children processes.
    workers: Integer. Maximum number of processes to spin up when using
      process-based threading. If unspecified, `workers` will default to 1. If
      0, will execute the generator on the main thread.
    steps_per_epoch: Total number of steps (batches of samples) before declaring
      one epoch finished and starting the next epoch. Ignored with the default
      value of `None`.
    validation_data: Either a tuple of NumPy/Tensor inputs (i.e. `(x,)` or `(x,
      y)` or `(x, y, sample_weights)`) or a generator or
      `keras.utils.data_utils.Sequence` object or Eager Iterator or Dataset.
    validation_steps: Total number of steps (batches of samples) before
      declaring validation finished.
    mode: One of ModeKeys.TRAIN/ModeKeys.TEST/ModeKeys.PREDICT.
    kwargs: Additional arguments for backwards compatibility.

  Raises:
    ValueError: If `steps_per_epoch` or `validation_steps` are not passed
      for data types that require them, or if unrecognized keyword
      arguments are passed.
  """
  if not is_sequence and use_multiprocessing and workers > 1:
    logging.warning(
        UserWarning('Using a generator with `use_multiprocessing=True`'
                    ' and multiple workers may duplicate your data.'
                    ' Please consider using the `keras.utils.Sequence`'
                    ' class.'))

  if steps_per_epoch is None and not is_dataset:
    arg_name = 'steps_per_epoch' if mode == ModeKeys.TRAIN else 'steps'
    raise ValueError('Please specify the number of steps via the '
                     '`{}` argument.'.format(arg_name))

  val_gen = (
      data_utils.is_generator_or_sequence(validation_data) or
      isinstance(validation_data, iterator_ops.IteratorBase))
  if (val_gen and not isinstance(validation_data, data_utils.Sequence) and
      not validation_steps):
    raise ValueError('Please specify the `validation_steps` argument.')

  if any(k != 'steps' for k in kwargs):
    raise ValueError('Invalid arguments passed: {}'.format(
        [k for k in kwargs if k != 'steps']))


def convert_to_generator_like(data,
                              batch_size=None,
                              steps_per_epoch=None,
                              epochs=1,
                              shuffle=False):
  """Make a generator out of NumPy or EagerTensor inputs.

  Args:
    data: Either a generator or `keras.utils.data_utils.Sequence` object or
      `Dataset`, `Iterator`, or a {1,2,3}-tuple of NumPy arrays or EagerTensors.
      If a tuple, the elements represent `(x, y, sample_weights)` and may be
      `None` or `[None]`.
    batch_size: Used when creating a generator out of tuples of NumPy arrays or
      EagerTensors.
    steps_per_epoch: Steps of the generator to run each epoch. If `None` the
      number of steps will be read from the data (for
      `keras.utils.data_utils.Sequence` types).
    epochs: Total number of epochs to run.
    shuffle: Whether the data should be shuffled.

  Returns:
    - Generator, `keras.utils.data_utils.Sequence`, or `Iterator`.

  Raises:
    - ValueError: If `batch_size` is not provided for NumPy or EagerTensor
      inputs.
  """
  if isinstance(data, tuple):
    # Scrub `Nones` that might have been passed for `targets`, `sample_weights`.
    data = tuple(
        ele for ele in data if not all(e is None for e in nest.flatten(ele)))

  if data_utils.is_generator_or_sequence(data) or isinstance(
      data, iterator_ops.IteratorBase):
    if isinstance(data, data_utils.Sequence):
      if steps_per_epoch is None:
        steps_per_epoch = len(data)
    return data, steps_per_epoch
  if isinstance(data, dataset_ops.DatasetV2):
    return dataset_ops.make_one_shot_iterator(data), steps_per_epoch

  # Create generator from NumPy or EagerTensor Input.
  num_samples = int(nest.flatten(data)[0].shape[0])
  if batch_size is None:
    raise ValueError(
        'When passing input data as arrays, do not specify '
        '`steps_per_epoch`/`steps` argument. Please use `batch_size` instead.')
  steps_per_epoch = int(math.ceil(num_samples / batch_size))

  def _gen(data):
    """Makes a generator out of a structure of NumPy/EagerTensors."""
    index_array = np.arange(num_samples)
    for _ in range(epochs):
      if shuffle:
        np.random.shuffle(index_array)
      batches = generic_utils.make_batches(num_samples, batch_size)
      for (batch_start, batch_end) in batches:
        batch_ids = index_array[batch_start:batch_end]
        flat_batch_data = training_utils.slice_arrays(
            nest.flatten(data), batch_ids, contiguous=(not shuffle))
        yield nest.pack_sequence_as(data, flat_batch_data)

  return _gen(data), steps_per_epoch


def _make_enqueued_generator(generator,
                             workers=1,
                             use_multiprocessing=False,
                             max_queue_size=10,
                             shuffle=False):
  """Create a buffered queue of next elements of the generator."""
  is_sequence = isinstance(generator, data_utils.Sequence)
  enqueuer = None
  if workers > 0:
    if is_sequence:
      enqueuer = data_utils.OrderedEnqueuer(
          generator, use_multiprocessing=use_multiprocessing, shuffle=shuffle)
    else:
      enqueuer = data_utils.GeneratorEnqueuer(
          generator, use_multiprocessing=use_multiprocessing)
    enqueuer.start(workers=workers, max_queue_size=max_queue_size)
    output_generator = enqueuer.get()
  else:
    if is_sequence:
      output_generator = data_utils.iter_sequence_infinite(generator)
    else:
      output_generator = generator
  return output_generator, enqueuer


def _make_execution_function(model, mode, class_weight=None):
  """Makes function to run one step of model execution."""
  if mode == ModeKeys.TRAIN:
    f = functools.partial(model.train_on_batch, class_weight=class_weight)
  elif mode == ModeKeys.TEST:
    f = model.test_on_batch
  else:
    # Match signature of other modes to allow
    # 1, 2, or 3-tuples from generator
    def predict_on_batch(x, y=None, sample_weights=None):  # pylint: disable=unused-argument
      return model.predict_on_batch(x)

    f = predict_on_batch

  # Maintain stateful metrics across batch-level calls.
  if mode != ModeKeys.PREDICT:
    f = functools.partial(f, reset_metrics=False)

  return f


def _get_num_samples_or_steps(data, steps_per_epoch):
  """Returns number of samples or steps, and whether to use steps count mode."""
  flat_inputs = nest.flatten(data)
  if hasattr(flat_inputs[0], 'shape'):
    return int(flat_inputs[0].shape[0]), False
  return steps_per_epoch, True


class GeneratorOrSequenceTrainingLoop(training_utils_v1.TrainingLoop):
  """Generator-like.

  Input is Python generator, or Sequence object.

  The difference between this class and `GeneratorLikeTrainingFunction` is that
  this class only handles inputs that with x, y and sample_weight fused into one
  param.
  """

  def fit(self,
          model,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False):
    model._validate_or_infer_batch_size(batch_size, steps_per_epoch, x)
    training_utils_v1.check_generator_arguments(
        y, sample_weight, validation_split=validation_split)
    return fit_generator(
        model,
        x,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_steps=validation_steps,
        validation_freq=validation_freq,
        class_weight=class_weight,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        shuffle=shuffle,
        initial_epoch=initial_epoch,
        steps_name='steps_per_epoch')

  def evaluate(self,
               model,
               x=None,
               y=None,
               batch_size=None,
               verbose=1,
               sample_weight=None,
               steps=None,
               callbacks=None,
               max_queue_size=10,
               workers=1,
               use_multiprocessing=False):
    model._validate_or_infer_batch_size(batch_size, steps, x)
    training_utils_v1.check_generator_arguments(y, sample_weight)
    return evaluate_generator(
        model,
        x,
        steps=steps,
        verbose=verbose,
        callbacks=callbacks,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing)

  def predict(self,
              model,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False):
    model._validate_or_infer_batch_size(batch_size, steps, x)
    return predict_generator(
        model,
        x,
        steps=steps,
        verbose=verbose,
        callbacks=callbacks,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing)


class EagerDatasetOrIteratorTrainingLoop(training_utils_v1.TrainingLoop):
  """A non-distributed Dataset or iterator in eager execution."""

  def fit(self,
          model,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_freq=1,
          **kwargs):
    model._validate_or_infer_batch_size(batch_size, steps_per_epoch, x)
    # Make sure that y, sample_weights, validation_split are not passed.
    training_utils_v1.validate_dataset_input(x, y, sample_weight,
                                             validation_split)
    if (isinstance(x, (dataset_ops.DatasetV1, dataset_ops.DatasetV2)) and
        shuffle):
      training_utils_v1.verify_dataset_shuffled(x)

    return fit_generator(
        model,
        x,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_steps=validation_steps,
        validation_freq=validation_freq,
        class_weight=class_weight,
        workers=0,
        shuffle=shuffle,
        initial_epoch=initial_epoch,
        steps_name='steps_per_epoch')

  def evaluate(self,
               model,
               x=None,
               y=None,
               batch_size=None,
               verbose=1,
               sample_weight=None,
               steps=None,
               callbacks=None,
               **kwargs):
    model._validate_or_infer_batch_size(batch_size, steps, x)
    # Make sure that y, sample_weights, validation_split are not passed.
    training_utils_v1.validate_dataset_input(x, y, sample_weight)
    return evaluate_generator(
        model, x, steps=steps, verbose=verbose, workers=0, callbacks=callbacks)

  def predict(self,
              model,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              **kwargs):
    model._validate_or_infer_batch_size(batch_size, steps, x)
    return predict_generator(
        model, x, steps=steps, verbose=verbose, workers=0, callbacks=callbacks)


class GeneratorLikeTrainingLoop(training_utils_v1.TrainingLoop):
  """TrainingLoop that handle inputs like python generator.

  This is the default handler for most of the input data types, includes
  symbolic tensors or Numpy array-like, Datasets and iterators in graph mode
  (since they generate symbolic tensors). This Function is used to handle model
  with `run_eagerly` = True.
  """

  def fit(self,
          model,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_freq=1,
          **kwargs):
    batch_size = model._validate_or_infer_batch_size(batch_size,
                                                     steps_per_epoch, x)
    x, y, sample_weights = model._standardize_user_data(
        x,
        y,
        sample_weight=sample_weight,
        class_weight=class_weight,
        batch_size=batch_size,
        check_steps=True,
        steps_name='steps_per_epoch',
        steps=steps_per_epoch,
        validation_split=validation_split,
        shuffle=shuffle)

    if validation_data:
      validation_data = model._prepare_validation_data(validation_data,
                                                       batch_size,
                                                       validation_steps)
    elif validation_split and 0. < validation_split < 1.:
      (x, y, sample_weights, val_x, val_y,
       val_sample_weights) = (
           training_utils_v1.split_training_and_validation_data(
               x, y, sample_weights, validation_split))
      validation_data = (val_x, val_y, val_sample_weights)
    else:
      if validation_steps:
        raise ValueError('`validation_steps` should not be specified if '
                         '`validation_data` is None.')

    return fit_generator(
        model, (x, y, sample_weights),
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_steps=validation_steps,
        validation_freq=validation_freq,
        workers=0,
        shuffle=shuffle,
        initial_epoch=initial_epoch,
        steps_name='steps_per_epoch')

  def evaluate(self,
               model,
               x=None,
               y=None,
               batch_size=None,
               verbose=1,
               sample_weight=None,
               steps=None,
               callbacks=None,
               **kwargs):
    batch_size = model._validate_or_infer_batch_size(batch_size, steps, x)
    x, y, sample_weights = model._standardize_user_data(
        x,
        y,
        sample_weight=sample_weight,
        batch_size=batch_size,
        check_steps=True,
        steps_name='steps',
        steps=steps)
    return evaluate_generator(
        model, (x, y, sample_weights),
        steps=steps,
        batch_size=batch_size,
        verbose=verbose,
        workers=0,
        callbacks=callbacks)

  def predict(self,
              model,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              **kwargs):
    batch_size = model._validate_or_infer_batch_size(batch_size, steps, x)
    x, _, _ = model._standardize_user_data(
        x, check_steps=True, steps_name='steps', steps=steps)
    return predict_generator(
        model,
        x,
        steps=steps,
        batch_size=batch_size,
        verbose=verbose,
        workers=0,
        callbacks=callbacks)
