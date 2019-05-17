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

import functools

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils.generic_utils import make_batches
from tensorflow.python.keras.utils.generic_utils import slice_arrays
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging

try:
  from scipy.sparse import issparse  # pylint: disable=g-import-not-at-top
except ImportError:
  issparse = None


def model_iteration(model,
                    inputs,
                    targets=None,
                    sample_weights=None,
                    batch_size=None,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    val_inputs=None,
                    val_targets=None,
                    val_sample_weights=None,
                    shuffle=True,
                    initial_epoch=0,
                    steps_per_epoch=None,
                    validation_steps=None,
                    validation_freq=1,
                    mode=ModeKeys.TRAIN,
                    validation_in_fit=False,
                    prepared_feed_values_from_dataset=False,
                    steps_name='steps',
                    **kwargs):
  """Loop function for arrays of data with modes TRAIN/TEST/PREDICT.

  Arguments:
      model: Keras Model instance.
      inputs: Either a list or dictionary of arrays, or a dataset instance.
      targets: List/dictionary of input arrays.
      sample_weights: Optional list of sample weight arrays.
      batch_size: Integer batch size or None if unknown.
      epochs: Number of times to iterate over the data
      verbose: 0, 1, or 2. Verbosity mode.
        0 = silent, 1 = progress bar, 2 = one line per epoch.
        Note that the progress bar is not particularly useful when
        logged to a file, so verbose=2 is recommended when not running
        interactively (eg, in a production environment).
      callbacks: List of callbacks to be called during training
      val_inputs: Either a list or dictionary of arrays, or a dataset instance.
      val_targets: List/dictionary of target arrays.
      val_sample_weights: Optional list of sample weight arrays.
      shuffle: Whether to shuffle the data at the beginning of each epoch
        concatenation of list the display names of the outputs of `f` and the
        list of display names of the outputs of `f_val`.
      initial_epoch: Epoch at which to start training (useful for resuming a
        previous training run)
      steps_per_epoch: Total number of steps (batches of samples) before
        declaring one epoch finished and starting the next epoch. Ignored with
        the default value of `None`.
      validation_steps: Number of steps to run validation for (only if doing
        validation from data tensors). Ignored with the default value of `None`.
      validation_freq: Only relevant if validation data is provided. Integer or
        `collections.Container` instance (e.g. list, tuple, etc.). If an
        integer, specifies how many training epochs to run before a new
        validation run is performed, e.g. `validation_freq=2` runs
        validation every 2 epochs. If a Container, specifies the epochs on
        which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
        validation at the end of the 1st, 2nd, and 10th epochs.
      mode: One of ModeKeys.TRAIN/ModeKeys.TEST/ModeKeys.PREDICT.
      validation_in_fit: if true, then this method is invoked from within
        training iteration (for validation). In the case where `val_inputs` is a
        dataset, this flag indicates that its iterator and feed values are
        already created so should properly reuse resources.
      prepared_feed_values_from_dataset: if True, `inputs` is a list of feed
        tensors returned from `_prepare_feed_values` call on the validation
        dataset, so do not call it again on `inputs`. Should only be used for
        inline validation (i.e., only if `validation_in_fit` is also True).
      steps_name: The string name of the steps argument, either `steps`,
        `validation_steps`, or `steps_per_epoch`. Only used for error message
        formatting.
      **kwargs: Additional arguments for backwards compatibility.

  Returns:
      - In TRAIN mode: `History` object.
      - In TEST mode: Evaluation metrics.
      - In PREDICT mode: Outputs of the Model called on inputs.

  Raises:
      ValueError: in case of invalid arguments.
  """
  # Backwards compatibility.
  if 'steps' in kwargs:
    steps_per_epoch = kwargs.pop('steps')
  if kwargs:
    raise TypeError('Unknown arguments: %s' % (kwargs,))

  # In case we were passed a dataset, we extract symbolic tensors from it.
  reset_dataset_after_each_epoch = False
  input_iterator = None
  is_dataset = isinstance(inputs,
                          (dataset_ops.DatasetV1, dataset_ops.DatasetV2))
  # TODO(fchollet): consider moving `steps_per_epoch` inference to
  # _standardize_user_data and set reset_dataset_after_each_epoch as an
  # attribute on the dataset instance.
  if is_dataset:
    if steps_per_epoch is None:
      reset_dataset_after_each_epoch = True
      steps_per_epoch = training_utils.infer_steps_for_dataset(
          inputs, steps_per_epoch, epochs=epochs, steps_name=steps_name)
    input_iterator = _get_iterator(inputs, model._distribution_strategy)

  if mode == ModeKeys.TRAIN:
    _print_train_info(inputs, val_inputs, steps_per_epoch, verbose)

  # Enter tf.distribute.Strategy scope.
  if model._distribution_strategy:
    scope = distributed_training_utils.distributed_scope(
        strategy=model._distribution_strategy,
        learning_phase=(1 if mode == ModeKeys.TRAIN else 0))
    scope.__enter__()

  model._update_sample_weight_modes(sample_weights=sample_weights)

  # Get step function and loop type.
  f = _make_execution_function(model, mode)
  use_steps = is_dataset or steps_per_epoch is not None
  do_validation = val_inputs is not None

  # Convert Eager Tensors to NumPy arrays to support batching/shuffling.
  inputs, targets, sample_weights = training_utils. \
      convert_eager_tensors_to_numpy((inputs, targets, sample_weights))

  # Prepare input data.
  inputs = input_iterator or inputs
  if validation_in_fit and prepared_feed_values_from_dataset:
    # When invoking validation in training loop, avoid creating iterator and
    # list of feed values for the same validation dataset multiple times (which
    # essentially would call `iterator.get_next()` that slows down execution and
    # leads to OOM errors eventually.
    ins = inputs
  else:
    ins = _prepare_feed_values(model, inputs, targets, sample_weights, mode)
    # `ins` is a function when a distribute strategy is used in Eager mode.  In
    # that case `is_dataset` is True.  The code branches that have requirements
    # about the type of `ins` do not trigger in the distributed case.
  if not is_dataset:
    num_samples_or_steps = _get_num_samples_or_steps(ins, batch_size,
                                                     steps_per_epoch)
  else:
    num_samples_or_steps = steps_per_epoch

  # Prepare validation data. Hold references to the iterator and the input list
  # to properly reinitialize and reuse in multiple validation passes.
  val_iterator = None
  if isinstance(val_inputs, (dataset_ops.DatasetV1, dataset_ops.DatasetV2)):
    if validation_steps is None:
      # Because we pass an iterator feed instead of a Dataset to the eval
      # model_iteration() call, it will not trigger the dataset-input path
      # that determines the number of steps required. To avoid this issue,
      # set validation_steps here if validation_steps is None.
      validation_steps = training_utils.infer_steps_for_dataset(
          val_inputs,
          validation_steps,
          epochs=epochs,
          steps_name='validation_steps')
    val_iterator = _get_iterator(val_inputs, model._distribution_strategy)
    val_inputs = _prepare_feed_values(
        model, val_iterator, val_targets, val_sample_weights, ModeKeys.TEST)

  # Configure callbacks.
  count_mode = 'steps' if use_steps else 'samples'
  callbacks = cbks.configure_callbacks(
      callbacks,
      model,
      do_validation=do_validation,
      batch_size=batch_size,
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      samples=num_samples_or_steps,
      verbose=0,  # Handle ProgBarLogger separately in this loop.
      mode=mode)
  # TODO(omalleyt): Handle ProgBar as part of Callbacks once hooks are ready.
  progbar = training_utils.get_progbar(model, count_mode)
  progbar.params = callbacks.params
  progbar.params['verbose'] = verbose

  # Find beforehand arrays that need sparse-to-dense conversion.
  if issparse is not None and not use_steps:
    indices_for_conversion_to_dense = []
    feed = _get_model_feed(model, mode)
    for i, (input_data, feed_tensor) in enumerate(zip(ins, feed)):
      if issparse(input_data) and not K.is_sparse(feed_tensor):
        indices_for_conversion_to_dense.append(i)

  # Select aggregation method.
  if mode == ModeKeys.PREDICT:
    aggregator = training_utils.OutputsAggregator(use_steps,
                                                  num_samples_or_steps)
  else:
    aggregator = training_utils.MetricsAggregator(use_steps,
                                                  num_samples_or_steps)

  if model._compile_distribution:
    distributed_training_utils._copy_weights_to_distributed_model(model, mode)

  callbacks.model.stop_training = False
  callbacks._call_begin_hook(mode)
  progbar.on_train_begin()

  initial_epoch = model._maybe_load_initial_epoch_from_ckpt(initial_epoch, mode)

  for epoch in range(initial_epoch, epochs):
    if callbacks.model.stop_training:
      break

    # Setup work for each epoch
    epoch_logs = {}
    model.reset_metrics()
    if mode == ModeKeys.TRAIN:
      callbacks.on_epoch_begin(epoch, epoch_logs)
    progbar.on_epoch_begin(epoch, epoch_logs)

    if use_steps:
      # Step-wise loop.
      if steps_per_epoch is None:
        # Loop over dataset until `OutOfRangeError` is raised.
        target_steps = np.inf
      else:
        # Loop over dataset for the specified number of steps.
        target_steps = steps_per_epoch

      step = 0
      while step < target_steps:
        batch_logs = {'batch': step, 'size': 1}
        callbacks._call_batch_hook(mode, 'begin', step, batch_logs)
        progbar.on_batch_begin(step, batch_logs)

        # Get outputs.
        try:
          # `ins` can be callable in tf.distribute.Strategy + eager case.
          actual_inputs = ins() if callable(ins) else ins
          batch_outs = f(actual_inputs)
        except errors.OutOfRangeError:
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
              aggregator.num_samples_or_steps = steps_per_epoch
              if mode == ModeKeys.TRAIN:
                progbar.params['steps'] = steps_per_epoch
                progbar.progbar.target = steps_per_epoch
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

        if not isinstance(batch_outs, list):
          batch_outs = [batch_outs]

        if model._distribution_strategy:
          batch_outs = distributed_training_utils._per_replica_aggregate_batch(
              batch_outs, model, mode)

        # Aggregate results.
        if step == 0:
          aggregator.create(batch_outs)
        aggregator.aggregate(batch_outs)

        # Callbacks batch end.
        batch_logs = cbks.make_logs(model, batch_logs, batch_outs, mode)
        callbacks._call_batch_hook(mode, 'end', step, batch_logs)
        progbar.on_batch_end(step, batch_logs)
        step += 1

        if callbacks.model.stop_training:
          break
    else:
      # Sample-wise loop.
      index_array = np.arange(num_samples_or_steps)
      if shuffle == 'batch':
        index_array = training_utils.batch_shuffle(index_array, batch_size)
      elif shuffle:
        np.random.shuffle(index_array)
      batches = make_batches(num_samples_or_steps, batch_size)

      for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]

        # Slice into a batch.
        try:
          if ins and isinstance(ins[-1], int):
            # Do not slice the training phase flag.
            ins_batch = slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
          else:
            ins_batch = slice_arrays(ins, batch_ids)
        except TypeError:
          raise TypeError('TypeError while preparing batch. '
                          'If using HDF5 input data, '
                          'pass shuffle="batch".')

        # Sparse to dense conversion.
        if issparse is not None:
          for i in indices_for_conversion_to_dense:
            ins_batch[i] = ins_batch[i].toarray()

        # Callbacks batch_begin.
        batch_logs = {'batch': batch_index, 'size': len(batch_ids)}
        callbacks._call_batch_hook(mode, 'begin', batch_index, batch_logs)
        progbar.on_batch_begin(batch_index, batch_logs)

        # Get outputs.
        batch_outs = f(ins_batch)
        if not isinstance(batch_outs, list):
          batch_outs = [batch_outs]

        # Aggregate results.
        if batch_index == 0:
          aggregator.create(batch_outs)
        aggregator.aggregate(batch_outs, batch_start, batch_end)

        # Callbacks batch end.
        batch_logs = cbks.make_logs(model, batch_logs, batch_outs, mode)
        callbacks._call_batch_hook(mode, 'end', batch_index, batch_logs)
        progbar.on_batch_end(batch_index, batch_logs)

        if callbacks.model.stop_training:
          break

    aggregator.finalize()
    results = aggregator.results
    epoch_logs = cbks.make_logs(model, epoch_logs, results, mode)
    if len(results) == 1:
      results = results[0]

    # Run the test loop every `validation_freq` epochs during training.
    if (do_validation and
        training_utils.should_run_validation(validation_freq, epoch) and
        not callbacks.model.stop_training):

      if model._compile_distribution:
        # Since we create a new clone from the original model we need to copy
        # the weights back to the original model before we can run validation.
        distributed_training_utils._copy_weights_to_original_model(
            model, ModeKeys.TRAIN)

      val_results = model_iteration(
          model,
          val_inputs,
          targets=val_targets,
          sample_weights=val_sample_weights,
          batch_size=batch_size,
          steps_per_epoch=validation_steps,
          callbacks=callbacks,
          verbose=0,
          mode=ModeKeys.TEST,
          validation_in_fit=True,
          prepared_feed_values_from_dataset=(val_iterator is not None),
          steps_name='validation_steps')
      if not isinstance(val_results, list):
        val_results = [val_results]
      epoch_logs = cbks.make_logs(
          model, epoch_logs, val_results, mode, prefix='val_')
      if val_iterator and epoch < epochs - 1:
        _reinitialize_iterator(val_iterator, model._distribution_strategy)

    if mode == ModeKeys.TRAIN:
      # Epochs only apply to `fit`.
      callbacks.on_epoch_end(epoch, epoch_logs)
    progbar.on_epoch_end(epoch, epoch_logs)

    # Reinitialize dataset iterator for the next epoch.
    if reset_dataset_after_each_epoch and epoch < epochs - 1:
      _reinitialize_iterator(input_iterator, model._distribution_strategy)

  callbacks._call_end_hook(mode)

  if model._distribution_strategy:
    if model._compile_distribution:
      # TODO(priyag, psv): Copy back metrics to the original model as well?
      distributed_training_utils._copy_weights_to_original_model(model, mode)
    scope.__exit__(None, None, None)

  if mode == ModeKeys.TRAIN:
    return model.history
  return results


def _get_model_feed(model, mode):
  if mode == ModeKeys.PREDICT:
    feed = model._feed_inputs
  else:
    feed = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights)
  return feed


def _print_train_info(inputs, val_inputs, steps_per_epoch, verbose):
  if (val_inputs and steps_per_epoch is None and verbose and inputs and
      hasattr(inputs[0], 'shape') and hasattr(val_inputs[0], 'shape')):
    print('Train on %d samples, validate on %d samples' %
          (inputs[0].shape[0], val_inputs[0].shape[0]))


def _get_num_samples_or_steps(ins, batch_size, steps_per_epoch):
  """Returns total number of samples (when training in batch mode) or steps."""
  if steps_per_epoch:
    return steps_per_epoch
  return training_utils.check_num_samples(ins, batch_size, steps_per_epoch,
                                          'steps_per_epoch')


def _prepare_feed_values(model, inputs, targets, sample_weights, mode):
  """Prepare feed values to the model execution function.

  Arguments:
    model: Model to prepare feed values for.
    inputs: List or dict of model inputs.
    targets: Optional list of model targets.
    sample_weights: Optional list of sample weight arrays.
    mode: One of ModeKeys.TRAIN/ModeKeys.TEST/ModeKeys.PREDICT.

  Returns:
    Feed values for the model in the given mode.
  """
  if model._distribution_strategy:
    if isinstance(inputs, (dataset_ops.DatasetV1, dataset_ops.DatasetV2)):
      inputs = distributed_training_utils.get_iterator(
          inputs, model._distribution_strategy)

    def get_distributed_inputs():
      return distributed_training_utils._prepare_feed_values(
          model, inputs, targets, sample_weights, mode)

    # In the eager case, we want to call the input method per step, so return
    # a lambda from here that can be called. Note that this is applicable only
    # in Distribution Strategy case as it follows the same code path for both
    # eager and graph modes.
    # TODO(priyag,omalleyt): Either we should move the training DS with
    # IteratorV2 to use training_generator code path, or figure out how to
    # set a symbolic Iterator out of a Dataset when in eager mode.
    if context.executing_eagerly():
      return get_distributed_inputs
    else:
      return get_distributed_inputs()

  if isinstance(inputs, (dataset_ops.DatasetV1, dataset_ops.DatasetV2,
                         iterator_ops.Iterator)):
    inputs, targets, sample_weights = model._standardize_user_data(
        inputs,
        extract_tensors_from_dataset=True)

  inputs = training_utils.ModelInputs(inputs).as_list()
  targets = targets or []
  sample_weights = sample_weights or []
  ins = inputs + targets + sample_weights
  if mode == ModeKeys.TRAIN and not isinstance(K.symbolic_learning_phase(),
                                               int):
    ins += [True]  # Add learning phase value.
  return ins


def _get_iterator(inputs, distribution_strategy=None):
  if distribution_strategy:
    return distributed_training_utils.get_iterator(
        inputs, distribution_strategy)
  return training_utils.get_iterator(inputs)


def _reinitialize_iterator(iterator, distribution_strategy=None):
  if distribution_strategy:
    distributed_training_utils.initialize_iterator(
        iterator, distribution_strategy)
  else:
    training_utils.initialize_iterator(iterator)


def _make_execution_function(model, mode):
  """Makes function to run one step of model execution."""
  if model._distribution_strategy:
    return distributed_training_utils._make_execution_function(model, mode)
  return model._make_execution_function(mode)


# For backwards compatibility for internal users of these loops.
fit_loop = functools.partial(model_iteration, mode=ModeKeys.TRAIN)
test_loop = functools.partial(
    model_iteration, mode=ModeKeys.TEST, shuffle=False)
predict_loop = functools.partial(
    model_iteration, mode=ModeKeys.PREDICT, shuffle=False)
