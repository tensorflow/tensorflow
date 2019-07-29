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
"""Training related logic for Keras model in TF 2.0 context.

Note that all the code under this module is under active development, please DO
NOT use it unless you are really sure what you are doing.
"""

# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import errors
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_v2_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_contextlib


# The list of DataAdapter that support validation_split, only numpy and data
# tensor support validation_split for now.
_ADAPTER_FOR_VALIDATION_SPLIT = [data_adapter.TensorLikeDataAdapter]

# The list of DataAdapter that support model._standardize_user_data. Currently
# keras.sequence/python generator will cause error when calling
# model._standardize_user_data, this should be updated in future cl, eg, the
# dataset/generate/sequence input will be peeked and processed by
# model._standardize_user_data()
_ADAPTER_FOR_STANDARDIZE_USER_DATA = [
    data_adapter.TensorLikeDataAdapter, data_adapter.DatasetAdapter
]


def run_one_epoch(model,
                  iterator,
                  execution_function,
                  dataset_size=None,
                  batch_size=None,
                  strategy=None,
                  steps_per_epoch=None,
                  num_samples=None,
                  mode=ModeKeys.TRAIN,
                  training_context=None,
                  total_epochs=None):
  """Run the execution function with the data from iterator.

  Given the dataset iterator and execution function, get the data from iterator
  and call it with the execution function to get the result (metric/loss).
  It will run for steps_per_epoch or until to the iterator is fully consumed.

  Args:
    model: The keras model to run.
    iterator: the dataset iterator to fetch the data.
    execution_function: a tf.function that can be called with data.
    dataset_size: the size of iterator, None when unknown.
    batch_size: The size of the current batch.
    strategy: the distribution strategy instance from the model.
    steps_per_epoch: the number of steps to run for the epoch.
    num_samples: the number of samples for the whole epoch if known. This can be
      used to calculate the final partial batch, and scale the loss.
    mode: the mode for the current epoch.
    training_context: the context that contains callbacks and progress bar.
    total_epochs: the total number of epochs that will be run.
      Used when throw error when the iterator unexpectedly
      reaches its end.
  Returns:
    The loss and metric value from the model.
  """
  # Only use the sample to count if there is a partial batch at the end.
  use_steps = num_samples is None

  if mode == ModeKeys.PREDICT:
    aggregator = training_utils.OutputsAggregator(
        use_steps=use_steps,
        steps=steps_per_epoch,
        num_samples=num_samples,
        batch_size=batch_size)
  else:
    aggregator = training_utils.MetricsAggregator(
        use_steps=use_steps, steps=steps_per_epoch, num_samples=num_samples)
  callbacks = training_context.callbacks
  progbar = training_context.progbar

  if callbacks.model.stop_training:
    return

  target_steps = steps_per_epoch or np.inf
  step = 0

  while step < target_steps:
    if use_steps:
      current_batch_size = 1
    elif step < target_steps - 1:
      current_batch_size = batch_size
    else:
      current_batch_size = num_samples - step * batch_size

    # TODO(scottzhu): Maybe update the training context to take into account
    #  whether a batch of training happens. Then it could still use a
    #  context manager
    batch_logs = {'batch': step, 'size': current_batch_size}
    training_context.callbacks._call_batch_hook(
        mode, 'begin', step, batch_logs)
    training_context.progbar.on_batch_begin(step, batch_logs)
    try:
      batch_outs = execution_function(iterator)
    except (StopIteration, errors.OutOfRangeError):
      # TODO(kaftan): File bug about tf function and errors.OutOfRangeError?
      # Are there any other C++ errors tf function should recapture?
      # The only acceptable case here is that the input has a unknown
      # length, and configured to fully consume it.
      if (dataset_size is None
          and steps_per_epoch is None
          and step > 0):
        # The input passed by the user ran out of batches.
        # Now we know the cardinality of the input(dataset or generator).
        steps_per_epoch = step
        aggregator.steps = steps_per_epoch
        progbar.params['steps'] = steps_per_epoch
        progbar.progbar.target = steps_per_epoch
      else:
        callbacks.model.stop_training = True
        logging.warning(
            'Your input ran out of data; interrupting training. '
            'Make sure that your dataset or generator can generate at '
            'least `steps_per_epoch * epochs` batches (in this case, '
            '{} batches). You may need to use the repeat() function '
            'when building your dataset.'.format(
                total_epochs * steps_per_epoch))
      # In either case, break out the loop for training batch.
      break

    if not isinstance(batch_outs, list):
      batch_outs = [batch_outs]
    if strategy:
      batch_outs = dist_utils._per_replica_aggregate_batch(
          strategy, batch_outs, model, mode)

    if step == 0:
      aggregator.create(batch_outs)

    if use_steps:
      aggregator.aggregate(batch_outs)
    else:
      aggregator.aggregate(
          batch_outs,
          batch_start=step * batch_size,
          batch_end=step * batch_size + current_batch_size)
    cbks.make_logs(model, batch_logs, batch_outs, mode)

    training_context.callbacks._call_batch_hook(
        mode, 'end', step, batch_logs)
    training_context.progbar.on_batch_end(step, batch_logs)

    step += 1

    if callbacks.model.stop_training:
      break

  # End of an epoch.
  aggregator.finalize()
  results = aggregator.results
  return results


class Loop(training_utils.TrainingLoop):
  """The training loop for the TF 2.0.

  This class has some existing assumption for runtime, eg eager by default,
  have distribution strategy, etc.
  """

  def fit(
      self, model, x=None, y=None, batch_size=None, epochs=1, verbose=1,
      callbacks=None, validation_split=0., validation_data=None, shuffle=True,
      class_weight=None, sample_weight=None, initial_epoch=0,
      steps_per_epoch=None, validation_steps=None, validation_freq=1, **kwargs):
    batch_size = model._validate_or_infer_batch_size(
        batch_size, steps_per_epoch, x)

    strategy = _get_distribution_strategy(model)
    batch_size, steps_per_epoch = dist_utils.process_batch_and_step_size(
        strategy, x, batch_size, steps_per_epoch, ModeKeys.TRAIN)
    dist_utils.validate_callbacks(input_callbacks=callbacks,
                                  optimizer=model.optimizer)
    # Enter tf.distribute.Strategy scope.
    with strategy.scope():
      training_data_adapter, validation_adapter = _process_training_inputs(
          model,
          x,
          y,
          batch_size=batch_size,
          sample_weights=sample_weight,
          class_weights=class_weight,
          validation_split=validation_split,
          steps_per_epoch=steps_per_epoch,
          shuffle=shuffle,
          validation_data=validation_data,
          validation_steps=validation_steps,
          distribution_strategy=strategy)

      total_samples = _get_total_number_of_samples(training_data_adapter)
      use_sample = total_samples is not None
      do_validation = (validation_adapter is not None)

      if not steps_per_epoch:
        steps_per_epoch = training_data_adapter.get_size()

      # tf.print('{} on {} steps.'.format(ModeKeys.TRAIN, steps_per_epoch))
      training_context = TrainingContext()

      initial_epoch = model._maybe_load_initial_epoch_from_ckpt(
          initial_epoch, ModeKeys.TRAIN)

      training_dataset = training_data_adapter.get_dataset()
      # Raise an error if steps_per_epoch isn't specified but the dataset
      # is infinite.
      # TODO(scottzhu): This check should probably happen in the adapter
      training_utils.infer_steps_for_dataset(
          training_dataset, steps_per_epoch, steps_name='steps_per_epoch',
          epochs=0)

      training_dataset = strategy.experimental_distribute_dataset(
          training_dataset)

      _update_sample_weight_mode(model, ModeKeys.TRAIN, training_dataset)
      training_function = training_v2_utils._get_or_make_execution_function(
          model, ModeKeys.TRAIN)

      training_data_iter = None
      # Only recreate iterator when the data has a fixed length, which will be
      # fully consumed every epoch, or has a unknown length (dataset, generator)
      # and will be fully consumed (steps_per_epoch is None)
      recreate_training_iterator = (training_data_adapter.get_size() is not None
                                    or steps_per_epoch is None)

      if do_validation:
        if not validation_steps:
          validation_steps = validation_adapter.get_size()
        eval_function = training_v2_utils._get_or_make_execution_function(
            model, ModeKeys.TEST)
        eval_data_iter = None

        validation_dataset = validation_adapter.get_dataset()
        # Raise an error if validation_steps isn't specified but the validation
        # dataset is infinite.
        # TODO(scottzhu): This check should probably happen in the adapter
        training_utils.infer_steps_for_dataset(
            validation_dataset, validation_steps, steps_name='validation_steps',
            epochs=0)
        validation_dataset = strategy.experimental_distribute_dataset(
            validation_dataset)

      callbacks = cbks.configure_callbacks(
          callbacks,
          model,
          do_validation=do_validation,
          batch_size=batch_size,
          epochs=epochs,
          steps_per_epoch=steps_per_epoch,
          samples=total_samples,
          count_mode='samples' if use_sample else 'steps',
          verbose=0,  # Handle ProgBarLogger separately in this loop.
          mode=ModeKeys.TRAIN)

      with training_context.on_start(
          model, callbacks, use_sample, verbose, ModeKeys.TRAIN):
        # TODO(scottzhu): Handle TPUStrategy training loop
        for epoch in range(initial_epoch, epochs):
          if training_context.callbacks.model.stop_training:
            break

          # Training
          with training_context.on_epoch(epoch, ModeKeys.TRAIN) as epoch_logs:
            model.reset_metrics()
            if training_data_iter is None or recreate_training_iterator:
              if (training_data_iter is not None and
                  distribution_strategy_context.has_strategy()):
                # TODO(kaftan): remove this when MultiDeviceIterator is a
                ## compositetensor (unless this is more efficient)
                training_data_iter._initializer  # pylint: disable=pointless-statement
              else:
                training_data_iter = iter(training_dataset)

            training_result = run_one_epoch(
                model,
                training_data_iter,
                training_function,
                dataset_size=training_data_adapter.get_size(),
                batch_size=training_data_adapter.batch_size(),
                strategy=strategy,
                steps_per_epoch=steps_per_epoch,
                num_samples=total_samples,
                mode=ModeKeys.TRAIN,
                training_context=training_context,
                total_epochs=epochs)
            cbks.make_logs(model, epoch_logs, training_result, ModeKeys.TRAIN)

            # Evaluation
            if (do_validation and
                training_utils.should_run_validation(validation_freq, epoch) and
                not callbacks.model.stop_training):
              if (eval_data_iter is not None and
                  distribution_strategy_context.has_strategy()):
                # TODO(kaftan): remove this when MultiDeviceIterator is a
                ## compositetensor (unless this is more efficient)
                eval_data_iter._initializer  # pylint: disable=pointless-statement
              else:
                eval_data_iter = iter(validation_dataset)

              val_total_samples = _get_total_number_of_samples(
                  validation_adapter)
              eval_context = TrainingContext()
              with eval_context.on_start(
                  model, callbacks, use_sample, verbose=0, mode=ModeKeys.TEST):
                with eval_context.on_epoch(epoch, ModeKeys.TEST):
                  model.reset_metrics()
                  eval_result = run_one_epoch(
                      model,
                      eval_data_iter,
                      eval_function,
                      dataset_size=validation_adapter.get_size(),
                      batch_size=validation_adapter.batch_size(),
                      strategy=strategy,
                      steps_per_epoch=validation_steps,
                      num_samples=val_total_samples,
                      mode=ModeKeys.TEST,
                      training_context=eval_context,
                      total_epochs=1)
                  cbks.make_logs(model, epoch_logs, eval_result, ModeKeys.TEST,
                                 prefix='val_')

    return model.history

  def _model_iteration(
      self, model, mode, x=None, y=None, batch_size=None, verbose=1,
      sample_weight=None, steps=None, callbacks=None, **kwargs):

    batch_size = model._validate_or_infer_batch_size(
        batch_size, steps, x)
    strategy = _get_distribution_strategy(model)
    batch_size, steps = dist_utils.process_batch_and_step_size(
        strategy, x, batch_size, steps, mode)
    dist_utils.validate_callbacks(input_callbacks=callbacks,
                                  optimizer=model.optimizer)
    # Enter tf.distribute.Strategy scope.
    with strategy.scope():
      adapter = _process_inputs(
          model,
          x,
          y,
          batch_size=batch_size,
          sample_weights=sample_weight,
          steps=steps,
          distribution_strategy=strategy)
      total_samples = _get_total_number_of_samples(adapter)
      use_sample = total_samples is not None

      if not steps:
        steps = adapter.get_size()

      # tf.print('{} on {} steps.'.format(ModeKeys.TRAIN, steps_per_epoch))
      training_context = TrainingContext()

      dataset = adapter.get_dataset()
      # Raise an error if `steps` isn't specified but the dataset
      # is infinite.
      # TODO(scottzhu): This check should probably happen in the adapter
      training_utils.infer_steps_for_dataset(
          dataset, steps, steps_name='steps', epochs=0)
      dataset = strategy.experimental_distribute_dataset(dataset)

      _update_sample_weight_mode(model, mode, dataset)
      execution_function = training_v2_utils._get_or_make_execution_function(
          model, mode)

      data_iterator = iter(dataset)

      callbacks = cbks.configure_callbacks(
          callbacks,
          model,
          do_validation=False,
          batch_size=batch_size,
          epochs=1,
          steps_per_epoch=steps,
          samples=use_sample,
          count_mode='samples' if use_sample else 'steps',
          verbose=0,  # Handle ProgBarLogger separately in this loop.
          mode=mode)

      with training_context.on_start(
          model, callbacks, use_sample, verbose, mode):
        # TODO(scottzhu): Handle TPUStrategy training loop
        with training_context.on_epoch(0, mode) as epoch_logs:
          model.reset_metrics()
          result = run_one_epoch(
              model,
              data_iterator,
              execution_function,
              dataset_size=adapter.get_size(),
              batch_size=adapter.batch_size(),
              strategy=strategy,
              steps_per_epoch=steps,
              num_samples=total_samples,
              mode=mode,
              training_context=training_context,
              total_epochs=1)
          cbks.make_logs(model, epoch_logs, result, mode)

    if len(result) == 1:
      result = result[0]
    return result

  def evaluate(
      self, model, x=None, y=None, batch_size=None, verbose=1,
      sample_weight=None, steps=None, callbacks=None, **kwargs):
    return self._model_iteration(
        model, ModeKeys.TEST, x=x, y=y, batch_size=batch_size, verbose=verbose,
        sample_weight=sample_weight, steps=steps, callbacks=callbacks, **kwargs)

  def predict(self, model, x, batch_size=None, verbose=0, steps=None,
              callbacks=None, **kwargs):
    return self._model_iteration(
        model, ModeKeys.PREDICT, x=x, batch_size=batch_size, verbose=verbose,
        steps=steps, callbacks=callbacks, **kwargs)


def _get_distribution_strategy(model):
  """Get the model's distribution strategy."""
  if model._compile_time_distribution_strategy:
    strategy = model._compile_time_distribution_strategy
  else:
    # Grab the active strategy if the model was never compiled
    # but it is now predicting.
    strategy = distribution_strategy_context.get_strategy()
  return strategy


def _process_training_inputs(model, x, y, batch_size=None,
                             sample_weights=None, class_weights=None,
                             steps_per_epoch=None, validation_split=0.,
                             validation_data=None, validation_steps=None,
                             shuffle=True, distribution_strategy=None):
  """Process the data input for fit() with respect to validation_split."""
  if validation_split and 0. < validation_split < 1. and validation_data:
    raise ValueError('validation_data and validation_split cannot be used '
                     'at same time.')

  adapter_cls = data_adapter.select_data_adapter(x, y)

  # Handle validation_split, we want to split the data and get the training
  # section before we give it to data adapter.
  if validation_split and 0. < validation_split < 1.:
    if adapter_cls not in _ADAPTER_FOR_VALIDATION_SPLIT:
      raise ValueError(
          '`validation_split` argument is not supported when '
          'data adapter is {}. Received: x={}, validation_split={}'.format(
              adapter_cls, x, validation_split))
    # Retrieve the training section from x and y, and then construct dataset
    # from it.
    x, y, sample_weights = model._standardize_user_data(
        x, y, sample_weight=sample_weights,
        class_weight=class_weights,
        batch_size=batch_size,
        check_steps=True,
        steps=steps_per_epoch)
    (x, y, sample_weights,
     val_x, val_y,
     val_sample_weights) = training_utils.split_training_and_validation_data(
         x, y, sample_weights, validation_split)
    train_adapter = adapter_cls(x, y, batch_size=batch_size,
                                sample_weights=sample_weights, shuffle=shuffle,
                                distribution_strategy=distribution_strategy)
    val_adapter = adapter_cls(val_x, val_y,
                              sample_weights=val_sample_weights,
                              batch_size=batch_size,
                              distribution_strategy=distribution_strategy)
  else:
    train_adapter = _process_inputs(model, x, y, sample_weights=sample_weights,
                                    batch_size=batch_size,
                                    class_weights=class_weights,
                                    shuffle=shuffle, steps=steps_per_epoch,
                                    distribution_strategy=distribution_strategy)
    val_adapter = None
    if validation_data:
      (val_x, val_y,
       val_sample_weights) = training_utils.unpack_validation_data(
           validation_data)
      # For eval data, we use the training data batch_size it was unknown.
      # This is useful for generator/sequence training data input with numpy
      # validation data input.
      if not batch_size:
        batch_size = train_adapter.batch_size()
      val_adapter = _process_inputs(model, val_x, val_y,
                                    sample_weights=val_sample_weights,
                                    batch_size=batch_size,
                                    class_weights=class_weights,
                                    steps=validation_steps,
                                    distribution_strategy=distribution_strategy)
    elif validation_steps:
      raise ValueError('`validation_steps` should not be specified if '
                       '`validation_data` is None.')
  return train_adapter, val_adapter


def _process_inputs(model, x, y, batch_size=None, sample_weights=None,
                    class_weights=None, shuffle=False, steps=None,
                    distribution_strategy=None):
  """Process the inputs for fit/eval/predict()."""
  adapter_cls = data_adapter.select_data_adapter(x, y)
  if adapter_cls in _ADAPTER_FOR_STANDARDIZE_USER_DATA:
    x, y, sample_weights = model._standardize_user_data(
        x,
        y,
        sample_weight=sample_weights,
        class_weight=class_weights,
        batch_size=batch_size,
        check_steps=True,
        steps=steps)
    # TODO(scottzhu): The generator and keras.sequence does not work with
    # model._standardize_user_data() so far. However that method is very
    # important which contains on-fly model build/tensor align for dict input,
    # etc. We should still call the _standardize_user_data with the peeked data
    # from generator or sequence, and let model compile.
  return adapter_cls(x, y, batch_size=batch_size, steps=steps,
                     sample_weights=sample_weights, shuffle=shuffle,
                     distribution_strategy=distribution_strategy)


def _update_sample_weight_mode(model, mode, dataset):
  """Updates the sample_weight_mode of a given model."""
  # TODO(kaftan): This won't actually do anything right now because
  ## dist_utils._update_sample_weight_modes only does things when the model
  ## is distributed by cloning. We will need to revisit if a method here
  ## is needed at all, and if so how it should look.
  # Add a quick return to prevent us from calling model._feed_targets that
  # accesses certain model properties that may not be set in the `PREDICT` mode.
  if mode == ModeKeys.PREDICT:
    return

  # Get some sample inputs from the data_adapter
  iterator = iter(dataset)
  _, _, sample_weights = training_v2_utils._prepare_feed_values(
      model, iterator, mode)

  # Call the DistributionStrategy specific function to update the
  # sample_weight_mode on the model.
  dist_utils._update_sample_weight_modes(model, mode, sample_weights)

  # Force delete the iterator.
  del iterator


def _get_total_number_of_samples(adapter):
  if not adapter.get_size() or not adapter.batch_size():
    return None
  total_sample = adapter.get_size() * adapter.batch_size()
  if adapter.has_partial_batch():
    total_sample -= (adapter.batch_size() - adapter.partial_batch_size())
  return total_sample


class TrainingContext(object):
  """Utility object that wrap around callbacks and progress bars."""

  @tf_contextlib.contextmanager
  def on_start(self, model, callbacks=None, use_samples=False, verbose=0,
               mode=ModeKeys.TRAIN):
    """Provide a scope for the whole training process."""
    # TODO(omalleyt): Handle ProgBar as part of Callbacks once hooks are ready.
    progbar = training_utils.get_progbar(
        model, 'samples' if use_samples else 'steps')
    progbar.params = callbacks.params
    progbar.params['verbose'] = verbose
    callbacks.model.stop_training = False
    callbacks._call_begin_hook(mode)
    progbar.on_train_begin()

    # Cache those two instance so that it can be used in other functions.
    self.callbacks = callbacks
    self.progbar = progbar

    try:
      yield
    finally:
      # End of all epochs
      self.callbacks._call_end_hook(mode)

  @tf_contextlib.contextmanager
  def on_epoch(self, epoch=0, mode=ModeKeys.TRAIN):
    """Provide a scope for running one epoch."""
    epoch_logs = {}
    if mode == ModeKeys.TRAIN:
      self.callbacks.on_epoch_begin(epoch, epoch_logs)
    self.progbar.on_epoch_begin(epoch, epoch_logs)
    try:
      yield epoch_logs
    finally:
      if mode == ModeKeys.TRAIN:
        # Epochs only apply to `fit`.
        self.callbacks.on_epoch_end(epoch, epoch_logs)
      self.progbar.on_epoch_end(epoch, epoch_logs)

  @tf_contextlib.contextmanager
  def on_batch(self, step=0, mode=ModeKeys.TRAIN):
    """Provide a scope for running one batch."""
    batch_logs = {'batch': step, 'size': 1}
    self.callbacks._call_batch_hook(
        mode, 'begin', step, batch_logs)
    self.progbar.on_batch_begin(step, batch_logs)
    try:
      yield batch_logs
    finally:
      self.callbacks._call_batch_hook(
          mode, 'end', step, batch_logs)
      self.progbar.on_batch_end(step, batch_logs)
