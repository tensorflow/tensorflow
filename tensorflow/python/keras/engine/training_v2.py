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

import collections

import numpy as np


from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib


# The list of DataAdapter that support validation_split, only numpy and data
# tensor support validation_split for now.
_ADAPTER_FOR_VALIDATION_SPLIT = [data_adapter.NumpyArrayDataAdapter,
                                 data_adapter.TensorDataAdapter]

# The list of DataAdapter that support model._standardize_user_data. Currently
# keras.sequence/python generator will cause error when calling
# model._standardize_user_data, this should be updated in future cl, eg, the
# dataset/generate/sequence input will be peeked and processed by
# model._standardize_user_data()
_ADAPTER_FOR_STANDARDIZE_USER_DATA = [data_adapter.NumpyArrayDataAdapter,
                                      data_adapter.TensorDataAdapter,
                                      data_adapter.DatasetAdapter]


def run_one_epoch(model,
                  iterator,
                  execution_function,
                  dataset_size=None,
                  strategy=None,
                  steps_per_epoch=None,
                  mode=ModeKeys.TRAIN,
                  training_context=None,
                  current_epoch=1):
  """Run the execution function with the data from iterator.

  Given the dataset iterator and execution function, get the data from iterator
  and call it with the execution function to get the result (metric/loss).
  It will run for steps_per_epoch or until to the iterator is fully consumed.

  Args:
    model: The keras model to run.
    iterator: the dataset iterator to fetch the data.
    execution_function: a tf.function that can be called with data.
    dataset_size: the size of iterator, None when unknown.
    strategy: the distribution strategy instance from the model.
    steps_per_epoch: the number of steps to run for the epoch.
    mode: the mode for the current epoch.
    training_context: the context that contains callbacks and progress bar.
    current_epoch: the epoch number. Used when throw error when the
      the iterator is unexpected reach its end.
  Returns:
    The loss and metric value from the model.
  """
  if mode == ModeKeys.PREDICT:
    aggregator = training_utils.OutputsAggregator(
        use_steps=True, num_samples_or_steps=steps_per_epoch)
  else:
    aggregator = training_utils.MetricsAggregator(
        use_steps=True, num_samples_or_steps=steps_per_epoch)
  callbacks = training_context.callbacks
  progbar = training_context.progbar

  if callbacks.model.stop_training:
    return

  target_steps = steps_per_epoch or np.inf
  step = 0

  while step < target_steps:
    with training_context.on_batch(step, mode=mode) as batch_logs:
      try:
        batch_ins = create_batch_inputs(iterator, mode, model, strategy)
        batch_outs = execution_function(batch_ins)
      except StopIteration:
        # The only acceptable case here is that the input has a unknown
        # length, and configured to fully consume it.
        if (dataset_size is None
            and steps_per_epoch is None
            and step > 0):
          # The input passed by the user ran out of batches.
          # Now we know the cardinality of the input(dataset or generator).
          steps_per_epoch = step
          aggregator.num_samples_or_steps = steps_per_epoch
          progbar.params['steps'] = steps_per_epoch
          progbar.progbar.target = steps_per_epoch
        else:
          callbacks.model.stop_training = True
          logging.warning(
              'Your input ran out of data; interrupting training. '
              'Make sure that your dataset or generator can generate at '
              'least {} batches. You may need to use the repeat() function '
              'when building your dataset.'.format(
                  current_epoch * steps_per_epoch))
        # In either case, break out the loop for training batch.
        break

      if not isinstance(batch_outs, list):
        batch_outs = [batch_outs]
      if strategy:
        batch_outs = dist_utils._per_replica_aggregate_batch(
            batch_outs, model, mode)

      if step == 0:
        aggregator.create(batch_outs)
      aggregator.aggregate(batch_outs)
      cbks.make_logs(model, batch_logs, batch_outs, mode)
      step += 1

    if callbacks.model.stop_training:
      break

  # End of an epoch.
  aggregator.finalize()
  results = aggregator.results
  return results


def create_batch_inputs(iterator, mode, model, strategy):
  """Create the input data from the iterator based on the model and strategy."""
  if strategy:
    # Note that the batch_ins is a function to avoid the tf.function
    # retrace.
    def distribute_batch_ins():
      return dist_utils._prepare_feed_values(model, iterator, None, None, mode)
    batch_ins = distribute_batch_ins
  else:
    batch_ins = next(iterator)
    if (mode == ModeKeys.TRAIN
        and not model.run_eagerly
        and not isinstance(backend.symbolic_learning_phase(), int)):
      # Add learning phase value.
      if not isinstance(batch_ins, collections.Sequence):
        batch_ins = (batch_ins, True)
      else:
        batch_ins += (True,)
  return batch_ins


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
    if strategy:
      batch_size, steps_per_epoch = dist_utils.process_batch_and_step_size(
          strategy, x, batch_size, steps_per_epoch, ModeKeys.TRAIN)
      dist_utils.validate_callbacks(input_callbacks=callbacks,
                                    optimizer=model.optimizer)
      # Enter tf.distribute.Strategy scope.
      scope = dist_utils.distributed_scope(
          strategy=strategy, learning_phase=1)
      scope.__enter__()

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

    do_validation = (validation_adapter is not None)

    if not steps_per_epoch:
      steps_per_epoch = training_data_adapter.get_size()

    # tf.print('{} on {} steps.'.format(ModeKeys.TRAIN, steps_per_epoch))
    training_context = TrainingContext()

    initial_epoch = model._maybe_load_initial_epoch_from_ckpt(
        initial_epoch, ModeKeys.TRAIN)

    _update_sample_weight_mode(model, ModeKeys.TRAIN, training_data_adapter)
    training_function = _make_execution_function(model, ModeKeys.TRAIN)

    training_data_iter = None
    # Only recreate iterator when the data has a fixed length, which will be
    # fully consumed every epoch, or has a unknown length (dataset, generator)
    # and will be fully consumed (steps_per_epoch is None)
    recreate_training_iterator = (training_data_adapter.get_size() is not None
                                  or steps_per_epoch is None)

    if do_validation:
      if not validation_steps:
        validation_steps = validation_adapter.get_size()
      eval_function = _make_execution_function(model, ModeKeys.TEST)
      eval_data_iter = None
      recreate_eval_iterator = (validation_adapter.get_size() is not None
                                or validation_steps is None)

    callbacks = cbks.configure_callbacks(
        callbacks,
        model,
        do_validation=do_validation,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        samples=None,
        verbose=0,  # Handle ProgBarLogger separately in this loop.
        mode=ModeKeys.TRAIN)

    with training_context.on_start(model, callbacks, verbose, ModeKeys.TRAIN):
      # TODO(scottzhu): Handle TPUStrategy training loop
      for epoch in range(initial_epoch, epochs):
        if training_context.callbacks.model.stop_training:
          break

        # Training
        with training_context.on_epoch(epoch, ModeKeys.TRAIN) as epoch_logs:
          model.reset_metrics()
          if training_data_iter is None or recreate_training_iterator:
            training_data_iter = _create_dataset_iterator(
                strategy, training_data_adapter.get_dataset())

          training_result = run_one_epoch(
              model,
              training_data_iter,
              training_function,
              dataset_size=training_data_adapter.get_size(),
              strategy=strategy,
              steps_per_epoch=steps_per_epoch,
              mode=ModeKeys.TRAIN,
              training_context=training_context,
              current_epoch=epoch)
          cbks.make_logs(model, epoch_logs, training_result, ModeKeys.TRAIN)

          # Evaluation
          if (do_validation and
              training_utils.should_run_validation(validation_freq, epoch) and
              not callbacks.model.stop_training):
            if eval_data_iter is None or recreate_eval_iterator:
              eval_data_iter = _create_dataset_iterator(
                  strategy, validation_adapter.get_dataset())
            eval_context = TrainingContext()
            with eval_context.on_start(
                model, callbacks, verbose=0, mode=ModeKeys.TEST):
              with eval_context.on_epoch(epoch, ModeKeys.TEST):
                model.reset_metrics()
                eval_result = run_one_epoch(
                    model,
                    eval_data_iter,
                    eval_function,
                    dataset_size=validation_adapter.get_size(),
                    strategy=strategy,
                    steps_per_epoch=validation_steps,
                    mode=ModeKeys.TEST,
                    training_context=eval_context,
                    current_epoch=epochs)
                cbks.make_logs(model, epoch_logs, eval_result, ModeKeys.TRAIN,
                               prefix='val_')

    if strategy:
      scope.__exit__(None, None, None)

    return model.history

  def _model_iteration(
      self, model, mode, x=None, y=None, batch_size=None, verbose=1,
      sample_weight=None, steps=None, callbacks=None, **kwargs):

    batch_size = model._validate_or_infer_batch_size(
        batch_size, steps, x)
    strategy = _get_distribution_strategy(model)
    if strategy:
      batch_size, steps = dist_utils.process_batch_and_step_size(
          strategy, x, batch_size, steps, mode)
      dist_utils.validate_callbacks(input_callbacks=callbacks,
                                    optimizer=model.optimizer)
      # Enter tf.distribute.Strategy scope.
      scope = dist_utils.distributed_scope(
          strategy=strategy, learning_phase=0)
      scope.__enter__()

    adapter = _process_inputs(
        model,
        x,
        y,
        batch_size=batch_size,
        sample_weights=sample_weight,
        steps=steps,
        distribution_strategy=strategy)

    if not steps:
      steps = adapter.get_size()

    # tf.print('{} on {} steps.'.format(ModeKeys.TRAIN, steps_per_epoch))
    training_context = TrainingContext()

    _update_sample_weight_mode(model, mode, adapter)
    execution_function = _make_execution_function(model, mode)
    data_iterator = _create_dataset_iterator(
        strategy, adapter.get_dataset())

    callbacks = cbks.configure_callbacks(
        callbacks,
        model,
        do_validation=False,
        batch_size=batch_size,
        epochs=1,
        steps_per_epoch=steps,
        samples=None,
        verbose=0,  # Handle ProgBarLogger separately in this loop.
        mode=mode)

    with training_context.on_start(model, callbacks, verbose, mode):
      # TODO(scottzhu): Handle TPUStrategy training loop
      with training_context.on_epoch(0, mode) as epoch_logs:
        model.reset_metrics()
        result = run_one_epoch(
            model,
            data_iterator,
            execution_function,
            dataset_size=adapter.get_size(),
            strategy=strategy,
            steps_per_epoch=steps,
            mode=mode,
            training_context=training_context,
            current_epoch=1)
        cbks.make_logs(model, epoch_logs, result, mode)

    if strategy:
      scope.__exit__(None, None, None)

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
  if model._distribution_strategy:
    return model._distribution_strategy
  # TODO(scottzhu): might want to just get the default strategy in future.
  elif distribution_strategy_context.has_strategy():
    return distribution_strategy_context.get_strategy()
  else:
    return None


def _create_dataset_iterator(strategy, training_dataset):
  if strategy:
    training_data_iter = strategy.make_dataset_iterator(training_dataset)
  else:
    training_data_iter = iter(training_dataset)
  return training_data_iter


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
  return adapter_cls(x, y, batch_size=batch_size,
                     sample_weights=sample_weights, shuffle=shuffle,
                     distribution_strategy=distribution_strategy)


def _make_execution_function(model, mode):
  """Makes function to run one step of model execution."""
  if model._distribution_strategy:
    return dist_utils._make_execution_function(model, mode)
  else:
    return model._make_execution_function(mode)


def _update_sample_weight_mode(model, mode, adapter):
  """Updates the sample_weight_mode of a given model."""
  # Add a quick return to prevent us from calling model._feed_targets that
  # accesses certain model properties that may not be set in the `PREDICT` mode.
  if mode == ModeKeys.PREDICT:
    return

  sample_weights = None

  # Get some sample inputs from the data_adapter
  iterator = _create_dataset_iterator(model._distribution_strategy,
                                      adapter.get_dataset())
  inputs = create_batch_inputs(iterator, mode, model,
                               model._distribution_strategy)
  # `inputs` is the model's inputs + targets + sample_weights +
  # learning phase placeholder if specified. To update the sample_weight_mode
  # we need to determine if the user has passed sample weights as part of the
  # input.
  if not callable(inputs):
    # if not isinstance(inputs, collections.Sequence):
    #   inputs = (inputs,)
    # Note that the batch inputs should be a tuple of 2, 3 or 4 items.
    # (input, target, {sample_weights}, {learning_phase})
    sample_weights_index = 0
    if model._feed_inputs:
      sample_weights_index += 1
    if model._feed_targets:
      sample_weights_index += 1

    sample_weights = inputs[sample_weights_index:]
    has_learning_phase_pl = (mode == ModeKeys.TRAIN and
                             not isinstance(backend.symbolic_learning_phase(),
                                            int))
    if has_learning_phase_pl:
      sample_weights = sample_weights[:-1]
    model._update_sample_weight_modes(nest.flatten(sample_weights))

  # Call the DistributionStrategy specific function to update the
  # sample_weight_mode on the model.
  if model._distribution_strategy:
    dist_utils._update_sample_weight_modes(model, mode, sample_weights)

  # Force delete the iterator.
  del iterator


class TrainingContext(object):
  """Utility object that wrap around callbacks and progress bars."""

  @tf_contextlib.contextmanager
  def on_start(self, model, callbacks=None, verbose=0, mode=ModeKeys.TRAIN):
    """Provide a scope for the whole training process."""
    # TODO(omalleyt): Handle ProgBar as part of Callbacks once hooks are ready.
    progbar = training_utils.get_progbar(model, 'steps')
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
