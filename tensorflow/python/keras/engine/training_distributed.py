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
"""Part of the Keras training engine related to distributed training.
"""
# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.engine import distributed_training_utils
from tensorflow.python.keras.engine import training_arrays
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.mode_keys import ModeKeys
from tensorflow.python.util import nest


def fit_distributed(model,
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
                    validation_freq=1):
  """Fit loop for Distribution Strategies."""
  distributed_training_utils.validate_callbacks(callbacks, model.optimizer)
  distributed_training_utils.validate_inputs(
      x, y, model._distribution_strategy)

  first_x_value = nest.flatten(x)[0]
  if isinstance(first_x_value, np.ndarray):
    steps_per_epoch, batch_size = (
        distributed_training_utils.get_input_params(
            model._distribution_strategy, first_x_value, steps_per_epoch,
            batch_size, is_training=True))
  batch_size = model._validate_or_infer_batch_size(
      batch_size, steps_per_epoch, x)
  dataset = model._distribution_standardize_user_data(
      x, y,
      sample_weight=sample_weight,
      class_weight=class_weight,
      batch_size=batch_size,
      check_steps=True,
      steps_name='steps_per_epoch',
      steps=steps_per_epoch,
      validation_split=validation_split,
      shuffle=shuffle)

  val_dataset = None
  if validation_data:
    val_x, val_y, val_sample_weights = model._unpack_validation_data(
        validation_data)
    distributed_training_utils.validate_inputs(
        val_x, val_y, model._distribution_strategy)
    first_valx_value = nest.flatten(val_x)[0]
    if isinstance(first_valx_value, np.ndarray):
      validation_steps, _ = distributed_training_utils.get_input_params(
          model._distribution_strategy, first_valx_value, validation_steps,
          batch_size)
    val_dataset = model._distribution_standardize_user_data(
        val_x, val_y,
        sample_weight=val_sample_weights,
        class_weight=None,
        batch_size=batch_size,
        check_steps=True,
        steps_name='validation_steps',
        steps=validation_steps,
        validation_split=validation_split,
        shuffle=shuffle)
  elif validation_split:
    raise ValueError('validation_split argument is not supported with '
                     'distribution strategies.')

  if distributed_training_utils.is_tpu_strategy(model._distribution_strategy):
    return experimental_tpu_fit_loop(
        model,
        dataset,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        val_dataset=val_dataset,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_freq=1)
  else:
    return training_arrays.fit_loop(
        model,
        dataset,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        val_inputs=val_dataset,
        shuffle=shuffle,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_freq=validation_freq)


def evaluate_distributed(model,
                         x=None,
                         y=None,
                         batch_size=None,
                         verbose=1,
                         sample_weight=None,
                         steps=None,
                         callbacks=None):
  """Evaluate loop for Distribution Strategies."""
  distributed_training_utils.validate_inputs(x, y, model._distribution_strategy)
  first_x_value = nest.flatten(x)[0]
  if isinstance(first_x_value, np.ndarray):
    steps, batch_size = distributed_training_utils.get_input_params(
        model._distribution_strategy, first_x_value, steps, batch_size)
  batch_size = model._validate_or_infer_batch_size(batch_size, steps, x)
  dataset = model._distribution_standardize_user_data(
      x, y,
      sample_weight=sample_weight,
      batch_size=batch_size,
      check_steps=True,
      steps_name='steps',
      steps=steps)

  if distributed_training_utils.is_tpu_strategy(model._distribution_strategy):
    # TODO(fchollet): why aren't callbacks supported here?
    return experimental_tpu_test_loop(
        model, dataset, verbose=verbose, steps=steps)
  else:
    return training_arrays.test_loop(
        model,
        inputs=dataset,
        batch_size=batch_size,
        verbose=verbose,
        steps=steps,
        callbacks=callbacks)


def predict_distributed(model,
                        x=None,
                        batch_size=None,
                        verbose=0,
                        steps=None,
                        callbacks=None):
  """Predict loop for Distribution Strategies."""
  distributed_training_utils.validate_inputs(
      x, None, model._distribution_strategy)
  first_x_value = nest.flatten(x)[0]
  if isinstance(first_x_value, np.ndarray):
    steps, batch_size = distributed_training_utils.get_input_params(
        model._distribution_strategy, first_x_value, steps, batch_size)
  batch_size = model._validate_or_infer_batch_size(batch_size, steps, x)
  dataset = model._distribution_standardize_user_data(
      x,
      batch_size=batch_size,
      check_steps=True,
      steps_name='steps',
      steps=steps)
  if distributed_training_utils.is_tpu_strategy(model._distribution_strategy):
    # TODO(fchollet): why aren't callbacks supported here?
    return experimental_tpu_predict_loop(
        model, dataset, verbose=verbose, steps=steps)
  else:
    return training_arrays.predict_loop(
        model,
        dataset,
        batch_size=batch_size,
        verbose=verbose,
        steps=steps,
        callbacks=callbacks)


def experimental_tpu_fit_loop(model,
                              dataset,
                              epochs=100,
                              verbose=1,
                              callbacks=None,
                              initial_epoch=0,
                              steps_per_epoch=None,
                              val_dataset=None,
                              validation_steps=None,
                              validation_freq=1):
  """Fit loop for training with TPU DistributionStrategy.

  Arguments:
      model: Keras Model instance.
      dataset: Dataset that returns inputs and targets
      epochs: Number of times to iterate over the data
      verbose: Integer, Verbosity mode, 0, 1 or 2
      callbacks: List of callbacks to be called during training
      initial_epoch: Epoch at which to start training
          (useful for resuming a previous training run)
      steps_per_epoch: Total number of steps (batches of samples)
          before declaring one epoch finished and starting the
          next epoch. Ignored with the default value of `None`.
      val_dataset: Dataset for validation data.
      validation_steps: Number of steps to run validation for
          (only if doing validation from data tensors).
          Ignored with the default value of `None`.
      validation_freq: Only relevant if validation data is provided. Integer or
          `collections.Container` instance (e.g. list, tuple, etc.). If an
          integer, specifies how many training epochs to run before a new
          validation run is performed, e.g. `validation_freq=2` runs
          validation every 2 epochs. If a Container, specifies the epochs on
          which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
          validation at the end of the 1st, 2nd, and 10th epochs.

  Returns:
      Returns `None`.

  Raises:
      ValueError: in case of invalid arguments.
  """
  # TODO(fchollet): add support for `steps_per_epoch=None` in TPU loops.
  current_strategy = model._distribution_strategy
  iterator = distributed_training_utils.get_iterator(dataset, current_strategy)
  scope = current_strategy.scope()
  scope.__enter__()

  def _per_device_fit_function(model):
    model._make_fit_function()
    return (model._fit_function.inputs, model._fit_function.outputs,
            model._fit_function.updates_op, model._fit_function.session_kwargs)

  # TODO(priyag, sourabhbajaj): This should likely not be hardcoded here.
  K.set_learning_phase(1)
  out_labels = model.metrics_names or []

  def step_fn(ctx, inputs):
    """Clones the model and calls make_fit_function."""
    inputs, targets = inputs
    if model._compile_distribution:
      distributed_training_utils.clone_model_on_replicas(
          model, current_strategy,
          make_callback_model=True, inputs=inputs,
          targets=targets, mode=distributed_training_utils.ModeKeys.TRAIN)
    else:
      distributed_training_utils._build_distributed_network(
          model, current_strategy, inputs,
          targets, mode=distributed_training_utils.ModeKeys.TRAIN)

    (grouped_inputs, grouped_outputs, grouped_updates,
     grouped_session_args) = current_strategy.extended.call_for_each_replica(
         _per_device_fit_function, args=(model._distributed_model_train,))
    (all_inputs, all_outputs, all_updates,
     all_session_args) = distributed_training_utils.unwrap_values(
         current_strategy, grouped_inputs, grouped_outputs,
         grouped_updates, grouped_session_args)
    combined_fn = K.function(
        all_inputs,
        all_outputs,
        updates=all_updates,
        name='distributed_fit_function',
        **all_session_args)

    for label, output in zip(out_labels, combined_fn.outputs):
      if label == 'loss':
        reduce_op = distribute_lib.get_loss_reduction()
      else:
        # We reduce all other metrics using mean for now. This is temporary
        # workaround until new metrics are in place.
        reduce_op = ds_reduce_util.ReduceOp.MEAN
      ctx.set_last_step_output(label, output, reduce_op)

    # TODO(priyag, sourabhbajaj): Ignoring these things from the combined_fn:
    # feed_dict, session kwargs, run options, run_metadata for now. These should
    # be handled appropriately
    return combined_fn.updates_op

  # Add initial dummy values for loss and other metric tensors.
  initial_loop_values = {}
  initial_loop_values['loss'] = constant_op.constant(1e7)
  for name in model.metrics_names[1:]:
    tensor = model._all_stateful_metrics_tensors[name]
    initial_loop_values[name] = array_ops.zeros(tensor.shape, tensor.dtype)

  if steps_per_epoch is None:
    raise ValueError('`steps_per_epoch` should be specified when calling '
                     '`fit` on the model.')
  steps_per_run = K.variable(
      value=min(steps_per_epoch, current_strategy.extended.steps_per_run),
      dtype='int32',
      name='steps_per_run')

  with current_strategy.scope():
    ctx = current_strategy.extended.experimental_run_steps_on_iterator(
        step_fn, iterator, iterations=steps_per_run,
        initial_loop_values=initial_loop_values)

  train_op = ctx.run_op
  output_tensors = ctx.last_step_outputs

  do_validation = bool(validation_steps)

  if model._compile_distribution:
    with current_strategy.scope():
      distributed_training_utils._copy_weights_to_distributed_model(
          model, model._distributed_model_train)

  callbacks = cbks.configure_callbacks(
      callbacks,
      model,
      do_validation=do_validation,
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      verbose=verbose)

  # Calculate the steps each time on the device.
  steps_to_run = [current_strategy.extended.steps_per_run] * (
      steps_per_epoch // current_strategy.extended.steps_per_run)
  if steps_per_epoch % current_strategy.extended.steps_per_run:
    steps_to_run.append(
        steps_per_epoch % current_strategy.extended.steps_per_run)

  callbacks.on_train_begin()
  for epoch in range(initial_epoch, epochs):
    with current_strategy.scope():
      distributed_training_utils._reset_metrics(
          model, model._distributed_model_train)
    callbacks.on_epoch_begin(epoch)
    epoch_logs = {}
    step_index = 0
    prev_step_count = None
    for step_count in steps_to_run:
      batch_logs = {'batch': step_index, 'size': 1, 'num_steps': step_count}
      callbacks.on_batch_begin(step_index, batch_logs)
      if prev_step_count is None or step_count != prev_step_count:
        steps_per_run.load(step_count, K.get_session())
        prev_step_count = step_count
      try:
        _, outputs = K.get_session().run([train_op, output_tensors])
      except errors.OutOfRangeError:
        logging.warning('Your dataset iterator ran out of data; '
                        'interrupting training. Make sure that your dataset '
                        'can generate at least `steps_per_epoch * epochs` '
                        'batches (in this case, %d batches).' %
                        steps_per_epoch * epochs)
        break

      batch_logs.update(outputs)
      callbacks.on_batch_end(step_index, batch_logs)
      step_index = step_index + step_count
      if callbacks.model.stop_training:
        break

    if (do_validation and
        training_utils.should_run_validation(validation_freq, epoch)):
      logging.info('Running validation at fit epoch: %s', epoch)

      if model._compile_distribution:
        # Since we create a new clone from the original model we need to copy
        # the weights back to the original model before we can run validation.
        with current_strategy.scope():
          distributed_training_utils._copy_weights_to_original_model(
              model, model._distributed_model_train, ModeKeys.TRAIN)

      val_outs = experimental_tpu_test_loop(  # pylint: disable=undefined-variable
          model,
          val_dataset,
          steps=validation_steps,
          verbose=verbose)
      if not isinstance(val_outs, list):
        val_outs = [val_outs]
      # Same labels assumed.
      for label, val_out in zip(out_labels, val_outs):
        epoch_logs['val_' + label] = val_out

    callbacks.on_epoch_end(epoch, epoch_logs)
    if callbacks.model.stop_training:
      break
  callbacks.on_train_end()

  if model._compile_distribution:
    # Copy the weights back from the replicated model to the original model.
    with current_strategy.scope():
      distributed_training_utils._copy_weights_to_original_model(
          model, model._distributed_model_train, ModeKeys.TRAIN)
  scope.__exit__(None, None, None)
  return model.history


def experimental_tpu_test_loop(model,
                               dataset,
                               verbose=0,
                               steps=None):
  """Test loop for evaluating with TPU DistributionStrategy.

  Arguments:
      model: Keras Model instance.
      dataset: Dataset for input data.
      verbose: Integer, Verbosity mode 0 or 1.
      steps: Total number of steps (batches of samples)
          before declaring predictions finished.
          Ignored with the default value of `None`.

  Returns:
      Scalar loss (if the model has a single output and no metrics)
      or list of scalars (if the model has multiple outputs
      and/or metrics). The attribute `model.metrics_names` will give you
      the display labels for the outputs.
  """
  current_strategy = model._distribution_strategy
  iterator = distributed_training_utils.get_iterator(dataset, current_strategy)
  scope = current_strategy.scope()
  scope.__enter__()

  def _per_device_eval_function(model):
    model._make_eval_function()
    return (model._eval_function.inputs, model._eval_function.outputs,
            model._eval_function.updates_op,
            model._eval_function.session_kwargs)

  # TODO(priyag, sourabhbajaj): This should likely not be hardcoded here.
  K.set_learning_phase(0)

  def step_fn(ctx, inputs):
    """Clones the model and calls make_eval_function."""
    inputs, targets = inputs
    if model._compile_distribution:
      distributed_training_utils. clone_model_on_replicas(
          model, current_strategy,
          make_callback_model=False, inputs=inputs,
          targets=targets, mode=distributed_training_utils.ModeKeys.TEST)
    else:
      distributed_training_utils._build_distributed_network(
          model, current_strategy, inputs, targets,
          mode=distributed_training_utils.ModeKeys.TEST)

    (grouped_inputs, grouped_outputs, grouped_updates,
     grouped_session_args) = current_strategy.extended.call_for_each_replica(
         _per_device_eval_function, args=(model._distributed_model_test,))

    (all_inputs, all_outputs, all_updates,
     all_session_args) = distributed_training_utils.unwrap_values(
         current_strategy, grouped_inputs, grouped_outputs, grouped_updates,
         grouped_session_args)

    combined_fn = K.function(
        all_inputs, all_outputs,
        updates=all_updates,
        name='distributed_test_function',
        **all_session_args)

    for label, output in zip(model.metrics_names, combined_fn.outputs):
      if label == 'loss':
        reduce_op = distribute_lib.get_loss_reduction()
      else:
        # We reduce all other metrics using mean for now. This is temporary
        # workaround until new metrics are in place.
        reduce_op = ds_reduce_util.ReduceOp.MEAN
      ctx.set_last_step_output(label, output, reduce_op)

    return combined_fn.updates_op

  # Add initial dummy values for loss and other metric tensors.
  initial_loop_values = {}
  initial_loop_values['loss'] = constant_op.constant(1e7)
  for name in model.metrics_names[1:]:
    tensor = model._all_stateful_metrics_tensors[name]
    initial_loop_values[name] = array_ops.zeros(tensor.shape, tensor.dtype)

  with current_strategy.scope():
    # TODO(priyag): Use steps_per_run when we use new metrics as they will
    # allow handling metric computation at each step using variables.
    ctx = current_strategy.extended.experimental_run_steps_on_iterator(
        step_fn, iterator, iterations=1,
        initial_loop_values=initial_loop_values)

  test_op = ctx.run_op
  output_tensors = ctx.last_step_outputs

  if verbose == 1:
    progbar = Progbar(target=steps)

  if model._compile_distribution:
    with current_strategy.scope():
      distributed_training_utils._copy_weights_to_distributed_model(
          model, model._distributed_model_test)
  with current_strategy.scope():
    distributed_training_utils._reset_metrics(
        model, model._distributed_model_test)

  assert steps is not None
  outs = [0.] * len(model.metrics_names)
  for step in range(steps):
    _, batch_outs = K.get_session().run([test_op, output_tensors])
    for i, label in enumerate(model.metrics_names):
      if i == 0:
        # Loss is stateless metrics.
        outs[i] += batch_outs[label]
      else:
        # For all stateful metrics, the aggregation is handled by mirrored vars.
        outs[i] = batch_outs[label]

    if verbose >= 1:
      progbar.update(step + 1)

  scope.__exit__(None, None, None)
  if len(outs) >= 0:
    outs[0] /= (steps)

  if len(outs) == 1:
    return outs[0]
  return outs


def experimental_tpu_predict_loop(model, dataset, verbose=0, steps=None):
  """Predict loop for predicting with TPU DistributionStrategy.

  Arguments:
      model: Keras Model instance.
      dataset: Dataset for input data.
      verbose: Integer, Verbosity mode 0 or 1.
      steps: Total number of steps (batches of samples)
          before declaring `_predict_loop` finished.
          Ignored with the default value of `None`.

  Returns:
      Array of predictions (if the model has a single output)
      or list of arrays of predictions
      (if the model has multiple outputs).
  """
  current_strategy = model._distribution_strategy
  iterator = distributed_training_utils.get_iterator(dataset, current_strategy)
  scope = current_strategy.scope()
  scope.__enter__()

  # TODO(priyag, sourabhbajaj): This should likely not be hardcoded here.
  K.set_learning_phase(0)

  def _per_device_predict_function(model):
    model._make_predict_function()
    return (model.predict_function.inputs,
            model.predict_function.outputs,
            model.predict_function.updates_op,
            model.predict_function.session_kwargs)

  def step_fn(ctx, inputs):
    """Clones the model and calls make_predict_function."""
    if model._compile_distribution:
      distributed_training_utils. clone_model_on_replicas(
          model, current_strategy,
          make_callback_model=False, inputs=inputs,
          mode=distributed_training_utils.ModeKeys.PREDICT)
    else:
      distributed_training_utils._build_distributed_network(
          model, current_strategy, inputs,
          mode=distributed_training_utils.ModeKeys.PREDICT)

    (grouped_inputs, grouped_outputs, grouped_updates,
     grouped_session_args) = current_strategy.extended.call_for_each_replica(
         _per_device_predict_function, args=(model._distributed_model_predict,))

    (all_inputs, all_outputs, all_updates,
     all_session_args) = distributed_training_utils.unwrap_values(
         current_strategy, grouped_inputs, grouped_outputs, grouped_updates,
         grouped_session_args)

    combined_fn = K.function(
        all_inputs, all_outputs,
        updates=all_updates,
        name='distributed_predict_function',
        **all_session_args)

    for label, output in zip(model.output_names, combined_fn.outputs):
      ctx.set_last_step_output(label, output)

    return combined_fn.updates_op

  # Add initial dummy values for outputs.
  initial_loop_values = {}
  batch_dimension = distributed_training_utils.get_batch_dimension(iterator)
  for name, tensor in zip(model.output_names, model.outputs):
    # TODO(priyag): This is a workaround as we do not know the batch dimension
    # of the model's output at this point.
    shape = tensor_shape.TensorShape(tensor.shape.dims)
    shape.dims = [batch_dimension] + shape.dims[1:]
    initial_loop_values[name] = array_ops.zeros(shape, tensor.dtype)

  with current_strategy.scope():
    # TODO(priyag, sourabhbajaj): Support steps_per_run if/when we add outfeed.
    ctx = current_strategy.extended.experimental_run_steps_on_iterator(
        step_fn, iterator, iterations=1,
        initial_loop_values=initial_loop_values)

  predict_op = ctx.run_op
  output_tensors = ctx.last_step_outputs

  if verbose == 1:
    progbar = Progbar(target=steps)

  if model._compile_distribution:
    with current_strategy.scope():
      distributed_training_utils._copy_weights_to_distributed_model(
          model, model._distributed_model_predict)
  with current_strategy.scope():
    distributed_training_utils._reset_metrics(
        model, model._distributed_model_predict)

  assert steps is not None
  # Since we do not know how many samples we will see, we cannot pre-allocate
  # the returned Numpy arrays. Instead, we store one array per batch seen
  # and concatenate them upon returning.
  unconcatenated_outs = [[] for _ in model.outputs]
  for step in range(steps):
    _, batch_outs = K.get_session().run([predict_op, output_tensors])
    # TODO(priyag): maybe need to unwrap the outputs first for MirroredStrategy.
    for i, label in enumerate(model.output_names):
      unconcatenated_outs[i].extend(batch_outs[label])
    if verbose >= 1:
      progbar.update(step + 1)

  scope.__exit__(None, None, None)
  if len(unconcatenated_outs) == 1:
    return np.concatenate(unconcatenated_outs[0], axis=0)
  return [
      np.concatenate(unconcatenated_outs[i], axis=0)
      for i in range(len(unconcatenated_outs))
  ]
