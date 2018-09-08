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
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import distributed_training_utils
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import distribute as distribute_lib


def fit_loop(
    model,
    iterator,
    epochs=100,
    verbose=1,
    callbacks=None,
    val_iterator=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None):
  """fit function when using DistributionStrategy for training.

  Arguments:
      model: Keras Model instance.
      iterator: Iterator for input data.
      epochs: Number of times to iterate over the data
      verbose: Verbosity mode, 0, 1 or 2
      callbacks: List of callbacks to be called during training
      val_iterator: Iterator for validation data.
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
  current_strategy = model._distribution_strategy

  # TODO(priyag, sourabhbajaj): Remove this when the codepaths are merged.
  if current_strategy.__class__.__name__ == 'TPUStrategy':
    return _experimental_fit_loop(
        model, iterator, epochs, verbose, callbacks, initial_epoch,
        steps_per_epoch)

  clone_model_on_towers(
      model, current_strategy, make_callback_model=True)

  def _per_device_train_function(model):
    model._make_train_function()
    return (model.train_function.inputs,
            model.train_function.outputs,
            model.train_function.updates_op,
            model.train_function.session_kwargs)

  inputs, targets = _get_input_from_iterator(iterator, model)
  with current_strategy.scope():
    # Create train ops on each of the devices when we call
    # `_per_device_train_function`.
    (grouped_inputs, grouped_outputs, grouped_updates,
     grouped_session_args) = current_strategy.call_for_each_tower(
         _per_device_train_function, model._grouped_model)
    # Unwrap all the per device values returned from `call_for_each_tower`.
    # Unwrapping per device values gives you a list of values that can be
    # used to construct a new train function that is composed of update ops on
    # all the devices over which the model is distributed.
    (all_inputs, all_outputs, all_updates,
     all_session_args) = distributed_training_utils.unwrap_values(
         current_strategy, grouped_inputs, grouped_outputs,
         grouped_updates, grouped_session_args, with_loss_tensor=True)

    # Dataset inputs and targets are also per devices values that need to be
    # unwrapped.
    dataset_inputs = distributed_training_utils.flatten_perdevice_values(
        current_strategy, inputs)
    dataset_targets = distributed_training_utils.flatten_perdevice_values(
        current_strategy, targets)

  # Create a train function that is composed of all the parameters above.
  distributed_train_function = K.Function(
      all_inputs, all_outputs,
      updates=all_updates,
      name='distributed_train_function',
      **all_session_args)

  # We need to set sample_weights to None since there are sample weight
  # placeholders that are created with default values.
  sample_weights = [None for _ in range(len(model.outputs) *
                                        current_strategy.num_towers)]
  if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
    ins = dataset_inputs + dataset_targets + sample_weights + [1]
  else:
    ins = dataset_inputs + dataset_targets

  do_validation = False
  if validation_steps:
    do_validation = True

  # Copy the weights from the original model to each of the replicated models.
  orig_model_weights = model.get_weights()
  with current_strategy.scope():
    distributed_model = current_strategy.unwrap(model._grouped_model)[0]
    distributed_training_utils.set_weights(
        current_strategy, distributed_model, orig_model_weights)

  callbacks = cbks.configure_callbacks(
      callbacks,
      model,
      do_validation=do_validation,
      val_inputs=None,
      val_targets=None,
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      verbose=verbose)
  out_labels = model.metrics_names or []
  callbacks.on_train_begin()

  assert steps_per_epoch is not None

  for epoch in range(initial_epoch, epochs):
    callbacks.on_epoch_begin(epoch)
    epoch_logs = {}
    for step_index in range(steps_per_epoch):
      batch_logs = {'batch': step_index, 'size': 1}
      callbacks.on_batch_begin(step_index, batch_logs)
      try:
        outs = distributed_train_function(ins)
      except errors.OutOfRangeError:
        logging.warning('Your dataset iterator ran out of data; '
                        'interrupting training. Make sure that your dataset '
                        'can generate at least `steps_per_epoch * epochs` '
                        'batches (in this case, %d batches).' %
                        steps_per_epoch * epochs)
        break

      if not isinstance(outs, list):
        outs = [outs]

      outs = _aggregate_metrics_across_towers(
          current_strategy.num_towers, out_labels, outs)
      for l, o in zip(out_labels, outs):
        batch_logs[l] = o
      callbacks.on_batch_end(step_index, batch_logs)
      if callbacks.model.stop_training:
        break
    if do_validation:
      val_outs = test_loop(
          model,
          val_iterator,
          steps=validation_steps,
          verbose=0)
      if not isinstance(val_outs, list):
        val_outs = [val_outs]
      # Same labels assumed.
      for l, o in zip(out_labels, val_outs):
        epoch_logs['val_' + l] = o

    callbacks.on_epoch_end(epoch, epoch_logs)
    if callbacks.model.stop_training:
      break
  callbacks.on_train_end()

  # Copy the weights back from the replicated model to the original model.
  with current_strategy.scope():
    updated_weights = current_strategy.unwrap(
        model._grouped_model)[0].get_weights()
    model.set_weights(updated_weights)
  return model.history


def _experimental_fit_loop(
    model,
    iterator,
    epochs=100,
    verbose=1,
    callbacks=None,
    initial_epoch=0,
    steps_per_epoch=None):
  """fit function when using TPU DistributionStrategy for training.

  Arguments:
      model: Keras Model instance.
      iterator: Iterator that returns inputs and targets
      epochs: Number of times to iterate over the data
      verbose: Verbosity mode, 0, 1 or 2
      callbacks: List of callbacks to be called during training
      initial_epoch: Epoch at which to start training
          (useful for resuming a previous training run)
      steps_per_epoch: Total number of steps (batches of samples)
          before declaring one epoch finished and starting the
          next epoch. Ignored with the default value of `None`.

  Returns:
      Returns `None`.

  Raises:
      ValueError: in case of invalid arguments.
  """
  current_strategy = model._distribution_strategy

  # TODO(priyag): Add validation that shapes are fully defined for TPU case.

  K.get_session().run(current_strategy.initialize())

  def _per_device_train_function(model):
    model._make_train_function()
    return (model.train_function.inputs,
            model.train_function.outputs,
            model.train_function.updates_op,
            model.train_function.session_kwargs)

  # TODO(priyag, sourabhbajaj): This should likely not be hardcoded here.
  K.set_learning_phase(1)

  def step_fn(ctx, inputs, targets):
    """Clones the model and calls make_train_function."""
    # TODO(priyag, sourabhbajaj): Should cache this keyed on input shapes.
    clone_model_on_towers(
        model,
        current_strategy,
        make_callback_model=True,
        inputs=inputs,
        targets=targets)

    (grouped_inputs, grouped_outputs, grouped_updates,
     grouped_session_args) = current_strategy.call_for_each_tower(
         _per_device_train_function, model._grouped_model)
    (all_inputs, all_outputs, all_updates,
     all_session_args) = distributed_training_utils.unwrap_values(
         current_strategy, grouped_inputs, grouped_outputs,
         grouped_updates, grouped_session_args, with_loss_tensor=True)
    combined_fn = K.Function(
        all_inputs, all_outputs,
        updates=all_updates,
        name='distributed_train_function',
        **all_session_args)

    # TODO(priyag, sourabhbajaj): Perhaps the aggregation type needs to be
    # something else for different outputs.
    out_labels = model.metrics_names or []
    for label, output in zip(out_labels, combined_fn.outputs):
      ctx.set_last_step_output(label, output,
                               aggregation=distribute_lib.get_loss_reduction())

    # TODO(priyag, sourabhbajaj): Ignoring these things from the combined_fn:
    # feed_dict, session kwargs, run options, run_metadata for now. These should
    # be handled appropriately
    return combined_fn.updates_op

  # Add initial dummy values for loss and other metric tensors.
  initial_loop_values = {}
  initial_loop_values['loss'] = constant_op.constant(1e7)
  for name, tensor in zip(model.metrics_names[1:], model.metrics_tensors):
    initial_loop_values[name] = array_ops.zeros(tensor.shape, tensor.dtype)

  with current_strategy.scope():
    # TODO(priyag, sourabhbajaj): Adjust steps_per_run appropriately based on
    # steps_per_epoch and number of epochs.
    ctx = current_strategy.run_steps_on_dataset(
        step_fn, iterator, iterations=current_strategy.steps_per_run,
        initial_loop_values=initial_loop_values)

  train_op = ctx.run_op
  output_tensors = ctx.last_step_outputs

  # Copy the weights from the original model to each of the replicated models.
  orig_model_weights = model.get_weights()
  with current_strategy.scope():
    distributed_model = current_strategy.unwrap(model._grouped_model)[0]
    distributed_training_utils.set_weights(
        current_strategy, distributed_model, orig_model_weights)

  assert steps_per_epoch is not None

  # TODO(sourabhbajaj): Convert this into a proper validation function
  if callbacks:
    raise NotImplementedError(
        'Callbacks are not supported with TPUStrategy right now.')

  callbacks = cbks.configure_callbacks(
      callbacks,
      model,
      do_validation=False,
      val_inputs=None,
      val_targets=None,
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      verbose=verbose)
  # TODO(priyag, sourabhbajaj): Add callbacks support for per step callback
  # TODO(priyag, sourabhbajaj): Fix the number of steps run with steps_per_run
  # TODO(priyag, sourabhbajaj): Add validation.
  callbacks.on_train_begin()
  for epoch in range(initial_epoch, epochs):
    callbacks.on_epoch_begin(epoch)
    epoch_logs = {}
    for step_index in range(0, steps_per_epoch, current_strategy.steps_per_run):
      # TODO(sourabhbajaj): Add the size parameter in batch_logs once callbacks
      # are fixed as we need to replace size with a combination of steps_per_run
      # and batch_size
      batch_logs = {'batch': step_index}
      callbacks.on_batch_begin(step_index, batch_logs)
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
      if callbacks.model.stop_training:
        break

    callbacks.on_epoch_end(epoch, epoch_logs)
    if callbacks.model.stop_training:
      break
  callbacks.on_train_end()

  # Copy the weights back from the replicated model to the original model.
  with current_strategy.scope():
    updated_weights = current_strategy.unwrap(
        model._grouped_model)[0].get_weights()
    model.set_weights(updated_weights)

  K.get_session().run(current_strategy.finalize())
  return model.history


def test_loop(model, iterator, verbose=0, steps=None):
  """evaluate method to validate a model that uses DistributionStrategy.

  Arguments:
      model: Keras Model instance.
      iterator: Iterator for input data.
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
  current_strategy = model._distribution_strategy

  clone_model_on_towers(model, current_strategy)

  def _per_device_test_function(model):
    model._make_test_function()
    return (model.test_function.inputs,
            model.test_function.outputs,
            model.test_function.updates_op,
            model.test_function.session_kwargs)

  inputs, targets = _get_input_from_iterator(iterator, model)
  with current_strategy.scope():
    (grouped_inputs, grouped_outputs, grouped_updates,
     grouped_session_args) = current_strategy.call_for_each_tower(
         _per_device_test_function, model._grouped_model)

    (all_inputs, all_outputs, all_updates,
     all_session_args) = distributed_training_utils.unwrap_values(
         current_strategy, grouped_inputs, grouped_outputs, grouped_updates,
         grouped_session_args, with_loss_tensor=True)

    dataset_inputs = distributed_training_utils.flatten_perdevice_values(
        current_strategy, inputs)
    dataset_targets = distributed_training_utils.flatten_perdevice_values(
        current_strategy, targets)

  distributed_test_function = K.Function(
      all_inputs, all_outputs,
      updates=all_updates,
      name='distributed_test_function',
      **all_session_args)

  # We need to set sample_weights to None since there are sample weight
  # placeholders that are created with default values.
  sample_weights = [None for _ in range(len(model.outputs) *
                                        current_strategy.num_towers)]
  if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
    ins = dataset_inputs + dataset_targets + sample_weights + [0]
  else:
    ins = dataset_inputs + dataset_targets

  outs = []
  if verbose == 1:
    progbar = Progbar(target=steps)

  # Copy the weights from the original model to each of the replicated models.
  orig_model_weights = model.get_weights()
  with current_strategy.scope():
    distributed_model = current_strategy.unwrap(model._grouped_model)[0]
    distributed_training_utils.set_weights(
        current_strategy, distributed_model, orig_model_weights)

  if steps is not None:
    for step in range(steps):
      batch_outs = distributed_test_function(ins)
      batch_outs = _aggregate_metrics_across_towers(
          current_strategy.num_towers, model.metrics_names, batch_outs)
      if isinstance(batch_outs, list):
        if step == 0:
          for _ in enumerate(batch_outs):
            outs.append(0.)
        for i, batch_out in enumerate(batch_outs):
          outs[i] += batch_out
      else:
        if step == 0:
          outs.append(0.)
        outs[0] += batch_outs
      if verbose == 1:
        progbar.update(step + 1)
    for i in range(len(outs)):
      outs[i] /= steps

  if len(outs) == 1:
    return outs[0]
  return outs


def predict_loop(model, iterator, verbose=0, steps=None):
  """Abstract method to loop over some data in batches.

  Arguments:
      model: Keras Model instance.
      iterator: Iterator for input data.
      verbose: verbosity mode.
      steps: Total number of steps (batches of samples)
          before declaring `_predict_loop` finished.
          Ignored with the default value of `None`.

  Returns:
      Array of predictions (if the model has a single output)
      or list of arrays of predictions
      (if the model has multiple outputs).
  """
  current_strategy = model._distribution_strategy

  clone_model_on_towers(model, current_strategy)

  def _per_device_predict_function(model):
    model._make_predict_function()
    return (model.predict_function.inputs,
            model.predict_function.outputs,
            model.predict_function.updates_op,
            model.predict_function.session_kwargs)

  inputs, _ = _get_input_from_iterator(iterator, model)
  with current_strategy.scope():
    (grouped_inputs, grouped_outputs, grouped_updates,
     grouped_session_args) = current_strategy.call_for_each_tower(
         _per_device_predict_function, model._grouped_model)

    (all_inputs, all_outputs, all_updates,
     all_session_args) = distributed_training_utils.unwrap_values(
         current_strategy, grouped_inputs, grouped_outputs, grouped_updates,
         grouped_session_args)

    dataset_inputs = distributed_training_utils.flatten_perdevice_values(
        current_strategy, inputs)

  distributed_predict_function = K.Function(
      all_inputs, all_outputs,
      updates=all_updates,
      name='distributed_predict_function',
      **all_session_args)

  if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
    ins = dataset_inputs + [0]
  else:
    ins = dataset_inputs

  if verbose == 1:
    progbar = Progbar(target=steps)

  # Copy the weights from the original model to each of the replicated models.
  orig_model_weights = model.get_weights()
  with current_strategy.scope():
    distributed_model = current_strategy.unwrap(model._grouped_model)[0]
    distributed_training_utils.set_weights(
        current_strategy, distributed_model, orig_model_weights)

  if steps is not None:
    # Since we do not know how many samples we will see, we cannot pre-allocate
    # the returned Numpy arrays. Instead, we store one array per batch seen
    # and concatenate them upon returning.
    unconcatenated_outs = []
    for step in range(steps):
      batch_outs = distributed_predict_function(ins)
      if not isinstance(batch_outs, list):
        batch_outs = [batch_outs]
      if step == 0:
        for _ in batch_outs:
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


def _clone_and_build_model(model, inputs=None, targets=None):
  """Clone and build the given keras_model."""
  # We need to set the import here since we run into a circular dependency
  # error.
  from tensorflow.python.keras import models  # pylint: disable=g-import-not-at-top
  cloned_model = models.clone_model(model, input_tensors=inputs)

  # Compile and build model.
  if isinstance(model.optimizer, optimizers.TFOptimizer):
    optimizer = model.optimizer
  else:
    optimizer_config = model.optimizer.get_config()
    optimizer = model.optimizer.__class__.from_config(optimizer_config)

  # TODO(priyag): Is there a cleaner way to do this? The API doc suggests a
  # single tensor should be OK but it throws an error in that case.
  if (targets is not None and not isinstance(targets, list) and
      not isinstance(targets, dict)):
    targets = [targets]
  cloned_model.compile(
      optimizer,
      model.loss,
      metrics=model.metrics,
      loss_weights=model.loss_weights,
      sample_weight_mode=model.sample_weight_mode,
      weighted_metrics=model.weighted_metrics,
      target_tensors=targets)
  return cloned_model


def clone_model_on_towers(
    model, strategy, make_callback_model=False, inputs=None, targets=None):
  """Create a cloned model on each tower, unless already created."""
  if not model._grouped_model:
    with strategy.scope():
      model._grouped_model = strategy.call_for_each_tower(
          _clone_and_build_model, model, inputs, targets)
    if make_callback_model:
      model._make_callback_model()


def _aggregate_metrics_across_towers(num_devices, out_labels, outs):
  """Aggregate metrics values across all towers.

  When using `MirroredStrategy`, the number of towers is equal to the
  number of devices over which training is distributed. This may not always be
  the case.

  Args:
    num_devices: Number of devices over which the model is being distributed.
    out_labels: The list of metric names passed to `compile`.
    outs: The output from all the towers.

  Returns:
    The average value of each metric across the towers.
  """
  # TODO(anjalisridhar): Temporary workaround for aggregating metrics
  # across towers. Replace with the new metrics module eventually.
  merged_output = []
  # The first output is the total loss.
  merged_output.append(outs[0])
  current_index = 1
  # Each label in `out_labels` corresponds to one set of metrics. The
  # number of metric values corresponds to the number of devices. We
  # currently take the mean of the values.
  for _ in out_labels[1:]:
    m = np.mean(outs[current_index:current_index + num_devices])
    merged_output.append(m)
    current_index += num_devices
  return merged_output


def _get_input_from_iterator(iterator, model):
  """Get elements from the iterator and verify the input shape and type."""
  next_element = iterator.get_next()
  # TODO(anjalisridhar): Support predict input correctly as it will not contain
  # targets, only inputs.
  if not isinstance(next_element, (list, tuple)) or len(next_element) != 2:
    raise ValueError('Please provide model inputs as a list or tuple of 2 '
                     'elements: input and target pair. '
                     'Received %s' % next_element)

  x, y = next_element
  # Validate that all the elements in x and y are of the same type and shape.
  # We can then pass the first element of x and y to `_standardize_weights`
  # below and be confident of the output.
  x_values, y_values = distributed_training_utils.\
    validate_distributed_dataset_inputs(model._distribution_strategy, x, y)
  # TODO(sourabhbajaj): Add support for sample weights in distribution
  # strategy.
  model._standardize_weights(x_values, y_values)
  return x, y
