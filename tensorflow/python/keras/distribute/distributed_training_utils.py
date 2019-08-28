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
"""Utilities related to distributed training."""
# pylint:disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib


def set_weights(distribution_strategy, dist_model, weights):
  """Sets the weights of the replicated models.

  The weights of the replicated models are set to the weights of the original
  model. The weights of the replicated model are Mirrored variables and hence
  we need to use the `update` call within a DistributionStrategy scope.

  Args:
    distribution_strategy: DistributionStrategy used to distribute training
        and validation.
    dist_model: The replicated models on the different devices.
    weights: The weights of the original model.
  """
  assign_ops = []
  for layer in dist_model.layers:
    num_param = len(layer.weights)
    layer_weights = weights[:num_param]
    for sw, w in zip(layer.weights, layer_weights):
      if ops.executing_eagerly_outside_functions():
        sw.assign(w)
      else:
        assign_ops.append(distribution_strategy.unwrap(sw.assign(w)))
    weights = weights[num_param:]

  if not ops.executing_eagerly_outside_functions():
    K.get_session(assign_ops).run(assign_ops)


def unwrap_values(distribution_strategy, grouped_inputs, grouped_outputs,
                  grouped_updates=None, grouped_session_args=None,
                  with_loss_tensor=False):
  """Unwrap the list of values contained in the PerReplica parameters.

  This function calls `flatten_per_replica_values` to parse each of the input
  parameters into a list of values on the different devices. If we set
  `with_loss_tensor` to be True, we also call `reduce` on the list of losses on
  the different devices to give us one loss tensor.

  Args:
    distribution_strategy: DistributionStrategy used to distribute training and
        validation.
    grouped_inputs: PerReplica inputs returned from the train or test function
        that we ran on each device.
    grouped_outputs: PerReplica outputs returned from the train or test function
        that we ran on each device.
    grouped_updates: PerReplica updates returned from the train or test function
        that we ran on each device.
    grouped_session_args: PerReplica session args returned from the train or
        test function that we ran on each device.
    with_loss_tensor: Boolean that indicates if we need to add the reduced loss
        tensor as one of the outputs.

  Returns:
    Values of each of the PerReplica parameters.

  """
  # Unwrap per device values returned from each model's train function.
  # This will be used to construct the main train function.
  all_inputs = flatten_per_replica_values(distribution_strategy,
                                          grouped_inputs)
  all_outputs = unwrap_outputs(distribution_strategy, grouped_outputs,
                               with_loss_tensor)

  if grouped_updates:
    all_updates = flatten_per_replica_values(distribution_strategy,
                                             grouped_updates)
  else:
    all_updates = None

  all_session_args = {}
  if grouped_session_args:
    grouped_feed_dict = grouped_session_args.get('feed_dict')
    if grouped_feed_dict:
      all_session_args['feed_dict'] = flatten_per_replica_values(
          distribution_strategy, grouped_feed_dict)

    grouped_fetches = grouped_session_args.get('fetches')
    if grouped_fetches:
      all_session_args['fetches'] = flatten_per_replica_values(
          distribution_strategy, grouped_fetches)

  # TODO(priyag): Return only non empty/None values
  return all_inputs, all_outputs, all_updates, all_session_args


def unwrap_output_dict(strategy, grouped_outputs, mode):
  """Unwrap the list of outputs contained in the PerReplica parameters."""
  if mode == ModeKeys.PREDICT:
    return flatten_per_replica_values(strategy, grouped_outputs)

  # In the case of fit/eval, the grouped_outputs is a dict, whereas in predict,
  # the output is as same structure as model output. They need to be treated
  # differently
  total_loss = strategy.reduce(reduce_util.ReduceOp.SUM,
                               grouped_outputs['total_loss'][0], axis=None)
  output_losses = flatten_per_replica_values(strategy,
                                             grouped_outputs['output_losses'])
  metrics = flatten_per_replica_values(strategy,
                                       grouped_outputs['metrics'])
  batch_size = strategy.reduce(reduce_util.ReduceOp.SUM,
                               grouped_outputs['batch_size'], axis=None)
  if (is_tpu_strategy(strategy) and
      ops.executing_eagerly_outside_functions()):
    # Choose 1 value per replica in the TPU case since all replicas produce the
    # same output.
    # We only do this in eager mode for now since this function is used in
    # both graph and eager mode and in the graph case we currently don't use
    # experimental_run so would need to be removed when we converge the graph
    # code path as well.
    output_losses = output_losses[::strategy.num_replicas_in_sync]
    metrics = metrics[::strategy.num_replicas_in_sync]
  return {'total_loss': [total_loss],
          'output_losses': output_losses,
          'metrics': metrics,
          'batch_size': batch_size}


def unwrap_outputs(distribution_strategy, grouped_outputs,
                   with_loss_tensor=False):
  """Unwrap the list of outputs contained in the PerReplica parameters.

  This function calls `flatten_per_replica_values` to parse each of the input
  parameters into a list of outputs on the different devices. If we set
  `with_loss_tensor` to be True, we also call `reduce` on the list of losses on
  the different devices to give us one loss tensor.

  Args:
    distribution_strategy: DistributionStrategy used to distribute training and
        validation.
    grouped_outputs: PerReplica outputs returned from the train or test function
        that we ran on each device.
    with_loss_tensor: Boolean that indicates if we need to add the reduced loss
        tensor as one of the outputs.

  Returns:
    Values of each of the PerReplica outputs.

  """
  if not with_loss_tensor:
    return flatten_per_replica_values(distribution_strategy,
                                      grouped_outputs)

  if not isinstance(grouped_outputs, list):
    grouped_outputs = [grouped_outputs]
  # reduce loss tensor before adding it to the list of fetches
  loss = distribution_strategy.reduce(reduce_util.ReduceOp.SUM,
                                      grouped_outputs[0], axis=None)
  all_outputs = flatten_per_replica_values(distribution_strategy,
                                           grouped_outputs[1:])
  if (is_tpu_strategy(distribution_strategy) and
      ops.executing_eagerly_outside_functions()):
    # Choose 1 value per replica in the TPU case since all replicas produce the
    # same output.
    # We only do this in eager mode for now since this function is used in
    # both graph and eager mode and in the graph case we currently don't use
    # experimental_run so would need to be removed when we converge the graph
    # code path as well.
    all_outputs = all_outputs[::distribution_strategy.num_replicas_in_sync]
  return [loss] + all_outputs


def flatten_per_replica_values(distribution_strategy, per_replica_values):
  """Unwraps and flattens a nest of PerReplica parameters.

  PerReplica values have one value associated with each device. Each entry in
  the PerReplica dict has a device `key` and the corresponding value on the
  device as the `value`. In this function we take a PerReplica value or a list
  of PerReplica values and return all the values in the PerReplica dict.

  Args:
    distribution_strategy: DistributionStrategy used to distribute training and
      validation.
    per_replica_values: List of PerReplica object or a single PerReplica object.

  Returns:
    List of values of all the PerReplica objects.

  """
  # pylint: disable=g-complex-comprehension
  # This function takes a PerReplica object or a list of PerReplica objects and
  # returns all the values associated with it.
  return [e for flattened in nest.flatten(per_replica_values)
          for e in distribution_strategy.unwrap(flattened)]


def validate_callbacks(input_callbacks, optimizer):
  """Validate whether given callbacks are supported by DistributionStrategy.

  Args:
    input_callbacks: List of callbacks passed by the user to fit.
    optimizer: Optimizer instance used to train the model.

  Raises:
    ValueError: If `LearningRateScheduler` or `ReduceLROnPlateau` is one of the
        callbacks passed.
    ValueError: If `write_grads` is one of the parameters passed as part of the
        TensorBoard callback.
  """
  if input_callbacks:
    for callback in input_callbacks:
      if isinstance(callback, (callbacks.LearningRateScheduler,
                               callbacks.ReduceLROnPlateau)):

        if not isinstance(optimizer, optimizer_v2.OptimizerV2):
          raise ValueError('You must specify a Keras Optimizer V2 when using '
                           '%s callback with DistributionStrategy.' % callback)

      # If users want to use the TensorBoard callback they cannot use certain
      # features of the callback that involve accessing model attributes and
      # running ops.
      if isinstance(callback, callbacks.TensorBoard):
        if getattr(callback, 'write_grads', False):
          logging.warning(
              UserWarning(
                  '`write_grads` in the TensorBoard callback is not supported '
                  'when using DistributionStrategy. Setting `write_grads` '
                  'to `False`.'))
          callback.write_grads = False


def validate_distributed_dataset_inputs(distribution_strategy, x, y,
                                        sample_weights=None):
  """Validate all the components of a DistributedValue Dataset input.

  Args:
    distribution_strategy: The current DistributionStrategy used to call
        `fit`/`evaluate`.
    x: Input Dataset DistributedValue object. For example, when we use
        `MirroredStrategy` this is a PerReplica object with a tensor for each
        device set in the dict. x can also be a tuple or dict. The keys of the
        dict should match the names of the input layers of the model.
    y: Target Dataset DistributedValue object. For example, when we use
        `MirroredStrategy` this is a PerReplica object with a tensor for each
        device set in the dict. y can also be a tuple or dict. The keys of the
        dict should match the names of the output layers of the model.
    sample_weights: Sample weights Dataset DistributedValue object. For example,
        when we use `MirroredStrategy` this is a PerReplica object with a tensor
        for each device set in the dict.

  Returns:
    The unwrapped values list of the x and y DistributedValues inputs.

  Raises:
    ValueError: If x and y do not have support for being evaluated as tensors.
        or if x and y contain elements that are not tensors or if x and y
        contain elements that have a shape or dtype mismatch.
  """
  # If the input and target used to call the model are not dataset tensors,
  # we need to raise an error. When using a DistributionStrategy, the input
  # and targets to a model should be from a `tf.data.Dataset`.

  # If each element of x and y are not tensors, we cannot standardize and
  # validate the input and targets.
  x_values_list = validate_per_replica_inputs(distribution_strategy, x)

  if y is not None:
    y_values_list = validate_per_replica_inputs(distribution_strategy, y)
  else:
    y_values_list = None

  if sample_weights is not None:
    sample_weights_list = validate_per_replica_inputs(distribution_strategy,
                                                      sample_weights)
  else:
    sample_weights_list = None

  # Return the unwrapped values to avoid calling `unwrap` a second time.
  return x_values_list, y_values_list, sample_weights_list


def validate_per_replica_inputs(distribution_strategy, x):
  """Validates PerReplica dataset input list.

  Args:
    distribution_strategy: The current DistributionStrategy used to call
      `fit`, `evaluate` and `predict`.
    x: A list of PerReplica objects that represent the input or
      target values.

  Returns:
    List containing the first element of each of the PerReplica objects in
    the input list.

  Raises:
    ValueError: If any of the objects in the `per_replica_list` is not a tensor.

  """
  # Convert the inputs and targets into a list of PerReplica objects.
  per_replica_list = nest.flatten(x, expand_composites=True)
  x_values_list = []
  for x in per_replica_list:
    if not tensor_util.is_tensor(x):
      raise ValueError('Dataset input to the model should be tensors instead '
                       'they are of type {}'.format(type(x)))

    # At this point both x and y contain tensors in the `DistributedValues`
    # structure.
    x_values = distribution_strategy.unwrap(x)

    if not context.executing_eagerly():
      # Validate that the shape and dtype of all the elements in x are the same.
      validate_all_tensor_shapes(x, x_values)
    validate_all_tensor_types(x, x_values)

    x_values_list.append(x_values[0])
  return x_values_list


def validate_all_tensor_types(x, x_values):
  x_dtype = x_values[0].dtype
  for i in range(1, len(x_values)):
    if x_dtype != x_values[i].dtype:
      raise ValueError('Input tensor dtypes do not match for distributed tensor'
                       ' inputs {}'.format(x))


def validate_all_tensor_shapes(x, x_values):
  # Validate that the shape of all the elements in x have the same shape
  x_shape = x_values[0].shape.as_list()
  for i in range(1, len(x_values)):
    if x_shape != x_values[i].shape.as_list():
      raise ValueError('Input tensor shapes do not match for distributed tensor'
                       ' inputs {}'.format(x))


def _wait_for_variable_initialization(session):
  """Utility to wait for variables to be initialized."""
  all_variables = K._get_variables(K.get_graph())  # pylint: disable=protected-access
  candidate_vars = []
  for v in all_variables:
    if not getattr(v, '_keras_initialized', False):
      candidate_vars.append(v)

  if not candidate_vars:
    return

  while True:
    is_initialized = session.run(
        [variables.is_variable_initialized(v) for v in candidate_vars])
    uninitialized_vars = []
    for flag, v in zip(is_initialized, candidate_vars):
      if not flag:
        uninitialized_vars.append(v)
      v._keras_initialized = True  # pylint: disable=protected-access
    if not uninitialized_vars:
      break


def init_restore_or_wait_for_variables():
  """Initialize or restore variables or wait for variables to be initialized."""
  session = K._get_session()  # pylint: disable=protected-access
  if not multi_worker_util.has_worker_context(
  ) or multi_worker_util.should_load_checkpoint():
    # TODO(yuefengz): if checkpoints exist, restore from checkpoint.
    K._initialize_variables(session)  # pylint: disable=protected-access
  else:
    _wait_for_variable_initialization(session)


def validate_inputs(x, y):
  """Validate inputs when using DistributionStrategy.

  Args:
    x: Model Inputs.
    y: Model Targets.

  Raises:
    ValueError: if input is not a Dataset or a numpy array(when we use
      MirroredStrategy).
  """
  if (isinstance(x, iterator_ops.Iterator) or
      isinstance(y, iterator_ops.Iterator)):
    raise ValueError('`DistributionStrategy` does not support inputs of type '
                     'Iterator. You must pass a `tf.data.Dataset` object or a '
                     'numpy array as input.')


# TODO(b/118776054): Currently we support global batch size for TPUStrategy and
# core MirroredStrategy only. Remove this check when contrib MirroredStrategy is
# no longer needed.
def global_batch_size_supported(distribution_strategy):
  return distribution_strategy.extended._global_batch_size  # pylint: disable=protected-access


# TODO(sourabhbajaj): Remove this once we use the same API for all strategies.
def is_tpu_strategy(strategy):
  """We're executing TPU Strategy."""
  return (strategy is not None and
          strategy.__class__.__name__.startswith('TPUStrategy'))


def is_dataset_shape_fully_defined(dataset):
  """Returns whether a dataset contains a final partial batch."""
  shapes = nest.flatten(dataset_ops.get_legacy_output_shapes(dataset))
  unknown_shapes = [s for s in shapes if not s.is_fully_defined()]
  return not unknown_shapes


def process_batch_and_step_size(strategy,
                                inputs,
                                batch_size,
                                steps_per_epoch,
                                mode,
                                validation_split=0.):
  """Process the batch size and step size based on input and dist strategy."""
  first_x_value = nest.flatten(inputs)[0]
  if isinstance(first_x_value, np.ndarray):
    num_samples = first_x_value.shape[0]
    if validation_split and 0. < validation_split < 1.:
      num_samples = int(num_samples * (1 - validation_split))
    # Until support for partial batch is implemented across all
    # functions and distribution strategy, we pass `mode` to selectively
    # relax the constraint to consume all the training samples.
    steps_per_epoch, batch_size = get_input_params(
        strategy, num_samples, steps_per_epoch, batch_size, mode=mode)
  return batch_size, steps_per_epoch


def get_input_params(distribution_strategy,
                     num_samples,
                     steps,
                     batch_size,
                     mode=None):
  """Calculate the number of batches and steps/steps_per_epoch.

  Args:
    distribution_strategy: The DistributionStrategy used to compile the model.
    num_samples: The number of samples from which we determine the batch size
      and steps.
    steps:  The specified number of steps.
    batch_size: The specified batch_size.
    mode: ModeKey representing whether input will be used for training,
      evaluation, or prediction. This is used to relax the constraints on
      consuming all the training samples to keep compatibility till we support
      partial batches. If none, then partial batches are not allowed.

  Returns:
    steps: The steps or steps_per_epoch argument depending on if a user is
        calling `fit`, `evaluate` or `predict`. If the is_training flag is set
        we don't require the number of samples to be used completely.
    batch_size: The batch size to be used in model iterations.

  Raises:
    ValueError: If the number of batches or steps evaluates to 0.

  """
  # TODO(b/118776054): Use global batch size for Keras/DS support.
  # Currently this is only supported in TPUStrategy and CoreMirroredStrategy.
  use_per_replica_batch = not global_batch_size_supported(
      distribution_strategy)

  # TODO(b/128995245): In eager mode, uneven batch sizes are allowed except for
  # `fit()` on TPUStrategy.
  # In graph mode, the zero batch case in batch norm is not handled due to
  # XLA-GPU regression. Uneven batch sizes are not allowed except
  # for `test()` and `predict()` on TPUStrategy.
  if context.executing_eagerly():
    allow_partial_batch = (mode != ModeKeys.TRAIN or
                           not is_tpu_strategy(distribution_strategy))
  else:
    allow_partial_batch = (mode == ModeKeys.TRAIN or
                           ((mode == ModeKeys.PREDICT or mode == ModeKeys.TEST)
                            and is_tpu_strategy(distribution_strategy)))

  if steps is None:
    if batch_size is None:
      # If neither the batch size or number of steps are set. We choose the
      # global batch size as the minimum of number of samples and 32. 32 is
      # chosen to provide backward compatibility.
      global_batch_size = min(num_samples, 32)
    else:
      # If the user provided the batch size we need to handle the case
      # between different strategies that use the global/per-replica batch size
      global_batch_size = batch_size
      if use_per_replica_batch:
        global_batch_size *= distribution_strategy.num_replicas_in_sync
    if allow_partial_batch:
      steps = np.ceil(num_samples / global_batch_size).astype(int)
    else:
      if num_samples % global_batch_size:
        raise ValueError('The number of samples %s is not divisible by '
                         'batch size %s.' % (num_samples, global_batch_size))
      steps = num_samples // global_batch_size
  else:
    if batch_size is None:
      # We calculate the batch size based on the number of steps specified
      if num_samples % steps:
        raise ValueError('The number of samples %s is not divisible by '
                         'steps %s. Please change the number of steps to a '
                         'value that can consume all the samples' % (
                             num_samples, steps))
      global_batch_size = num_samples // steps
    else:
      # If the user provided the batch size we need to handle the case
      # between different strategies that use the global/per-replica batch size
      global_batch_size = batch_size
      if use_per_replica_batch:
        global_batch_size *= distribution_strategy.num_replicas_in_sync

      min_num_samples = global_batch_size * steps
      if allow_partial_batch:
        min_num_samples = global_batch_size * (steps-1) + 1 if steps > 1 else 0

      if num_samples < min_num_samples:
        raise ValueError('Number of samples %s is less than samples required '
                         'for specified batch_size %s and steps %s' % (
                             num_samples, global_batch_size, steps))

  # We need to return the per replica or global batch size based on the strategy
  if use_per_replica_batch:
    if global_batch_size % distribution_strategy.num_replicas_in_sync:
      raise ValueError(
          'The batch size (%s) could not be sharded evenly across the sync '
          'replicas (%s) in the distribution strategy.' % (
              global_batch_size, distribution_strategy.num_replicas_in_sync))
    batch_size = global_batch_size // distribution_strategy.num_replicas_in_sync
  else:
    batch_size = global_batch_size

  return steps, batch_size


def get_batch_dimension(iterator):
  shapes = nest.flatten(dataset_ops.get_legacy_output_shapes(iterator))
  # Take the batch size from the first element, as it should be the same for
  # all.
  dims = shapes[0].dims
  return dims[0] if dims else None


def get_iterator(dataset, distribution_strategy):
  with distribution_strategy.scope():
    iterator = distribution_strategy.make_dataset_iterator(dataset)
  initialize_iterator(iterator, distribution_strategy)
  return iterator


def initialize_iterator(iterator, distribution_strategy):
  with distribution_strategy.scope():
    init_op = control_flow_ops.group(iterator.initialize())
    if not context.executing_eagerly():
      K.get_session((init_op,)).run(init_op)


def _get_input_from_iterator(iterator, model):
  """Get elements from the iterator and verify the input shape and type."""
  next_element = iterator.get_next()

  # `len(nest.flatten(x))` is going to not count empty elements such as {}.
  # len(nest.flatten([[0,1,2], {}])) is 3 and not 4.   The `next_element` is
  # going to get flattened in `_prepare_feed_values` to work around that. Empty
  # elements are going to get filtered out as part of the flattening.
  if len(nest.flatten(next_element)) == len(model.inputs):
    x = next_element
    y = None
    sample_weights = None
  elif len(nest.flatten(next_element)) == (len(model.inputs) +
                                           len(model.outputs)):
    x, y = next_element
    sample_weights = None
  else:
    x, y, sample_weights = next_element

  # Validate that all the elements in x and y are of the same type and shape.
  validate_distributed_dataset_inputs(
      model._distribution_strategy, x, y, sample_weights)
  return x, y, sample_weights


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
  strategy = model._distribution_strategy
  inputs, targets, sample_weights = _get_input_from_iterator(inputs, model)
  if is_tpu_strategy(strategy):
    if sample_weights is not None:
      raise ValueError('TPUStrategy does not support sample weights.')

  # When the inputs are dict, then we want to flatten it in the same order as
  # the input layers, such that the data are fed into the input layers in the
  # correct order.
  if isinstance(inputs, dict):
    inputs = [inputs[key] for key in model._feed_input_names]
  if is_distributing_by_cloning(model):
    inputs = flatten_per_replica_values(strategy, inputs)
    targets = flatten_per_replica_values(strategy, targets)
    # Expand 1-dimensional inputs.
    # TODO(b/124535720): Remove once this standarize data logic is shared with
    # main flow.
    inputs, targets = nest.map_structure(
        training_utils.standardize_single_array, (inputs, targets))
  else:
    inputs = training_utils.ModelInputs(inputs).as_list()

  if mode == ModeKeys.PREDICT:
    sample_weights = []
    targets = []
  elif sample_weights is not None and is_distributing_by_cloning(model):
    if context.executing_eagerly() and not model._compile_distribution:
      raise NotImplementedError('`sample_weight` is not supported when using '
                                'tf.distribute.Strategy in eager mode and '
                                'cloning=True.')
    sample_weights = flatten_per_replica_values(strategy, sample_weights)

  ins = [inputs, targets, sample_weights]
  return tuple(ins)


def is_distributing_by_cloning(model):
  """Decide whether this model is going to be distributed via cloning.

  We are going to distribute the model by cloning in graph mode.

  Args:
    model: Keras model to distribute.

  Returns:
    True if the `model` is going to be distributed using cloning and False
    otherwise.
  """
  if (is_tpu_strategy(model._distribution_strategy) and
      context.executing_eagerly):  # b/137580852
    return False
  elif ops.executing_eagerly_outside_functions():
    return bool(model._compile_distribution)
  return True


def _custom_compile_for_predict(model):
  """Custom compile for TPU predict mode."""
  if not model.built:
    # Model is not compilable because it does not know its number of inputs
    # and outputs, nor their shapes and names. We will compile after the first
    # time the model gets called on training data.
    return
  model._is_compiled = True
  model.total_loss = None
  model.train_function = None
  model.test_function = None
  model.predict_function = None


def _build_network_on_replica(model, mode, inputs=None, targets=None):
  """Build an updated model on replicas.

  We create a new Keras model while sharing the variables from the old graph.
  Building a new sub-graph is required since the original keras model creates
  placeholders for the input and the output that are not accessible till we
  call iterator.get_next() inside the step_fn for `fit`/`evaluate`/`predict`.

  The sharing of weights and layers between the old and the new model gaurantee
  that we're using Strategy variables and any updates on either model are
  reflected correctly in callbacks and loop iterations.

  We need to make sure we share the optimizers between the old and the new model
  as well so that optimizer state is not lost if the user is running fit
  multiple times.

  Args:
    model: Model to be replicated across Replicas
    mode: Which of fit/eval/predict is building the distributed network
    inputs: Input variables to be passed to the model
    targets: Target tensor to be passed to model.compile

  Returns:
    A new model with shared layers with the old model.
  """
  # Need to do imports here since we run into a circular dependency error.
  from tensorflow.python.keras import models  # pylint: disable=g-import-not-at-top
  from tensorflow.python.keras.engine import sequential  # pylint: disable=g-import-not-at-top

  # We rely on the internal methods to avoid having share_weights weights in the
  # public API.
  if isinstance(model, sequential.Sequential):
    updated_model = models._clone_sequential_model(
        model, input_tensors=inputs, layer_fn=models.share_weights)
  else:
    updated_model = models._clone_functional_model(
        model, input_tensors=inputs, layer_fn=models.share_weights)
    # Callable losses added directly to a functional Model need to be added
    # here.
    updated_model._callable_losses = model._callable_losses

  # Recast all low precision outputs back to float32 since we only casted
  # the inputs to bfloat16 and not targets. This is done so that we can preserve
  # precision when calculating the loss value.
  def _upcast_low_precision_outputs(output):
    if output.dtype == dtypes.bfloat16:
      return math_ops.cast(output, dtypes.float32)
    else:
      return output
  updated_model.outputs = [_upcast_low_precision_outputs(o)
                           for o in updated_model.outputs]

  if isinstance(targets, tuple):
    targets = nest.flatten(targets)

  if mode == ModeKeys.PREDICT and inputs is not None:  # TPU predict case
    _custom_compile_for_predict(updated_model)
  else:
    updated_model.compile(
        model.optimizer,
        model.loss,
        metrics=metrics_module.clone_metrics(model._compile_metrics),
        loss_weights=model.loss_weights,
        sample_weight_mode=model.sample_weight_mode,
        weighted_metrics=metrics_module.clone_metrics(
            model._compile_weighted_metrics),
        target_tensors=targets)
  return updated_model


def _build_distributed_network(model, strategy, mode, inputs=None,
                               targets=None):
  """Create a cloned model on each replica."""
  with K.get_graph().as_default(), strategy.scope():
    distributed_model = strategy.extended.call_for_each_replica(
        _build_network_on_replica,
        args=(model, mode, inputs, targets))
    set_distributed_model(model, mode, distributed_model)


def _clone_and_build_model(model, mode, inputs=None, targets=None):
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

  # Recast all low precision outputs back to float32 since we only casted
  # the inputs to bfloat16 and not targets. This is done so that we can preserve
  # precision when calculating the loss value.
  def _upcast_low_precision_outputs(output):
    if output.dtype == dtypes.bfloat16:
      return math_ops.cast(output, dtypes.float32)
    else:
      return output
  cloned_model.outputs = [_upcast_low_precision_outputs(o)
                          for o in cloned_model.outputs]

  if isinstance(targets, tuple):
    targets = nest.flatten(targets)
  if mode == ModeKeys.PREDICT and inputs is not None:  # TPU predict case
    _custom_compile_for_predict(cloned_model)
  else:
    cloned_model.compile(
        optimizer,
        model.loss,
        metrics=metrics_module.clone_metrics(model._compile_metrics),
        loss_weights=model.loss_weights,
        sample_weight_mode=model.sample_weight_mode,
        weighted_metrics=metrics_module.clone_metrics(
            model._compile_weighted_metrics),
        target_tensors=targets)
  return cloned_model


def clone_model_on_replicas(model, strategy, mode, inputs=None, targets=None):
  """Create a cloned model on each replica."""
  with K.get_graph().as_default(), strategy.scope():
    distributed_model = strategy.extended.call_for_each_replica(
        _clone_and_build_model, args=(model, mode, inputs, targets))
    set_distributed_model(model, mode, distributed_model)
  if mode == ModeKeys.TRAIN:
    model._make_callback_model(distributed_model)


def _make_execution_function(model, mode):
  """Makes or reuses function to run one step of distributed model execution."""
  if is_distributing_by_cloning(model):
    return _make_execution_function_with_cloning(model, mode)

  distributed_function = get_distributed_function(model, mode)
  if distributed_function:
    return distributed_function

  distribution_function = _make_execution_function_without_cloning(model, mode)
  set_distributed_function(model, mode, distribution_function)
  return distribution_function


def _make_execution_function_without_cloning(model, mode):
  """Creates a function to run one step of distributed model execution."""
  strategy = model._distribution_strategy

  with strategy.scope():
    per_replica_function = _make_replica_execution_function(model, mode)

    def distributed_function(input_fn):
      """A single step of the distributed execution across replicas."""
      x, y, sample_weights = input_fn()
      # Call `Model.{train,test,predict}_on_batch` on every replica passing
      # PerReplicas as arguments.  On every replica inside this call, each
      # PerReplica object will return the value for that replica.  The outputs
      # are PerReplicas too.
      outputs = strategy.experimental_run_v2(
          per_replica_function, args=(x, y, sample_weights))
      # Out of PerReplica outputs reduce or pick values to return.
      all_outputs = unwrap_outputs(
          strategy, outputs, with_loss_tensor=(mode != ModeKeys.PREDICT))
      return all_outputs

    if not model.run_eagerly:
      distributed_function = def_function.function(distributed_function)
      def execution_function(input_fn):
        # `numpy` translates Tensors to values in Eager mode.
        return [out.numpy() for out in distributed_function(input_fn)]
    else:
      execution_function = distributed_function

    return execution_function


def _make_replica_execution_function(model, mode):
  """A single step of the distributed execution on a replica."""
  if mode == ModeKeys.TRAIN:
    func = model.train_on_batch
  elif mode == ModeKeys.TEST:
    func = model.test_on_batch
  else:

    def predict_on_batch(x, y=None, sample_weights=None):
      del y, sample_weights
      return model.predict_on_batch(x)

    func = predict_on_batch

  if mode != ModeKeys.PREDICT:
    # `reset_metrics` is set to False to maintain stateful metrics across
    # batch-level calls.
    func = functools.partial(func, reset_metrics=False)

  return func


def _make_replicated_models_with_cloning(model, mode):
  """Build models on each replica."""
  strategy = model._distribution_strategy

  # If distributed_model is not built, create one for `mode`.
  if model._compile_distribution:
    clone_model_on_replicas(model, strategy, mode)
  else:
    _build_distributed_network(model, strategy, mode)


def _make_execution_function_with_cloning(model, mode):
  """Clones or re-uses models to run one step of distributed model execution."""
  distributed_model = get_distributed_model(model, mode)
  # TODO(b/134069401): Create a cache for the distributed model and exec
  # function that incorporates additional attributes to be part of the cache key
  # than just the mode.
  # If distributed model for a particular `mode` is already built, use the
  # `_distribution_function` on that distributed model.
  # If you have updated the sample_weight_mode on the model, then you will need
  # to recompile metrics and recreate the execution function. This is indicated
  # by the `_recompile_exec_function` property.
  if (distributed_model and hasattr(distributed_model, '_distribution_function')
      and not (hasattr(distributed_model, '_recompile_exec_function') and
               distributed_model._recompile_exec_function)):
    return distributed_model._distributed_function

  if not distributed_model:
    _make_replicated_models_with_cloning(model, mode)
    distributed_model = get_distributed_model(model, mode)
  assert distributed_model

  # Also create an execution fuction on that distributed model.
  if context.executing_eagerly():
    distributed_function = _make_eager_execution_function(model, mode)
  else:
    distributed_function = _make_graph_execution_function(model, mode)

  # We cache the distributed execution function on the model since creating
  # distributed models and execution functions are expensive.
  distributed_model._distributed_function = distributed_function
  distributed_model._recompile_exec_function = False
  return distributed_function


def _make_graph_execution_function(model, mode):
  """Makes function to run one step of distributed model in graph mode."""

  def _per_replica_function(model):
    f = model._make_execution_function(mode)
    return (f.inputs, f.outputs, f.updates_op, f.session_kwargs)

  strategy = model._distribution_strategy
  with strategy.scope():
    # Create train ops on each of the devices when we call
    # `_per_replica_fit_function`.
    (grouped_inputs, grouped_outputs, grouped_updates,
     grouped_session_args) = strategy.extended.call_for_each_replica(
         _per_replica_function, args=(get_distributed_model(model, mode),))

    # Initialize the variables in the replicated model. This is necessary for
    # multi-worker training because on some workers, initialization is not
    # needed. This method does initialization or waiting for initialization
    # according to the context object of distribute coordinator.
    init_restore_or_wait_for_variables()

    # Unwrap all the per device values returned from `call_for_each_replica`.
    # Unwrapping per device values gives you a list of values that can be
    # used to construct a new train function that is composed of update ops on
    # all the devices over which the model is distributed.
    (all_inputs, all_outputs, all_updates, all_session_args) = unwrap_values(
        strategy,
        grouped_inputs,
        grouped_outputs,
        grouped_updates,
        grouped_session_args,
        with_loss_tensor=(mode != ModeKeys.PREDICT))

    return K.function(
        all_inputs,
        all_outputs,
        updates=all_updates,
        name='distributed_{}_function'.format(mode),
        **all_session_args)


def _make_eager_execution_function(model, mode):
  """Makes function to run one step of distributed model eager execution."""
  def _per_replica_function(model):
    f = model._make_execution_function(mode)
    return (f.inputs, f.outputs)

  # NOTE(priyag): Try creating a new FuncGraph within DS scope instead of using
  # the global one.
  strategy = model._distribution_strategy
  global_graph = K.get_graph()

  with global_graph.as_default(), strategy.scope():
    # First we gather the relevant portions of the model across all replicas.
    # `K._scratch_graph(global_graph)` signals to Keras that it should not
    # lift to a separate graph when creating the per-replica functions.
    with K._scratch_graph(global_graph):
      # Create train ops on each of the devices when we call
      # `_per_replica_fit_function`.
      grouped = strategy.extended.call_for_each_replica(
          _per_replica_function, args=(get_distributed_model(model, mode),))
      grouped_inputs, grouped_outputs = grouped

      # Unwrap all the per device values returned from `call_for_each_replica`.
      # Unwrapping per device values gives you a list of values that can be
      # used to construct a new train function that is composed of
      # inputs/outputs on all the devices over which the model is distributed.
      (all_inputs, all_outputs, _, _) = unwrap_values(
          strategy,
          grouped_inputs,
          grouped_outputs,
          with_loss_tensor=(mode != ModeKeys.PREDICT))

    # Finally, a joint Keras function is created; this one will be created in
    # a separate FuncGraph.
    return K.function(
        all_inputs,
        all_outputs,
        name='eager_distributed_{}_function'.format(mode))


def _copy_weights_to_distributed_model(original_model, mode):
  """Copies weights from original model to distributed models."""
  strategy = original_model._distribution_strategy
  distributed_model = get_distributed_model(original_model, mode)
  if strategy:
    # Copy the weights from the original model to each of the replicated
    # models.
    orig_model_weights = original_model.get_weights()
    first_model = strategy.unwrap(distributed_model)[0]
    set_weights(strategy, first_model, orig_model_weights)


def _copy_weights_to_original_model(model, mode):
  """Copies weights from first distributed model back to original model."""
  if model._distribution_strategy and mode == ModeKeys.TRAIN:
    distributed_model = get_distributed_model(model, mode)
    updated_weights = model._distribution_strategy.unwrap(
        distributed_model)[0].get_weights()
    model.set_weights(updated_weights)


def _per_replica_aggregate_batch(strategy, batch_outs, model, mode):
  """Aggregates the per-replica batch-level outputs from a distributed step."""
  if strategy is not None and mode == ModeKeys.PREDICT:
    total_batch_outs = []
    for i in range(len(model.outputs)):
      num_replicas = strategy.num_replicas_in_sync
      nested_outs = batch_outs[i * num_replicas:i * num_replicas + num_replicas]
      total_batch_outs.append(
          concat_along_batch_dimension(nest.flatten(nested_outs)))
    return total_batch_outs
  return batch_outs


def _reset_metrics(model):
  if model._distribution_strategy:
    for mode in [ModeKeys.TRAIN, ModeKeys.TEST, ModeKeys.PREDICT]:
      distributed_model = get_distributed_model(model, mode)
      if distributed_model:
        first_model = model._distribution_strategy.unwrap(distributed_model)[0]
        first_model.reset_metrics()


def get_distributed_model(model, mode):
  key = _generate_cache_key(mode)
  return model._distributed_model_cache.get(key, None)


def set_distributed_model(model, mode, distributed_model):
  key = _generate_cache_key(mode)
  model._distributed_model_cache[key] = distributed_model


def get_distributed_function(model, mode):
  key = _generate_cache_key(mode)
  return model._distributed_function_cache.get(key, None)


def set_distributed_function(model, mode, distributed_function):
  key = _generate_cache_key(mode)
  model._distributed_function_cache[key] = distributed_function


def _generate_cache_key(mode):
  key = hash(mode)
  return key


@tf_contextlib.contextmanager
def distributed_scope(strategy, learning_phase):
  with strategy.scope(), K.learning_phase_scope(learning_phase):
    yield


def call_replica_local_fn(fn, *args, **kwargs):
  """Call a function that uses replica-local variables.

  This function correctly handles calling `fn` in a cross-replica
  context.

  Arguments:
    fn: The function to call.
    *args: Positional arguments to the `fn`.
    **kwargs: Keyword argument to `fn`.

  Returns:
    The result of calling `fn`.
  """
  # TODO(b/132666209): Remove this function when we support assign_*
  # for replica-local variables.
  strategy = None
  if 'strategy' in kwargs:
    strategy = kwargs.pop('strategy')
  else:
    if ds_context.has_strategy():
      strategy = ds_context.get_strategy()

  # TODO(b/120571621): TPUStrategy does not implement replica-local variables.
  is_tpu = is_tpu_strategy(strategy)
  if ((not is_tpu) and strategy and ds_context.in_cross_replica_context()):
    with strategy.scope():
      return strategy.extended.call_for_each_replica(fn, args, kwargs)
  return fn(*args, **kwargs)


def is_current_worker_chief():
  return dc_context.get_current_worker_context().is_chief


def filter_distributed_callbacks(callbacks_list, model):
  """Filter Callbacks based on the worker context when running multi-worker.

  Arguments:
    callbacks_list: A list of `Callback` instances.
    model: Keras model instance.

  Returns:
    The list of `Callback` instances that should be run on this worker.
  """

  if not model._in_multi_worker_mode():
    raise ValueError(
        'filter_distributed_callbacks() should only be called when Keras '
        'is in multi worker mode.')

  callbacks_list = callbacks_list or []
  if not [
      c for c in callbacks_list if isinstance(c, callbacks.ModelCheckpoint)
  ]:
    # TODO(rchao): Consider providing a ModelCheckpoint here if the user
    # fails to (possibly with tempfile directory).
    logging.warning('ModelCheckpoint callback is not provided. '
                    'Workers will need to restart training if any fails.')

  if callbacks_list is None or is_current_worker_chief():
    return callbacks_list

  # Some Callbacks should only run on the chief worker.
  return [
      callback for callback in callbacks_list if not callback._chief_worker_only
  ]  # pylint: disable=protected-access


def _update_sample_weight_modes(model, mode, sample_weights):
  """Update sample_weight_mode of the distributed model."""
  if is_distributing_by_cloning(model):
    distributed_model = get_distributed_model(model, mode)
    if not distributed_model:
      _make_replicated_models_with_cloning(model, mode)
      distributed_model = get_distributed_model(model, mode)
    distributed_model._recompile_exec_function = any(
        [e.sample_weights_mismatch() for e in model._training_endpoints])

    if sample_weights:
      distributed_models = flatten_per_replica_values(
          model._distribution_strategy, distributed_model)
      # sample_weights is a tuple of 1 list where the number of elements in the
      # list is equal to the number of replicas in sync.
      sample_weights = sample_weights[0]
      if sample_weights and None not in sample_weights:
        for m, sw in zip(distributed_models, sample_weights):
          m._update_sample_weight_modes(sample_weights=[sw])


def concat_along_batch_dimension(outputs):
  """Concats prediction outputs along the batch dimension."""
  if isinstance(outputs[0], sparse_tensor.SparseTensor):
    return sparse_ops.sparse_concat_v2(axis=0, sp_inputs=outputs)
  if isinstance(outputs[0], ragged_tensor.RaggedTensor):
    return ragged_concat_ops.concat(outputs, axis=0)
  return np.concatenate(outputs)
