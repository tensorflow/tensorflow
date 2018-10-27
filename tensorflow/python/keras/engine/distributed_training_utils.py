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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session as session_module
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.util import nest


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
      assign_ops.append(distribution_strategy.unwrap(sw.assign(w)))

    weights = weights[num_param:]
  K.get_session().run(assign_ops)


def unwrap_values(distribution_strategy, grouped_inputs, grouped_outputs,
                  grouped_updates, grouped_session_args,
                  with_loss_tensor=False):
  """Unwrap and return the list of values contained in the PerDevice parameters.

  This function calls `flatten_perdevice_values` to parse each of the input
  parameters into a list of values on the different devices. If we set
  `with_loss_tensor` to be True, we also call `reduce` on the list of losses on
  the different devices to give us one loss tensor.

  Args:
    distribution_strategy: DistributionStrategy used to distribute training and
        validation.
    grouped_inputs: PerDevice inputs returned from the train or test function
        that we ran on each device.
    grouped_outputs: PerDevice outputs returned from the train or test function
        that we ran on each device.
    grouped_updates: PerDevice updates returned from the train or test function
        that we ran on each device.
    grouped_session_args: PerDevice session args returned from the train or
        test function that we ran on each device.
    with_loss_tensor: Boolean that indicates if we need to add the reduced loss
        tensor as one of the outputs.

  Returns:
    Values of each of the PerDevice parameters.

  """
  # Unwrap per device values returned from each model's train function.
  # This will be used to construct the main train function.
  all_inputs = flatten_perdevice_values(distribution_strategy,
                                        grouped_inputs)
  if with_loss_tensor:
    # reduce loss tensor before adding it to the list of fetches
    loss = distribution_strategy.unwrap(
        distribution_strategy.reduce(distribute_lib.get_loss_reduction(),
                                     grouped_outputs[0],
                                     destinations='/device:CPU:0'))[0]

    all_outputs = flatten_perdevice_values(distribution_strategy,
                                           grouped_outputs[1:])
    all_outputs = [loss] + all_outputs
  else:
    all_outputs = flatten_perdevice_values(distribution_strategy,
                                           grouped_outputs)

  all_updates = flatten_perdevice_values(distribution_strategy,
                                         grouped_updates)

  all_session_args = {}
  grouped_feed_dict = grouped_session_args.get('feed_dict')
  if grouped_feed_dict:
    all_session_args['feed_dict'] = flatten_perdevice_values(
        distribution_strategy, grouped_feed_dict)

  grouped_fetches = grouped_session_args.get('fetches')
  if grouped_fetches:
    all_session_args['fetches'] = flatten_perdevice_values(
        distribution_strategy, grouped_fetches)

  return all_inputs, all_outputs, all_updates, all_session_args


def flatten_perdevice_values(distribution_strategy, perdevice_values):
  """Unwraps and flattens a nest of PerDevice parameters.

  PerDevice values have one value associated with each device. Each entry in
  the PerDevice dict has a device `key` and the corresponding value on the
  device as the `value`. In this function we take a PerDevice value or a list of
  PerDevice values and return all the values in the PerDevice dict.

  Args:
    distribution_strategy: DistributionStrategy used to distribute training and
        validation.
    perdevice_values: List of PerDevice object or a single PerDevice object.

  Returns:
    List of values of all the PerDevice objects.

  """
  # This function takes a PerDevice object or a list of PerDevice objects and
  # returns all the values associated with it.
  return [e for flattened in nest.flatten(perdevice_values)
          for e in distribution_strategy.unwrap(flattened)]


def validate_callbacks(input_callbacks):
  """Validate whether given callbacks are supported by DistributionStrategy.

  Args:
    input_callbacks: List of callbacks passed by the user to fit.

  Raises:
    ValueError: If `LearningRateScheduler` or `ReduceLROnPlateau` is one of the
        callbacks passed.
    ValueError: If `histogram_freq` or `write_grads` is one of the parameters
        passed as part of the TensorBoard callback.
  """
  if input_callbacks:
    for callback in input_callbacks:
      if callback not in [callbacks.TensorBoard, callbacks.ReduceLROnPlateau,
                          callbacks.LearningRateScheduler, callbacks.CSVLogger,
                          callbacks.EarlyStopping, callbacks.ModelCheckpoint,
                          callbacks.TerminateOnNaN, callbacks.ProgbarLogger,
                          callbacks.History, callbacks.RemoteMonitor]:
        logging.warning('Your input callback is not one of the predefined '
                        'Callbacks that supports DistributionStrategy. You '
                        'might encounter an error if you access one of the '
                        'model\'s attributes as part of the callback since '
                        'these attributes are not set. You can access each of '
                        'the individual distributed models using the '
                        '`_grouped_model` attribute of your original model.')
      if isinstance(callback, callbacks.LearningRateScheduler):
        raise ValueError('LearningRateScheduler callback is not supported with '
                         'DistributionStrategy.')
      if isinstance(callback, callbacks.ReduceLROnPlateau):
        raise ValueError('ReduceLROnPlateau callback is not supported with '
                         'DistributionStrategy.')

      # If users want to use the TensorBoard callback they cannot use certain
      # features of the callback that involve accessing model attributes and
      # running ops.
      if isinstance(callback, callbacks.TensorBoard):
        if callback.__getattribute__('histogram_freq'):
          raise ValueError('histogram_freq in the TensorBoard callback is not '
                           'supported when using DistributionStrategy.')
        if callback.__getattribute__('write_grads'):
          raise ValueError('write_grads in the TensorBoard callback is not '
                           'supported when using DistributionStrategy.')


def validate_distributed_dataset_inputs(distribution_strategy, x, y,
                                        sample_weights=None):
  """Validate all the components of a DistributedValue Dataset input.

  Args:
    distribution_strategy: The current DistributionStrategy used to call
        `fit`/`evaluate`.
    x: Input Dataset DistributedValue object. For example, when we use
        `MirroredStrategy` this is a PerDevice object with a tensor for each
        device set in the dict. x can also be a tuple or dict. The keys of the
        dict should match the names of the input layers of the model.
    y: Target Dataset DistributedValue object. For example, when we use
        `MirroredStrategy` this is a PerDevice object with a tensor for each
        device set in the dict. y can also be a tuple or dict. The keys of the
        dict should match the names of the output layers of the model.
    sample_weights: Sample weights Dataset DistributedValue object. For example,
        when we use `MirroredStrategy` this is a PerDevice object with a tensor
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
  x_values_list = validate_per_device_inputs(distribution_strategy, x)

  if y is not None:
    y_values_list = validate_per_device_inputs(distribution_strategy, y)
  else:
    y_values_list = None

  if sample_weights is not None:
    sample_weights_list = validate_per_device_inputs(distribution_strategy,
                                                     sample_weights)
  else:
    sample_weights_list = None

  # Return the unwrapped values to avoid calling `unwrap` a second time.
  return x_values_list, y_values_list, sample_weights_list


def validate_per_device_inputs(distribution_strategy, x):
  """Validates PerDevice dataset input list.

  Args:
    distribution_strategy: The current DistributionStrategy used to call
      `fit`, `evaluate` and `predict`.
    x: A list of PerDevice objects that represent the input or
      target values.

  Returns:
    List containing the first element of each of the PerDevice objects in
    the input list.

  Raises:
    ValueError: If any of the objects in the `per_device_list` is not a tensor.

  """
  # Convert the inputs and targets into a list of PerDevice objects.
  per_device_list = nest.flatten(x)
  x_values_list = []
  for x in per_device_list:
    if not tensor_util.is_tensor(x):
      raise ValueError('Dataset input to the model should be tensors instead '
                       'they are of type {}'.format(type(x)))

    # At this point both x and y contain tensors in the `DistributedValues`
    # structure.
    x_values = distribution_strategy.unwrap(x)

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
  x_shape = x_values[0].get_shape().as_list()
  for i in range(1, len(x_values)):
    if x_shape != x_values[i].get_shape().as_list():
      raise ValueError('Input tensor shapes do not match for distributed tensor'
                       ' inputs {}'.format(x))


def configure_and_create_session(distribution_strategy):
  """Configure session config and create a session with it."""
  # TODO(priyag): Throw error if a session already exists.
  session_config = K.get_default_session_config()
  distribution_strategy.configure(session_config)

  if distribution_strategy.__class__.__name__ == 'TPUStrategy':
    # TODO(priyag): Remove this workaround when Distributed Coordinator is
    # integrated with keras and we can create a session from there.
    master = distribution_strategy._tpu_cluster_resolver.master()  # pylint: disable=protected-access
    session = session_module.Session(config=session_config, target=master)
  else:
    session = session_module.Session(config=session_config)

  K.set_session(session)


def validate_inputs(x, y, distribution_strategy):
  """Validate inputs when using DistributionStrategy.

  Args:
    x: Model Inputs.
    y: Model Targets.
    distribution_strategy: The DistributionStrategy with which the model is
      compiled.

  Raises:
    ValueError: if input is not a Dataset or a numpy array(when we use
      MirroredStrategy).
  """
  if isinstance(x, dict) or isinstance(y, dict):
    raise ValueError('`DistributionStrategy` does not support inputs of type '
                     'dict. You must pass a `tf.data.Dataset` object or a '
                     'numpy array as input.')

  if (isinstance(x, iterator_ops.Iterator) or
      isinstance(y, iterator_ops.Iterator)):
    raise ValueError('`DistributionStrategy` does not support inputs of type '
                     'Iterator. You must pass a `tf.data.Dataset` object or a '
                     'numpy array as input.')

  if distribution_strategy.__class__.__name__ == 'TPUStrategy':
    for i in [x, y]:
      if isinstance(i, dataset_ops.Dataset):
        shapes = nest.flatten(i.output_shapes)
        if any([not s.is_fully_defined() for s in shapes]):
          raise ValueError(
              'Using TPUs currently requires fully defined shapes. Either use '
              'set_shape() on the input tensors or use '
              'dataset.batch(..., drop_remainder=True).'
              'Found unknown shape {} in input {}.'.format(s, i))


def get_input_batch_params(first_x_value, batch_size, distribution_strategy):
  """Calculate the number of batches and steps/steps_per_epoch.

  Args:
    first_x_value: This is the first input numpy array that is passed in as the
      model input.
    batch_size: The specified batch_size or the default batch_size of 32.
    distribution_strategy: The current DistributionStrategy used to compile the
      model.

  Returns:
    The steps or steps_per_epoch argument depending on if a user is
    calling `fit`, `evaluate` or `predict`.

  Raises:
    ValueError: If the number of batches or steps evaluates to 0.

  """
  num_batches = first_x_value.shape[0] // batch_size
  if not num_batches:
    raise ValueError('Please specify a batch_size that is smaller than'
                     'the number of input samples %d.' % first_x_value.shape[0])
  # TODO(anjalisridhar): TPU currently supports using the num_replicas property.
  # We might want to look into implementing worker_devices. In multi worker
  # strategy, perhaps num_replicas works better?
  steps = num_batches // distribution_strategy.num_replicas
  if not steps:
    # TODO(anjalisridhar): Number of replicas in the error message may not
    # convey what we want to the user. Is there another terminology that we can
    # use that is consistent across different strategies?
    raise ValueError('The number of batches %d is smaller than the number '
                     'of replicas %d used for DistributionStrategy. ' %
                     (num_batches, distribution_strategy.num_replicas))
  return steps


def get_batch_dimension(iterator):
  shapes = nest.flatten(iterator.output_shapes)
  # Take the batch size from the first element, as it should be the same for
  # all.
  dims = shapes[0].dims
  return dims[0] if dims else None


def get_batch_size(num_replicas, num_samples, steps):
  """Calculate and return batch size for numpy inputs.

  Args:
    num_replicas: Number of devices over which the model input is distributed.
    num_samples: Total number of input samples in the input numpy arrays.
    steps: Number of steps that we run the model for.

  Returns:
    batch size used to create the Dataset object from the input numpy arrays.

  """
  if num_samples % steps != 0:
    logging.warning('The number of input samples %d is not evenly '
                    'divisible by the number of steps %d. '
                    'Some samples will not be processed as expected.' %
                    (num_samples, steps))
  global_batch_size = num_samples // steps
  if global_batch_size % num_replicas != 0:
    logging.warning('The total number of batches per step %d is not evenly '
                    'divisible by the number of replicas %d used in '
                    'DistributionStrategy. Some samples will not be processed '
                    'as expected.' %
                    (global_batch_size, num_replicas))
  return global_batch_size // num_replicas


def get_cpu_device(distribution_strategy):
  """Returns the CPU device of the TPU host or the default CPU device string.

  Args:
    distribution_strategy: The DistributionStrategy used to compile the model.

  Returns:
    A device string which is the TPU host's CPU device in case of
    TPUDistributionStrategy or the default CPU device string in all other
    cases.

  Raises:
    NotImplementedError: We currently don't support copying numpy data to
    multiple hosts in the case of Cloud TPU pods.
  """
  if distribution_strategy.__class__.__name__ == 'TPUStrategy':
    if distribution_strategy.num_hosts > 1:
      raise NotImplementedError('TPUDistributionStrategy does not '
                                'support numpy inputs when running on Cloud'
                                'TPU pods.')
    return distribution_strategy.get_host_cpu_device(0)
  else:
    # For all strategies except TPUDistributionStrategy
    # TODO(anjalisridhar): We may need to modify this when we add support for
    # multi-worker strategy.
    return '/CPU:0'


def get_var_for_numpy(distribution_strategy, x):
  if isinstance(x, list):
    var_x = tuple([_get_var_for_numpy(distribution_strategy, single_input)
                   for single_input in x])
  else:
    var_x = _get_var_for_numpy(distribution_strategy, x)
  return var_x


def _get_var_for_numpy(distribution_strategy, input_array):
  """Creates a variable and assigns the value of the numpy array to it.

  Args:
    distribution_strategy: The DistributionStrategy used to compile the model.
    input_array: The input numpy array whose value will be assigned to the
      variable we create.

  Returns:
    The variable to which we will copy the value of the input numpy array.

  """
  with ops.device(get_cpu_device(distribution_strategy)):
    # Create and initialize a variable on the CPU device. This is the CPU
    # device of the host in the case of TPUDistributionStrategy.
    input_var = variables.VariableV1(array_ops.zeros(input_array.shape,
                                                     input_array.dtype),
                                     trainable=False, use_resource=True)
  K.get_session().run(input_var.initializer)

  # Create a placeholder for the numpy array input slices. We copy the value
  # of the input numpy array to the variable in slices of size 64 MB to avoid
  # running into memory issues or RPC message limits.
  start_placeholder = array_ops.placeholder(dtypes.int64, ())
  end_placeholder = array_ops.placeholder(dtypes.int64, ())
  slice_placeholder = array_ops.placeholder(input_var.dtype)
  assign_slice_op = input_var[start_placeholder:end_placeholder].assign(
      slice_placeholder)

  # If each batch element is > 64 MB, then we copy each batch element
  # individually. Otherwise, the slices will be < 128 MB. There might be padding
  # which might mean that the slices are 128 MB even if the size of the
  # tensor allocated is less than 128 MB.
  # This formula gives slices with size:
  # ceil(64 MB / byte size per batch element) bytes.
  # Using ceil() guarantees we get a number >= 1.

  # Calculate the size of each batch element.
  byte_size_per_batch_element = np.prod(input_array.shape[1:]) * \
                                input_var.dtype.size

  # Calculate number of elements we want to copy per slice.
  batch_size_per_slice = np.ceil((64 << 20) / byte_size_per_batch_element)

  # Copy slices of the above size starting at 0, except the last slice will be
  # smaller.
  start = 0
  limit = input_array.shape[0]
  while start < limit:
    end = min(start + batch_size_per_slice, limit)
    K.get_session().run(assign_slice_op, feed_dict={
        start_placeholder: start,
        end_placeholder: end,
        slice_placeholder: input_array[start:end]})
    start = end

  return input_var
