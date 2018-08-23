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

from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks
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
  backend.get_session().run(assign_ops)


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


def validate_distributed_dataset_inputs(distribution_strategy, x, y):
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

  y_values_list = validate_per_device_inputs(distribution_strategy, y)

  # Return the unwrapped values to avoid calling `unwrap` a second time.
  return x_values_list, y_values_list


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
