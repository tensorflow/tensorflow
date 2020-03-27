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
# pylint: disable=protected-access
"""Utilities related to layer/model functionality.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.conv_utils import convert_kernel
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.utils.get_source_inputs')
def get_source_inputs(tensor, layer=None, node_index=None):
  """Returns the list of input tensors necessary to compute `tensor`.

  Output will always be a list of tensors
  (potentially with 1 element).

  Arguments:
      tensor: The tensor to start from.
      layer: Origin layer of the tensor. Will be
          determined via tensor._keras_history if not provided.
      node_index: Origin node index of the tensor.

  Returns:
      List of input tensors.
  """
  if not hasattr(tensor, '_keras_history'):
    return tensor

  if layer is None or node_index:
    layer, node_index, _ = tensor._keras_history
  if not layer._inbound_nodes:
    return [tensor]
  else:
    node = layer._inbound_nodes[node_index]
    if not node.inbound_layers:
      # Reached an Input layer, stop recursion.
      return nest.flatten(node.input_tensors)
    else:
      source_tensors = []
      for layer, node_index, _, tensor in node.iterate_inbound():
        previous_sources = get_source_inputs(tensor, layer, node_index)
        # Avoid input redundancy.
        for x in previous_sources:
          if all(x is not t for t in source_tensors):
            source_tensors.append(x)
      return source_tensors


def validate_string_arg(input_data,
                        allowable_strings,
                        layer_name,
                        arg_name,
                        allow_none=False,
                        allow_callables=False):
  """Validates the correctness of a string-based arg."""
  if allow_none and input_data is None:
    return
  elif allow_callables and callable(input_data):
    return
  elif isinstance(input_data,
                  six.string_types) and input_data in allowable_strings:
    return
  else:
    allowed_args = '`None`, ' if allow_none else ''
    allowed_args += 'a `Callable`, ' if allow_callables else ''
    allowed_args += 'or one of the following values: %s' % (allowable_strings,)
    raise ValueError(("%s's %s arg received an invalid value %s. " +
                      'Allowed values are %s.') %
                     (layer_name, arg_name, input_data, allowed_args))


def count_params(weights):
  """Count the total number of scalars composing the weights.

  Arguments:
      weights: An iterable containing the weights on which to compute params

  Returns:
      The total number of scalars composing the weights
  """
  unique_weights = object_identity.ObjectIdentitySet(weights)
  weight_shapes = [w.shape.as_list() for w in unique_weights]
  standardized_weight_shapes = [
      [0 if w_i is None else w_i for w_i in w] for w in weight_shapes
  ]
  return int(sum(np.prod(p) for p in standardized_weight_shapes))


def print_summary(model, line_length=None, positions=None, print_fn=None):
  """Prints a summary of a model.

  Arguments:
      model: Keras model instance.
      line_length: Total length of printed lines
          (e.g. set this to adapt the display to different
          terminal window sizes).
      positions: Relative or absolute positions of log elements in each line.
          If not provided, defaults to `[.33, .55, .67, 1.]`.
      print_fn: Print function to use.
          It will be called on each line of the summary.
          You can set it to a custom function
          in order to capture the string summary.
          It defaults to `print` (prints to stdout).
  """
  if print_fn is None:
    print_fn = print

  if model.__class__.__name__ == 'Sequential':
    sequential_like = True
  elif not model._is_graph_network:
    # We treat subclassed models as a simple sequence of layers, for logging
    # purposes.
    sequential_like = True
  else:
    sequential_like = True
    nodes_by_depth = model._nodes_by_depth.values()
    nodes = []
    for v in nodes_by_depth:
      if (len(v) > 1) or (len(v) == 1 and
                          len(nest.flatten(v[0].inbound_layers)) > 1):
        # if the model has multiple nodes
        # or if the nodes have multiple inbound_layers
        # the model is no longer sequential
        sequential_like = False
        break
      nodes += v
    if sequential_like:
      # search for shared layers
      for layer in model.layers:
        flag = False
        for node in layer._inbound_nodes:
          if node in nodes:
            if flag:
              sequential_like = False
              break
            else:
              flag = True
        if not sequential_like:
          break

  if sequential_like:
    line_length = line_length or 65
    positions = positions or [.45, .85, 1.]
    if positions[-1] <= 1:
      positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ['Layer (type)', 'Output Shape', 'Param #']
  else:
    line_length = line_length or 98
    positions = positions or [.33, .55, .67, 1.]
    if positions[-1] <= 1:
      positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Connected to']
    relevant_nodes = []
    for v in model._nodes_by_depth.values():
      relevant_nodes += v

  def print_row(fields, positions):
    line = ''
    for i in range(len(fields)):
      if i > 0:
        line = line[:-1] + ' '
      line += str(fields[i])
      line = line[:positions[i]]
      line += ' ' * (positions[i] - len(line))
    print_fn(line)

  print_fn('Model: "{}"'.format(model.name))
  print_fn('_' * line_length)
  print_row(to_display, positions)
  print_fn('=' * line_length)

  def print_layer_summary(layer):
    """Prints a summary for a single layer.

    Arguments:
        layer: target layer.
    """
    try:
      output_shape = layer.output_shape
    except AttributeError:
      output_shape = 'multiple'
    except RuntimeError:  # output_shape unknown in Eager mode.
      output_shape = '?'
    name = layer.name
    cls_name = layer.__class__.__name__
    fields = [name + ' (' + cls_name + ')', output_shape, layer.count_params()]
    print_row(fields, positions)

  def print_layer_summary_with_connections(layer):
    """Prints a summary for a single layer (including topological connections).

    Arguments:
        layer: target layer.
    """
    try:
      output_shape = layer.output_shape
    except AttributeError:
      output_shape = 'multiple'
    connections = []
    for node in layer._inbound_nodes:
      if relevant_nodes and node not in relevant_nodes:
        # node is not part of the current network
        continue

      for inbound_layer, node_index, tensor_index, _ in node.iterate_inbound():
        connections.append('{}[{}][{}]'.format(inbound_layer.name, node_index,
                                               tensor_index))

    name = layer.name
    cls_name = layer.__class__.__name__
    if not connections:
      first_connection = ''
    else:
      first_connection = connections[0]
    fields = [
        name + ' (' + cls_name + ')', output_shape,
        layer.count_params(), first_connection
    ]
    print_row(fields, positions)
    if len(connections) > 1:
      for i in range(1, len(connections)):
        fields = ['', '', '', connections[i]]
        print_row(fields, positions)

  layers = model.layers
  for i in range(len(layers)):
    if sequential_like:
      print_layer_summary(layers[i])
    else:
      print_layer_summary_with_connections(layers[i])
    if i == len(layers) - 1:
      print_fn('=' * line_length)
    else:
      print_fn('_' * line_length)

  if hasattr(model, '_collected_trainable_weights'):
    trainable_count = count_params(model._collected_trainable_weights)
  else:
    trainable_count = count_params(model.trainable_weights)

  non_trainable_count = count_params(model.non_trainable_weights)

  print_fn('Total params: {:,}'.format(trainable_count + non_trainable_count))
  print_fn('Trainable params: {:,}'.format(trainable_count))
  print_fn('Non-trainable params: {:,}'.format(non_trainable_count))
  print_fn('_' * line_length)


def gather_trainable_weights(trainable, sub_layers, extra_variables):
  """Lists the trainable weights for an object with sub-layers.

  Args:
    trainable: Whether the object collecting the variables is trainable.
    sub_layers: A flat list of Layer objects owned by this object, to collect
      variables from.
    extra_variables: Any extra variables to include. Their `.trainable` property
      is used to categorize them.

  Returns:
    A list of collected trainable weights/variables.
  """
  if not trainable:
    return []
  weights = []
  for layer in sub_layers:
    weights += layer.trainable_weights
  trainable_extra_variables = [
      v for v in extra_variables if v.trainable]
  return weights + trainable_extra_variables


def gather_non_trainable_weights(trainable, sub_layers, extra_variables):
  """Lists the non-trainable weights for an object with sub-layers.

  Args:
    trainable: Whether the object collecting the variables is trainable.
    sub_layers: A flat list of Layer objects owned by this object, to collect
      variables from.
    extra_variables: Any extra variables to include. Their `.trainable` property
      is used to categorize them.

  Returns:
    A list of collected non-trainable weights/variables.
  """
  trainable_extra_variables = []
  non_trainable_extra_variables = []
  for v in extra_variables:
    if v.trainable:
      trainable_extra_variables.append(v)
    else:
      non_trainable_extra_variables.append(v)
  weights = []
  for layer in sub_layers:
    weights += layer.non_trainable_weights
  if not trainable:
    trainable_weights = []
    for layer in sub_layers:
      trainable_weights += layer.trainable_weights
    return (trainable_weights + trainable_extra_variables
            + weights + non_trainable_extra_variables)
  return weights + non_trainable_extra_variables


@deprecation.deprecated('2020-06-23',
                        'The Theano kernel format is legacy; '
                        'this utility will be removed.')
@keras_export('keras.utils.convert_all_kernels_in_model')
def convert_all_kernels_in_model(model):
  """Converts all convolution kernels in a model from Theano to TensorFlow.

  Also works from TensorFlow to Theano.

  This is used for converting legacy Theano-saved model files.

  Arguments:
      model: target model for the conversion.
  """
  # Note: SeparableConvolution not included
  # since only supported by TF.
  conv_classes = {
      'Conv1D',
      'Conv2D',
      'Conv3D',
      'Conv2DTranspose',
  }
  to_assign = []
  for layer in model.layers:
    if layer.__class__.__name__ in conv_classes:
      original_kernel = K.get_value(layer.kernel)
      converted_kernel = convert_kernel(original_kernel)
      to_assign.append((layer.kernel, converted_kernel))
  K.batch_set_value(to_assign)


def convert_dense_weights_data_format(dense,
                                      previous_feature_map_shape,
                                      target_data_format='channels_first'):
  """Utility useful when changing a convnet's `data_format`.

  When porting the weights of a convnet from one data format to the other,
  if the convnet includes a `Flatten` layer
  (applied to the last convolutional feature map)
  followed by a `Dense` layer, the weights of that `Dense` layer
  should be updated to reflect the new dimension ordering.

  Arguments:
      dense: The target `Dense` layer.
      previous_feature_map_shape: A shape tuple of 3 integers,
          e.g. `(512, 7, 7)`. The shape of the convolutional
          feature map right before the `Flatten` layer that
          came before the target `Dense` layer.
      target_data_format: One of "channels_last", "channels_first".
          Set it "channels_last"
          if converting a "channels_first" model to "channels_last",
          or reciprocally.
  """
  assert target_data_format in {'channels_last', 'channels_first'}
  kernel, bias = dense.get_weights()
  for i in range(kernel.shape[1]):
    if target_data_format == 'channels_first':
      c, h, w = previous_feature_map_shape
      original_fm_shape = (h, w, c)
      ki = kernel[:, i].reshape(original_fm_shape)
      ki = np.transpose(ki, (2, 0, 1))  # last -> first
    else:
      h, w, c = previous_feature_map_shape
      original_fm_shape = (c, h, w)
      ki = kernel[:, i].reshape(original_fm_shape)
      ki = np.transpose(ki, (1, 2, 0))  # first -> last
    kernel[:, i] = np.reshape(ki, (np.prod(previous_feature_map_shape),))
  dense.set_weights([kernel, bias])


def is_builtin_layer(layer):
  if not getattr(layer, '_keras_api_names', None):
    return False

  # Subclasses of `Layer` that are not exported inherit the export name
  # of the base layer class.
  return (layer._keras_api_names != ('keras.layers.Layer',) and
          layer._keras_api_names_v1 != ('keras.layers.Layer',))
