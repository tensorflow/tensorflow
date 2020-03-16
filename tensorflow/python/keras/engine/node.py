# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Contains the `Node` class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.util import nest


class Node(object):
  """A `Node` describes the connectivity between two layers.

  Each time a layer is connected to some new input,
  a node is added to `layer._inbound_nodes`.
  Each time the output of a layer is used by another layer,
  a node is added to `layer._outbound_nodes`.

  Arguments:
      outbound_layer: the layer that takes
          `input_tensors` and turns them into `output_tensors`
          (the node gets created when the `call`
          method of the layer was called).
      inbound_layers: a list of layers, the same length as `input_tensors`,
          the layers from where `input_tensors` originate.
      node_indices: a list of integers, the same length as `inbound_layers`.
          `node_indices[i]` is the origin node of `input_tensors[i]`
          (necessary since each inbound layer might have several nodes,
          e.g. if the layer is being shared with a different data stream).
      tensor_indices: a list of integers,
          the same length as `inbound_layers`.
          `tensor_indices[i]` is the index of `input_tensors[i]` within the
          output of the inbound layer
          (necessary since each inbound layer might
          have multiple tensor outputs, with each one being
          independently manipulable).
      input_tensors: list of input tensors.
      output_tensors: list of output tensors.
      arguments: dictionary of keyword arguments that were passed to the
          `call` method of the layer at the call that created the node.

  `node_indices` and `tensor_indices` are basically fine-grained coordinates
  describing the origin of the `input_tensors`.

  A node from layer A to layer B is added to:
    - A._outbound_nodes
    - B._inbound_nodes
  """

  def __init__(self,
               outbound_layer,
               inbound_layers,
               node_indices,
               tensor_indices,
               input_tensors,
               output_tensors,
               arguments=None):
    # Layer instance (NOT a sequence)
    if isinstance(outbound_layer, (list, tuple, dict)):
      raise ValueError('`outbound_layer` should be a layer instance, '
                       'not a list, tuple, or, dict.')

    # These arguments are user-provided. Copy them here so that future
    # user modifications do not affect the node's metadata.
    input_tensors = nest.map_structure(lambda t: t, input_tensors)
    output_tensors = nest.map_structure(lambda t: t, output_tensors)
    arguments = nest.map_structure(lambda t: t, arguments)

    # this is the layer that takes a nested structure of input tensors
    # and turns them into a nested structure of output tensors.
    # the current node will be added to
    # the inbound_nodes of outbound_layer.
    self.outbound_layer = outbound_layer

    # The following 3 properties describe where
    # the input tensors come from: which layers,
    # and for each layer, which node and which
    # tensor output of each node.

    # Nested structure of layer instances.
    self.inbound_layers = inbound_layers
    # Nested structure of integers, 1:1 mapping with inbound_layers.
    self.node_indices = node_indices
    # Nested of integers, 1:1 mapping with inbound_layers.
    self.tensor_indices = tensor_indices

    # Following 2 properties:
    # tensor inputs and outputs of outbound_layer.

    # Nested structure of tensors. 1:1 mapping with inbound_layers.
    self.input_tensors = input_tensors
    # Nested structure of tensors, created by outbound_layer.call().
    self.output_tensors = output_tensors

    # Following 2 properties: input and output shapes.

    # Nested structure of shape tuples, shapes of input_tensors.
    self.input_shapes = nest.map_structure(backend.int_shape, input_tensors)
    # Nested structure of shape tuples, shapes of output_tensors.
    self.output_shapes = nest.map_structure(backend.int_shape, output_tensors)

    # Optional keyword arguments to layer's `call`.
    self.arguments = arguments

    # Create Keras History for any Keras Tensors in `arguments`.
    tensor_arguments = [
        t for t in nest.flatten(self.arguments) if isinstance(t, ops.Tensor)
    ]
    for tensor_argument in tensor_arguments:
      if base_layer_utils.needs_keras_history(
          tensor_argument, ignore_call_context=True):
        base_layer_utils.create_keras_history(tensor_argument)

    # Add nodes to all layers involved.
    for layer in nest.flatten(inbound_layers):
      if layer is not None:
        # For compatibility with external Keras, we use the deprecated
        # accessor here.
        layer.outbound_nodes.append(self)
    # For compatibility with external Keras, we use the deprecated
    # accessor here.
    outbound_layer.inbound_nodes.append(self)

  def iterate_inbound(self, include_arguments=False):
    """Returns a list of tuples representing the inbound data.

    Arguments:
      include_arguments: Whether to also iterate over any Keras Tensors
        passed as args, kwargs.

    Returns:
      List of tuples like: (inbound_layer, node_index, tensor_index, tensor).
    """
    inputs_inbound = list(
        zip(
            nest.flatten(self.inbound_layers),
            nest.flatten(self.node_indices),
            nest.flatten(self.tensor_indices),
            nest.flatten(self.input_tensors)))

    if include_arguments:
      keras_tensor_arguments = [
          kt for kt in nest.flatten(self.arguments)
          if hasattr(kt, '_keras_history')
      ]

      def _get_inbound(keras_tensor):
        kh = keras_tensor._keras_history
        return kh.layer, kh.node_index, kh.tensor_index, keras_tensor

      arguments_inbound = nest.map_structure(_get_inbound,
                                             keras_tensor_arguments)

      return inputs_inbound + arguments_inbound
    else:
      return inputs_inbound

  def _get_all_node_dependencies(self):
    """Returns all of the nodes this node immediately depends on."""
    node_deps = []
    for layer, node_index, _, _ in self.iterate_inbound():
      node_deps.append(layer._inbound_nodes[node_index])

    for arg in nest.flatten(self.arguments):
      if isinstance(arg, ops.Tensor) and hasattr(arg, '_keras_history'):
        kh = arg._keras_history
        node_deps.append(kh.layer._inbound_nodes[kh.node_index])

    return node_deps

  def get_config(self):
    inbound_names = nest.map_structure(
        lambda layer: layer.name if layer else None, self.inbound_layers)
    return {
        'outbound_layer': self.outbound_layer.name,
        'inbound_layers': inbound_names,
        'node_indices': self.node_indices,
        'tensor_indices': self.tensor_indices
    }
