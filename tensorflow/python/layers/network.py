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
# =============================================================================
"""Contains Network, a composition of layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.eager import context
from tensorflow.python.estimator import util as estimator_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils as layers_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


class InputLayer(base.Layer):
  """Layer to be used as an entry point into a Network (a graph of layers).

  It can either wrap an existing tensor (pass an `input_tensor` argument)
  or create its a placeholder tensor (pass arguments `input_shape`
  as well as `dtype`).

  It is generally recommend to use the functional layer API via `Input`,
  (which creates an `InputLayer`) without directly using `InputLayer`.

  Arguments:
      input_shape: Shape tuple (not including the batch axis), or `TensorShape`
        instance (not including the batch axis).
      batch_size: Optional input batch size (integer or None).
      dtype: Datatype of the input.
      input_tensor: Optional tensor to use as layer input
          instead of creating a placeholder.
      sparse: Boolean, whether the placeholder created
          is meant to be sparse.
      name: Name of the layer (string).

    Raises:
      RuntimeError: If created in Eager mode.
  """

  def __init__(self,
               input_shape=None,
               batch_size=None,
               dtype=dtypes.float32,
               input_tensor=None,
               sparse=False,
               name=None):
    super(InputLayer, self).__init__(dtype=dtype, name=name)
    self.built = True
    self.sparse = sparse
    self.batch_size = batch_size

    if isinstance(input_shape, tensor_shape.TensorShape):
      input_shape = tuple(input_shape.as_list())

    if input_tensor is None:
      if input_shape is not None:
        batch_input_shape = (batch_size,) + tuple(input_shape)
      else:
        batch_input_shape = None

      if context.in_eager_mode():
        # In eager mode, create a temporary placeholder to call the layer on.
        input_tensor = base._DeferredTensor(  # pylint: disable=protected-access
            shape=batch_input_shape,
            dtype=dtype,
            name=self.name)
      else:
        # In graph mode, create a graph placeholder to call the layer on.
        if sparse:
          input_tensor = array_ops.sparse_placeholder(
              shape=batch_input_shape,
              dtype=dtype,
              name=self.name)
        else:
          input_tensor = array_ops.placeholder(
              shape=batch_input_shape,
              dtype=dtype,
              name=self.name)

      # For compatibility with Keras API.
      self.is_placeholder = True
      self._batch_input_shape = batch_input_shape
    else:
      # For compatibility with Keras API.
      self.is_placeholder = False
      self._batch_input_shape = tuple(input_tensor.get_shape().as_list())

    # Create an input node to add to self.outbound_node
    # and set output_tensors' _keras_history.
    input_tensor._keras_history = (self, 0, 0)  # pylint: disable=protected-access
    base.Node(
        self,
        inbound_layers=[],
        node_indices=[],
        tensor_indices=[],
        input_tensors=[input_tensor],
        output_tensors=[input_tensor])


def Input(  # pylint: disable=invalid-name
    shape=None,
    batch_size=None,
    name=None,
    dtype=dtypes.float32,
    sparse=False,
    tensor=None):
  """`Input()` is used to instantiate an input tensor for use with a `Network`.

  For instance, if a, b and c are tensors created via `Input`,
  it becomes possible to do:

  `network = Network(inputs=[a, b], outputs=c)`

  Example:

      ```python
      # This is a logistic regression
      x = tf.layers.Input(shape=(32,))
      y = tf.layers.Dense(16, activation='softmax')(x)
      network = tf.layers.Network(x, y)
      ```

  Arguments:
      shape: A shape tuple (integer), not including the batch size.
          For instance, `shape=(32,)` indicates that the expected input
          will be batches of 32-dimensional vectors.
      batch_size: Optional input batch size (integer or None).
      name: An optional name string for the layer.
          Should be unique in a model (do not reuse the same name twice).
          It will be autogenerated if it isn't provided.
      dtype: The data type expected by the input, as a string
          (`float32`, `float64`, `int32`...)
      sparse: A boolean specifying whether the placeholder
          to be created is sparse.
      tensor: Optional existing tensor to wrap into the `Input` layer.
          If set, the layer will not create a placeholder tensor.

  Returns:
      A tensor: either a new placeholder (with history metadata) or
      `tensor` (if passed), with added history metadata.

  Raises:
    RuntimeError: If called in Eager mode.
  """
  input_layer = InputLayer(
      input_shape=shape,
      batch_size=batch_size,
      name=name,
      dtype=dtype,
      sparse=sparse,
      input_tensor=tensor)
  # Return tensor including `_keras_history` metadata.
  # Note that in this case train_output and test_output are the same pointer.
  outputs = input_layer._inbound_nodes[0].output_tensors  # pylint: disable=protected-access
  if len(outputs) == 1:
    return outputs[0]
  else:
    return outputs


class GraphNetwork(base.Layer):
  """A GraphNetwork is a directed acyclic graph of layers.

  It is the topological form of a "model".
  A Model is simply a GraphNetwork with added training/evaluation routines.

  A GraphNetwork instance implements the full Layer API. In particular, a
  GraphNetwork can be called on new inputs.

  Example:

      ```python
      # This is a logistic regression
      x = tf.layers.Input(shape=(32,))
      y = tf.layers.Dense(16, activation='softmax')(x)
      network = tf.layers.GraphNetwork(x, y)

      # It is then possible to call the network on compatible inputs:
      z = tf.layers.Input(shape=(32,))
      w = network(z)

      # It is possible to retrieve the same properties as a layer:
      weights = network.trainable_weights
      ```

  Arguments:
      inputs: Input tensor or list of input tensors.
        Must come from `tf.layers.Input`.
      output: Output tensor or list of output tensors. Must come from
        tf.layers Layers or Keras layers.
      name: Optional name of the model (string).

  Attributes:
    GraphNetwork has the same attributes as Layer. On top of it, it also has:
      - layers: a list of the children layers of the network,
        a list of layer instances, ordered from "earlier in the graph"
        to "later in the graph".

  Methods:
    GraphNetwork has the same methods as Layer. On top of it, it also has:
      - get_layer: retrieves a child layer by name or index in the graph.

  Raises:
    RuntimeError: If created in Eager mode.
  """

  def __init__(self, inputs, outputs, name=None):  # pylint: disable=super-init-not-called
    if context.in_eager_mode():
      # TODO(fchollet): check that all inputs and outputs are DeferredTensors.
      pass

    self._init_set_name(name)
    self._activity_regularizer = None
    with vs.variable_scope(
        None, default_name=self._base_name) as captured_scope:
      self._scope = captured_scope
    call_fn_args = estimator_util.fn_args(self.call)
    self._compute_previous_mask = ('mask' in call_fn_args or
                                   hasattr(self, 'compute_mask'))
    self._call_has_scope_arg = 'scope' in call_fn_args

    # This acts just like the `trainable` attribute of any layer instance.
    # It does not affect users of the underlying layers, only users of the
    # GraphNetwork instance.
    self.trainable = True
    # A GraphNetwork does not create weights of its own, thus it is already
    # built.
    self.built = True
    # A GraphNetwork does not create weights of its own, thus has no dtype.
    self._dtype = None
    # The following are implemented as property functions:
    # self.trainable_weights
    # self.non_trainable_weights
    # self.input_spec

    # Private attributes to implement compatibility with Layer.
    self._per_input_losses = {}
    self._per_input_updates = {}
    self._updates = []
    self._losses = []
    self._scope = None
    self._reuse = None
    self._graph = ops.get_default_graph()

    # GraphNetwork-specific properties.
    if isinstance(inputs, (list, tuple)):
      self.inputs = list(inputs)  # Tensor or list of tensors.
    else:
      self.inputs = [inputs]
    if isinstance(outputs, (list, tuple)):
      self.outputs = list(outputs)
    else:
      self.outputs = [outputs]
    # All layers in order of horizontal graph traversal.
    # Entries are unique. Includes input and output layers.
    self.layers = []

    # Check for redundancy in inputs.
    if len(set(self.inputs)) != len(self.inputs):
      raise ValueError('The list of inputs passed to the model '
                       'is redundant. '
                       'All inputs should only appear once.'
                       ' Found: ' + str(self.inputs))

    # # List of initial layers (1 to 1 mapping with self.inputs,
    # # hence the same layer might appear twice)
    # self._input_layers = []
    # self._input_layers_node_indices = []
    # self._input_layers_tensor_indices = []
    # # list of layers (1 to 1 mapping with self.inputs,
    # # hence the same layer might appear twice)
    # self._output_layers = []
    # self._output_layers_node_indices = []
    # self._output_layers_tensor_indices = []

    self._input_layers = []
    self._output_layers = []
    self._input_coordinates = []
    self._output_coordinates = []

    # This is for performance optimization when calling the GraphNetwork on new
    # inputs. Every time the GraphNetwork is called on a set on input tensors,
    # we compute the output tensors, output masks and output shapes in one pass,
    # then cache them here. When any of these outputs is queried later, we
    # retrieve it from there instead of recomputing it.
    self._output_mask_cache = {}
    self._output_tensor_cache = {}
    self._output_shape_cache = {}

    # User-provided arguments validation.
    for x in self.inputs:
      # Check that x has appropriate `_keras_history` metadata.
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Input tensors to a ' + cls_name + ' ' +
                         'must come from `tf.layers.Input`. '
                         'Received: ' + str(x) +
                         ' (missing previous layer metadata).')
      # Check that x is an input tensor.
      # pylint: disable=protected-access
      layer, node_index, tensor_index = x._keras_history
      if len(layer._inbound_nodes) > 1 or (
          layer._inbound_nodes and layer._inbound_nodes[0].inbound_layers):
        cls_name = self.__class__.__name__
        logging.warning(cls_name + ' inputs must come from '
                        '`tf.layers.Input` (thus holding past layer metadata), '
                        'they cannot be the output of '
                        'a previous non-Input layer. '
                        'Here, a tensor specified as '
                        'input to "' + self.name + '" was not an Input tensor, '
                        'it was generated by layer ' + layer.name + '.\n'
                        'Note that input tensors are '
                        'instantiated via `tensor = tf.layers.Input(shape)`.\n'
                        'The tensor that caused the issue was: ' + str(x.name))
      # pylint: enable=protected-access
    for x in self.outputs:
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Output tensors to a ' + cls_name + ' must be '
                         'the output of a TensorFlow `Layer` '
                         '(thus holding past layer metadata). Found: ' + str(x))

    # Build self._output_layers:
    for x in self.outputs:
      layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      self._output_layers.append(layer)
      self._output_coordinates.append((layer, node_index, tensor_index))

    # Build self._input_layers:
    for x in self.inputs:
      layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      # It's supposed to be an input layer, so only one node
      # and one tensor output.
      assert node_index == 0
      assert tensor_index == 0
      self._input_layers.append(layer)
      self._input_coordinates.append((layer, node_index, tensor_index))

    # Network_nodes: set of nodes included in the graph
    # (not all nodes included in the layers
    # are relevant to the current graph).
    network_nodes = set()  # ids of all nodes relevant to the GraphNetwork
    nodes_depths = {}  # dict {node: depth value}
    layers_depths = {}  # dict {layer: depth value}
    layer_indices = {}  # dict {layer: index in traversal}
    nodes_in_decreasing_depth = []

    def build_map_of_graph(tensor,
                           finished_nodes,
                           nodes_in_progress,
                           layer,
                           node_index,
                           tensor_index):
      """Builds a map of the graph of layers.

      This recursively updates the map `layer_indices`,
      the list `nodes_in_decreasing_depth` and the set `network_nodes`.

      Arguments:
          tensor: Some tensor in a graph.
          finished_nodes: Set of nodes whose subgraphs have been traversed
              completely. Useful to prevent duplicated work.
          nodes_in_progress: Set of nodes that are currently active on the
              recursion stack. Useful to detect cycles.
          layer: Layer from which `tensor` comes from. If not provided,
              will be obtained from `tensor._keras_history`.
          node_index: Node index from which `tensor` comes from.
          tensor_index: Tensor_index from which `tensor` comes from.

      Raises:
          ValueError: if a cycle is detected.
      """
      node = layer._inbound_nodes[node_index]  # pylint: disable=protected-access

      # Prevent cycles.
      if node in nodes_in_progress:
        raise ValueError('The tensor ' + str(tensor) + ' at layer "' +
                         layer.name + '" is part of a cycle.')

      # Don't repeat work for shared subgraphs
      if node in finished_nodes:
        return

      node_key = _make_node_key(layer.name, node_index)
      # Update network_nodes.
      network_nodes.add(node_key)

      # Store the traversal order for layer sorting.
      if layer not in layer_indices:
        layer_indices[layer] = len(layer_indices)

      nodes_in_progress.add(node)

      # Propagate to all previous tensors connected to this node.
      for i in range(len(node.inbound_layers)):
        x = node.input_tensors[i]
        layer = node.inbound_layers[i]
        node_index = node.node_indices[i]
        tensor_index = node.tensor_indices[i]
        build_map_of_graph(x, finished_nodes, nodes_in_progress, layer,
                           node_index, tensor_index)

      finished_nodes.add(node)
      nodes_in_progress.remove(node)
      nodes_in_decreasing_depth.append(node)

    finished_nodes = set()
    nodes_in_progress = set()
    for x in self.outputs:
      layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      build_map_of_graph(x, finished_nodes, nodes_in_progress,
                         layer=layer,
                         node_index=node_index,
                         tensor_index=tensor_index)

    for node in reversed(nodes_in_decreasing_depth):
      # If the depth is not set, the node has no outbound nodes (depth 0).
      depth = nodes_depths.setdefault(node, 0)

      # Update the depth of the corresponding layer
      previous_depth = layers_depths.get(node.outbound_layer, 0)
      # If we've seen this layer before at a higher depth,
      # we should use that depth instead of the node depth.
      # This is necessary for shared layers that have inputs at different
      # depth levels in the graph.
      depth = max(depth, previous_depth)
      layers_depths[node.outbound_layer] = depth
      nodes_depths[node] = depth

      # Update the depth of inbound nodes.
      # The "depth" of a node is the max of the depths
      # of all layers it is connected to.
      for i in range(len(node.inbound_layers)):
        inbound_layer = node.inbound_layers[i]
        node_index = node.node_indices[i]
        inbound_node = inbound_layer._inbound_nodes[node_index]  # pylint: disable=protected-access
        previous_depth = nodes_depths.get(inbound_node, 0)
        nodes_depths[inbound_node] = max(depth + 1, previous_depth)

    # Build a dict {depth: list of nodes with this depth}
    nodes_by_depth = {}
    for node, depth in nodes_depths.items():
      if depth not in nodes_by_depth:
        nodes_by_depth[depth] = []
      nodes_by_depth[depth].append(node)

    # Build a dict {depth: list of layers with this depth}
    layers_by_depth = {}
    for layer, depth in layers_depths.items():
      if depth not in layers_by_depth:
        layers_by_depth[depth] = []
      layers_by_depth[depth].append(layer)

    # Get sorted list of layer depths.
    depth_keys = list(layers_by_depth.keys())
    depth_keys.sort(reverse=True)

    # Set self.layers and self._layers_by_depth.
    layers = []
    for depth in depth_keys:
      layers_for_depth = layers_by_depth[depth]
      # GraphNetwork.layers needs to have a deterministic order:
      # here we order them by traversal order.
      layers_for_depth.sort(key=lambda x: layer_indices[x])
      layers.extend(layers_for_depth)
    self.layers = layers
    self._layers_by_depth = layers_by_depth

    # Get sorted list of node depths.
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    # Check that all tensors required are computable.
    # computable_tensors: all tensors in the graph
    # that can be computed from the inputs provided.
    computable_tensors = []
    for x in self.inputs:
      computable_tensors.append(x)

    layers_with_complete_input = []  # To provide a better error msg.
    for depth in depth_keys:
      for node in nodes_by_depth[depth]:
        layer = node.outbound_layer
        if layer:
          for x in node.input_tensors:
            if x not in computable_tensors:
              raise ValueError('Graph disconnected: '
                               'cannot obtain value for tensor ' + str(x) +
                               ' at layer "' + layer.name + '". '
                               'The following previous layers '
                               'were accessed without issue: ' +
                               str(layers_with_complete_input))
          for x in node.output_tensors:
            computable_tensors.append(x)
          layers_with_complete_input.append(layer.name)

    # Keep track of the network's nodes.
    self._network_nodes = network_nodes
    self._nodes_by_depth = nodes_by_depth

    # Ensure name unicity, which will be crucial for serialization
    # (since serialized nodes refer to layers by their name).
    all_names = [layer.name for layer in self.layers]
    for name in all_names:
      if all_names.count(name) != 1:
        raise ValueError('The name "' + name + '" is used ' +
                         str(all_names.count(name)) + ' times in the model. '
                         'All layer names should be unique.')

    # Layer parameters.
    # The new network starts with a single inbound node
    # for its inputs, and no outbound nodes.
    self._outbound_nodes = []  # Will be appended to by future calls to __call__
    self._inbound_nodes = [
    ]  # Will be appended to below, and by future calls to __call__
    # Create the node linking internal inputs to internal outputs.
    base.Node(
        outbound_layer=self,
        inbound_layers=[],
        node_indices=[],
        tensor_indices=[],
        input_tensors=self.inputs,
        output_tensors=self.outputs)

  def get_layer(self, name=None, index=None):
    """Retrieves a layer based on either its name (unique) or index.

    Indices are based on order of horizontal graph traversal (bottom-up).

    Arguments:
        name: String, name of layer.
        index: Integer, index of layer.

    Returns:
        A layer instance.

    Raises:
        ValueError: In case of invalid layer name or index.
    """
    # TODO(fchollet): We could build a dictionary based on layer names
    # since they are constant, but we have not done that yet.
    if index is not None:
      if len(self.layers) <= index:
        raise ValueError('Was asked to retrieve layer at index ' + str(index) +
                         ' but model only has ' + str(len(self.layers)) +
                         ' layers.')
      else:
        return self.layers[index]
    else:
      if not name:
        raise ValueError('Provide either a layer name or layer index.')
    for layer in self.layers:
      if layer.name == name:
        return layer
    raise ValueError('No such layer: ' + name)

  @property
  def updates(self):
    """Retrieve the network's updates.

    Will only include updates that are either
    unconditional, or conditional on inputs to this model
    (e.g. will not include updates that depend on tensors
    that aren't inputs to this model).

    Returns:
        A list of update ops.
    """
    updates = []
    for layer in self.layers:
      if hasattr(layer, 'updates'):
        # Collect updates that are dependent on inputs
        # that are part of the model.
        for node_index, node in enumerate(layer._inbound_nodes):  # pylint: disable=protected-access
          node_key = _make_node_key(layer.name, node_index)
          if node_key in self._network_nodes:
            # The model owns this layer node.
            inputs = node.input_tensors
            updates += layer.get_updates_for(inputs)
        # Collect unconditional updates.
        updates += layer.get_updates_for(None)
    return updates

  @property
  def losses(self):
    """Retrieve the network's losses.

    Will only include losses that are either
    unconditional, or conditional on inputs to this model
    (e.g. will not include losses that depend on tensors
    that aren't inputs to this model).

    Returns:
        A list of loss tensors.
    """
    losses = []
    # Retrieve losses for all internal layers.
    for layer in self.layers:
      if hasattr(layer, 'losses'):
        # Collect losses that are dependent on inputs
        # that are part of the model.
        for node_index, node in enumerate(layer._inbound_nodes):  # pylint: disable=protected-access
          node_key = _make_node_key(layer.name, node_index)
          if node_key in self._network_nodes:
            # The model owns this layer node.
            inputs = node.input_tensors
            losses += layer.get_losses_for(inputs)
        # Collect unconditional losses.
        losses += layer.get_losses_for(None)
    # Add any potential unconditional model-level loss.
    losses += self.get_losses_for(None)
    return losses

  @property
  def trainable_weights(self):
    if not self.trainable:
      return []
    weights = []
    for layer in self.layers:
      weights += layer.trainable_weights
    return weights

  @property
  def non_trainable_weights(self):
    weights = []
    for layer in self.layers:
      weights += layer.non_trainable_weights
    if not self.trainable:
      trainable_weights = []
      for layer in self.layers:
        trainable_weights += layer.trainable_weights
      return trainable_weights + weights
    return weights

  @property
  def input_spec(self):
    """Gets the network's input specs.

    Returns:
        A list of `InputSpec` instances (one per input to the model)
            or a single instance if the model has only one input.
    """
    specs = []
    for layer in self._input_layers:
      if layer.input_spec is None:
        specs.append(None)
      else:
        if not isinstance(layer.input_spec, list):
          raise TypeError('Layer ' + layer.name +
                          ' has an input_spec attribute that '
                          'is not a list. We expect a list. '
                          'Found input_spec = ' + str(layer.input_spec))
        specs += layer.input_spec
    if len(specs) == 1:
      return specs[0]
    return specs

  def call(self, inputs, mask=None):
    """Call the model on new inputs.

    In this case `call` just reapplies
    all ops in the graph to the new inputs
    (e.g. build a new computational graph from the provided inputs).

    Arguments:
        inputs: A tensor or list of tensors.
        mask: A mask or list of masks. A mask can be
            either a tensor or None (no mask).

    Returns:
        A tensor if there is a single output, or
        a list of tensors if there are more than one outputs.
    """
    inputs = nest.flatten(inputs)
    if mask is None:
      masks = [None for _ in range(len(inputs))]
    else:
      masks = nest.flatten(mask)

    if context.in_graph_mode():
      # Try to retrieve cached outputs if the layer has already been called
      # on these exact inputs.
      cache_key = (layers_util.object_list_uid(inputs)
                   + '_' + layers_util.object_list_uid(masks))
      if cache_key in self._output_tensor_cache:
        # Cache hit.
        return self._output_tensor_cache[cache_key]
    # Actually apply the network graph to the new inputs.
    outputs, _ = self._run_internal_graph(inputs, masks)
    return outputs

  def _compute_output_shape(self, input_shape):
    if isinstance(input_shape, list):
      input_shapes = []
      for shape in input_shape:
        if shape is not None:
          input_shapes.append(tuple(tensor_shape.TensorShape(shape).as_list()))
        else:
          input_shapes.append(None)
    else:
      if input_shape is not None:
        input_shapes = [tuple(tensor_shape.TensorShape(input_shape).as_list())]
      else:
        input_shapes = [None]

    if len(input_shapes) != len(self._input_layers):
      raise ValueError('Invalid input_shape argument ' + str(input_shape) +
                       ': model has ' + str(len(self._input_layers)) +
                       ' tensor inputs.')

    cache_key = layers_util.object_list_uid(input_shapes)
    if cache_key not in self._output_shape_cache:
      # Cache miss. We have to run the network graph manually (recursive calls
      # to `_compute_output_shape`).
      layers_to_output_shapes = {}
      for i in range(len(input_shapes)):
        layer = self._input_layers[i]
        input_shape = input_shapes[i]
        # It's an input layer: then `_compute_output_shape` is identity,
        # and there is only one node and one tensor output.
        shape_key = layer.name + '_0_0'
        layers_to_output_shapes[shape_key] = input_shape

      depth_keys = list(self._nodes_by_depth.keys())
      depth_keys.sort(reverse=True)
      # Iterate over nodes, by depth level.
      if len(depth_keys) > 1:
        for depth in depth_keys:
          nodes = self._nodes_by_depth[depth]
          for node in nodes:
            # This is always a single layer, never a list.
            layer = node.outbound_layer
            if layer in self._input_layers:
              # We've already covered the input layers
              # a few lines above.
              continue
            # Potentially redundant list,
            # same size as node.input_tensors.
            input_shapes = []
            for j in range(len(node.inbound_layers)):
              inbound_layer = node.inbound_layers[j]
              node_index = node.node_indices[j]
              tensor_index = node.tensor_indices[j]
              shape_key = inbound_layer.name + '_%s_%s' % (node_index,
                                                           tensor_index)
              input_shape = layers_to_output_shapes[shape_key]
              input_shapes.append(input_shape)

            if len(input_shapes) == 1:
              output_shape = layer._compute_output_shape(input_shapes[0])  # pylint: disable=protected-access
            else:
              output_shape = layer._compute_output_shape(input_shapes)  # pylint: disable=protected-access
            if isinstance(output_shape, list):
              output_shapes = [
                  tuple(tensor_shape.TensorShape(shape).as_list())
                  for shape in output_shape
              ]
            else:
              output_shapes = [
                  tuple(tensor_shape.TensorShape(output_shape).as_list())
              ]

            node_index = layer._inbound_nodes.index(node)  # pylint: disable=protected-access
            for j in range(len(output_shapes)):
              shape_key = layer.name + '_%s_%s' % (node_index, j)
              layers_to_output_shapes[shape_key] = output_shapes[j]

        # Read final output shapes from layers_to_output_shapes.
        output_shapes = []
        for i in range(len(self._output_layers)):
          layer, node_index, tensor_index = self._output_coordinates[i]
          shape_key = layer.name + '_%s_%s' % (node_index, tensor_index)
          output_shapes.append(layers_to_output_shapes[shape_key])

        # Store in cache.
        self._output_shape_cache[cache_key] = output_shapes
      else:
        # Cache hit.
        output_shapes = self._output_shape_cache[cache_key]

      if isinstance(output_shapes, list):
        if len(output_shapes) == 1:
          return tensor_shape.TensorShape(output_shapes[0])
        else:
          return [tensor_shape.TensorShape(shape) for shape in output_shapes]
      else:
        return tensor_shape.TensorShape(output_shapes)

  def _run_internal_graph(self, inputs, masks=None):
    """Computes output tensors for new inputs.

    # Note:
        - Expects `inputs` to be a list (potentially with 1 element).
        - Can be run on non-Keras tensors.

    Arguments:
        inputs: List of tensors
        masks: List of masks (tensors or None).

    Returns:
        Three lists: output_tensors, output_masks, output_shapes
    """
    # Note: masking support is relevant mainly for Keras.
    # It cannot be factored out without having the fully reimplement the network
    # calling logic on the Keras side. We choose to incorporate it in
    # GraphNetwork because 1) it may be useful to fully support in tf.layers in
    # the future and 2) Keras is a major user of GraphNetwork.  If you don't
    # use masking, it does not interfere with regular behavior at all and you
    # can ignore it.
    if masks is None:
      masks = [None for _ in range(len(inputs))]

    # Dictionary mapping reference tensors to tuples
    # (computed tensor, compute mask)
    # we assume a 1:1 mapping from tensor to mask
    # TODO(fchollet): raise exception when a `.compute_mask()` call
    # does not return a list the same size as `call`
    tensor_map = {}
    for x, y, mask in zip(self.inputs, inputs, masks):
      tensor_map[str(id(x))] = (y, mask)

    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    for depth in depth_keys:
      nodes = self._nodes_by_depth[depth]
      for node in nodes:
        # This is always a single layer, never a list.
        layer = node.outbound_layer

        reference_input_tensors = node.input_tensors
        reference_output_tensors = node.output_tensors

        # If all previous input tensors are available in tensor_map,
        # then call node.inbound_layer on them.
        computed_data = []  # List of tuples (input, mask).
        for x in reference_input_tensors:
          if str(id(x)) in tensor_map:
            computed_data.append(tensor_map[str(id(x))])

        if len(computed_data) == len(reference_input_tensors):
          # Call layer (reapplying ops to new inputs).
          with ops.name_scope(layer.name):
            if node.arguments:
              kwargs = node.arguments
            else:
              kwargs = {}
            if len(computed_data) == 1:
              computed_tensor, computed_mask = computed_data[0]
              # Ensure mask propagation if applicable.
              if 'mask' in estimator_util.fn_args(layer.call):
                if 'mask' not in kwargs:
                  kwargs['mask'] = computed_mask

              output_tensors = nest.flatten(
                  layer.call(computed_tensor, **kwargs))
              if hasattr(layer, 'compute_mask'):
                output_masks = nest.flatten(
                    layer.compute_mask(computed_tensor, computed_mask))
              else:
                output_masks = [None for _ in range(len(output_tensors))]
              computed_tensors = [computed_tensor]
              computed_masks = [computed_mask]
            else:
              computed_tensors = [x[0] for x in computed_data]
              computed_masks = [x[1] for x in computed_data]
              if 'mask' in estimator_util.fn_args(layer.call):
                if 'mask' not in kwargs:
                  kwargs['mask'] = computed_masks
              output_tensors = nest.flatten(
                  layer.call(computed_tensors, **kwargs))
              if hasattr(layer, 'compute_mask'):
                output_masks = nest.flatten(
                    layer.compute_mask(computed_tensors, computed_masks))
              else:
                output_masks = [None for _ in range(len(output_tensors))]

            # Apply activity regularizer if any:
            if layer.activity_regularizer is not None:
              regularization_losses = [
                  layer.activity_regularizer(x) for x in computed_tensors
              ]
              layer.add_loss(regularization_losses, computed_tensors)

          if context.in_graph_mode():
            # Update model updates and losses:
            # Keep track of updates that depend on the inputs
            # (e.g. BN updates).
            self.add_update(layer.get_updates_for(computed_tensors), inputs)
            # Keep track of unconditional updates (e.g. a counter).
            self.add_update(layer.get_updates_for(None), None)
            # Keep track of losses that depend on the inputs
            # (e.g. activity regularizers).
            self.add_loss(layer.get_losses_for(computed_tensors), inputs)
            # Keep track of unconditional losses
            # (e.g. weight regularizers).
            self.add_loss(layer.get_losses_for(None), None)

          # Update tensor_map.
          for x, y, mask in zip(reference_output_tensors, output_tensors,
                                output_masks):
            tensor_map[str(id(x))] = (y, mask)

    output_tensors = []
    output_masks = []
    output_shapes = []
    for x in self.outputs:
      assert str(id(x)) in tensor_map, 'Could not compute output ' + str(x)
      tensor, mask = tensor_map[str(id(x))]
      output_shapes.append(layers_util.static_shape(x))
      output_tensors.append(tensor)
      output_masks.append(mask)

    if len(output_tensors) == 1:
      output_tensors = output_tensors[0]
      if output_shapes is not None:
        output_shapes = output_shapes[0]
      if output_masks is not None:
        output_masks = output_masks[0]

    if context.in_graph_mode():
      # Update cache;
      # keys are based on ids on input tensors and inputs masks.
      cache_key = (layers_util.object_list_uid(inputs)
                   + '_' + layers_util.object_list_uid(masks))
      self._output_tensor_cache[cache_key] = output_tensors
      if output_masks is not None:
        self._output_mask_cache[cache_key] = output_masks
      if output_shapes is not None:
        input_shapes = [layers_util.static_shape(x) for x in inputs]
        cache_key = layers_util.object_list_uid(input_shapes)
        self._output_shape_cache[cache_key] = output_shapes

    return output_tensors, output_masks


def _make_node_key(layer_name, node_index):
  return layer_name + '_ib-' + str(node_index)
