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
"""A `Network` is way to compose layers: the topological form of a `Model`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
import weakref

import numpy as np
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import function as eager_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import saving
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.training.checkpointable import data_structures
from tensorflow.python.training.checkpointable import layer_utils as checkpointable_layer_utils
from tensorflow.python.training.checkpointable import util as checkpointable_utils
from tensorflow.python.util import tf_inspect


# pylint: disable=g-import-not-at-top
try:
  import h5py
except ImportError:
  h5py = None

try:
  import yaml
except ImportError:
  yaml = None
# pylint: enable=g-import-not-at-top


class Network(base_layer.Layer):
  """A `Network` is a composition of layers.

  It is the topological form of a "model". A `Model`
  is simply a `Network` with added training routines.
  """

  def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called
    # Signature detection
    if (len(args) == 2 or
        len(args) == 1 and 'outputs' in kwargs or
        'inputs' in kwargs and 'outputs' in kwargs):
      # Graph network
      self._init_graph_network(*args, **kwargs)
    else:
      # Subclassed network
      self._init_subclassed_network(**kwargs)

  # Several Network methods have "no_automatic_dependency_tracking"
  # annotations. Since Network does automatic dependency tracking on attribute
  # assignment, including for common data structures such as lists, by default
  # we'd have quite a few empty dependencies which users don't care about (or
  # would need some way to ignore dependencies automatically, which is confusing
  # when applied to user code). Some attributes, such as _layers, would cause
  # structural issues (_layers being the place where Layers assigned to tracked
  # attributes are stored).
  #
  # Aside from these aesthetic and structural issues, useless dependencies on
  # empty lists shouldn't cause issues; adding or removing them will not break
  # checkpoints, but may cause "all Python objects matched" assertions to fail
  # (in which case less strict assertions may be substituted if necessary).
  @checkpointable.no_automatic_dependency_tracking
  def _base_init(self, name=None):
    # The following are implemented as property functions:
    # self.trainable_weights
    # self.non_trainable_weights
    # self.input_spec
    # self.losses
    # self.updates

    self._init_set_name(name, zero_based=True)
    self._activity_regularizer = None
    # This acts just like the `trainable` attribute of any layer instance.
    # It does not affect users of the underlying layers, only users of the
    # Network instance.
    self.trainable = True
    self._is_compiled = False
    self._expects_training_arg = False
    # A list of "extra" variables assigned to attributes of this class, included
    # in self.weights and self.variables. Always empty for graph networks (but
    # included in base_init to avoid excessive special casing when retrieving
    # the value).
    self._extra_variables = []
    # In many internal cases one needs to compute both the model's output
    # and its output mask without relying on `__call__` (which would do both and
    # set mask metadata), but for models, computing the mask requires to
    # recompute the output.
    # Hence the pattern `output = model.call(); mask = model.compute_mask()`
    # would be redundant, and internal logic
    # (susceptible to use `call` directly) should prefer using the
    # internal method `output, mask = _call_and_compute_mask()`.
    # This is True for Sequential networks and graph networks.
    self._compute_output_and_mask_jointly = False

    self.supports_masking = False
    if not hasattr(self, 'optimizer'):
      # Don't reset optimizer if already set.
      self.optimizer = None

    # Private attributes to implement compatibility with Layer.
    self._updates = []  # Used in symbolic mode only.
    self._losses = []   # Used in symbolic mode only.
    self._scope = None  # Never used.
    self._reuse = None  # Never used.
    if context.executing_eagerly():
      self._graph = None
    else:
      self._graph = ops.get_default_graph()  # Used in symbolic mode only.
      # A Network does not create weights of its own, thus has no dtype.
    self._dtype = None

    # All layers in order of horizontal graph traversal.
    # Entries are unique. Includes input and output layers.
    self._layers = []

    # Used in symbolic mode only, only in conjunction with graph-networks
    self._outbound_nodes = []
    self._inbound_nodes = []

    self._checkpointable_saver = checkpointable_utils.CheckpointableSaver(
        weakref.ref(self))

  @checkpointable.no_automatic_dependency_tracking
  def _init_graph_network(self, inputs, outputs, name=None):
    self._call_convention = base_layer.CallConvention.EXPLICIT_INPUTS_ARGUMENT
    # Normalize and set self.inputs, self.outputs.
    if isinstance(inputs, (list, tuple)):
      self.inputs = list(inputs)  # Tensor or list of tensors.
    else:
      self.inputs = [inputs]
    if isinstance(outputs, (list, tuple)):
      self.outputs = list(outputs)
    else:
      self.outputs = [outputs]

    # User-provided argument validation.
    if context.executing_eagerly():
      # Check that all inputs/outputs are DeferredTensors.
      for tensor in self.inputs:
        if not isinstance(tensor, base_layer.DeferredTensor):  # pylint: disable=protected-access
          raise TypeError('When eager execution is enabled, '
                          'inputs must come from a call to '
                          '`tf.keras.Input` (called after '
                          'tf.enable_eager_execution()). '
                          'Received invalid input: ' + str(tensor))
      for tensor in self.outputs:
        if not isinstance(tensor, base_layer.DeferredTensor):  # pylint: disable=protected-access
          raise TypeError('When eager execution is enabled, '
                          'outputs must come from a call to '
                          'a layer (called after '
                          'tf.enable_eager_execution()). '
                          'Received invalid output: ' + str(tensor))
    # Check for redundancy in inputs.
    if len(set(self.inputs)) != len(self.inputs):
      raise ValueError('The list of inputs passed to the model '
                       'is redundant. '
                       'All inputs should only appear once.'
                       ' Found: ' + str(self.inputs))
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
    for x in self.outputs:
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Output tensors to a ' + cls_name + ' must be '
                         'the output of a TensorFlow `Layer` '
                         '(thus holding past layer metadata). Found: ' + str(x))

    self._base_init(name=name)
    self._compute_previous_mask = (
        'mask' in tf_inspect.getfullargspec(self.call).args or
        hasattr(self, 'compute_mask'))
    # A Network does not create weights of its own, thus it is already
    # built.
    self.built = True
    self._compute_output_and_mask_jointly = True
    self._is_graph_network = True

    self._input_layers = []
    self._output_layers = []
    self._input_coordinates = []
    self._output_coordinates = []

    # This is for performance optimization when calling the Network on new
    # inputs. Every time the Network is called on a set on input tensors,
    # we compute the output tensors, output masks and output shapes in one pass,
    # then cache them here. When any of these outputs is queried later, we
    # retrieve it from there instead of recomputing it.
    self._output_mask_cache = {}
    self._output_tensor_cache = {}
    self._output_shape_cache = {}

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

    # Keep track of the network's nodes and layers.
    nodes, nodes_by_depth, layers, layers_by_depth = _map_graph_network(
        self.inputs, self.outputs)
    self._network_nodes = nodes
    self._nodes_by_depth = nodes_by_depth
    self._layers = layers
    self._layers_by_depth = layers_by_depth

    self._track_layers(layers)

    # Create the node linking internal inputs to internal outputs.
    base_layer.Node(
        outbound_layer=self,
        inbound_layers=[],
        node_indices=[],
        tensor_indices=[],
        input_tensors=self.inputs,
        output_tensors=self.outputs)

    # Build self.input_names and self.output_names.
    self.input_names = []
    self.output_names = []
    self._feed_input_names = []
    self._feed_inputs = []
    self._feed_input_shapes = []
    for i, layer in enumerate(self._input_layers):
      self.input_names.append(layer.name)
      if layer.is_placeholder:
        self._feed_input_names.append(layer.name)
        self._feed_input_shapes.append(backend.int_shape(self.inputs[i]))
        # layer.input gives an error in eager mode
        if not context.executing_eagerly():
          self._feed_inputs.append(layer.input)
    for layer in self._output_layers:
      self.output_names.append(layer.name)

  @checkpointable.no_automatic_dependency_tracking
  def _init_subclassed_network(self, name=None):
    self._base_init(name=name)
    self._is_graph_network = False
    call_argspec = tf_inspect.getfullargspec(self.call)
    if 'training' in call_argspec.args:
      self._expects_training_arg = True
    else:
      self._expects_training_arg = False
    self._call_convention = self._determine_call_convention(call_argspec)
    self.outputs = []
    self.inputs = []
    self.built = False

  def _determine_call_convention(self, call_argspec):
    """Decides how `self.call()` is invoked. See base_layer.CallConvention."""
    if call_argspec.varargs:
      may_take_single_argument = False
    else:
      try:
        # Note: tf_inspect doesn't raise a TypeError when regular inspect would,
        # so we need to keep in mind that "getcallargs" may have returned
        # something even though we under-specified positional arguments.
        all_args = tf_inspect.getcallargs(self.call, None)
        self_args = set()
        for arg_name, obj in all_args.items():
          if obj is self:
            self_args.add(arg_name)
        may_take_single_argument = True
      except TypeError:
        may_take_single_argument = False
    if may_take_single_argument:
      # A single positional argument (plus "self") is considered equivalent to
      # an "inputs" argument.
      all_positional_args = len(call_argspec.args)
      if call_argspec.defaults is not None:
        all_positional_args -= len(call_argspec.defaults)
      non_self_positional_args = all_positional_args
      for positional_arg_name in call_argspec.args[:all_positional_args]:
        if positional_arg_name in self_args:
          non_self_positional_args -= 1
      if non_self_positional_args == 1:
        if 'inputs' in call_argspec.args[all_positional_args:]:
          raise TypeError(
              "Model.call() takes a single positional argument (to which "
              "inputs are passed by convention) and a separate 'inputs' "
              "argument. Unable to determine which arguments are inputs.")
        return base_layer.CallConvention.SINGLE_POSITIONAL_ARGUMENT
    if 'inputs' in call_argspec.args:
      return base_layer.CallConvention.EXPLICIT_INPUTS_ARGUMENT
    else:
      return base_layer.CallConvention.POSITIONAL_ARGUMENTS_ARE_INPUTS

  def _track_layers(self, layers):
    """Add Checkpointable dependencies on a list of Layers."""
    weight_layer_index = 0
    for layer_index, layer in enumerate(layers):
      if layer.weights:
        # Keep a separate index for layers which have weights. This allows users
        # to insert Layers without weights anywhere in the network without
        # breaking checkpoints.
        self._track_checkpointable(
            layer, name='layer_with_weights-%d' % weight_layer_index,
            overwrite=True)
        weight_layer_index += 1
      # Even if it doesn't have weights, we should still track everything in
      # case it has/will have Checkpointable dependencies.
      self._track_checkpointable(
          layer, name='layer-%d' % layer_index, overwrite=True)

  def _no_dependency(self, value):
    """Override to allow `Layer` to disable dependency tracking.

    `CheckpointableBase` defines this method, whose semantics are "if a subclass
    does dependency tracking, this method exempts `value`." Layer uses
    `_no_dependency` to exempt some of its attribute assignments (conditional on
    attribute assignment causing tracking in the subclass).

    Args:
      value: An object which will be assigned to an object attribute, whose
        value should not be tracked.

    Returns:
      A wrapped object which, when assigned to an attribute, will not be
      tracked (`value` will be stored in the attribute).
    """
    return data_structures.NoDependency(value)

  def __setattr__(self, name, value):
    if not getattr(self, '_setattr_tracking', True):
      super(Network, self).__setattr__(name, value)
      return
    no_dependency = isinstance(value, data_structures.NoDependency)
    value = data_structures.sticky_attribute_assignment(
        checkpointable=self, value=value, name=name)
    if isinstance(value, (
        base_layer.Layer,
        Network,
        data_structures.CheckpointableDataStructure)):
      try:
        is_graph_network = self._is_graph_network
      except AttributeError:
        raise RuntimeError('It looks like you are subclassing `Model` and you '
                           'forgot to call `super(YourClass, self).__init__()`.'
                           ' Always start with this line.')
      if not is_graph_network:
        # We need to check object identity to avoid de-duplicating empty
        # container types which compare equal.
        if not any((layer is value for layer in self._layers)):
          self._layers.append(value)
          if hasattr(value, '_use_resource_variables'):
            # In subclassed models, legacy layers (tf.layers) must always use
            # resource variables.
            value._use_resource_variables = True
    if (not no_dependency
        and isinstance(value, checkpointable.CheckpointableBase)):
      if (  # For subclassed models only, users may add extra weights/variables
            # simply by assigning them to attributes.
          not self._is_graph_network
          and isinstance(value, variables.Variable)):
        self._extra_variables.append(value)
    super(Network, self).__setattr__(name, value)

  def add_variable(self, name, shape, dtype=None, initializer=None,
                   regularizer=None, trainable=True, constraint=None):
    if self._is_graph_network:
      raise NotImplementedError('`add_variable` is not supported on Networks.')
    else:
      raise NotImplementedError(
          '`add_variable` is not supported on Networks. However, you may '
          'assign variables to attributes and they will show up in the weights '
          'and variables properties.')

  def add_loss(self, *args, **kwargs):
    if context.executing_eagerly():
      raise NotImplementedError('`add_loss` is not supported on Networks '
                                'when eager execution is enabled.')
    super(Network, self).add_loss(*args, **kwargs)

  @property
  def uses_learning_phase(self):
    return any(
        [getattr(x, '_uses_learning_phase', False) for x in self.outputs])

  @property
  def stateful(self):
    return any([(hasattr(layer, 'stateful') and layer.stateful)
                for layer in self.layers])

  def reset_states(self):
    for layer in self.layers:
      if hasattr(layer, 'reset_states') and getattr(layer, 'stateful', False):
        layer.reset_states()

  @property
  def state_updates(self):
    """Returns the `updates` from all layers that are stateful.

    This is useful for separating training updates and
    state updates, e.g. when we need to update a layer's internal state
    during prediction.

    Returns:
        A list of update ops.
    """
    state_updates = []
    for layer in self.layers:
      if getattr(layer, 'stateful', False):
        if hasattr(layer, 'updates'):
          state_updates += layer.updates
    return state_updates

  def get_weights(self):
    """Retrieves the weights of the model.

    Returns:
        A flat list of Numpy arrays.
    """
    weights = []
    for layer in self.layers:
      weights += layer.weights
    return backend.batch_get_value(weights)

  def set_weights(self, weights):
    """Sets the weights of the model.

    Arguments:
        weights: A list of Numpy arrays with shapes and types matching
            the output of `model.get_weights()`.
    """
    tuples = []
    for layer in self.layers:
      num_param = len(layer.weights)
      layer_weights = weights[:num_param]
      for sw, w in zip(layer.weights, layer_weights):
        tuples.append((sw, w))
      weights = weights[num_param:]
    backend.batch_set_value(tuples)

  def compute_mask(self, inputs, mask):
    if not self._is_graph_network:
      return None

    inputs = generic_utils.to_list(inputs)
    if mask is None:
      masks = [None for _ in range(len(inputs))]
    else:
      masks = generic_utils.to_list(mask)

    _, output_masks = self._run_internal_graph(inputs, mask=masks)
    return output_masks

  @property
  def layers(self):
    return checkpointable_layer_utils.filter_empty_layer_containers(
        self._layers)

  def get_layer(self, name=None, index=None):
    """Retrieves a layer based on either its name (unique) or index.

    If `name` and `index` are both provided, `index` will take precedence.
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
  def _unfiltered_updates(self):
    if context.executing_eagerly():
      return []
    updates = []
    for layer in self.layers:
      if isinstance(layer, Network):
        updates += layer._unfiltered_updates
      else:
        updates += layer.updates
    return updates

  @property
  def _unfiltered_losses(self):
    losses = []
    for layer in self.layers:
      if isinstance(layer, Network):
        losses += layer._unfiltered_losses
      else:
        losses += layer.losses
    return losses

  @property
  def updates(self):
    """Retrieves the network's updates.

    Will only include updates that are either
    unconditional, or conditional on inputs to this model
    (e.g. will not include updates that were created by layers of this model
    outside of the model).

    When the network has no registered inputs, all updates are returned.

    Effectively, `network.updates` behaves like `layer.updates`.

    Concrete example:

    ```python
      bn = keras.layers.BatchNormalization()
      x1 = keras.layers.Input(shape=(10,))
      _ = bn(x1)  # This creates 2 updates.

      x2 = keras.layers.Input(shape=(10,))
      y2 = bn(x2)  # This creates 2 more updates.

      # The BN layer has now 4 updates.
      self.assertEqual(len(bn.updates), 4)

      # Let's create a model from x2 to y2.
      model = keras.models.Model(x2, y2)

      # The model does not list all updates from its underlying layers,
      # but only the updates that are relevant to it. Updates created by layers
      # outside of the model are discarded.
      self.assertEqual(len(model.updates), 2)

      # If you keep calling the model, you append to its updates, just like
      # what happens for a layer.
      x3 = keras.layers.Input(shape=(10,))
      y3 = model(x3)
      self.assertEqual(len(model.updates), 4)

      # But if you call the inner BN layer independently, you don't affect
      # the model's updates.
      x4 = keras.layers.Input(shape=(10,))
      _ = bn(x4)
      self.assertEqual(len(model.updates), 4)
    ```

    Returns:
        A list of update ops.
    """
    if context.executing_eagerly():
      return []

    if not self.trainable and not self.stateful:
      return []

    updates = self._unfiltered_updates

    # `updates` might contain irrelevant updates, so it needs to be filtered
    # with respect to inputs the model has been called on.
    relevant_inputs = []
    for i in range(0, len(self._inbound_nodes)):
      inputs = self.get_input_at(i)
      if isinstance(inputs, list):
        relevant_inputs += inputs
      else:
        relevant_inputs.append(inputs)
    if not relevant_inputs:
      return updates

    reachable = tf_utils.get_reachable_from_inputs(relevant_inputs, updates)
    relevant_conditional_updates = [x for x in updates if x in reachable]
    unconditional_updates = [
        x for x in updates if x._unconditional_update]  # pylint: disable=protected-access
    # A layer could be used multiple times in a nested structure,
    # so the updates list must be de-duped.
    return list(set(
        relevant_conditional_updates + unconditional_updates + self._updates))

  @property
  def losses(self):
    """Retrieves the network's losses.

    Will only include losses that are either
    unconditional, or conditional on inputs to this model
    (e.g. will not include losses that depend on tensors
    that aren't inputs to this model).

    When the network has no registered inputs, all losses are returned.

    Returns:
        A list of loss tensors.
    """
    losses = self._unfiltered_losses
    if context.executing_eagerly():
      return losses

    relevant_inputs = []
    for i in range(0, len(self._inbound_nodes)):
      inputs = self.get_input_at(i)
      if isinstance(inputs, list):
        relevant_inputs += inputs
      else:
        relevant_inputs.append(inputs)
    if not relevant_inputs:
      return losses

    reachable = tf_utils.get_reachable_from_inputs(relevant_inputs, losses)
    relevant_conditional_losses = [x for x in losses if x in reachable]
    unconditional_losses = [
        x for x in losses if x._unconditional_loss]  # pylint: disable=protected-access
    return list(set(
        relevant_conditional_losses + unconditional_losses + self._losses))

  @property
  def trainable_weights(self):
    return checkpointable_layer_utils.gather_trainable_weights(
        trainable=self.trainable,
        sub_layers=self.layers,
        extra_variables=self._extra_variables)

  @property
  def non_trainable_weights(self):
    return checkpointable_layer_utils.gather_non_trainable_weights(
        trainable=self.trainable,
        sub_layers=self.layers,
        extra_variables=self._extra_variables)

  @property
  def input_spec(self):
    """Gets the network's input specs.

    Returns:
        A list of `InputSpec` instances (one per input to the model)
            or a single instance if the model has only one input.
    """
    # If not a graph network, can't assume anything.
    if not self._is_graph_network:
      return None

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

  @base_layer.default
  def build(self, input_shape):
    """Builds the model based on input shapes received.

    This is to be used for subclassed models, which do not know at instantiation
    time what their inputs look like.

    Args:
     input_shape: Single tuple, TensorShape, or list of shapes, where shapes
         are tuples, integers, or TensorShapes.

    Raises:
      ValueError:
        1. In case of invalid user-provided data (not of type tuple,
           list, or TensorShape).
        2. If the model requires call arguments that are agnostic
           to the input shapes (positional or kwarg in call signature).
        3. If not all layers were properly built.
        4. If float type inputs are not supported within the layers.

      In each of these cases, the user should build their model by calling it
      on real tensor data.
    """
    if self._is_graph_network:
      self.built = True
      return

    # If subclass network
    if input_shape is None:
      raise ValueError('Input shape must be defined when calling build on a '
                       'model subclass network.')
    valid_types = (tuple, list, tensor_shape.TensorShape)
    if not isinstance(input_shape, valid_types):
      raise ValueError('Specified input shape is not one of the valid types. '
                       'Please specify a batch input shape of type tuple or '
                       'list of input shapes. User provided '
                       'input type: {}'.format(type(input_shape)))

    if input_shape and not self.inputs:
      # We create placeholders for the `None`s in the shape and build the model
      # in a Graph. Since tf.Variable is compatible with both eager execution
      # and graph building, the variables created after building the model in
      # a Graph are still valid when executing eagerly.
      with context.graph_mode():
        graph = eager_function.CapturingGraph()
        with graph.as_default():
          if isinstance(input_shape, list):
            x = [base_layer.generate_placeholders_from_shape(shape)
                 for shape in input_shape]
          else:
            x = base_layer.generate_placeholders_from_shape(input_shape)

          kwargs = {}
          num_call_args = len(tf_inspect.getfullargspec(self.call).args)
          if self._expects_training_arg and num_call_args == 3:
            # Has call signature of call(self, input, training)
            kwargs['training'] = False
          elif num_call_args > 2:
            # Has invalid call signature of call(self, input, *args, **kwargs)
            raise ValueError('Currently, you cannot build your model if it has '
                             'positional or keyword arguments that are not '
                             'inputs to the model, but are required for its '
                             '`call` method. Instead, in order to instantiate '
                             'and build your model, `call` your model on real '
                             'tensor data with all expected call arguments.')

          try:
            self.call(x, **kwargs)
          except (errors.InvalidArgumentError, TypeError):
            raise ValueError('You cannot build your model by calling `build` '
                             'if your layers do not support float type inputs. '
                             'Instead, in order to instantiate and build your '
                             'model, `call` your model on real tensor data (of '
                             'the correct dtype).')

    if self._layers:
      self._track_layers(self._layers)
    if self.layers:
      for layer in self.layers:
        if not layer.built:
          raise ValueError('Layer: {} was not built in your model. Calling '
                           '`build` manually on a subclassed model is only '
                           'allowed for models with a static topology. '
                           'In this case, you can build your model by '
                           'calling it on real tensor data.'.format(layer))
    self.built = True

  def call(self, inputs, training=None, mask=None):
    """Calls the model on new inputs.

    In this case `call` just reapplies
    all ops in the graph to the new inputs
    (e.g. build a new computational graph from the provided inputs).

    Arguments:
        inputs: A tensor or list of tensors.
        training: Boolean or boolean scalar tensor, indicating whether to run
          the `Network` in training mode or inference mode.
        mask: A mask or list of masks. A mask can be
            either a tensor or None (no mask).

    Returns:
        A tensor if there is a single output, or
        a list of tensors if there are more than one outputs.
    """
    if not self._is_graph_network:
      raise NotImplementedError('When subclassing the `Model` class, you should'
                                ' implement a `call` method.')

    inputs = generic_utils.to_list(inputs)
    if mask is None:
      masks = [None for _ in range(len(inputs))]
    else:
      masks = generic_utils.to_list(mask)
    outputs, _ = self._run_internal_graph(inputs,
                                          training=training,
                                          mask=masks)
    return outputs

  def _call_and_compute_mask(self, inputs, training=None, mask=None):
    inputs = generic_utils.to_list(inputs)
    if mask is None:
      masks = [None for _ in range(len(inputs))]
    else:
      masks = generic_utils.to_list(mask)
    return self._run_internal_graph(inputs,
                                    training=training,
                                    mask=masks)

  def compute_output_shape(self, input_shape):
    if not self._is_graph_network:
      if context.executing_eagerly():
        return super(Network, self).compute_output_shape(input_shape)
      raise NotImplementedError

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

    cache_key = generic_utils.object_list_uid(input_shapes)
    if cache_key in self._output_shape_cache:
      # Cache hit.
      output_shapes = self._output_shape_cache[cache_key]
    else:
      layers_to_output_shapes = {}
      for i in range(len(input_shapes)):
        layer = self._input_layers[i]
        input_shape = input_shapes[i]
        # It's an input layer: then `compute_output_shape` is identity,
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
              output_shape = layer.compute_output_shape(input_shapes[0])
            else:
              output_shape = layer.compute_output_shape(input_shapes)
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

    if isinstance(output_shapes, list):
      if len(output_shapes) == 1:
        return tensor_shape.TensorShape(output_shapes[0])
      else:
        return [tensor_shape.TensorShape(shape) for shape in output_shapes]
    else:
      return tensor_shape.TensorShape(output_shapes)

  def _run_internal_graph(self, inputs, training=None, mask=None):
    """Computes output tensors for new inputs.

    # Note:
        - Expects `inputs` to be a list (potentially with 1 element).
        - Can be run on non-Keras tensors.

    Arguments:
        inputs: List of tensors
        training: Boolean learning phase.
        mask: List of masks (tensors or None).

    Returns:
        Two lists: output_tensors, output_masks
    """
    # Note: masking support is relevant mainly for Keras.
    # It cannot be factored out without having the fully reimplement the network
    # calling logic on the Keras side. We choose to incorporate it in
    # Network because 1) it may be useful to fully support in tf.layers in
    # the future and 2) Keras is a major user of Network.  If you don't
    # use masking, it does not interfere with regular behavior at all and you
    # can ignore it.
    if mask is None:
      masks = [None for _ in range(len(inputs))]
    else:
      masks = mask

    # Dictionary mapping reference tensors to tuples
    # (computed tensor, compute mask)
    # we assume a 1:1 mapping from tensor to mask
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
            # Ensure `training` arg propagation if applicable.
            if 'training' in tf_inspect.getfullargspec(layer.call).args:
              kwargs.setdefault('training', training)

            if len(computed_data) == 1:
              computed_tensor, computed_mask = computed_data[0]
              # Ensure mask propagation if applicable.
              if 'mask' in tf_inspect.getfullargspec(layer.call).args:
                kwargs.setdefault('mask', computed_mask)

              # Compute outputs and masks.
              if (isinstance(layer, Network) and
                  layer._compute_output_and_mask_jointly):
                output_tensors, output_masks = layer._call_and_compute_mask(
                    computed_tensor, **kwargs)
              else:
                output_tensors = layer.call(computed_tensor, **kwargs)
                if hasattr(layer, 'compute_mask'):
                  output_masks = layer.compute_mask(computed_tensor,
                                                    computed_mask)
                else:
                  output_masks = [None for _ in output_tensors]
              computed_tensors = [computed_tensor]

            else:
              computed_tensors = [x[0] for x in computed_data]
              computed_masks = [x[1] for x in computed_data]
              # Ensure mask propagation if applicable.
              if 'mask' in tf_inspect.getfullargspec(layer.call).args:
                kwargs.setdefault('mask', computed_masks)

              # Compute outputs and masks.
              if (isinstance(layer, Network) and
                  layer._compute_output_and_mask_jointly):
                output_tensors, output_masks = layer._call_and_compute_mask(
                    computed_tensors, **kwargs)
              else:
                output_tensors = layer.call(computed_tensors, **kwargs)
                if hasattr(layer, 'compute_mask'):
                  output_masks = layer.compute_mask(computed_tensors,
                                                    computed_masks)
                else:
                  output_masks = [None for _ in output_tensors]

            output_tensors = generic_utils.to_list(output_tensors)
            if output_masks is None:
              output_masks = [None for _ in output_tensors]
            else:
              output_masks = generic_utils.to_list(output_masks)

            if not context.executing_eagerly():
              # Set mask metadata.
              for x, m in zip(output_tensors, output_masks):
                try:
                  x._keras_mask = m
                except AttributeError:
                  pass

              # Apply activity regularizer if any.
              if layer.activity_regularizer is not None:
                regularization_losses = [
                    layer.activity_regularizer(x) for x in output_tensors
                ]
                layer.add_loss(regularization_losses, computed_tensors)

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
      output_shapes.append(backend.int_shape(x))
      output_tensors.append(tensor)
      output_masks.append(mask)

    if len(output_tensors) == 1:
      output_tensors = output_tensors[0]
      if output_shapes is not None:
        output_shapes = output_shapes[0]
      if output_masks is not None:
        output_masks = output_masks[0]

    if output_shapes is not None:
      input_shapes = [backend.int_shape(x) for x in inputs]
      cache_key = generic_utils.object_list_uid(input_shapes)
      self._output_shape_cache[cache_key] = output_shapes

    return output_tensors, output_masks

  def get_config(self):
    if not self._is_graph_network:
      raise NotImplementedError

    config = {
        'name': self.name,
    }
    node_conversion_map = {}
    for layer in self.layers:
      if issubclass(layer.__class__, Network):
        # Networks start with a pre-existing node
        # linking their input to output.
        kept_nodes = 1
      else:
        kept_nodes = 0
      for original_node_index, node in enumerate(layer._inbound_nodes):
        node_key = _make_node_key(layer.name, original_node_index)
        if node_key in self._network_nodes:
          node_conversion_map[node_key] = kept_nodes
          kept_nodes += 1
    layer_configs = []
    for layer in self.layers:  # From the earliest layers on.
      layer_class_name = layer.__class__.__name__
      layer_config = layer.get_config()
      filtered_inbound_nodes = []
      for original_node_index, node in enumerate(layer._inbound_nodes):
        node_key = _make_node_key(layer.name, original_node_index)
        if node_key in self._network_nodes:
          # The node is relevant to the model:
          # add to filtered_inbound_nodes.
          if node.arguments:
            try:
              json.dumps(node.arguments)
              kwargs = node.arguments
            except TypeError:
              logging.warning(
                  'Layer ' + layer.name +
                  ' was passed non-serializable keyword arguments: ' +
                  str(node.arguments) + '. They will not be included '
                  'in the serialized model (and thus will be missing '
                  'at deserialization time).')
              kwargs = {}
          else:
            kwargs = {}
          if node.inbound_layers:
            node_data = []
            for i in range(len(node.inbound_layers)):
              inbound_layer = node.inbound_layers[i]
              node_index = node.node_indices[i]
              tensor_index = node.tensor_indices[i]
              node_key = _make_node_key(inbound_layer.name, node_index)
              new_node_index = node_conversion_map.get(node_key, 0)
              node_data.append(
                  [inbound_layer.name, new_node_index, tensor_index, kwargs])
            filtered_inbound_nodes.append(node_data)
      layer_configs.append({
          'name': layer.name,
          'class_name': layer_class_name,
          'config': layer_config,
          'inbound_nodes': filtered_inbound_nodes,
      })
    config['layers'] = layer_configs

    # Gather info about inputs and outputs.
    model_inputs = []
    for i in range(len(self._input_layers)):
      layer, node_index, tensor_index = self._input_coordinates[i]
      node_key = _make_node_key(layer.name, node_index)
      if node_key not in self._network_nodes:
        continue
      new_node_index = node_conversion_map[node_key]
      model_inputs.append([layer.name, new_node_index, tensor_index])
    config['input_layers'] = model_inputs
    model_outputs = []
    for i in range(len(self._output_layers)):
      layer, node_index, tensor_index = self._output_coordinates[i]
      node_key = _make_node_key(layer.name, node_index)
      if node_key not in self._network_nodes:
        continue
      new_node_index = node_conversion_map[node_key]
      model_outputs.append([layer.name, new_node_index, tensor_index])
    config['output_layers'] = model_outputs
    return copy.deepcopy(config)

  @classmethod
  def from_config(cls, config, custom_objects=None):
    """Instantiates a Model from its config (output of `get_config()`).

    Arguments:
        config: Model config dictionary.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    Returns:
        A model instance.

    Raises:
        ValueError: In case of improperly formatted config dict.
    """
    # Layer instances created during
    # the graph reconstruction process
    created_layers = {}

    # Dictionary mapping layer instances to
    # node data that specifies a layer call.
    # It acts as a queue that maintains any unprocessed
    # layer call until it becomes possible to process it
    # (i.e. until the input tensors to the call all exist).
    unprocessed_nodes = {}

    def add_unprocessed_node(layer, node_data):
      if layer not in unprocessed_nodes:
        unprocessed_nodes[layer] = [node_data]
      else:
        unprocessed_nodes[layer].append(node_data)

    def process_node(layer, node_data):
      """Deserialize a node.

      Arguments:
          layer: layer instance.
          node_data: node config dict.

      Raises:
          ValueError: In case of improperly formatted `node_data` dict.
      """
      input_tensors = []
      for input_data in node_data:
        inbound_layer_name = input_data[0]
        inbound_node_index = input_data[1]
        inbound_tensor_index = input_data[2]
        if len(input_data) == 3:
          kwargs = {}
        elif len(input_data) == 4:
          kwargs = input_data[3]
        else:
          raise ValueError('Improperly formatted model config.')
        if inbound_layer_name not in created_layers:
          add_unprocessed_node(layer, node_data)
          return
        inbound_layer = created_layers[inbound_layer_name]
        if len(inbound_layer._inbound_nodes) <= inbound_node_index:
          add_unprocessed_node(layer, node_data)
          return
        inbound_node = inbound_layer._inbound_nodes[inbound_node_index]
        input_tensors.append(inbound_node.output_tensors[inbound_tensor_index])
      # Call layer on its inputs, thus creating the node
      # and building the layer if needed.
      if input_tensors:
        if len(input_tensors) == 1:
          layer(input_tensors[0], **kwargs)
        else:
          layer(input_tensors, **kwargs)

    def process_layer(layer_data):
      """Deserializes a layer, then call it on appropriate inputs.

      Arguments:
          layer_data: layer config dict.

      Raises:
          ValueError: In case of improperly formatted `layer_data` dict.
      """
      layer_name = layer_data['name']

      # Instantiate layer.
      from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top

      layer = deserialize_layer(layer_data, custom_objects=custom_objects)
      created_layers[layer_name] = layer

      # Gather layer inputs.
      inbound_nodes_data = layer_data['inbound_nodes']
      for node_data in inbound_nodes_data:
        # We don't process nodes (i.e. make layer calls)
        # on the fly because the inbound node may not yet exist,
        # in case of layer shared at different topological depths
        # (e.g. a model such as A(B(A(B(x)))))
        add_unprocessed_node(layer, node_data)

    # First, we create all layers and enqueue nodes to be processed
    for layer_data in config['layers']:
      process_layer(layer_data)
    # Then we process nodes in order of layer depth.
    # Nodes that cannot yet be processed (if the inbound node
    # does not yet exist) are re-enqueued, and the process
    # is repeated until all nodes are processed.
    while unprocessed_nodes:
      for layer_data in config['layers']:
        layer = created_layers[layer_data['name']]
        if layer in unprocessed_nodes:
          for node_data in unprocessed_nodes.pop(layer):
            process_node(layer, node_data)

    name = config.get('name')
    input_tensors = []
    output_tensors = []
    for layer_data in config['input_layers']:
      layer_name, node_index, tensor_index = layer_data
      assert layer_name in created_layers
      layer = created_layers[layer_name]
      layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
      input_tensors.append(layer_output_tensors[tensor_index])
    for layer_data in config['output_layers']:
      layer_name, node_index, tensor_index = layer_data
      assert layer_name in created_layers
      layer = created_layers[layer_name]
      layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
      output_tensors.append(layer_output_tensors[tensor_index])
    return cls(inputs=input_tensors, outputs=output_tensors, name=name)

  def save(self, filepath, overwrite=True, include_optimizer=True):
    """Saves the model to a single HDF5 file.

    The savefile includes:
        - The model architecture, allowing to re-instantiate the model.
        - The model weights.
        - The state of the optimizer, allowing to resume training
            exactly where you left off.

    This allows you to save the entirety of the state of a model
    in a single file.

    Saved models can be reinstantiated via `keras.models.load_model`.
    The model returned by `load_model`
    is a compiled model ready to be used (unless the saved model
    was never compiled in the first place).

    Arguments:
        filepath: String, path to the file to save the weights to.
        overwrite: Whether to silently overwrite any existing file at the
            target location, or provide the user with a manual prompt.
        include_optimizer: If True, save optimizer's state together.

    Example:

    ```python
    from keras.models import load_model

    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model

    # returns a compiled model
    # identical to the previous one
    model = load_model('my_model.h5')
    ```
    """
    if not self._is_graph_network:
      raise NotImplementedError

    from tensorflow.python.keras.models import save_model  # pylint: disable=g-import-not-at-top
    save_model(self, filepath, overwrite, include_optimizer)

  def save_weights(self, filepath, overwrite=True, save_format=None):
    """Saves all layer weights.

    Either saves in HDF5 or in TensorFlow format based on the `save_format`
    argument.

    When saving in HDF5 format, the weight file has:
      - `layer_names` (attribute), a list of strings
          (ordered names of model layers).
      - For every layer, a `group` named `layer.name`
          - For every such layer group, a group attribute `weight_names`,
              a list of strings
              (ordered names of weights tensor of the layer).
          - For every weight in the layer, a dataset
              storing the weight value, named after the weight tensor.

    When saving in TensorFlow format, all objects referenced by the network are
    saved in the same format as `tf.train.Checkpoint`, including any `Layer`
    instances or `Optimizer` instances assigned to object attributes. For
    networks constructed from inputs and outputs using `tf.keras.Model(inputs,
    outputs)`, `Layer` instances used by the network are tracked/saved
    automatically. For user-defined classes which inherit from `tf.keras.Model`,
    `Layer` instances must be assigned to object attributes, typically in the
    constructor. See the documentation of `tf.train.Checkpoint` and
    `tf.keras.Model` for details.

    Arguments:
        filepath: String, path to the file to save the weights to. When saving
            in TensorFlow format, this is the prefix used for checkpoint files
            (multiple files are generated). Note that the '.h5' suffix causes
            weights to be saved in HDF5 format.
        overwrite: Whether to silently overwrite any existing file at the
            target location, or provide the user with a manual prompt.
        save_format: Either 'tf' or 'h5'. A `filepath` ending in '.h5' or
            '.keras' will default to HDF5 if `save_format` is `None`. Otherwise
            `None` defaults to 'tf'.

    Raises:
        ImportError: If h5py is not available when attempting to save in HDF5
            format.
        ValueError: For invalid/unknown format arguments.
    """
    filepath_is_h5 = _is_hdf5_filepath(filepath)
    if save_format is None:
      if filepath_is_h5:
        save_format = 'h5'
      else:
        save_format = 'tf'
    else:
      user_format = save_format.lower().strip()
      if user_format in ('tensorflow', 'tf'):
        save_format = 'tf'
      elif user_format in ('hdf5', 'h5', 'keras'):
        save_format = 'h5'
      else:
        raise ValueError(
            'Unknown format "%s". Was expecting one of {"tf", "h5"}.' % (
                save_format,))
    if save_format == 'tf' and filepath_is_h5:
      raise ValueError(
          ('save_weights got save_format="tf"/"tensorflow", but the '
           'filepath ("%s") looks like an HDF5 file. Omit the ".h5"/".keras" '
           'when saving in TensorFlow format.')
          % filepath)

    if save_format == 'h5' and h5py is None:
      raise ImportError(
          '`save_weights` requires h5py when saving in hdf5.')
    if save_format == 'tf':
      check_filepath = filepath + '.index'
    else:
      check_filepath = filepath
    # If file exists and should not be overwritten:
    if not overwrite and os.path.isfile(check_filepath):
      proceed = ask_to_proceed_with_overwrite(check_filepath)
      if not proceed:
        return
    if save_format == 'h5':
      with h5py.File(filepath, 'w') as f:
        saving.save_weights_to_hdf5_group(f, self.layers)
    else:
      if context.executing_eagerly():
        session = None
      else:
        session = backend.get_session()
      optimizer = getattr(self, 'optimizer', None)
      if (optimizer
          and not isinstance(optimizer, checkpointable.CheckpointableBase)):
        logging.warning(
            ('This model was compiled with a Keras optimizer (%s) but is being '
             'saved in TensorFlow format with `save_weights`. The model\'s '
             'weights will be saved, but unlike with TensorFlow optimizers in '
             'the TensorFlow format the optimizer\'s state will not be '
             'saved.\n\nConsider using a TensorFlow optimizer from `tf.train`.')
            % (optimizer,))
      self._checkpointable_saver.save(filepath, session=session)

  def load_weights(self, filepath, by_name=False):
    """Loads all layer weights, either from a TensorFlow or an HDF5 weight file.

    If `by_name` is False weights are loaded based on the network's
    topology. This means the architecture should be the same as when the weights
    were saved.  Note that layers that don't have weights are not taken into
    account in the topological ordering, so adding or removing layers is fine as
    long as they don't have weights.

    If `by_name` is True, weights are loaded into layers only if they share the
    same name. This is useful for fine-tuning or transfer-learning models where
    some of the layers have changed.

    Only topological loading (`by_name=False`) is supported when loading weights
    from the TensorFlow format. Note that topological loading differs slightly
    between TensorFlow and HDF5 formats for user-defined classes inheriting from
    `tf.keras.Model`: HDF5 loads based on a flattened list of weights, while the
    TensorFlow format loads based on the object-local names of attributes to
    which layers are assigned in the `Model`'s constructor.

    Arguments:
        filepath: String, path to the weights file to load. For weight files in
            TensorFlow format, this is the file prefix (the same as was passed
            to `save_weights`).
        by_name: Boolean, whether to load weights by name or by topological
            order. Only topological loading is supported for weight files in
            TensorFlow format.

    Returns:
        When loading a weight file in TensorFlow format, returns the same status
        object as `tf.train.Checkpoint.restore`. When graph building, restore
        ops are run automatically as soon as the network is built (on first call
        for user-defined classes inheriting from `Model`, immediately if it is
        already built).

        When loading weights in HDF5 format, returns `None`.

    Raises:
        ImportError: If h5py is not available and the weight file is in HDF5
            format.
    """
    if _is_hdf5_filepath(filepath):
      save_format = 'h5'
    else:
      try:
        pywrap_tensorflow.NewCheckpointReader(filepath)
        save_format = 'tf'
      except errors_impl.DataLossError:
        # The checkpoint is not readable in TensorFlow format. Try HDF5.
        save_format = 'h5'
    if save_format == 'tf':
      status = self._checkpointable_saver.restore(filepath)
      if by_name:
        raise NotImplementedError(
            'Weights may only be loaded based on topology into Models when '
            'loading TensorFlow-formatted weights (got by_name=True to '
            'load_weights).')
      if not context.executing_eagerly():
        session = backend.get_session()
        # Restore existing variables (if any) immediately, and set up a
        # streaming restore for any variables created in the future.
        checkpointable_utils.streaming_restore(status=status, session=session)
      return status
    if h5py is None:
      raise ImportError(
          '`load_weights` requires h5py when loading weights from HDF5.')
    if self._is_graph_network and not self.built:
      raise NotImplementedError(
          'Unable to load weights saved in HDF5 format into a subclassed '
          'Model which has not created its variables yet. Call the Model '
          'first, then load the weights.')
    with h5py.File(filepath, 'r') as f:
      if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
      if by_name:
        saving.load_weights_from_hdf5_group_by_name(f, self.layers)
      else:
        saving.load_weights_from_hdf5_group(f, self.layers)

  def _updated_config(self):
    """Util shared between different serialization methods.

    Returns:
        Model config with Keras version information added.
    """
    from tensorflow.python.keras import __version__ as keras_version  # pylint: disable=g-import-not-at-top

    config = self.get_config()
    model_config = {
        'class_name': self.__class__.__name__,
        'config': config,
        'keras_version': keras_version,
        'backend': backend.backend()
    }
    return model_config

  def to_json(self, **kwargs):
    """Returns a JSON string containing the network configuration.

    To load a network from a JSON save file, use
    `keras.models.model_from_json(json_string, custom_objects={})`.

    Arguments:
        **kwargs: Additional keyword arguments
            to be passed to `json.dumps()`.

    Returns:
        A JSON string.
    """
    def get_json_type(obj):
      # If obj is any numpy type
      if type(obj).__module__ == np.__name__:
        return obj.item()

      # If obj is a python 'type'
      if type(obj).__name__ == type.__name__:
        return obj.__name__

      raise TypeError('Not JSON Serializable:', obj)

    model_config = self._updated_config()
    return json.dumps(model_config, default=get_json_type, **kwargs)

  def to_yaml(self, **kwargs):
    """Returns a yaml string containing the network configuration.

    To load a network from a yaml save file, use
    `keras.models.model_from_yaml(yaml_string, custom_objects={})`.

    `custom_objects` should be a dictionary mapping
    the names of custom losses / layers / etc to the corresponding
    functions / classes.

    Arguments:
        **kwargs: Additional keyword arguments
            to be passed to `yaml.dump()`.

    Returns:
        A YAML string.

    Raises:
        ImportError: if yaml module is not found.
    """
    if yaml is None:
      raise ImportError(
          'Requires yaml module installed (`pip install pyyaml`).')
    return yaml.dump(self._updated_config(), **kwargs)

  def summary(self, line_length=None, positions=None, print_fn=None):
    """Prints a string summary of the network.

    Arguments:
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements
            in each line. If not provided,
            defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use. Defaults to `print`.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.

    Raises:
        ValueError: if `summary()` is called before the model is built.
    """
    if not self.built:
      raise ValueError('This model has never been called, thus its weights '
                       'have not yet been created, so no summary can be '
                       'displayed. Build the model first '
                       '(e.g. by calling it on some data).')
    layer_utils.print_summary(self,
                              line_length=line_length,
                              positions=positions,
                              print_fn=print_fn)


def _is_hdf5_filepath(filepath):
  return filepath.endswith('.h5') or filepath.endswith('.keras')


def _make_node_key(layer_name, node_index):
  return layer_name + '_ib-' + str(node_index)


def _map_graph_network(inputs, outputs):
  """Validates a network's topology and gather its layers and nodes.

  Arguments:
    inputs: List of input tensors.
    outputs: List of outputs tensors.

  Returns:
    A tuple `(nodes, nodes_by_depth, layers, layers_by_depth)`.
    - nodes: list of Node instances.
    - nodes_by_depth: dict mapping ints (depth) to lists of node instances.
    - layers: list of Layer instances.
    - layers_by_depth: dict mapping ints (depth) to lists of layer instances.

  Raises:
    ValueError: In case the network is not valid (e.g. disconnected graph).
  """
  # Network_nodes: set of nodes included in the graph of layers
  # (not all nodes included in the layers are relevant to the current graph).
  network_nodes = set()  # ids of all nodes relevant to the Network
  nodes_depths = {}  # dict {node: depth value}
  layers_depths = {}  # dict {layer: depth value}
  layer_indices = {}  # dict {layer: index in traversal}
  nodes_in_decreasing_depth = []

  def build_map(tensor,
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
      build_map(x, finished_nodes, nodes_in_progress, layer,
                node_index, tensor_index)

    finished_nodes.add(node)
    nodes_in_progress.remove(node)
    nodes_in_decreasing_depth.append(node)

  finished_nodes = set()
  nodes_in_progress = set()
  for x in outputs:
    layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
    build_map(x, finished_nodes, nodes_in_progress,
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
    # Network.layers needs to have a deterministic order:
    # here we order them by traversal order.
    layers_for_depth.sort(key=lambda x: layer_indices[x])
    layers.extend(layers_for_depth)

  # Get sorted list of node depths.
  depth_keys = list(nodes_by_depth.keys())
  depth_keys.sort(reverse=True)

  # Check that all tensors required are computable.
  # computable_tensors: all tensors in the graph
  # that can be computed from the inputs provided.
  computable_tensors = []
  for x in inputs:
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

  # Ensure name unicity, which will be crucial for serialization
  # (since serialized nodes refer to layers by their name).
  all_names = [layer.name for layer in layers]
  for name in all_names:
    if all_names.count(name) != 1:
      raise ValueError('The name "' + name + '" is used ' +
                       str(all_names.count(name)) + ' times in the model. '
                       'All layer names should be unique.')
  return network_nodes, nodes_by_depth, layers, layers_by_depth
