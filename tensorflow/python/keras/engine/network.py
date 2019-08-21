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

import collections
import copy
import itertools
import json
import os
import threading

import numpy as np
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras import saving
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import layer_utils as trackable_layer_utils
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util as trackable_utils
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import serialization
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

  `Network` is the topological form of a "model". A `Model`
  is simply a `Network` with added training routines.

  Two types of `Networks` exist: Graph Networks and Subclass Networks. Graph
  networks are used in the Keras Functional and Sequential APIs. Subclassed
  networks are used when a user subclasses the `Model` class. In general,
  more Keras features are supported with Graph Networks than with Subclassed
  Networks, specifically:

  - Model cloning (`keras.models.clone`)
  - Serialization (`model.get_config()/from_config`, `model.to_json()/to_yaml()`
  - Whole-model saving (`model.save()`)

  A Graph Network can be instantiated by passing two arguments to `__init__`.
  The first argument is the `keras.Input` Tensors that represent the inputs
  to the Network. The second argument specifies the output Tensors that
  represent the outputs of this Network. Both arguments can be a nested
  structure of Tensors.

  Example:

  ```
  inputs = {'x1': keras.Input(shape=(10,)), 'x2': keras.Input(shape=(1,))}
  t = keras.layers.Dense(1, activation='relu')(inputs['x1'])
  outputs = keras.layers.Add()([t, inputs['x2'])
  network = Network(inputs, outputs)
  ```

  A Graph Network constructed using the Functional API can also include raw
  TensorFlow functions, with the exception of functions that create Variables
  or assign ops.

  Example:

  ```
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(1)(inputs)
  outputs = tf.nn.relu(x)
  network = Network(inputs, outputs)
  ```

  Subclassed Networks can be instantiated via `name` and (optional) `dynamic`
  keyword arguments. Subclassed Networks keep track of their Layers, and their
  `call` method can be overridden. Subclassed Networks are typically created
  indirectly, by subclassing the `Model` class.

  Example:

  ```
  class MyModel(keras.Model):
    def __init__(self):
      super(MyModel, self).__init__(name='my_model', dynamic=False)

      self.layer1 = keras.layers.Dense(10, activation='relu')

    def call(self, inputs):
      return self.layer1(inputs)
  ```

  Allowed args in `super().__init__`:
    name: String name of the model.
    dynamic: (Subclassed models only) Set this to `True` if your model should
      only be run eagerly, and should not be used to generate a static
      computation graph. This attribute is automatically set for Functional API
      models.
    trainable: Boolean, whether the model's variables should be trainable.
    dtype: (Subclassed models only) Default dtype of the model's weights (
      default of `None` means use the type of the first input). This attribute
      has no effect on Functional API models, which do not have weights of their
      own.
  """

  # See tf.Module for the usage of this property.
  # The key of _layer_call_argspecs is a layer. tf.Module._flatten will fail to
  # flatten the key since it is trying to convert Trackable/Layer to a string.
  _TF_MODULE_IGNORED_PROPERTIES = frozenset(itertools.chain(
      ('_layer_call_argspecs',),
      base_layer.Layer._TF_MODULE_IGNORED_PROPERTIES
  ))

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

    tf_utils.assert_no_legacy_layers(self.layers)

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
  @trackable.no_automatic_dependency_tracking
  def _base_init(self, name=None, **kwargs):
    # The following are implemented as property functions:
    # self.trainable_weights
    # self.non_trainable_weights
    # self.input_spec
    # self.losses
    # self.updates

    generic_utils.validate_kwargs(kwargs, {'trainable', 'dtype', 'dynamic',
                                           'autocast'})

    # Object to store all thread local layer properties.
    self._thread_local = threading.local()

    self._init_set_name(name, zero_based=True)
    self._activity_regularizer = None
    # This acts just like the `trainable` attribute of any layer instance.
    self._trainable = kwargs.get('trainable', True)
    # This attribute has no effect if the model is created using the Functional
    # API. Instead, `model.dynamic` is determined based on the internal layers.
    self._dynamic = kwargs.get('dynamic', False)
    self._is_compiled = False
    self._layers = []

    # This is True for Sequential networks and Functional networks.
    self._compute_output_and_mask_jointly = False

    self.supports_masking = False
    if not hasattr(self, 'optimizer'):
      # Don't reset optimizer if already set.
      self.optimizer = None

    # Private attributes to implement compatibility with Layer.
    self._maybe_create_attribute('_trainable_weights', [])
    self._maybe_create_attribute('_non_trainable_weights', [])
    self._updates = []  # Used in symbolic mode only.
    self._losses = []
    self._callable_losses = []
    # A list of metric instances corresponding to the symbolic metric tensors
    # added using the `add_metric` API.
    self._metrics = []
    self._scope = None  # Never used.
    self._reuse = None  # Never used.
    if context.executing_eagerly():
      self._graph = None
    else:
      self._graph = ops.get_default_graph()  # Used in symbolic mode only.

    # Both graph and subclassed networks have a dtype policy. For graph
    # networks, the policy's compute and variable dtypes are ignored, but other
    # fields, like the loss scale, are used by Models. For subclassed networks,
    # the compute and variable dtypes are used as like any ordinary layer.
    self._set_dtype_policy(kwargs.get('dtype', None))

    # All layers in order of horizontal graph traversal.
    # Entries are unique. Includes input and output layers.
    self._maybe_create_attribute('_layers', [])

    # Used in symbolic mode only, only in conjunction with graph-networks
    self._outbound_nodes = []
    self._inbound_nodes = []

    self._trackable_saver = (
        trackable_utils.saver_with_op_caching(self))

  @trackable.no_automatic_dependency_tracking
  def _init_graph_network(self, inputs, outputs, name=None, **kwargs):
    generic_utils.validate_kwargs(
        kwargs, {'trainable'},
        'Functional models may only specify `name` and `trainable` keyword '
        'arguments during initialization. Got an unexpected argument:')
    # Normalize and set self.inputs, self.outputs.
    if isinstance(inputs, list) and len(nest.flatten(inputs)) == 1:
      inputs = inputs[0]
    if isinstance(outputs, list) and len(nest.flatten(outputs)) == 1:
      outputs = outputs[0]
    self._nested_outputs = outputs
    self._nested_inputs = inputs
    self.inputs = nest.flatten(inputs)
    self.outputs = nest.flatten(outputs)

    if any(not hasattr(tensor, '_keras_history') for tensor in self.outputs):
      base_layer_utils.create_keras_history(self._nested_outputs)

    self._base_init(name=name, **kwargs)
    self._validate_graph_inputs_and_outputs()

    # A Network does not create weights of its own, thus it is already
    # built.
    self.built = True
    self._compute_output_and_mask_jointly = True
    self._is_graph_network = True
    # `_expects_training_arg` is True since the `training` argument is always
    # present in the signature of the `call` method of a graph network.
    self._expects_training_arg = True
    self._expects_mask_arg = True
    # A graph network does not autocast inputs, as its layers will cast them
    # instead.
    self._autocast = False

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
    nodes, nodes_by_depth, layers, _ = _map_graph_network(
        self.inputs, self.outputs)
    self._network_nodes = nodes
    self._nodes_by_depth = nodes_by_depth
    self._layers = layers
    self._layer_call_argspecs = {}
    for layer in self._layers:
      self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)

    self._track_layers(layers)

    # Create the node linking internal inputs to internal outputs.
    node_module.Node(
        outbound_layer=self,
        inbound_layers=[],
        node_indices=[],
        tensor_indices=[],
        input_tensors=self._nested_inputs,
        output_tensors=self._nested_outputs)

    # Build self.input_names and self.output_names.
    self._set_output_names()
    self.input_names = []
    self._feed_input_names = []
    self._feed_inputs = []
    self._feed_input_shapes = []
    for i, layer in enumerate(self._input_layers):
      self.input_names.append(layer.name)
      if layer.is_placeholder:
        self._feed_input_names.append(layer.name)
        # Use batch_input_shape here because non-eager composite tensors may not
        # have a shape attribute that's meaningful (sparse, for instance, has
        # a tensor that's non-constant and needs to be fed). This means that
        # input layers that create placeholders will need to have the
        # batch_input_shape attr to allow for input shape validation.
        self._feed_input_shapes.append(layer._batch_input_shape)
        self._feed_inputs.append(layer.input)

  def _set_output_names(self):
    """Assigns unique names to the Network's outputs.

    Output layers with multiple output tensors would otherwise lead to duplicate
    names in self.output_names.
    """
    uniquified = []
    output_names = set()
    prefix_count = {}
    for layer in self._output_layers:
      proposal = layer.name
      while proposal in output_names:
        existing_count = prefix_count.get(layer.name, 1)
        proposal = '{}_{}'.format(layer.name, existing_count)
        prefix_count[layer.name] = existing_count + 1
      output_names.add(proposal)
      uniquified.append(proposal)
    self.output_names = uniquified

  @trackable.no_automatic_dependency_tracking
  def _init_subclassed_network(self, name=None, **kwargs):
    self._base_init(name=name, **kwargs)
    self._is_graph_network = False
    self._init_call_fn_args()
    self._autocast = kwargs.get('autocast',
                                base_layer_utils.v2_dtype_behavior_enabled())
    self.outputs = []
    self.inputs = []
    self.built = False

  @property
  def dynamic(self):
    if self._is_graph_network:
      return any(layer.dynamic for layer in self.layers)
    return self._dynamic or any(layer.dynamic for layer in self.layers)

  def _track_layers(self, layers):
    """Add Trackable dependencies on a list of Layers."""
    weight_layer_index = 0
    for layer_index, layer in enumerate(layers):
      try:
        if layer.weights:
          # Keep a separate index for layers which have weights. This allows
          # users to insert Layers without weights anywhere in the network
          # without breaking checkpoints.
          self._track_trackable(
              layer, name='layer_with_weights-%d' % weight_layer_index,
              overwrite=True)
          weight_layer_index += 1
      except ValueError:
        # The layer might have weights, but may not be built yet. We just treat
        # it as layer without weight.
        pass

      # Even if it doesn't have weights, we should still track everything in
      # case it has/will have Trackable dependencies.
      self._track_trackable(
          layer, name='layer-%d' % layer_index, overwrite=True)

  def __setattr__(self, name, value):
    if not getattr(self, '_self_setattr_tracking', True):
      super(Network, self).__setattr__(name, value)
      return

    if all(
        isinstance(v, (base_layer.Layer,
                       data_structures.TrackableDataStructure)) or
        trackable_layer_utils.has_weights(v) for v in nest.flatten(value)):
      try:
        self._is_graph_network
      except AttributeError:
        raise RuntimeError('It looks like you are subclassing `Model` and you '
                           'forgot to call `super(YourClass, self).__init__()`.'
                           ' Always start with this line.')

    super(Network, self).__setattr__(name, value)

    # Keep track of metric instance created in subclassed model/layer.
    # We do this so that we can maintain the correct order of metrics by adding
    # the instance to the `metrics` list as soon as it is created.
    from tensorflow.python.keras import metrics as metrics_module  # pylint: disable=g-import-not-at-top
    if isinstance(value, metrics_module.Metric):
      self._metrics.append(value)

  @property
  def stateful(self):
    return any((hasattr(layer, 'stateful') and layer.stateful)
               for layer in self.layers)

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

  @property
  def weights(self):
    """Returns the list of all layer variables/weights.

    Returns:
      A list of variables.
    """
    self._assert_weights_created()
    weights = []
    for layer in self._layers:
      weights += layer.weights
    weights += (self._trainable_weights + self._non_trainable_weights)
    return weights

  @property
  @tracking.cached_per_instance
  def _should_compute_mask(self):
    return self._is_graph_network and super(Network, self)._should_compute_mask

  def compute_mask(self, inputs, mask):
    if not self._is_graph_network:
      return None

    # TODO(omalleyt): b/123540974 This function is not really safe to call
    # by itself because it will duplicate any updates and losses in graph
    # mode by `call`ing the Layers again.
    output_tensors = self._run_internal_graph(inputs, mask=mask)
    return nest.map_structure(lambda t: t._keras_mask, output_tensors)

  @property
  def layers(self):
    return trackable_layer_utils.filter_empty_layer_containers(
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
  def trainable_weights(self):
    self._assert_weights_created()
    return trackable_layer_utils.gather_trainable_weights(
        trainable=self.trainable,
        sub_layers=self._layers,
        extra_variables=self._trainable_weights)

  @property
  def non_trainable_weights(self):
    self._assert_weights_created()
    return trackable_layer_utils.gather_non_trainable_weights(
        trainable=self.trainable,
        sub_layers=self._layers,
        extra_variables=self._non_trainable_weights + self._trainable_weights)

  @property
  def input_spec(self):
    """Gets the network's input specs.

    Returns:
        A list of `InputSpec` instances (one per input to the model)
            or a single instance if the model has only one input.
    """
    # If subclassed model, can't assume anything.
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

  @base_layer_utils.default
  def build(self, input_shape):
    """Builds the model based on input shapes received.

    This is to be used for subclassed models, which do not know at instantiation
    time what their inputs look like.

    This method only exists for users who want to call `model.build()` in a
    standalone way (as a substitute for calling the model on real data to
    build it). It will never be called by the framework (and thus it will
    never throw unexpected errors in an unrelated workflow).

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
      if context.executing_eagerly():
        graph = func_graph.FuncGraph('build_graph')
      else:
        graph = backend.get_graph()
      with graph.as_default():
        if isinstance(input_shape, list):
          x = [base_layer_utils.generate_placeholders_from_shape(shape)
               for shape in input_shape]
        else:
          x = base_layer_utils.generate_placeholders_from_shape(input_shape)

        kwargs = {}
        call_signature = tf_inspect.getfullargspec(self.call)
        call_args = call_signature.args
        # Exclude `self`, `inputs`, and any argument with a default value.
        if len(call_args) > 2:
          if call_signature.defaults:
            call_args = call_args[2:-len(call_signature.defaults)]
          else:
            call_args = call_args[2:]
          for arg in call_args:
            if arg == 'training':
              # Case where `training` is a positional arg with no default.
              kwargs['training'] = False
            else:
              # Has invalid call signature with unknown positional arguments.
              raise ValueError(
                  'Currently, you cannot build your model if it has '
                  'positional or keyword arguments that are not '
                  'inputs to the model, but are required for its '
                  '`call` method. Instead, in order to instantiate '
                  'and build your model, `call` your model on real '
                  'tensor data with all expected call arguments.')
        elif len(call_args) < 2:
          # Signature without `inputs`.
          raise ValueError('You can only call `build` on a model if its `call` '
                           'method accepts an `inputs` argument.')
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

    return self._run_internal_graph(inputs, training=training, mask=mask)

  def compute_output_shape(self, input_shape):
    if not self._is_graph_network:
      return super(Network, self).compute_output_shape(input_shape)

    # Convert any shapes in tuple format to TensorShapes.
    input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)

    if len(nest.flatten(input_shape)) != len(nest.flatten(self._input_layers)):
      raise ValueError('Invalid input_shape argument ' + str(input_shape) +
                       ': model has ' + str(len(self._input_layers)) +
                       ' tensor inputs.')

    cache_key = generic_utils.object_list_uid(input_shape)
    if cache_key in self._output_shape_cache:
      # Cache hit. Return shapes as TensorShapes.
      return self._output_shape_cache[cache_key]

    layers_to_output_shapes = {}
    for layer, shape in zip(self._input_layers, nest.flatten(input_shape)):
      # It's an input layer: then `compute_output_shape` is identity,
      # and there is only one node and one tensor..
      shape_key = layer.name + '_0_0'
      layers_to_output_shapes[shape_key] = shape

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
          layer_input_shapes = []
          for inbound_layer, node_id, tensor_id, _ in node.iterate_inbound():
            input_layer_key = inbound_layer.name + '_%s_%s' % (node_id,
                                                               tensor_id)
            layer_input_shapes.append(layers_to_output_shapes[input_layer_key])
          layer_input_shapes = nest.pack_sequence_as(node.inbound_layers,
                                                     layer_input_shapes)
          # Layers expect shapes to be tuples for `compute_output_shape`.
          layer_input_shapes = tf_utils.convert_shapes(
              layer_input_shapes, to_tuples=True)
          layer_output_shapes = layer.compute_output_shape(layer_input_shapes)
          # Convert back to TensorShapes.
          layer_output_shapes = tf_utils.convert_shapes(
              layer_output_shapes, to_tuples=False)

          node_index = layer._inbound_nodes.index(node)  # pylint: disable=protected-access
          for j, shape in enumerate(nest.flatten(layer_output_shapes)):
            shape_key = layer.name + '_%s_%s' % (node_index, j)
            layers_to_output_shapes[shape_key] = shape

      # Read final output shapes from layers_to_output_shapes.
      output_shapes = []
      for i in range(len(self._output_layers)):
        layer, node_index, tensor_index = self._output_coordinates[i]
        shape_key = layer.name + '_%s_%s' % (node_index, tensor_index)
        output_shapes.append(layers_to_output_shapes[shape_key])
      output_shapes = nest.pack_sequence_as(self._nested_outputs, output_shapes)
      # Store in cache.
      self._output_shape_cache[cache_key] = output_shapes

    # Return shapes as TensorShapes.
    return output_shapes

  def _run_internal_graph(self, inputs, training=None, mask=None):
    """Computes output tensors for new inputs.

    # Note:
        - Can be run on non-Keras tensors.

    Arguments:
        inputs: Tensor or nested structure of Tensors.
        training: Boolean learning phase.
        mask: (Optional) Tensor or nested structure of Tensors.

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
    inputs = nest.flatten(inputs)
    if mask is None:
      masks = [None for _ in range(len(inputs))]
    else:
      masks = nest.flatten(mask)

    for input_t, mask in zip(inputs, masks):
      input_t._keras_mask = mask

    # Dictionary mapping reference tensors to computed tensors.
    tensor_dict = {}

    for x, y in zip(self.inputs, inputs):
      tensor_dict[str(id(x))] = y

    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    # Ignore the InputLayers when computing the graph.
    depth_keys = depth_keys[1:]

    for depth in depth_keys:
      nodes = self._nodes_by_depth[depth]
      for node in nodes:
        # This is always a single layer, never a list.
        layer = node.outbound_layer

        if all(
            str(id(tensor)) in tensor_dict
            for tensor in nest.flatten(node.input_tensors)):

          # Call layer (reapplying ops to new inputs).
          computed_tensors = nest.map_structure(
              lambda t: tensor_dict[str(id(t))], node.input_tensors)

          # Ensure `training` arg propagation if applicable.
          kwargs = copy.copy(node.arguments) if node.arguments else {}
          argspec = self._layer_call_argspecs[layer].args
          if 'training' in argspec:
            kwargs.setdefault('training', training)
            if (type(kwargs['training']) is ops.Tensor and  # pylint: disable=unidiomatic-typecheck
                any([kwargs['training'] is x
                     for x in backend._GRAPH_LEARNING_PHASES.values()])):
              kwargs['training'] = training  # Materialize placeholder.

          # Map Keras tensors in kwargs to their computed value.
          def _map_tensor_if_from_keras_layer(t):
            if isinstance(t, ops.Tensor) and hasattr(t, '_keras_history'):
              t_id = str(id(t))
              return tensor_dict[t_id]
            return t

          kwargs = nest.map_structure(_map_tensor_if_from_keras_layer, kwargs)

          # Compute outputs.
          output_tensors = layer(computed_tensors, **kwargs)

          # Update tensor_dict.
          for x, y in zip(
              nest.flatten(node.output_tensors), nest.flatten(output_tensors)):
            tensor_dict[str(id(x))] = y

    output_tensors = []
    output_shapes = []
    for x in self.outputs:
      assert str(id(x)) in tensor_dict, 'Could not compute output ' + str(x)
      tensor = tensor_dict[str(id(x))]
      output_shapes.append(x.shape)
      output_tensors.append(tensor)

    if output_shapes is not None:
      input_shapes = [x.shape for x in inputs]
      cache_key = generic_utils.object_list_uid(input_shapes)
      self._output_shape_cache[cache_key] = nest.pack_sequence_as(
          self._nested_outputs, output_shapes)

    output_tensors = nest.pack_sequence_as(self._nested_outputs, output_tensors)
    return output_tensors

  def get_config(self):
    if not self._is_graph_network:
      raise NotImplementedError

    config = {
        'name': self.name,
    }
    node_conversion_map = {}
    for layer in self.layers:
      kept_nodes = 1 if _should_skip_first_node(layer) else 0
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
            kwargs = _serialize_tensors(node.arguments)
            try:
              json.dumps(kwargs)
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
            for inbound_layer, node_id, tensor_id, _ in node.iterate_inbound():
              node_key = _make_node_key(inbound_layer.name, node_id)
              new_node_index = node_conversion_map.get(node_key, 0)
              node_data.append(
                  tf_utils.ListWrapper(
                      [inbound_layer.name, new_node_index, tensor_id, kwargs]))
            node_data = nest.pack_sequence_as(node.input_tensors, node_data)
            if not nest.is_sequence(node_data):
              node_data = [node_data]
            # Convert ListWrapper to list for backwards compatible configs.
            node_data = tf_utils.convert_inner_node_data(node_data)
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
      model_inputs.append(
          tf_utils.ListWrapper([layer.name, new_node_index, tensor_index]))
    model_inputs = nest.pack_sequence_as(self._nested_inputs, model_inputs)
    # Preserve external Keras compat for Models with single input.
    if not nest.is_sequence(model_inputs):
      model_inputs = [model_inputs]
    model_inputs = tf_utils.convert_inner_node_data(model_inputs)
    config['input_layers'] = model_inputs

    model_outputs = []
    for i in range(len(self._output_layers)):
      layer, node_index, tensor_index = self._output_coordinates[i]
      node_key = _make_node_key(layer.name, node_index)
      if node_key not in self._network_nodes:
        continue
      new_node_index = node_conversion_map[node_key]
      model_outputs.append(
          tf_utils.ListWrapper([layer.name, new_node_index, tensor_index]))
    model_outputs = nest.pack_sequence_as(self._nested_outputs, model_outputs)
    # Preserve external Keras compat for Models with single output.
    if not nest.is_sequence(model_outputs):
      model_outputs = [model_outputs]
    model_outputs = tf_utils.convert_inner_node_data(model_outputs)
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
    # Layer instances created during the graph reconstruction process.
    created_layers = collections.OrderedDict()

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
          node_data: Nested structure of `ListWrapper`.

      Raises:
          ValueError: In case of improperly formatted `node_data`.
      """
      input_tensors = []
      for input_data in nest.flatten(node_data):
        input_data = input_data.as_list()
        inbound_layer_name = input_data[0]
        inbound_node_index = input_data[1]
        inbound_tensor_index = input_data[2]
        if len(input_data) == 3:
          kwargs = {}
        elif len(input_data) == 4:
          kwargs = input_data[3]
          kwargs = _deserialize_keras_tensors(kwargs, created_layers)
        else:
          raise ValueError('Improperly formatted model config.')

        inbound_layer = created_layers[inbound_layer_name]
        if len(inbound_layer._inbound_nodes) <= inbound_node_index:
          add_unprocessed_node(layer, node_data)
          return
        inbound_node = inbound_layer._inbound_nodes[inbound_node_index]
        input_tensors.append(
            nest.flatten(inbound_node.output_tensors)[inbound_tensor_index])
      input_tensors = nest.pack_sequence_as(node_data, input_tensors)
      # Call layer on its inputs, thus creating the node
      # and building the layer if needed.
      if input_tensors is not None:
        # Preserve compatibility with older configs
        flat_input_tensors = nest.flatten(input_tensors)
        # If this is a single element but not a dict, unwrap. If this is a dict,
        # assume the first layer expects a dict (as is the case with a
        # DenseFeatures layer); pass through.
        if not isinstance(input_tensors, dict) and len(flat_input_tensors) == 1:
          input_tensors = flat_input_tensors[0]
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

      # Gather layer inputs and convert to `ListWrapper` objects.
      inbound_nodes_data = layer_data['inbound_nodes']
      inbound_nodes_data = tf_utils.convert_inner_node_data(
          inbound_nodes_data, wrap=True)
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

    input_layers = tf_utils.convert_inner_node_data(
        config['input_layers'], wrap=True)
    for layer_data in nest.flatten(input_layers):
      layer_name, node_index, tensor_index = layer_data.as_list()
      assert layer_name in created_layers
      layer = created_layers[layer_name]
      layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
      input_tensors.append(nest.flatten(layer_output_tensors)[tensor_index])

    output_layers = tf_utils.convert_inner_node_data(
        config['output_layers'], wrap=True)
    for layer_data in nest.flatten(output_layers):
      layer_name, node_index, tensor_index = layer_data.as_list()
      assert layer_name in created_layers
      layer = created_layers[layer_name]
      layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
      output_tensors.append(nest.flatten(layer_output_tensors)[tensor_index])

    input_tensors = nest.pack_sequence_as(input_layers, input_tensors)
    output_tensors = nest.pack_sequence_as(output_layers, output_tensors)
    model = cls(inputs=input_tensors, outputs=output_tensors, name=name)

    # Layers not connected to outputs, such as those added in `add_loss`.
    ancillary_layers = [
        layer for layer in created_layers.values() if layer not in model.layers
    ]
    if ancillary_layers:
      relevant_nodes = nest.flatten([
          layer.inbound_nodes[1:]
          if _should_skip_first_node(layer) else layer.inbound_nodes
          for layer in created_layers.values()
      ])
      model._insert_layers(ancillary_layers, relevant_nodes)
    return model

  def save(self,
           filepath,
           overwrite=True,
           include_optimizer=True,
           save_format=None,
           signatures=None):
    """Saves the model to Tensorflow SavedModel or a single HDF5 file.

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
        filepath: String, path to SavedModel or H5 file to save the model.
        overwrite: Whether to silently overwrite any existing file at the
            target location, or provide the user with a manual prompt.
        include_optimizer: If True, save optimizer's state together.
        save_format: Either 'tf' or 'h5', indicating whether to save the model
          to Tensorflow SavedModel or HDF5. The default is currently 'h5', but
          will switch to 'tf' in TensorFlow 2.0. The 'tf' option is currently
          disabled (use `tf.keras.experimental.export_saved_model` instead).
      signatures: Signatures to save with the SavedModel. Applicable to the 'tf'
        format only. Please see the `signatures` argument in
        `tf.saved_model.save` for details.

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
    saving.save_model(self, filepath, overwrite, include_optimizer, save_format,
                      signatures)

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

    While the formats are the same, do not mix `save_weights` and
    `tf.train.Checkpoint`. Checkpoints saved by `Model.save_weights` should be
    loaded using `Model.load_weights`. Checkpoints saved using
    `tf.train.Checkpoint.save` should be restored using the corresponding
    `tf.train.Checkpoint.restore`. Prefer `tf.train.Checkpoint` over
    `save_weights` for training checkpoints.

    The TensorFlow format matches objects and variables by starting at a root
    object, `self` for `save_weights`, and greedily matching attribute
    names. For `Model.save` this is the `Model`, and for `Checkpoint.save` this
    is the `Checkpoint` even if the `Checkpoint` has a model attached. This
    means saving a `tf.keras.Model` using `save_weights` and loading into a
    `tf.train.Checkpoint` with a `Model` attached (or vice versa) will not match
    the `Model`'s variables. See the [guide to training
    checkpoints](https://www.tensorflow.org/alpha/guide/checkpoints) for details
    on the TensorFlow format.

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
    self._assert_weights_created()
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
          and not isinstance(optimizer, trackable.Trackable)):
        logging.warning(
            ('This model was compiled with a Keras optimizer (%s) but is being '
             'saved in TensorFlow format with `save_weights`. The model\'s '
             'weights will be saved, but unlike with TensorFlow optimizers in '
             'the TensorFlow format the optimizer\'s state will not be '
             'saved.\n\nConsider using a TensorFlow optimizer from `tf.train`.')
            % (optimizer,))
      self._trackable_saver.save(filepath, session=session)
      # Record this checkpoint so it's visible from tf.train.latest_checkpoint.
      checkpoint_management.update_checkpoint_state_internal(
          save_dir=os.path.dirname(filepath),
          model_checkpoint_path=filepath,
          save_relative_paths=True,
          all_model_checkpoint_paths=[filepath])

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
      status = self._trackable_saver.restore(filepath)
      if by_name:
        raise NotImplementedError(
            'Weights may only be loaded based on topology into Models when '
            'loading TensorFlow-formatted weights (got by_name=True to '
            'load_weights).')
      if not context.executing_eagerly():
        session = backend.get_session()
        # Restore existing variables (if any) immediately, and set up a
        # streaming restore for any variables created in the future.
        trackable_utils.streaming_restore(status=status, session=session)
      status.assert_nontrivial_match()
      return status
    if h5py is None:
      raise ImportError(
          '`load_weights` requires h5py when loading weights from HDF5.')
    if self._is_graph_network and not self.built:
      raise NotImplementedError(
          'Unable to load weights saved in HDF5 format into a subclassed '
          'Model which has not created its variables yet. Call the Model '
          'first, then load the weights.')
    self._assert_weights_created()
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
    model_config = self._updated_config()
    return json.dumps(
        model_config, default=serialization.get_json_type, **kwargs)

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
      raise ValueError('This model has not yet been built. '
                       'Build the model first by calling `build()` or calling '
                       '`fit()` with some data, or specify '
                       'an `input_shape` argument in the first layer(s) for '
                       'automatic build.')
    layer_utils.print_summary(self,
                              line_length=line_length,
                              positions=positions,
                              print_fn=print_fn)

  def _validate_graph_inputs_and_outputs(self):
    """Validates the inputs and outputs of a Graph Network."""
    # Check for redundancy in inputs.
    if len(object_identity.ObjectIdentitySet(self.inputs)) != len(self.inputs):
      raise ValueError('The list of inputs passed to the model '
                       'is redundant. '
                       'All inputs should only appear once.'
                       ' Found: ' + str(self.inputs))

    for x in self.inputs:
      # Check that x has appropriate `_keras_history` metadata.
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Input tensors to a ' + cls_name + ' ' +
                         'must come from `tf.keras.Input`. '
                         'Received: ' + str(x) +
                         ' (missing previous layer metadata).')
      # Check that x is an input tensor.
      # pylint: disable=protected-access
      layer = x._keras_history.layer
      if len(layer._inbound_nodes) > 1 or (
          layer._inbound_nodes and layer._inbound_nodes[0].inbound_layers):
        cls_name = self.__class__.__name__
        logging.warning(cls_name + ' inputs must come from '
                        '`tf.keras.Input` (thus holding past layer metadata), '
                        'they cannot be the output of '
                        'a previous non-Input layer. '
                        'Here, a tensor specified as '
                        'input to "' + self.name + '" was not an Input tensor, '
                        'it was generated by layer ' + layer.name + '.\n'
                        'Note that input tensors are '
                        'instantiated via `tensor = tf.keras.Input(shape)`.\n'
                        'The tensor that caused the issue was: ' + str(x.name))

    # Check compatibility of batch sizes of Input Layers.
    input_batch_sizes = [
        training_utils.get_static_batch_size(x._keras_history.layer)
        for x in self.inputs
    ]
    consistent_batch_size = None
    for batch_size in input_batch_sizes:
      if batch_size is not None:
        if (consistent_batch_size is not None and
            batch_size != consistent_batch_size):
          raise ValueError('The specified batch sizes of the Input Layers'
                           ' are incompatible. Found batch sizes: {}'.format(
                               input_batch_sizes))
        consistent_batch_size = batch_size

    for x in self.outputs:
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Output tensors to a ' + cls_name + ' must be '
                         'the output of a TensorFlow `Layer` '
                         '(thus holding past layer metadata). Found: ' + str(x))

  def _insert_layers(self, layers, relevant_nodes=None):
    """Inserts Layers into the Network after Network creation.

    This is only valid for Keras Graph Networks.  Layers added via this function
    will be included in the `call` computation and `get_config` of this Network.
    They will not be added to the Network's outputs.


    Arguments:
      layers: Arbitrary nested structure of Layers. Layers must be reachable
        from one or more of the `keras.Input` Tensors that correspond to this
        Network's inputs.
      relevant_nodes: Nodes from the Layers that should be considered part of
        this Network. If `None`, all Nodes will be considered part of this
        Network.

    Raises:
      ValueError: If the layers depend on `Input`s not found in this Model.
    """
    layers = nest.flatten(layers)
    tf_utils.assert_no_legacy_layers(layers)
    node_to_depth = {}
    for depth, nodes in self._nodes_by_depth.items():
      node_to_depth.update({node: depth for node in nodes})
    # The nodes of these Layers that are relevant to this Network. If not
    # provided, assume all Nodes are relevant
    if not relevant_nodes:
      relevant_nodes = nest.flatten([layer._inbound_nodes for layer in layers])
    network_nodes = set(relevant_nodes + list(node_to_depth.keys()))

    def _get_min_depth(node):
      """Gets the minimum depth at which node can be computed."""
      min_depth = 0
      for layer, node_id, _, _ in node.iterate_inbound(include_arguments=True):
        inbound_node = layer._inbound_nodes[node_id]
        if inbound_node in node_to_depth:
          min_depth = min(min_depth, node_to_depth[inbound_node])
        elif inbound_node not in network_nodes:
          continue
        else:
          # Previous relevant nodes haven't been processed yet.
          return None
      # New node is one shallower than its shallowest input.
      return min_depth - 1

    # Insert nodes into `_nodes_by_depth` and other node attrs.
    unprocessed_nodes = copy.copy(relevant_nodes)
    i = 0
    while unprocessed_nodes:
      i += 1
      # Do a sanity check. This can occur if `Input`s from outside this Model
      # are being relied on.
      if i > 10000:
        raise ValueError('Layers could not be added due to missing '
                         'dependencies.')

      node = unprocessed_nodes.pop(0)
      depth = _get_min_depth(node)
      if depth is None:  # Defer until inbound nodes are processed.
        unprocessed_nodes.append(node)
        continue
      node_key = _make_node_key(node.outbound_layer.name,
                                node.outbound_layer._inbound_nodes.index(node))
      if node_key not in self._network_nodes:
        node_to_depth[node] = depth
        self._network_nodes.add(node_key)
        self._nodes_by_depth[depth].append(node)

    # Insert layers and update other layer attrs.
    layer_set = set(self._layers)
    for layer in layers:
      if layer not in layer_set:
        self._layers.append(layer)
        self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)
        layer_set.add(layer)

  def _assert_weights_created(self):
    """Asserts that all the weights for the network have been created.

    For a non-dynamic network, the weights must already be created after the
    layer has been called. For a dynamic network, the exact list of weights can
    never be known for certain since it may change at any time during execution.

    We run this check right before accessing weights or getting the Numpy value
    for the current weights. Otherwise, if the layer has never been called,
    the user would just get an empty list, which is misleading.

    Raises:
      ValueError: if the weights of the network has not yet been created.
    """
    if self.dynamic:
      return
    if (not self._is_graph_network and
        'build' in self.__class__.__dict__ and
        not self.built):
      # For any model that has customized build() method but hasn't
      # been invoked yet, this will cover both sequential and subclass model.
      raise ValueError('Weights for model %s have not yet been created. '
                       'Weights are created when the Model is first called on '
                       'inputs or `build()` is called with an `input_shape`.' %
                       self.name)

  @property
  def _object_identifier(self):
    return '_tf_keras_network'

  def _graph_network_add_loss(self, symbolic_loss):
    new_nodes, new_layers = _map_subgraph_network(self.inputs, [symbolic_loss])
    # Losses must be keyed on inputs no matter what in order to be supported in
    # DistributionStrategy.
    add_loss_layer = base_layer.AddLoss(unconditional=False)
    add_loss_layer(symbolic_loss)
    new_nodes.extend(add_loss_layer.inbound_nodes)
    new_layers.append(add_loss_layer)
    self._insert_layers(new_layers, new_nodes)

  def _graph_network_add_metric(self, value, aggregation, name):
    new_nodes, new_layers = _map_subgraph_network(self.inputs, [value])
    add_metric_layer = base_layer.AddMetric(aggregation, name)
    add_metric_layer(value)
    new_nodes.extend(add_metric_layer.inbound_nodes)
    new_layers.append(add_metric_layer)
    self._insert_layers(new_layers, new_nodes)


def _is_hdf5_filepath(filepath):
  return (filepath.endswith('.h5') or filepath.endswith('.keras') or
          filepath.endswith('.hdf5'))


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
    for layer, node_index, tensor_index, tensor in node.iterate_inbound(
        include_arguments=True):
      build_map(tensor, finished_nodes, nodes_in_progress, layer, node_index,
                tensor_index)

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
    # of all nodes it is connected to + 1.
    for node_dep in node._get_all_node_dependencies():
      previous_depth = nodes_depths.get(node_dep, 0)
      nodes_depths[node_dep] = max(depth + 1, previous_depth)

  # Handle inputs that are not connected to outputs.
  # We do not error out here because the inputs may be used to compute losses
  # and metrics.
  for input_t in inputs:
    input_layer = input_t._keras_history[0]
    if input_layer not in layers_depths:
      layers_depths[input_layer] = 0
      layer_indices[input_layer] = -1
      nodes_depths[input_layer._inbound_nodes[0]] = 0
      network_nodes.add(_make_node_key(input_layer.name, 0))

  # Build a dict {depth: list of nodes with this depth}
  nodes_by_depth = collections.defaultdict(list)
  for node, depth in nodes_depths.items():
    nodes_by_depth[depth].append(node)

  # Build a dict {depth: list of layers with this depth}
  layers_by_depth = collections.defaultdict(list)
  for layer, depth in layers_depths.items():
    layers_by_depth[depth].append(layer)

  # Get sorted list of layer depths.
  depth_keys = list(layers_by_depth.keys())
  depth_keys.sort(reverse=True)

  # Set self.layers ordered by depth.
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
  computable_tensors = object_identity.ObjectIdentitySet()
  for x in inputs:
    computable_tensors.add(x)

  layers_with_complete_input = []  # To provide a better error msg.
  for depth in depth_keys:
    for node in nodes_by_depth[depth]:
      layer = node.outbound_layer
      if layer:
        for x in nest.flatten(node.input_tensors):
          if x not in computable_tensors:
            raise ValueError('Graph disconnected: '
                             'cannot obtain value for tensor ' + str(x) +
                             ' at layer "' + layer.name + '". '
                             'The following previous layers '
                             'were accessed without issue: ' +
                             str(layers_with_complete_input))
        for x in nest.flatten(node.output_tensors):
          computable_tensors.add(x)
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


def _map_subgraph_network(inputs, outputs):
  """Returns the nodes and layers in the topology from `inputs` to `outputs`.

  Args:
    inputs: List of input tensors.
    outputs: List of output tensors.

  Returns:
    A tuple of List{Node] and List[Layer].
  """
  base_layer_utils.create_keras_history(outputs)
  # Keep only nodes and layers in the topology betweeen inputs and outputs.
  _, nodes_by_depth, layers, _ = _map_graph_network(inputs, outputs)
  return nest.flatten([nodes for nodes in nodes_by_depth.values()]), layers


def _should_skip_first_node(layer):
  """Returns True if the first layer node should not be saved or loaded."""
  # Networks start with a pre-existing node linking their input to output.
  return issubclass(layer.__class__, Network) and layer._is_graph_network


def _serialize_tensors(kwargs):
  """Serializes Tensors passed to `call`."""

  def _serialize_keras_tensor(t):
    """Serializes a single Tensor passed to `call`."""
    if hasattr(t, '_keras_history'):
      kh = t._keras_history
      return [kh.layer.name, kh.node_index, kh.tensor_index]

    if isinstance(t, np.ndarray):
      return t.tolist()

    if isinstance(t, ops.Tensor):
      return backend.get_value(t).tolist()

    return t

  return nest.map_structure(_serialize_keras_tensor, kwargs)


def _deserialize_keras_tensors(kwargs, layer_map):
  """Deserializes Keras Tensors passed to `call`.."""

  def _deserialize_keras_tensor(t):
    """Deserializes a single Keras Tensor passed to `call`."""
    if isinstance(t, tf_utils.ListWrapper):
      t = t.as_list()
      layer_name = t[0]
      node_index = t[1]
      tensor_index = t[2]

      layer = layer_map[layer_name]
      node = layer._inbound_nodes[node_index]
      return nest.flatten(node.output_tensors)[tensor_index]
    return t

  kwargs = tf_utils.convert_inner_node_data(kwargs, wrap=True)
  return nest.map_structure(_deserialize_keras_tensor, kwargs)
