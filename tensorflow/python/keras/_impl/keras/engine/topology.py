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
"""Base layer code and base model (Network) code.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os

import numpy as np
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import constraints
from tensorflow.python.keras._impl.keras import initializers
from tensorflow.python.keras._impl.keras import regularizers
from tensorflow.python.keras._impl.keras.utils import conv_utils
from tensorflow.python.keras._impl.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.keras._impl.keras.utils.layer_utils import print_summary as print_layer_summary
from tensorflow.python.layers import base as tf_base_layers
from tensorflow.python.layers import network as tf_network
from tensorflow.python.layers import utils as tf_layers_util
from tensorflow.python.platform import tf_logging as logging


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

# pylint: disable=invalid-name
InputSpec = tf_base_layers.InputSpec
Node = tf_base_layers.Node
TFBaseLayer = tf_base_layers.Layer
# pylint: enable=invalid-name


class Layer(tf_base_layers.Layer):
  """Abstract base layer class.

  # Properties
      name: String, must be unique within a model.
      input_spec: List of InputSpec class instances
          each entry describes one required input:
              - ndim
              - dtype
          A layer with `n` input tensors must have
          an `input_spec` of length `n`.
      trainable: Boolean, whether the layer weights
          will be updated during training.
      uses_learning_phase: Whether any operation
          of the layer uses `K.in_training_phase()`
          or `K.in_test_phase()`.
      input_shape: Shape tuple. Provided for convenience,
          but note that there may be cases in which this
          attribute is ill-defined (e.g. a shared layer
          with multiple input shapes), in which case
          requesting `input_shape` will raise an Exception.
          Prefer using `layer.get_input_shape_for(input_shape)`,
          or `layer.get_input_shape_at(node_index)`.
      output_shape: Shape tuple. See above.
      inbound_nodes: List of nodes.
      outbound_nodes: List of nodes.
      input, output: Input/output tensor(s). Note that if the layer is used
          more than once (shared layer), this is ill-defined
          and will raise an exception. In such cases, use
          `layer.get_input_at(node_index)`.
      input_mask, output_mask: Same as above, for masks.
      trainable_weights: List of variables.
      non_trainable_weights: List of variables.
      weights: The concatenation of the lists trainable_weights and
          non_trainable_weights (in this order).

  # Methods
      call(x, mask=None): Where the layer's logic lives.
      __call__(x, mask=None): Wrapper around the layer logic (`call`).
          If x is a Keras tensor:
              - Connect current layer with last layer from tensor:
                  `self._add_inbound_node(last_layer)`
              - Add layer to tensor history
          If layer is not built:
              - Build from inputs shape
      get_weights()
      set_weights(weights)
      get_config()
      count_params()
      _compute_output_shape(input_shape)
      compute_mask(x, mask)
      get_input_at(node_index)
      get_output_at(node_index)
      get_input_shape_at(node_index)
      get_output_shape_at(node_index)
      get_input_mask_at(node_index)
      get_output_mask_at(node_index)

  # Class Methods
      from_config(config)

  # Internal methods:
      build(input_shape)
      _add_inbound_node(layer, index=0)
  """

  def __init__(self, **kwargs):
    # These properties should be set by the user via keyword arguments.
    # note that 'dtype', 'input_shape' and 'batch_input_shape'
    # are only applicable to input layers: do not pass these keywords
    # to non-input layers.
    allowed_kwargs = {
        'activity_regularizer',
        'input_shape',
        'batch_input_shape',
        'batch_size',
        'dtype',
        'name',
        'trainable',
        'weights',
    }
    # Validate optional keyword arguments.
    for kwarg in kwargs:
      if kwarg not in allowed_kwargs:
        raise TypeError('Keyword argument not understood:', kwarg)

    # Get layer name.
    name = kwargs.get('name')

    # Get `trainable` status.
    trainable = kwargs.get('trainable', True)

    # Get `dtype`.
    dtype = kwargs.get('dtype')
    if dtype is None:
      dtype = K.floatx()

    # Call super, which will set all properties common to Keras layers
    # and core TF layers.
    super(Layer, self).__init__(
        name=name, dtype=dtype, trainable=trainable,
        activity_regularizer=kwargs.get('activity_regularizer'))

    # Add properties that are Keras-only for now.
    self.supports_masking = False

    # Manage input shape information if passed.
    if 'input_shape' in kwargs or 'batch_input_shape' in kwargs:
      # In this case we will later create an input layer
      # to insert before the current layer
      if 'batch_input_shape' in kwargs:
        batch_input_shape = tuple(kwargs['batch_input_shape'])
      elif 'input_shape' in kwargs:
        if 'batch_size' in kwargs:
          batch_size = kwargs['batch_size']
        else:
          batch_size = None
        batch_input_shape = (batch_size,) + tuple(kwargs['input_shape'])
      self._batch_input_shape = batch_input_shape

    # Manage initial weight values if passed.
    if 'weights' in kwargs:
      self._initial_weights = kwargs['weights']
    else:
      self._initial_weights = None

  def add_weight(self,
                 name,
                 shape,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 constraint=None):
    """Adds a weight variable to the layer.

    Arguments:
        name: String, the name for the weight variable.
        shape: The shape tuple of the weight.
        dtype: The dtype of the weight.
        initializer: An Initializer instance (callable).
        regularizer: An optional Regularizer instance.
        trainable: A boolean, whether the weight should
            be trained via backprop or not (assuming
            that the layer itself is also trainable).
        constraint: An optional Constraint instance.

    Returns:
        The created weight variable.
    """
    if dtype is None:
      dtype = K.floatx()
    weight = self.add_variable(name, shape,
                               dtype=dtype,
                               initializer=initializers.get(initializer),
                               regularizer=regularizers.get(regularizer),
                               constraint=constraints.get(constraint),
                               trainable=trainable)
    return weight

  def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
    """This is where the layer's logic lives.

    Arguments:
        inputs: Input tensor, or list/tuple of input tensors.
        **kwargs: Additional keyword arguments.

    Returns:
        A tensor or list/tuple of tensors.
    """
    return inputs

  def __call__(self, inputs, **kwargs):
    """Wrapper around self.call(), for handling internal references.

    If a Keras tensor is passed:
        - We call self._add_inbound_node().
        - If necessary, we `build` the layer to match
            the shape of the input(s).
        - We update the _keras_history of the output tensor(s)
            with the current layer.
            This is done as part of _add_inbound_node().

    Arguments:
        inputs: Can be a tensor or list/tuple of tensors.
        **kwargs: Additional keyword arguments to be passed to `call()`.

    Returns:
        Output of the layer's `call` method.

    Raises:
        ValueError: in case the layer is missing shape information
            for its `build` call.
    """
    # Actually call the layer (optionally building it).
    output = super(Layer, self).__call__(inputs, **kwargs)
    if context.in_eager_mode():
      return output

    # Update learning phase info.
    output_tensors = _to_list(output)
    uses_lp = any(
        [getattr(x, '_uses_learning_phase', False) for x in _to_list(inputs)])
    uses_lp = getattr(self, 'uses_learning_phase', False) or uses_lp
    for i in range(len(output_tensors)):
      output_tensors[i]._uses_learning_phase = getattr(
          output_tensors[i], '_uses_learning_phase', False) or uses_lp

    # Optionally load weight values that were specified at layer instantiation.
    if hasattr(self, '_initial_weights') and self._initial_weights is not None:
      self.set_weights(self._initial_weights)
      del self._initial_weights
    return output

  def _compute_output_shape(self, input_shape):
    """Computes the output shape of the layer.

    Assumes that the layer will be built
    to match that input shape provided.

    Arguments:
        input_shape: Shape tuple (tuple of integers)
            or list of shape tuples (one per output tensor of the layer).
            Shape tuples can include None for free dimensions,
            instead of an integer.

    Returns:
        An input shape tuple.
    """
    if isinstance(input_shape, list):
      return [tensor_shape.TensorShape(shape) for shape in input_shape]
    else:
      return tensor_shape.TensorShape(input_shape)

  def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
    """Computes an output mask tensor.

    Arguments:
        inputs: Tensor or list of tensors.
        mask: Tensor or list of tensors.

    Returns:
        None or a tensor (or list of tensors,
            one per output tensor of the layer).
    """
    if not self.supports_masking:
      if mask is not None:
        if isinstance(mask, list):
          if any(m is not None for m in mask):
            raise TypeError('Layer ' + self.name + ' does not support masking, '
                            'but was passed an input_mask: ' + str(mask))
        else:
          raise TypeError('Layer ' + self.name + ' does not support masking, '
                          'but was passed an input_mask: ' + str(mask))
      # masking not explicitly supported: return None as mask
      return None
    # if masking is explicitly supported, by default
    # carry over the input mask
    return mask

  def get_input_mask_at(self, node_index):
    """Retrieves the input mask tensor(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A mask tensor
        (or list of tensors if the layer has multiple inputs).
    """
    inputs = self.get_input_at(node_index)
    if isinstance(inputs, list):
      return [getattr(x, '_keras_mask', None) for x in inputs]
    else:
      return getattr(inputs, '_keras_mask', None)

  def get_output_mask_at(self, node_index):
    """Retrieves the output mask tensor(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A mask tensor
        (or list of tensors if the layer has multiple outputs).
    """
    output = self.get_output_at(node_index)
    if isinstance(output, list):
      return [getattr(x, '_keras_mask', None) for x in output]
    else:
      return getattr(output, '_keras_mask', None)

  @property
  def input_mask(self):
    """Retrieves the input mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
        Input mask tensor (potentially None) or list of input
        mask tensors.

    Raises:
        AttributeError: if the layer is connected to
        more than one incoming layers.
    """
    inputs = self.input
    if isinstance(inputs, list):
      return [getattr(x, '_keras_mask', None) for x in inputs]
    else:
      return getattr(inputs, '_keras_mask', None)

  @property
  def output_mask(self):
    """Retrieves the output mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
        Output mask tensor (potentially None) or list of output
        mask tensors.

    Raises:
        AttributeError: if the layer is connected to
        more than one incoming layers.
    """
    output = self.output
    if isinstance(output, list):
      return [getattr(x, '_keras_mask', None) for x in output]
    else:
      return getattr(output, '_keras_mask', None)

  def set_weights(self, weights):
    """Sets the weights of the layer, from Numpy arrays.

    Arguments:
        weights: a list of Numpy arrays. The number
            of arrays and their shape must match
            number of the dimensions of the weights
            of the layer (i.e. it should match the
            output of `get_weights`).

    Raises:
        ValueError: If the provided weights list does not match the
            layer's specifications.
    """
    params = self.weights
    if len(params) != len(weights):
      raise ValueError('You called `set_weights(weights)` on layer "' +
                       self.name + '" with a  weight list of length ' +
                       str(len(weights)) + ', but the layer was expecting ' +
                       str(len(params)) + ' weights. Provided weights: ' +
                       str(weights)[:50] + '...')
    if not params:
      return
    weight_value_tuples = []
    param_values = K.batch_get_value(params)
    for pv, p, w in zip(param_values, params, weights):
      if pv.shape != w.shape:
        raise ValueError('Layer weight shape ' + str(pv.shape) +
                         ' not compatible with '
                         'provided weight shape ' + str(w.shape))
      weight_value_tuples.append((p, w))
    K.batch_set_value(weight_value_tuples)

  def get_weights(self):
    """Returns the current weights of the layer.

    Returns:
        Weights values as a list of numpy arrays.
    """
    params = self.weights
    return K.batch_get_value(params)

  def get_config(self):
    """Returns the config of the layer.

    A layer config is a Python dictionary (serializable)
    containing the configuration of a layer.
    The same layer can be reinstantiated later
    (without its trained weights) from this configuration.

    The config of a layer does not include connectivity
    information, nor the layer class name. These are handled
    by `Network` (one layer of abstraction above).

    Returns:
        Python dictionary.
    """
    config = {'name': self.name, 'trainable': self.trainable}
    if hasattr(self, '_batch_input_shape'):
      config['batch_input_shape'] = self._batch_input_shape
    if hasattr(self, 'dtype'):
      config['dtype'] = self.dtype
    return config

  @classmethod
  def from_config(cls, config):
    """Creates a layer from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same layer from the config
    dictionary. It does not handle layer connectivity
    (handled by Network), nor weights (handled by `set_weights`).

    Arguments:
        config: A Python dictionary, typically the
            output of get_config.

    Returns:
        A layer instance.
    """
    return cls(**config)

  @tf_base_layers.Layer.activity_regularizer.setter
  def activity_regularizer(self, activity_regularizer):
    self._activity_regularizer = activity_regularizer


class InputLayer(tf_network.InputLayer, Layer):
  """Layer to be used as an entry point into a graph.

  It can either wrap an existing tensor (pass an `input_tensor` argument)
  or create its a placeholder tensor (pass argument `input_shape`.

  Arguments:
      input_shape: Shape tuple, not including the batch axis.
      batch_size: Optional input batch size (integer or None).
      dtype: Datatype of the input.
      input_tensor: Optional tensor to use as layer input
          instead of creating a placeholder.
      sparse: Boolean, whether the placeholder created
          is meant to be sparse.
      name: Name of the layer (string).
  """

  def __init__(self,
               input_shape=None,
               batch_size=None,
               dtype=None,
               input_tensor=None,
               sparse=False,
               name=None,
               **kwargs):
    if 'batch_input_shape' in kwargs:
      batch_input_shape = kwargs.pop('batch_input_shape')
      if input_shape and batch_input_shape:
        raise ValueError('Only provide the input_shape OR '
                         'batch_input_shape argument to '
                         'InputLayer, not both at the same time.')
      batch_size = batch_input_shape[0]
      input_shape = batch_input_shape[1:]
    if kwargs:
      raise ValueError('Unrecognized keyword arguments:', kwargs.keys())

    if not name:
      prefix = 'input'
      name = prefix + '_' + str(K.get_uid(prefix))

    if not dtype:
      if input_tensor is None:
        dtype = K.floatx()
      else:
        dtype = K.dtype(input_tensor)
    super(InputLayer, self).__init__(input_shape=input_shape,
                                     batch_size=batch_size,
                                     dtype=dtype,
                                     input_tensor=input_tensor,
                                     sparse=sparse,
                                     name=name)

  def get_config(self):
    config = {
        'batch_input_shape': self._batch_input_shape,
        'dtype': self.dtype,
        'sparse': self.sparse,
        'name': self.name
    }
    return config


def Input(  # pylint: disable=invalid-name
    shape=None,
    batch_size=None,
    name=None,
    dtype=None,
    sparse=False,
    tensor=None,
    **kwargs):
  """`Input()` is used to instantiate a Keras tensor.

  A Keras tensor is a tensor object from the underlying backend
  (Theano or TensorFlow), which we augment with certain
  attributes that allow us to build a Keras model
  just by knowing the inputs and outputs of the model.

  For instance, if a, b and c are Keras tensors,
  it becomes possible to do:
  `model = Model(input=[a, b], output=c)`

  The added Keras attribute is:
      `_keras_history`: Last layer applied to the tensor.
          the entire layer graph is retrievable from that layer,
          recursively.

  Arguments:
      shape: A shape tuple (integers), not including the batch size.
          For instance, `shape=(32,)` indicates that the expected input
          will be batches of 32-dimensional vectors.
      batch_size: optional static batch size (integer).
      name: An optional name string for the layer.
          Should be unique in a model (do not reuse the same name twice).
          It will be autogenerated if it isn't provided.
      dtype: The data type expected by the input, as a string
          (`float32`, `float64`, `int32`...)
      sparse: A boolean specifying whether the placeholder
          to be created is sparse.
      tensor: Optional existing tensor to wrap into the `Input` layer.
          If set, the layer will not create a placeholder tensor.
      **kwargs: deprecated arguments support.

  Returns:
      A tensor.

  Example:

      ```python
      # this is a logistic regression in Keras
      x = Input(shape=(32,))
      y = Dense(16, activation='softmax')(x)
      model = Model(x, y)
      ```

  Raises:
    ValueError: in case of invalid arguments.
  """
  if 'batch_shape' in kwargs:
    batch_shape = kwargs.pop('batch_shape')
    if shape and batch_shape:
      raise ValueError('Only provide the shape OR '
                       'batch_shape argument to '
                       'Input, not both at the same time.')
    batch_size = batch_shape[0]
    shape = batch_shape[1:]
  if kwargs:
    raise ValueError('Unrecognized keyword arguments:', kwargs.keys())

  if dtype is None:
    dtype = K.floatx()
  if not shape and tensor is None:
    raise ValueError('Please provide to Input either a `shape`'
                     ' or a `tensor` argument. Note that '
                     '`shape` does not include the batch '
                     'dimension.')
  input_layer = InputLayer(
      input_shape=shape,
      batch_size=batch_size,
      name=name,
      dtype=dtype,
      sparse=sparse,
      input_tensor=tensor)
  # Return tensor including `_keras_history`.
  # Note that in this case train_output and test_output are the same pointer.
  outputs = input_layer._inbound_nodes[0].output_tensors
  if len(outputs) == 1:
    return outputs[0]
  else:
    return outputs


class Network(tf_network.GraphNetwork, Layer):
  """A Network is a directed acyclic graph of layers.

  It is the topological form of a "model". A Model
  is simply a Network with added training routines.

  # Properties
      name
      inputs
      outputs
      input_layers
      output_layers
      input_spec (list of class instances)
          each entry describes one required input:
              - ndim
              - dtype
      trainable (boolean)
      input_shape
      output_shape
      inbound_nodes: list of nodes
      outbound_nodes: list of nodes
      trainable_weights (list of variables)
      non_trainable_weights (list of variables)

  # Methods
      summary
      get_layer
      get_weights
      set_weights
      get_config
      compute_output_shape

  # Class Methods
      from_config
  """

  def __init__(self, inputs, outputs, name=None):
    super(Network, self).__init__(inputs, outputs, name=name)

    self.supports_masking = False
    # Fill in the output mask cache.
    masks = []
    for x in self.inputs:
      mask = x._keras_mask if hasattr(x, '_keras_mask') else None
      masks.append(mask)
    mask_cache_key = (tf_layers_util.object_list_uid(self.inputs) + '_' +
                      tf_layers_util.object_list_uid(masks))
    masks = []
    for x in self.outputs:
      mask = x._keras_mask if hasattr(x, '_keras_mask') else None
      masks.append(mask)
    if len(masks) == 1:
      mask = masks[0]
    else:
      mask = masks
    self._output_mask_cache[mask_cache_key] = mask

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
        self._feed_inputs.append(layer.input)
        self._feed_input_shapes.append(K.int_shape(self.inputs[i]))
    for layer in self._output_layers:
      self.output_names.append(layer.name)

    self.internal_input_shapes = [K.int_shape(x) for x in self.inputs]
    self.internal_output_shapes = [K.int_shape(x) for x in self.outputs]

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
    return K.batch_get_value(weights)

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
    K.batch_set_value(tuples)

  def compute_mask(self, inputs, mask):
    inputs = _to_list(inputs)
    if mask is None:
      masks = [None for _ in range(len(inputs))]
    else:
      masks = _to_list(mask)
    cache_key = ','.join([str(id(x)) for x in inputs])
    cache_key += '_' + ','.join([str(id(x)) for x in masks])
    if cache_key in self._output_mask_cache:
      return self._output_mask_cache[cache_key]
    else:
      _, output_masks = self._run_internal_graph(inputs, masks)
      return output_masks

  def get_config(self):
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
        node_key = tf_network._make_node_key(layer.name,
                                             original_node_index)
        if node_key in self._network_nodes:
          node_conversion_map[node_key] = kept_nodes
          kept_nodes += 1
    layer_configs = []
    for layer in self.layers:  # From the earliest layers on.
      layer_class_name = layer.__class__.__name__
      layer_config = layer.get_config()
      filtered_inbound_nodes = []
      for original_node_index, node in enumerate(layer._inbound_nodes):
        node_key = tf_network._make_node_key(layer.name,
                                             original_node_index)
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
              node_key = tf_network._make_node_key(inbound_layer.name,
                                                   node_index)
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
      node_key = tf_network._make_node_key(layer.name,
                                           node_index)
      if node_key not in self._network_nodes:
        continue
      new_node_index = node_conversion_map[node_key]
      model_inputs.append([layer.name, new_node_index, tensor_index])
    config['input_layers'] = model_inputs
    model_outputs = []
    for i in range(len(self._output_layers)):
      layer, node_index, tensor_index = self._output_coordinates[i]
      node_key = tf_network._make_node_key(layer.name,
                                           node_index)
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
      """Deserialize a layer, then call it on appropriate inputs.

      Arguments:
          layer_data: layer config dict.

      Raises:
          ValueError: In case of improperly formatted `layer_data` dict.
      """
      layer_name = layer_data['name']

      # Instantiate layer.
      from tensorflow.python.keras._impl.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top

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
    """Save the model to a single HDF5 file.

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
    from tensorflow.python.keras._impl.keras.models import save_model  # pylint: disable=g-import-not-at-top
    save_model(self, filepath, overwrite, include_optimizer)

  def save_weights(self, filepath, overwrite=True):
    """Dumps all layer weights to a HDF5 file.

    The weight file has:
        - `layer_names` (attribute), a list of strings
            (ordered names of model layers).
        - For every layer, a `group` named `layer.name`
            - For every such layer group, a group attribute `weight_names`,
                a list of strings
                (ordered names of weights tensor of the layer).
            - For every weight in the layer, a dataset
                storing the weight value, named after the weight tensor.

    Arguments:
        filepath: String, path to the file to save the weights to.
        overwrite: Whether to silently overwrite any existing file at the
            target location, or provide the user with a manual prompt.

    Raises:
        ImportError: If h5py is not available.
    """
    if h5py is None:
      raise ImportError('`save_weights` requires h5py.')
    # If file exists and should not be overwritten:
    if not overwrite and os.path.isfile(filepath):
      proceed = ask_to_proceed_with_overwrite(filepath)
      if not proceed:
        return
    f = h5py.File(filepath, 'w')
    save_weights_to_hdf5_group(f, self.layers)
    f.flush()
    f.close()

  def load_weights(self, filepath, by_name=False):
    """Loads all layer weights from a HDF5 save file.

    If `by_name` is False (default) weights are loaded
    based on the network's topology, meaning the architecture
    should be the same as when the weights were saved.
    Note that layers that don't have weights are not taken
    into account in the topological ordering, so adding or
    removing layers is fine as long as they don't have weights.

    If `by_name` is True, weights are loaded into layers
    only if they share the same name. This is useful
    for fine-tuning or transfer-learning models where
    some of the layers have changed.

    Arguments:
        filepath: String, path to the weights file to load.
        by_name: Boolean, whether to load weights by name
            or by topological order.

    Raises:
        ImportError: If h5py is not available.
    """
    if h5py is None:
      raise ImportError('`load_weights` requires h5py.')
    f = h5py.File(filepath, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
      f = f['model_weights']
    if by_name:
      load_weights_from_hdf5_group_by_name(f, self.layers)
    else:
      load_weights_from_hdf5_group(f, self.layers)

    if hasattr(f, 'close'):
      f.close()

  def _updated_config(self):
    """Util hared between different serialization methods.

    Returns:
        Model config with Keras version information added.
    """
    from tensorflow.python.keras._impl.keras import __version__ as keras_version  # pylint: disable=g-import-not-at-top

    config = self.get_config()
    model_config = {
        'class_name': self.__class__.__name__,
        'config': config,
        'keras_version': keras_version,
        'backend': K.backend()
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
      raise ImportError('Requires yaml module installed.')
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
    """
    print_layer_summary(self,
                        line_length=line_length,
                        positions=positions,
                        print_fn=print_fn)


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
      return node.input_tensors
    else:
      source_tensors = []
      for i in range(len(node.inbound_layers)):
        x = node.input_tensors[i]
        layer = node.inbound_layers[i]
        node_index = node.node_indices[i]
        previous_sources = get_source_inputs(x, layer, node_index)
        # Avoid input redundancy.
        for x in previous_sources:
          if x not in source_tensors:
            source_tensors.append(x)
      return source_tensors


def _to_list(x):
  """Normalizes a list/tensor into a list.

  If a tensor is passed, we return
  a list of size 1 containing the tensor.

  Arguments:
      x: target object to be normalized.

  Returns:
      A list.
  """
  if isinstance(x, list):
    return x
  return [x]


def save_weights_to_hdf5_group(f, layers):
  from tensorflow.python.keras._impl.keras import __version__ as keras_version  # pylint: disable=g-import-not-at-top

  f.attrs['layer_names'] = [layer.name.encode('utf8') for layer in layers]
  f.attrs['backend'] = K.backend().encode('utf8')
  f.attrs['keras_version'] = str(keras_version).encode('utf8')

  for layer in layers:
    g = f.create_group(layer.name)
    symbolic_weights = layer.weights
    weight_values = K.batch_get_value(symbolic_weights)
    weight_names = []
    for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
      if hasattr(w, 'name') and w.name:
        name = str(w.name)
      else:
        name = 'param_' + str(i)
      weight_names.append(name.encode('utf8'))
    g.attrs['weight_names'] = weight_names
    for name, val in zip(weight_names, weight_values):
      param_dset = g.create_dataset(name, val.shape, dtype=val.dtype)
      if not val.shape:
        # scalar
        param_dset[()] = val
      else:
        param_dset[:] = val


def preprocess_weights_for_loading(layer,
                                   weights,
                                   original_keras_version=None,
                                   original_backend=None):
  """Converts layers weights from Keras 1 format to Keras 2.

  Arguments:
      layer: Layer instance.
      weights: List of weights values (Numpy arrays).
      original_keras_version: Keras version for the weights, as a string.
      original_backend: Keras backend the weights were trained with,
          as a string.

  Returns:
      A list of weights values (Numpy arrays).
  """
  if original_keras_version == '1':
    if layer.__class__.__name__ == 'Bidirectional':
      num_weights_per_layer = len(weights) // 2

      forward_weights = preprocess_weights_for_loading(
          layer.forward_layer, weights[:num_weights_per_layer],
          original_keras_version, original_backend)
      backward_weights = preprocess_weights_for_loading(
          layer.backward_layer, weights[num_weights_per_layer:],
          original_keras_version, original_backend)
      weights = forward_weights + backward_weights

    if layer.__class__.__name__ == 'TimeDistributed':
      weights = preprocess_weights_for_loading(
          layer.layer, weights, original_keras_version, original_backend)

    if layer.__class__.__name__ == 'Conv1D':
      shape = weights[0].shape
      # Handle Keras 1.1 format
      if shape[:2] != (layer.kernel_size[0], 1) or shape[3] != layer.filters:
        # Legacy shape:
        # (filters, input_dim, filter_length, 1)
        assert shape[0] == layer.filters and shape[2:] == (layer.kernel_size[0],
                                                           1)
        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
      weights[0] = weights[0][:, 0, :, :]

    if layer.__class__.__name__ == 'Conv2D':
      if layer.data_format == 'channels_first':
        # old: (filters, stack_size, kernel_rows, kernel_cols)
        # new: (kernel_rows, kernel_cols, stack_size, filters)
        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))

    if layer.__class__.__name__ == 'Conv2DTranspose':
      if layer.data_format == 'channels_last':
        # old: (kernel_rows, kernel_cols, stack_size, filters)
        # new: (kernel_rows, kernel_cols, filters, stack_size)
        weights[0] = np.transpose(weights[0], (0, 1, 3, 2))
      if layer.data_format == 'channels_first':
        # old: (filters, stack_size, kernel_rows, kernel_cols)
        # new: (kernel_rows, kernel_cols, filters, stack_size)
        weights[0] = np.transpose(weights[0], (2, 3, 0, 1))

    if layer.__class__.__name__ == 'Conv3D':
      if layer.data_format == 'channels_first':
        # old: (filters, stack_size, ...)
        # new: (..., stack_size, filters)
        weights[0] = np.transpose(weights[0], (2, 3, 4, 1, 0))

    if layer.__class__.__name__ == 'GRU':
      if len(weights) == 9:
        kernel = np.concatenate([weights[0], weights[3], weights[6]], axis=-1)
        recurrent_kernel = np.concatenate(
            [weights[1], weights[4], weights[7]], axis=-1)
        bias = np.concatenate([weights[2], weights[5], weights[8]], axis=-1)
        weights = [kernel, recurrent_kernel, bias]

    if layer.__class__.__name__ == 'LSTM':
      if len(weights) == 12:
        # old: i, c, f, o
        # new: i, f, c, o
        kernel = np.concatenate(
            [weights[0], weights[6], weights[3], weights[9]], axis=-1)
        recurrent_kernel = np.concatenate(
            [weights[1], weights[7], weights[4], weights[10]], axis=-1)
        bias = np.concatenate(
            [weights[2], weights[8], weights[5], weights[11]], axis=-1)
        weights = [kernel, recurrent_kernel, bias]

    if layer.__class__.__name__ == 'ConvLSTM2D':
      if len(weights) == 12:
        kernel = np.concatenate(
            [weights[0], weights[6], weights[3], weights[9]], axis=-1)
        recurrent_kernel = np.concatenate(
            [weights[1], weights[7], weights[4], weights[10]], axis=-1)
        bias = np.concatenate(
            [weights[2], weights[8], weights[5], weights[11]], axis=-1)
        if layer.data_format == 'channels_first':
          # old: (filters, stack_size, kernel_rows, kernel_cols)
          # new: (kernel_rows, kernel_cols, stack_size, filters)
          kernel = np.transpose(kernel, (2, 3, 1, 0))
          recurrent_kernel = np.transpose(recurrent_kernel, (2, 3, 1, 0))
        weights = [kernel, recurrent_kernel, bias]

    if layer.__class__.__name__ in ['Model', 'Sequential']:
      new_weights = []
      # trainable weights
      for sublayer in layer.layers:
        num_weights = len(sublayer.trainable_weights)
        if num_weights > 0:
          new_weights.extend(
              preprocess_weights_for_loading(
                  layer=sublayer,
                  weights=weights[:num_weights],
                  original_keras_version=original_keras_version,
                  original_backend=original_backend))
          weights = weights[num_weights:]

      # non-trainable weights
      for sublayer in layer.layers:
        num_weights = len([
            l for l in sublayer.weights if l not in sublayer.trainable_weights
        ])
        if num_weights > 0:
          new_weights.extend(
              preprocess_weights_for_loading(
                  layer=sublayer,
                  weights=weights[:num_weights],
                  original_keras_version=original_keras_version,
                  original_backend=original_backend))
          weights = weights[num_weights:]
      weights = new_weights

  conv_layers = ['Conv1D', 'Conv2D', 'Conv3D', 'Conv2DTranspose', 'ConvLSTM2D']
  if layer.__class__.__name__ in conv_layers:
    if original_backend and K.backend() != original_backend:
      weights[0] = conv_utils.convert_kernel(weights[0])
      if layer.__class__.__name__ == 'ConvLSTM2D':
        weights[1] = conv_utils.convert_kernel(weights[1])
    if K.int_shape(layer.weights[0]) != weights[0].shape:
      weights[0] = np.transpose(weights[0], (3, 2, 0, 1))
      if layer.__class__.__name__ == 'ConvLSTM2D':
        weights[1] = np.transpose(weights[1], (3, 2, 0, 1))

  # convert the weights of CuDNNLSTM so that they could be loaded into LSTM
  if layer.__class__.__name__ == 'LSTM':
    # determine if we're loading a CuDNNLSTM layer from the number of bias
    # weights:
    # CuDNNLSTM has (units * 8) weights; while LSTM has (units * 4)
    units = weights[1].shape[0]
    bias = weights[2]
    if len(bias) == units * 8:
      # reshape the kernels
      kernels = np.split(weights[0], 4, axis=1)
      kernels = [
          kernel.reshape(-1).reshape(kernel.shape, order='F')
          for kernel in kernels
      ]
      weights[0] = np.concatenate(kernels, axis=1)

      # transpose the recurrent kernels
      recurrent_kernels = np.split(weights[1], 4, axis=1)
      recurrent_kernels = [kernel.T for kernel in recurrent_kernels]
      weights[1] = np.concatenate(recurrent_kernels, axis=1)

      # split the bias into half and merge
      weights[2] = bias[:units * 4] + bias[units * 4:]

  return weights


def load_weights_from_hdf5_group(f, layers):
  """Implements topological (order-based) weight loading.

  Arguments:
      f: A pointer to a HDF5 group.
      layers: a list of target layers.

  Raises:
      ValueError: in case of mismatch between provided layers
          and weights file.
  """
  if 'keras_version' in f.attrs:
    original_keras_version = f.attrs['keras_version'].decode('utf8')
  else:
    original_keras_version = '1'
  if 'backend' in f.attrs:
    original_backend = f.attrs['backend'].decode('utf8')
  else:
    original_backend = None

  filtered_layers = []
  for layer in layers:
    weights = layer.weights
    if weights:
      filtered_layers.append(layer)

  layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
  filtered_layer_names = []
  for name in layer_names:
    g = f[name]
    weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
    if weight_names:
      filtered_layer_names.append(name)
  layer_names = filtered_layer_names
  if len(layer_names) != len(filtered_layers):
    raise ValueError('You are trying to load a weight file '
                     'containing ' + str(len(layer_names)) +
                     ' layers into a model with ' + str(len(filtered_layers)) +
                     ' layers.')

  # We batch weight value assignments in a single backend call
  # which provides a speedup in TensorFlow.
  weight_value_tuples = []
  for k, name in enumerate(layer_names):
    g = f[name]
    weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
    weight_values = [g[weight_name] for weight_name in weight_names]
    layer = filtered_layers[k]
    symbolic_weights = layer.weights
    weight_values = preprocess_weights_for_loading(
        layer, weight_values, original_keras_version, original_backend)
    if len(weight_values) != len(symbolic_weights):
      raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
                       '" in the current model) was found to '
                       'correspond to layer ' + name + ' in the save file. '
                       'However the new layer ' + layer.name + ' expects ' +
                       str(len(symbolic_weights)) +
                       ' weights, but the saved weights have ' +
                       str(len(weight_values)) + ' elements.')
    weight_value_tuples += zip(symbolic_weights, weight_values)
  K.batch_set_value(weight_value_tuples)


def load_weights_from_hdf5_group_by_name(f, layers):
  """Implements name-based weight loading.

  (instead of topological weight loading).

  Layers that have no matching name are skipped.

  Arguments:
      f: A pointer to a HDF5 group.
      layers: a list of target layers.

  Raises:
      ValueError: in case of mismatch between provided layers
          and weights file.
  """
  if 'keras_version' in f.attrs:
    original_keras_version = f.attrs['keras_version'].decode('utf8')
  else:
    original_keras_version = '1'
  if 'backend' in f.attrs:
    original_backend = f.attrs['backend'].decode('utf8')
  else:
    original_backend = None

  # New file format.
  layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

  # Reverse index of layer name to list of layers with name.
  index = {}
  for layer in layers:
    if layer.name:
      index.setdefault(layer.name, []).append(layer)

  # We batch weight value assignments in a single backend call
  # which provides a speedup in TensorFlow.
  weight_value_tuples = []
  for k, name in enumerate(layer_names):
    g = f[name]
    weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
    weight_values = [g[weight_name] for weight_name in weight_names]

    for layer in index.get(name, []):
      symbolic_weights = layer.weights
      weight_values = preprocess_weights_for_loading(
          layer, weight_values, original_keras_version, original_backend)
      if len(weight_values) != len(symbolic_weights):
        raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
                         '") expects ' + str(len(symbolic_weights)) +
                         ' weight(s), but the saved weights' + ' have ' +
                         str(len(weight_values)) + ' element(s).')
      # Set values.
      for i in range(len(weight_values)):
        weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
  K.batch_set_value(weight_value_tuples)
