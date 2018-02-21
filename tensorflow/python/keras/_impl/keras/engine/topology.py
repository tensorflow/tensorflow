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
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import constraints
from tensorflow.python.keras._impl.keras import initializers
from tensorflow.python.keras._impl.keras import regularizers
from tensorflow.python.keras._impl.keras.utils import conv_utils
from tensorflow.python.keras._impl.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.keras._impl.keras.utils.layer_utils import print_summary as print_layer_summary
from tensorflow.python.layers import base as tf_base_layers
from tensorflow.python.layers import utils as tf_layers_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


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


@tf_export('keras.layers.Layer')
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
      compute_output_shape(input_shape)
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

    # Un-built subclassed network: build it
    if isinstance(self, Network) and not self.inputs:
      self._set_inputs(inputs, training=kwargs.get('training'))

    # Update learning phase info.
    output_tensors = to_list(output)
    uses_lp = any(
        [getattr(x, '_uses_learning_phase', False) for x in to_list(inputs)])
    uses_lp = getattr(self, 'uses_learning_phase', False) or uses_lp
    for i in range(len(output_tensors)):
      output_tensors[i]._uses_learning_phase = getattr(
          output_tensors[i], '_uses_learning_phase', False) or uses_lp

    # Optionally load weight values that were specified at layer instantiation.
    if hasattr(self, '_initial_weights') and self._initial_weights is not None:
      self.set_weights(self._initial_weights)
      del self._initial_weights
    return output

  def compute_output_shape(self, input_shape):
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
    logging.warning(
        'All custom layers should implement the '
        '`compute_output_shape` method. This layer (' + self.name + ') '
        'is relying on the base `Layer.compute_output_shape` implementation, '
        'which will start raising a `NotImplementedError` '
        'as of July 1st, 2018.')
    return input_shape

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


class InputLayer(Layer):
  """Layer to be used as an entry point into a Network (a graph of layers).

  It can either wrap an existing tensor (pass an `input_tensor` argument)
  or create its a placeholder tensor (pass arguments `input_shape`, and
  optionally, `dtype`).

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
        input_tensor = tf_base_layers._DeferredTensor(  # pylint: disable=protected-access
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
    tf_base_layers.Node(
        self,
        inbound_layers=[],
        node_indices=[],
        tensor_indices=[],
        input_tensors=[input_tensor],
        output_tensors=[input_tensor])

  def get_config(self):
    config = {
        'batch_input_shape': self._batch_input_shape,
        'dtype': self.dtype,
        'sparse': self.sparse,
        'name': self.name
    }
    return config


@tf_export('keras.layers.Input', 'keras.Input')
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


class Network(Layer):
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

  def _base_init(self, name=None):
    # The following are implemented as property functions:
    # self.trainable_weights
    # self.non_trainable_weights
    # self.input_spec
    # self.losses
    # self.updates

    self._init_set_name(name)
    self._activity_regularizer = None
    # This acts just like the `trainable` attribute of any layer instance.
    # It does not affect users of the underlying layers, only users of the
    # Network instance.
    self.trainable = True
    self._is_compiled = False
    self._expects_training_arg = False

    self.supports_masking = False
    self.optimizer = None

    # Private attributes to implement compatibility with Layer.
    self._updates = []  # Used in symbolic mode only.
    self._losses = []   # Used in symbolic mode only.
    self._scope = None  # Never used.
    self._reuse = None  # Never used.
    if context.in_eager_mode:
      self._graph = None
    else:
      self._graph = ops.get_default_graph()  # Used in symbolic mode only.
        # A Network does not create weights of its own, thus has no dtype.
    self._dtype = None

    # All layers in order of horizontal graph traversal.
    # Entries are unique. Includes input and output layers.
    self._layers = []

    # Used in symbolic mode only, only in conjonction with graph-networks
    self._outbound_nodes = []
    self._inbound_nodes = []

  def _init_graph_network(self, inputs, outputs, name=None):
    # Normalize and set self.inputs, self.outputs.
    if isinstance(inputs, (list, tuple)):
      self.inputs = list(inputs)  # Tensor or list of tensors.
    else:
      self.inputs = [inputs]
    if isinstance(outputs, (list, tuple)):
      self.outputs = list(outputs)
    else:
      self.outputs = [outputs]

    # User-prodived argument validation.
    if context.in_eager_mode():
      # Check that all inputs/outputs are DeferredTensors.
      for tensor in self.inputs:
        if not isinstance(tensor, tf_base_layers._DeferredTensor):  # pylint: disable=protected-access
          raise TypeError('When eager execution is enabled, '
                          'inputs must come from a call to '
                          '`tf.keras.Input` (called after '
                          'tfe.enable_eager_execution()). '
                          'Received invalid input: ' + str(tensor))
      for tensor in self.outputs:
        if not isinstance(tensor, tf_base_layers._DeferredTensor):  # pylint: disable=protected-access
          raise TypeError('When eager execution is enabled, '
                          'outputs must come from a call to '
                          'a layer (called after '
                          'tfe.enable_eager_execution()). '
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
        'mask' in tf_inspect.getargspec(self.call).args or
        hasattr(self, 'compute_mask'))
    # A Network does not create weights of its own, thus it is already
    # built.
    self.built = True
    self._is_graph_network = True

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

    # Create the node linking internal inputs to internal outputs.
    tf_base_layers.Node(
        outbound_layer=self,
        inbound_layers=[],
        node_indices=[],
        tensor_indices=[],
        input_tensors=self.inputs,
        output_tensors=self.outputs)

    # Fill in the output mask cache.
    masks = []
    for x in self.inputs:
      mask = x._keras_mask if hasattr(x, '_keras_mask') else None  # pylint: disable=protected-access
      masks.append(mask)
    mask_cache_key = (tf_layers_util.object_list_uid(self.inputs) + '_' +
                      tf_layers_util.object_list_uid(masks))
    masks = []
    for x in self.outputs:
      mask = x._keras_mask if hasattr(x, '_keras_mask') else None  # pylint: disable=protected-access
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
        self._feed_input_shapes.append(K.int_shape(self.inputs[i]))
        # layer.input gives an error in eager mode
        if context.in_graph_mode():
          self._feed_inputs.append(layer.input)
    for layer in self._output_layers:
      self.output_names.append(layer.name)

  def _init_subclassed_network(self, name=None):
    self._base_init(name=name)
    self._is_graph_network = False
    if 'training' in tf_inspect.getargspec(self.call).args:
      self._expects_training_arg = True
    else:
      self._expects_training_arg = False

    self.outputs = None
    self.inputs = None
    self.built = False

  def __setattr__(self, name, value):
    if isinstance(value, (tf_base_layers.Layer, Network)):
      try:
        is_graph_network = self._is_graph_network
      except AttributeError:
        raise RuntimeError('It looks like you are subclassing `Model` and you '
                           'forgot to call `super(YourClass, self).__init__()`.'
                           ' Always start with this line.')
      if not is_graph_network:
        if value not in self._layers:
          self._layers.append(value)
    super(Network, self).__setattr__(name, value)

  def add_variable(self, name, shape, dtype=None, initializer=None,
                   regularizer=None, trainable=True, constraint=None):
    raise NotImplementedError('`add_variable` is not supported on Networks.')

  def add_loss(self, *args, **kwargs):
    if context.in_eager_mode():
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
    if not self._is_graph_network:
      return None

    inputs = to_list(inputs)
    if mask is None:
      masks = [None for _ in range(len(inputs))]
    else:
      masks = to_list(mask)
    cache_key = (tf_layers_util.object_list_uid(inputs)
                 + '_' + tf_layers_util.object_list_uid(masks))
    if cache_key in self._output_mask_cache:
      return self._output_mask_cache[cache_key]
    else:
      _, output_masks = self._run_internal_graph(inputs, masks)
      return output_masks

  @property
  def layers(self):
    return self._layers

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
    (e.g. will not include updates that were created by layers of this model
    outside of the model).

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
    if context.in_eager_mode():
      return []

    if not self.trainable and not self.stateful:
      return []

    updates = []
    for layer in self.layers:
      updates += layer.updates

    # `updates` might contain irrelevant updates, so it needs to be filtered
    # with respect to inputs the model has been called on.
    relevant_inputs = self.inputs or []
    for i in range(1, len(self._inbound_nodes)):
      inputs = self.get_input_at(i)
      if isinstance(inputs, list):
        relevant_inputs += inputs
      else:
        relevant_inputs.append(inputs)
    reachable = tf_layers_util.get_reachable_from_inputs(relevant_inputs,
                                                         updates)
    relevant_conditional_updates = [x for x in updates if x in reachable]
    unconditional_updates = [
        x for x in updates if x._unconditional_update]  # pylint: disable=protected-access
    # A layer could be used multiple times in a nested structure,
    # so the updates list must be de-duped.
    return list(set(
        relevant_conditional_updates + unconditional_updates + self._updates))

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
    for layer in self.layers:
      losses += layer.losses
    if context.in_eager_mode():
      return losses

    relevant_inputs = self.inputs or []
    for i in range(1, len(self._inbound_nodes)):
      inputs = self.get_input_at(i)
      if isinstance(inputs, list):
        relevant_inputs += inputs
      else:
        relevant_inputs.append(inputs)
    reachable = tf_layers_util.get_reachable_from_inputs(relevant_inputs,
                                                         losses)
    relevant_conditional_losses = [x for x in losses if x in reachable]
    unconditional_losses = [
        x for x in losses if x._unconditional_loss]  # pylint: disable=protected-access
    return list(set(
        relevant_conditional_losses + unconditional_losses + self._losses))

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
      cache_key = (tf_layers_util.object_list_uid(inputs)
                   + '_' + tf_layers_util.object_list_uid(masks))
      if cache_key in self._output_tensor_cache:
        # Cache hit.
        return self._output_tensor_cache[cache_key]
    # Actually apply the network graph to the new inputs.
    outputs, _ = self._run_internal_graph(inputs, masks)
    return outputs

  def compute_output_shape(self, input_shape):
    if not self._is_graph_network:
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

    cache_key = tf_layers_util.object_list_uid(input_shapes)
    if cache_key not in self._output_shape_cache:
      # Cache miss. We have to run the network graph manually (recursive calls
      # to `compute_output_shape`).
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
    # Network because 1) it may be useful to fully support in tf.layers in
    # the future and 2) Keras is a major user of Network.  If you don't
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
              if 'mask' in tf_inspect.getargspec(layer.call).args:
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
              if 'mask' in tf_inspect.getargspec(layer.call).args:
                if 'mask' not in kwargs:
                  kwargs['mask'] = computed_masks
              output_tensors = nest.flatten(
                  layer.call(computed_tensors, **kwargs))
              if hasattr(layer, 'compute_mask'):
                output_masks = nest.flatten(
                    layer.compute_mask(computed_tensors, computed_masks))
              else:
                output_masks = [None for _ in range(len(output_tensors))]

            if context.in_graph_mode():
              if layer.activity_regularizer is not None:
                regularization_losses = [
                    layer.activity_regularizer(x) for x in output_tensors
                ]
                # Apply activity regularizer if any:
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
      output_shapes.append(tf_layers_util.static_shape(x))
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
      cache_key = (tf_layers_util.object_list_uid(inputs)
                   + '_' + tf_layers_util.object_list_uid(masks))
      self._output_tensor_cache[cache_key] = output_tensors
      self._output_mask_cache[cache_key] = output_masks

      if output_shapes is not None:
        input_shapes = [tf_layers_util.static_shape(x) for x in inputs]
        cache_key = tf_layers_util.object_list_uid(input_shapes)
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
    if not self._is_graph_network:
      raise NotImplementedError

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
    with h5py.File(filepath, 'w') as f:
      save_weights_to_hdf5_group(f, self.layers)

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
    with h5py.File(filepath, 'r') as f:
      if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
      if by_name:
        load_weights_from_hdf5_group_by_name(f, self.layers)
      else:
        load_weights_from_hdf5_group(f, self.layers)

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
    if not self._is_graph_network:
      raise NotImplementedError

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
    if not self._is_graph_network:
      raise NotImplementedError

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


def to_list(x):
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
  if layer.__class__.__name__ == 'Bidirectional':
    num_weights_per_layer = len(weights) // 2
    forward_weights = preprocess_weights_for_loading(
        layer.forward_layer, weights[:num_weights_per_layer],
        original_keras_version, original_backend)
    backward_weights = preprocess_weights_for_loading(
        layer.backward_layer, weights[num_weights_per_layer:],
        original_keras_version, original_backend)
    weights = forward_weights + backward_weights

  if original_keras_version == '1':
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
    if original_backend == 'theano':
      weights[0] = conv_utils.convert_kernel(weights[0])
      if layer.__class__.__name__ == 'ConvLSTM2D':
        weights[1] = conv_utils.convert_kernel(weights[1])
    if K.int_shape(layer.weights[0]) != weights[0].shape:
      weights[0] = np.transpose(weights[0], (3, 2, 0, 1))
      if layer.__class__.__name__ == 'ConvLSTM2D':
        weights[1] = np.transpose(weights[1], (3, 2, 0, 1))

  # Convert the weights of CuDNNLSTM so that they could be loaded into LSTM
  if layer.__class__.__name__ == 'LSTM' and len(weights) == 3:
    # Determine if loading a CuDNNLSTM layer from the number of bias weights:
    # CuDNNLSTM has (units * 8) weights; while LSTM has (units * 4)
    # if there's no bias weight in the file, skip this conversion
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


def shape_type_conversion(fn):
  """Decorator that handles tuple/TensorShape conversion.

  Used in `compute_output_shape` and `build`.

  Arguments:
    fn: function to wrap.

  Returns:
    Wrapped function.
  """

  def wrapper(instance, input_shape):
    if input_shape is not None:
      if isinstance(input_shape, list):
        input_shape = [
            tuple(tensor_shape.TensorShape(x).as_list()) for x in input_shape]
      else:
        input_shape = tuple(tensor_shape.TensorShape(input_shape).as_list())
    output_shape = fn(instance, input_shape)
    if output_shape is not None:
      if isinstance(output_shape, list):
        return [tensor_shape.TensorShape(x) for x in output_shape]
      return tensor_shape.TensorShape(output_shape)

  return wrapper


def _make_node_key(layer_name, node_index):
  return layer_name + '_ib-' + str(node_index)


def _map_graph_network(inputs, outputs):
  """Validate a network's topology and gather its layers and nodes.

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
