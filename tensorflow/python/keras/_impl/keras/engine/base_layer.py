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
"""Base layer code (`Layer`).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect  # Necessary supplement to tf_inspect to deal with variadic args.

from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import constraints
from tensorflow.python.keras._impl.keras import initializers
from tensorflow.python.keras._impl.keras import regularizers
from tensorflow.python.keras._impl.keras.utils import generic_utils
from tensorflow.python.layers import base as tf_base_layers
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


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
    self._uses_inputs_arg = True

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

  def _inputs_from_call_args(self, call_args, call_kwargs):
    """Get Layer inputs from __call__ *args and **kwargs.

    Args:
      call_args: The positional arguments passed to __call__.
      call_kwargs: The keyword argument dict passed to __call__.

    Returns:
      A tuple of (inputs, non_input_kwargs). These may be the same objects as
      were passed in (call_args and call_kwargs).
    """
    if getattr(self, '_uses_inputs_arg', True):
      assert len(call_args) == 1  # TypeError raised earlier in __call__.
      return call_args[0], call_kwargs
    else:
      call_arg_spec = tf_inspect.getargspec(self.call)
      # There is no explicit "inputs" argument expected or provided to
      # call(). Arguments which have default values are considered non-inputs,
      # and arguments without are considered inputs.
      if call_arg_spec.defaults:
        if call_arg_spec.varargs is not None:
          raise TypeError(
              'Layer.call() may not accept both *args and arguments with '
              'default values (unable to determine which are inputs to the '
              'Layer).')
        keyword_arg_names = set(
            call_arg_spec.args[-len(call_arg_spec.defaults):])
      else:
        keyword_arg_names = set()
        # Training is never an input argument name, to allow signatures like
        # call(x, training).
      keyword_arg_names.add('training')
      _, unwrapped_call = tf_decorator.unwrap(self.call)
      bound_args = inspect.getcallargs(
          unwrapped_call, *call_args, **call_kwargs)
      if call_arg_spec.keywords is not None:
        var_kwargs = bound_args.pop(call_arg_spec.keywords)
        bound_args.update(var_kwargs)
        keyword_arg_names = keyword_arg_names.union(var_kwargs.keys())
      all_args = call_arg_spec.args
      if all_args and bound_args[all_args[0]] is self:
        # Ignore the 'self' argument of methods
        bound_args.pop(call_arg_spec.args[0])
        all_args = all_args[1:]
      non_input_arg_values = {}
      input_arg_values = []
      remaining_args_are_keyword = False
      for argument_name in all_args:
        if argument_name in keyword_arg_names:
          remaining_args_are_keyword = True
        else:
          if remaining_args_are_keyword:
            raise TypeError(
                'Found a positional argument to call() after a non-input '
                'argument. All arguments after "training" must be keyword '
                'arguments, and are not tracked as inputs to the Layer.')
        if remaining_args_are_keyword:
          non_input_arg_values[argument_name] = bound_args[argument_name]
        else:
          input_arg_values.append(bound_args[argument_name])
      if call_arg_spec.varargs is not None:
        input_arg_values.extend(bound_args[call_arg_spec.varargs])
      return input_arg_values, non_input_arg_values

  def __call__(self, inputs, *args, **kwargs):
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
        *args: Additional positional arguments to be passed to `call()`. Only
          allowed in subclassed Models with custom call() signatures. In other
          cases, `Layer` inputs must be passed using the `inputs` argument and
          non-inputs must be keyword arguments.
        **kwargs: Additional keyword arguments to be passed to `call()`.

    Returns:
        Output of the layer's `call` method.

    Raises:
        ValueError: in case the layer is missing shape information
            for its `build` call.
        TypeError: If positional arguments are passed and this `Layer` is not a
            subclassed `Model`.
    """
    # Actually call the layer (optionally building it).
    output = super(Layer, self).__call__(inputs, *args, **kwargs)

    if args and getattr(self, '_uses_inputs_arg', True):
      raise TypeError(
          'This Layer takes an `inputs` argument to call(), and only the '
          '`inputs` argument may be specified as a positional argument. Pass '
          'everything else as a keyword argument (those arguments will not be '
          'tracked as inputs to the Layer).')

    if context.executing_eagerly():
      return output

    inputs, kwargs = self._inputs_from_call_args(
        call_args=(inputs,) + args, call_kwargs=kwargs)

    if hasattr(self, '_symbolic_set_inputs') and not self.inputs:
      # Subclassed network: explicitly set metadata normally set by a call to
      # self._set_inputs().
      self._symbolic_set_inputs(inputs, output)

    # Update learning phase info.
    output_tensors = generic_utils.to_list(output)
    uses_lp = any(
        [getattr(x, '_uses_learning_phase', False)
         for x in generic_utils.to_list(inputs)])
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
