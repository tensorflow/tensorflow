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

# pylint: disable=unused-import,g-bad-import-order
"""Contains the base Layer class, from which all layers inherit.

This is a private class and its internal implementation is subject to changes
in the future.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import functools
import re
import weakref

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import six

from tensorflow.python.estimator import util as estimator_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.platform import tf_logging as logging


class Layer(object):
  """Base layer class.

  WARNING: Do not subclass this layer unless you know what you are doing:
  the API is subject to future changes.

  This is the class from which all layers inherit, implementing common
  infrastructure functionality.

  A layer is a class implementing common neural networks operations, such
  as convolution, batch norm, etc. These operations require managing variables,
  losses, and updates, as well as applying TensorFlow ops to input tensors.

  Properties:
    trainable: Whether the layer should be trained (boolean).
    name: The name of the layer (string).
    dtype: Default dtype of the layer (dtypes.float32).
    trainable_variables: List of trainable variables.
    non_trainable_variables: List of non-trainable variables.
    variables: List of all variables of this layer, trainable and non-trainable.
    updates: List of update ops of this layer.
    losses: List of losses added by this layer.
    input_spec: Object specifying the constraints on inputs that can be
      accepted by the layer.
  """

  def __init__(self, trainable=True, name=None,
               dtype=dtypes.float32, **kwargs):
    # We use a kwargs dict here because these kwargs only exist
    # for compatibility reasons.
    # The list of kwargs is subject to changes in the future.
    # We do not want to commit to it or to expose the list to users at all.
    # Note this is exactly as safe as defining kwargs in the function signature,
    # the only difference being that the list of valid kwargs is defined
    # below rather rather in the signature, and default values are defined
    # in calls to kwargs.get().
    allowed_kwargs = {
        '_scope',
        '_reuse',
        'input_shape',  # For compatibility with Keras `Sequential` model.
        'batch_size',  # For compatibility with Keras `Sequential` model.
    }
    for kwarg in kwargs:
      if kwarg not in allowed_kwargs:
        raise TypeError('Keyword argument not understood:', kwarg)

    self.trainable = trainable
    self.built = False
    self._trainable_weights = []
    self._non_trainable_weights = []
    self._updates = []
    self._losses = []
    self._reuse = kwargs.get('_reuse')
    self._graph = ops.get_default_graph()
    self._per_input_losses = {}
    self._per_input_updates = {}
    self.dtype = dtypes.as_dtype(dtype).name
    self.input_spec = None

    # These lists will be filled via successive calls
    # to self._add_inbound_node().
    self.inbound_nodes = []
    self.outbound_nodes = []

    # Determine layer name (non-unique).
    if isinstance(name, vs.VariableScope):
      base_name = name.name
    else:
      base_name = name
      self.name = name
    if not name:
      base_name = _to_snake_case(self.__class__.__name__)
      self.name = _unique_layer_name(base_name)
    self._base_name = base_name

    # Determine variable scope.
    scope = kwargs.get('_scope')
    if scope:
      self._scope = next(vs.variable_scope(scope).gen)
    else:
      self._scope = None

    # Set `batch_input_shape` attribute
    # for compatibility with Keras `Sequential` model.
    if 'input_shape' in kwargs:
      batch_size = kwargs.get('batch_size')
      self.batch_input_shape = (batch_size,) + tuple(kwargs['input_shape'])

  @property
  def scope_name(self):
    if not self._scope:
      raise ValueError('No name available for layer scope because the layer "' +
                       self.name + '" has not been used yet. The scope name ' +
                       ' is determined the first time the layer instance is ' +
                       'called. You must therefore call the layer before ' +
                       'querying `scope_name`.')
    return self._scope.name

  @property
  def trainable_weights(self):
    return self._trainable_weights if self.trainable else []

  @property
  def non_trainable_weights(self):
    if self.trainable:
      return self._non_trainable_weights
    else:
      return self._trainable_weights + self._non_trainable_weights

  @property
  def trainable_variables(self):
    return self.trainable_weights

  @property
  def non_trainable_variables(self):
    return self.non_trainable_weights

  @property
  def weights(self):
    """Returns the list of all layer variables/weights.

    Returns:
      A list of variables.
    """
    return self.trainable_weights + self.non_trainable_weights

  @property
  def variables(self):
    """Returns the list of all layer variables/weights.

    Returns:
      A list of variables.
    """
    return self.weights

  @property
  def updates(self):
    return self._updates

  def add_update(self, updates, inputs=None):
    """Add update op(s), potentially dependent on layer inputs.

    Weight updates (for instance, the updates of the moving mean and variance
    in a BatchNormalization layer) may be dependent on the inputs passed
    when calling a layer. Hence, when reusing a same layer on
    different inputs `a` and `b`, some entries in `layer.updates` may be
    dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    The `get_updates_for` method allows to retrieve the updates relevant to a
    specific set of inputs.

    Arguments:
      updates: Update op, or list/tuple of update ops.
      inputs: Optional input tensor(s) that the update(s) depend on. Must
        match the `inputs` argument passed to the `__call__` method at the time
        the updates are created. If `None` is passed, the updates are assumed
        to be unconditional, and will apply across all dataflows of the layer.
    """
    updates = _to_list(updates)
    if not updates:
      return
    self._updates += updates
    if inputs is not None:
      inputs = _to_list(inputs)
    if not inputs:
      inputs = None
    if inputs is not None:
      # We compute an ID that uniquely identifies the list of tensors.
      # This ID is order-sensitive.
      inputs_hash = _object_list_uid(inputs)
    else:
      inputs_hash = None
    if inputs_hash not in self._per_input_updates:
      self._per_input_updates[inputs_hash] = []
    self._per_input_updates[inputs_hash] += updates

  def get_updates_for(self, inputs):
    """Retrieves updates relevant to a specific set of inputs.

    Arguments:
      inputs: Input tensor or list/tuple of input tensors.
        Must match the `inputs` argument passed to the `__call__` method
        at the time the updates were created.
        If you pass `inputs=None`, unconditional updates are returned.

    Returns:
      List of update ops of the layer that depend on `inputs`.
    """
    if inputs is not None:
      inputs = _to_list(inputs)
    if not inputs:
      inputs = None
    if inputs is not None:
      inputs_hash = _object_list_uid(inputs)
    else:
      inputs_hash = None
    return self._per_input_updates.get(inputs_hash, [])

  @property
  def losses(self):
    return self._losses

  def add_loss(self, losses, inputs=None):
    """Add loss tensor(s), potentially dependent on layer inputs.

    Some losses (for instance, activity regularization losses) may be dependent
    on the inputs passed when calling a layer. Hence, when reusing a same layer
    on different inputs `a` and `b`, some entries in `layer.losses` may be
    dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    The `get_losses_for` method allows to retrieve the losses relevant to a
    specific set of inputs.

    Arguments:
      losses: Loss tensor, or list/tuple of tensors.
      inputs: Optional input tensor(s) that the loss(es) depend on. Must
        match the `inputs` argument passed to the `__call__` method at the time
        the losses are created. If `None` is passed, the losses are assumed
        to be unconditional, and will apply across all dataflows of the layer
        (e.g. weight regularization losses).
    """
    losses = _to_list(losses)
    if not losses:
      return
    self._losses += losses
    if inputs is not None:
      inputs = _to_list(inputs)
    if not inputs:
      inputs = None
    if inputs is not None:
      # We compute an ID that uniquely identifies the list of tensors.
      # This ID is order-sensitive.
      inputs_hash = _object_list_uid(inputs)
    else:
      inputs_hash = None
    if inputs_hash not in self._per_input_losses:
      self._per_input_losses[inputs_hash] = []
    self._per_input_losses[inputs_hash] += losses

  def get_losses_for(self, inputs):
    """Retrieves losses relevant to a specific set of inputs.

    Arguments:
      inputs: Input tensor or list/tuple of input tensors.
        Must match the `inputs` argument passed to the `__call__`
        method at the time the losses were created.
        If you pass `inputs=None`, unconditional losses are returned,
        such as weight regularization losses.

    Returns:
      List of loss tensors of the layer that depend on `inputs`.
    """
    if inputs is not None:
      inputs = _to_list(inputs)
    if not inputs:
      inputs = None
    if inputs is not None:
      inputs_hash = _object_list_uid(inputs)
    else:
      inputs_hash = None
    return self._per_input_losses.get(inputs_hash, [])

  def build(self, _):
    """Creates the variables of the layer.
    """
    self.built = True

  def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
    """The logic of the layer lives here.

    Arguments:
      inputs: input tensor(s).
     **kwargs: additional keyword arguments.

    Returns:
      Output tensor(s).
    """
    return inputs

  def _compute_output_shape(self, input_shape):
    """Computes the output shape of the layer given the input shape.

    Assumes that the layer will be built to match that input shape.
    If this method is not implemented by child classes, the default
    assumption will be that the layer does not alter the shape of the tensors
    passing through it.

    Args:
      input_shape: A (possibly nested tuple of) `TensorShape`.  It need not
        be fully defined (e.g. the batch size may be unknown).

    Returns:
      A (possibly nested tuple of) `TensorShape`.

    Raises:
      TypeError: if `input_shape` is not a (possibly nested tuple of)
        `TensorShape`.
      ValueError: if `input_shape` is incomplete or is incompatible with the
        the layer.
    """
    return input_shape

  def _set_scope(self, scope=None):
    if self._scope is None:
      # If constructed with _scope=None, lazy setting of scope.
      if self._reuse:
        self._scope = next(vs.variable_scope(
            scope if scope is not None else self._base_name).gen)
      else:
        self._scope = next(vs.variable_scope(
            scope, default_name=self._base_name).gen)

  def add_variable(self, name, shape, dtype=None,
                   initializer=None, regularizer=None,
                   trainable=True, constraint=None):
    """Adds a new variable to the layer, or gets an existing one; returns it.

    Arguments:
      name: variable name.
      shape: variable shape.
      dtype: The type of the variable. Defaults to `self.dtype`.
      initializer: initializer instance (callable).
      regularizer: regularizer instance (callable).
      trainable: whether the variable should be part of the layer's
        "trainable_variables" (e.g. variables, biases)
        or "non_trainable_variables" (e.g. BatchNorm mean, stddev).
      constraint: constraint instance (callable).

    Returns:
      The created variable.
    """
    if dtype is None:
      dtype = self.dtype
    existing_variables = set(tf_variables.global_variables())

    self._set_scope(None)

    with vs.variable_scope(self._scope,
                           reuse=self.built or self._reuse) as scope:
      with ops.name_scope(scope.original_name_scope):
        variable = vs.get_variable(name,
                                   shape=shape,
                                   initializer=initializer,
                                   dtype=dtypes.as_dtype(dtype),
                                   constraint=constraint,
                                   trainable=trainable and self.trainable)
        if variable in existing_variables:
          return variable
        if regularizer:
          # To match the behavior of tf.get_variable(), we only
          # apply regularization if the variable is newly created.
          if isinstance(variable, tf_variables.PartitionedVariable):
            for v in variable:
              with ops.colocate_with(v.op):
                with ops.name_scope(name + '/Regularizer'):
                  regularization = regularizer(v)
              if regularization is not None:
                self.add_loss(regularization)
                _add_elements_to_collection(
                    regularization, ops.GraphKeys.REGULARIZATION_LOSSES)
          else:
            with ops.colocate_with(variable.op):
              with ops.name_scope(name + '/Regularizer'):
                regularization = regularizer(variable)
            if regularization is not None:
              self.add_loss(regularization)
              _add_elements_to_collection(
                  regularization, ops.GraphKeys.REGULARIZATION_LOSSES)
    if trainable:
      self._trainable_weights.append(variable)
    else:
      self._non_trainable_weights.append(variable)
    return variable

  def __call__(self, inputs, *args, **kwargs):
    """Wraps `call`, applying pre- and post-processing steps.

    Arguments:
      inputs: input tensor(s).
      *args: additional positional arguments to be passed to `self.call`.
      **kwargs: additional keyword arguments to be passed to `self.call`.
        **Note**: kwarg `scope` is reserved for use by the layer.

    Returns:
      Output tensor(s).

    Note:
      - If the layer's `call` method takes a `scope` keyword argument,
        this argument will be automatically set to the current variable scope.
      - If the layer's `call` method takes a `mask` argument (as some Keras
        layers do), its default value will be set to the mask generated
        for `inputs` by the previous layer (if `input` did come from
        a layer that generated a corresponding mask, i.e. if it came from
        a Keras layer with masking support.

    Raises:
      ValueError: if the layer's `call` method returns None (an invalid value).
    """
    self._set_scope(kwargs.pop('scope', None))

    # Ensure the Layer, if being reused, is working with inputs from
    # the same graph as where it was created.
    try:
      ops._get_graph_from_inputs(nest.flatten(inputs), graph=self.graph)  # pylint: disable=protected-access
    except ValueError as e:
      raise ValueError('Input graph and Layer graph are not the same: %s' % e)

    # Handle Keras mask propagation from previous layer to current layer.
    previous_mask = _collect_previous_mask(inputs)
    user_kwargs = copy.copy(kwargs)
    if not _is_all_none(previous_mask):
      # The previous layer generated a mask, which we should use as default
      # for any "mask" argument in `call`.
      if 'mask' in estimator_util.fn_args(self.call):
        if 'mask' not in kwargs:
          # If mask is explicitly passed to __call__,
          # we should override the default mask.
          kwargs['mask'] = previous_mask

    with vs.variable_scope(self._scope,
                           reuse=self.built or self._reuse) as scope:
      with ops.name_scope(scope.original_name_scope):
        if not self.built:
          # Check input assumptions set before layer building, e.g. input rank.
          self._assert_input_compatibility(inputs)
          input_list = nest.flatten(inputs)
          input_shapes = [x.get_shape() for x in input_list]
          if len(input_shapes) == 1:
            self.build(input_shapes[0])
          else:
            self.build(input_shapes)
        if 'scope' in estimator_util.fn_args(self.call):
          kwargs['scope'] = scope
        # Check input assumptions set after layer building, e.g. input shape.
        self._assert_input_compatibility(inputs)
        outputs = self.call(inputs, *args, **kwargs)

        if outputs is None:
          raise ValueError('A layer\'s `call` method should return a Tensor '
                           'or a list of Tensors, not None.')

        # Apply activity regularization.
        # Note that it should be applied every time the layer creates a new
        # output, since it is output-specific.
        if hasattr(self, 'activity_regularizer') and self.activity_regularizer:
          output_list = _to_list(outputs)
          for output in output_list:
            with ops.name_scope('ActivityRegularizer'):
              activity_regularization = self.activity_regularizer(output)
            self.add_loss(activity_regularization)
            _add_elements_to_collection(
                activity_regularization, ops.GraphKeys.REGULARIZATION_LOSSES)

        # Handle mask computation and propagation to the next layer.
        if hasattr(self, 'compute_mask'):
          output_mask = self.compute_mask(inputs, previous_mask)
          if isinstance(outputs, list):
            if output_mask is None:
              output_mask = [None for _ in range(len(outputs))]
            for x, m in zip(outputs, output_mask):
              x._keras_mask = m  # pylint: disable=protected-access
          else:
            outputs._keras_mask = output_mask  # pylint: disable=protected-access

    # If all input tensors have history metadata,
    # we update the output tensors
    # with corresponding history metadata, thus eventually allowing to use
    # these tensors to instantiate a Network.
    if _have_all_keras_metadata(inputs):
      # If the layer returns tensors from its inputs, unmodified,
      # we copy them to avoid loss of tensor metadata.
      output_ls = _to_list(outputs)
      inputs_ls = _to_list(inputs)
      output_ls_copy = []
      for x in output_ls:
        if x in inputs_ls:
          with ops.name_scope(scope.original_name_scope):
            x = array_ops.identity(x)
        output_ls_copy.append(x)
      if len(output_ls_copy) == 1:
        outputs = output_ls_copy[0]
      else:
        outputs = output_ls_copy

      # Add an inbound node to the layer, so it can keep track of this call.
      # This updates the layer history of the output tensor(s).
      self._add_inbound_node(
          input_tensors=inputs,
          output_tensors=outputs,
          arguments=user_kwargs)

    # Update global default collections.
    _add_elements_to_collection(self.updates, ops.GraphKeys.UPDATE_OPS)
    self.built = True
    return outputs

  @property
  def graph(self):
    return self._graph

  def __deepcopy__(self, memo):
    no_copy = set(['_graph'])
    shallow_copy = set(['_scope'])
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    for k, v in self.__dict__.items():
      if k in no_copy:
        setattr(result, k, v)
      elif k in shallow_copy:
        setattr(result, k, copy.copy(v))
      elif _is_tensor_or_tensor_list(v):
        setattr(result, k, v)
      else:
        setattr(result, k, copy.deepcopy(v, memo))
    return result

  def apply(self, inputs, *args, **kwargs):
    """Apply the layer on a input.

    This simply wraps `self.__call__`.

    Arguments:
      inputs: Input tensor(s).
      *args: additional positional arguments to be passed to `self.call`.
      **kwargs: additional keyword arguments to be passed to `self.call`.

    Returns:
      Output tensor(s).
    """
    return self.__call__(inputs, *args, **kwargs)

  def _add_inbound_node(self,
                        input_tensors,
                        output_tensors,
                        arguments=None):
    """Internal method to create an inbound node for the layer.

    Arguments:
        input_tensors: list of input tensors.
        output_tensors: list of output tensors.
        arguments: dictionary of keyword arguments that were passed to the
            `call` method of the layer at the call that created the node.
    """
    input_tensors = _to_list(input_tensors)
    output_tensors = _to_list(output_tensors)

    # Collect input tensor(s) coordinates.
    inbound_layers = []
    node_indices = []
    tensor_indices = []
    for x in input_tensors:
      assert hasattr(x, '_keras_history')
      inbound_layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      inbound_layers.append(inbound_layer)
      node_indices.append(node_index)
      tensor_indices.append(tensor_index)

    # Create node, add it to inbound nodes.
    Node(
        self,
        inbound_layers=inbound_layers,
        node_indices=node_indices,
        tensor_indices=tensor_indices,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        arguments=arguments)

    # Update tensor history metadata.
    for i in range(len(output_tensors)):
      # The metadata attribute consists of 1) a layer instance
      # 2) a node index for the layer, 3) a tensor index for the node.
      # The allows layer reuse (multiple nodes per layer) and multi-output
      # or multi-input layers (e.g. a layer can return multiple tensors,
      # and each can be sent to a different layer).
      output_tensors[i]._keras_history = (self, len(self.inbound_nodes) - 1, i)  # pylint: disable=protected-access

  def _get_node_attribute_at_index(self, node_index, attr, attr_name):
    """Private utility to retrieves an attribute (e.g. inputs) from a node.

    This is used to implement the methods:
        - get_input_shape_at
        - get_output_shape_at
        - get_input_at
        etc...

    Arguments:
        node_index: Integer index of the node from which
            to retrieve the attribute.
        attr: Exact node attribute name.
        attr_name: Human-readable attribute name, for error messages.

    Returns:
        The layer's attribute `attr` at the node of index `node_index`.

    Raises:
        RuntimeError: If the layer has no inbound nodes.
        ValueError: If the index provided does not match any node.
    """
    if not self.inbound_nodes:
      raise RuntimeError('The layer has never been called '
                         'and thus has no defined ' + attr_name + '.')
    if not len(self.inbound_nodes) > node_index:
      raise ValueError('Asked to get ' + attr_name + ' at node ' +
                       str(node_index) + ', but the layer has only ' +
                       str(len(self.inbound_nodes)) + ' inbound nodes.')
    values = getattr(self.inbound_nodes[node_index], attr)
    if len(values) == 1:
      return values[0]
    else:
      return values

  def get_input_shape_at(self, node_index):
    """Retrieves the input shape(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A shape tuple
        (or list of shape tuples if the layer has multiple inputs).
    """
    return self._get_node_attribute_at_index(node_index, 'input_shapes',
                                             'input shape')

  def get_output_shape_at(self, node_index):
    """Retrieves the output shape(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A shape tuple
        (or list of shape tuples if the layer has multiple outputs).
    """
    return self._get_node_attribute_at_index(node_index, 'output_shapes',
                                             'output shape')

  def get_input_at(self, node_index):
    """Retrieves the input tensor(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A tensor (or list of tensors if the layer has multiple inputs).
    """
    return self._get_node_attribute_at_index(node_index, 'input_tensors',
                                             'input')

  def get_output_at(self, node_index):
    """Retrieves the output tensor(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A tensor (or list of tensors if the layer has multiple outputs).
    """
    return self._get_node_attribute_at_index(node_index, 'output_tensors',
                                             'output')

  @property
  def input(self):
    """Retrieves the input tensor(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer.

    Returns:
        Input tensor or list of input tensors.

    Raises:
        AttributeError: if the layer is connected to
        more than one incoming layers.
    """
    if not self.inbound_nodes:
      raise AttributeError('Layer ' + self.name +
                           ' is not connected, no input to return.')
    return self._get_node_attribute_at_index(0, 'input_tensors', 'input')

  @property
  def output(self):
    """Retrieves the output tensor(s) of a layer.

    Only applicable if the layer has exactly one output,
    i.e. if it is connected to one incoming layer.

    Returns:
        Output tensor or list of output tensors.

    Raises:
        AttributeError: if the layer is connected to
        more than one incoming layers.
    """
    if not self.inbound_nodes:
      raise AttributeError('Layer ' + self.name + ' has no inbound nodes.')
    return self._get_node_attribute_at_index(0, 'output_tensors', 'output')

  @property
  def input_shape(self):
    """Retrieves the input shape(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer, or if all inputs
    have the same shape.

    Returns:
        Input shape, as an integer shape tuple
        (or list of shape tuples, one tuple per input tensor).

    Raises:
        AttributeError: if the layer has no defined input_shape.
    """
    if not self.inbound_nodes:
      raise AttributeError('The layer has never been called '
                           'and thus has no defined input shape.')
    all_input_shapes = set(
        [str(node.input_shapes) for node in self.inbound_nodes])
    if len(all_input_shapes) == 1:
      input_shapes = self.inbound_nodes[0].input_shapes
      if len(input_shapes) == 1:
        return tuple(tensor_shape.TensorShape(input_shapes[0]).as_list())
      else:
        return [
            tuple(tensor_shape.TensorShape(shape).as_list())
            for shape in input_shapes
        ]
    else:
      raise AttributeError('The layer "' + str(self.name) +
                           ' has multiple inbound nodes, '
                           'with different input shapes. Hence '
                           'the notion of "input shape" is '
                           'ill-defined for the layer. '
                           'Use `get_input_shape_at(node_index)` '
                           'instead.')

  def count_params(self):
    """Count the total number of scalars composing the weights.

    Returns:
        An integer count.

    Raises:
        ValueError: if the layer isn't yet built
          (in which case its weights aren't yet defined).
    """
    if not self.built:
      if self.__class__.__name__ == 'Sequential':
        self.build()  # pylint: disable=no-value-for-parameter
      else:
        raise ValueError('You tried to call `count_params` on ' + self.name +
                         ', but the layer isn\'t built. '
                         'You can build it manually via: `' + self.name +
                         '.build(batch_input_shape)`.')
    weight_shapes = [w.get_shape().as_list() for w in self.weights]
    return int(sum([np.prod(w) for w in weight_shapes]))

  @property
  def output_shape(self):
    """Retrieves the output shape(s) of a layer.

    Only applicable if the layer has one output,
    or if all outputs have the same shape.

    Returns:
        Output shape, as an integer shape tuple
        (or list of shape tuples, one tuple per output tensor).

    Raises:
        AttributeError: if the layer has no defined output shape.
    """
    if not self.inbound_nodes:
      raise AttributeError('The layer has never been called '
                           'and thus has no defined output shape.')
    all_output_shapes = set(
        [str(node.output_shapes) for node in self.inbound_nodes])
    if len(all_output_shapes) == 1:
      output_shapes = self.inbound_nodes[0].output_shapes
      if len(output_shapes) == 1:
        return tuple(tensor_shape.TensorShape(output_shapes[0]).as_list())
      else:
        return [
            tuple(tensor_shape.TensorShape(shape).as_list())
            for shape in output_shapes
        ]
    else:
      raise AttributeError('The layer "%s"'
                           ' has multiple inbound nodes, '
                           'with different output shapes. Hence '
                           'the notion of "output shape" is '
                           'ill-defined for the layer. '
                           'Use `get_output_shape_at(node_index)` '
                           'instead.' % self.name)

  def _assert_input_compatibility(self, inputs):
    """Checks compatibility between the layer and provided inputs.

    This checks that the tensor(s) `inputs` verify the input assumptions
    of the layer (if any). If not, a clear and actional exception gets raised.

    Arguments:
        inputs: input tensor or list of input tensors.

    Raises:
        ValueError: in case of mismatch between
            the provided inputs and the expectations of the layer.
    """
    if not self.input_spec:
      return
    if not isinstance(self.input_spec, (list, tuple)):
      input_spec = _to_list(self.input_spec)
    else:
      input_spec = self.input_spec
    inputs = _to_list(inputs)
    if len(inputs) != len(input_spec):
      raise ValueError('Layer ' + self.name + ' expects ' +
                       str(len(input_spec)) + ' inputs, '
                       'but it received ' + str(len(inputs)) +
                       ' input tensors. Inputs received: ' + str(inputs))
    for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
      if spec is None:
        continue

      if (spec.ndim is not None or
          spec.min_ndim is not None or
          spec.max_ndim is not None):
        if x.get_shape().ndims is None:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'its rank is undefined, but the layer requires a '
                           'defined rank.')

      # Check ndim.
      if spec.ndim is not None:
        ndim = x.get_shape().ndims
        if ndim != spec.ndim:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'expected ndim=' + str(spec.ndim) + ', found ndim='
                           + str(ndim) + '. Full shape received: ' +
                           str(x.get_shape().as_list()))
      if spec.max_ndim is not None:
        ndim = x.get_shape().ndims
        if ndim is not None and ndim > spec.max_ndim:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'expected max_ndim=' + str(spec.max_ndim) +
                           ', found ndim=' + str(ndim))
      if spec.min_ndim is not None:
        ndim = x.get_shape().ndims
        if ndim is not None and ndim < spec.min_ndim:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           ': expected min_ndim=' + str(spec.min_ndim) +
                           ', found ndim=' + str(ndim) +
                           '. Full shape received: ' +
                           str(x.get_shape().as_list()))
      # Check dtype.
      if spec.dtype is not None:
        if x.dtype != spec.dtype:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'expected dtype=' + str(spec.dtype) +
                           ', found dtype=' + str(x.dtype))
      # Check specific shape axes.
      if spec.axes:
        shape = x.get_shape().as_list()
        if shape is not None:
          for axis, value in spec.axes.items():
            if hasattr(value, 'value'):
              value = value.value
            if value is not None and shape[int(axis)] not in {value, None}:
              raise ValueError(
                  'Input ' + str(input_index) + ' of layer ' + self.name + ' is'
                  ' incompatible with the layer: expected axis ' + str(axis) +
                  ' of input shape to have value ' + str(value) +
                  ' but received input with shape ' + str(shape))
      # Check shape.
      if spec.shape is not None:
        shape = x.get_shape().as_list()
        if shape is not None:
          for spec_dim, dim in zip(spec.shape, shape):
            if spec_dim is not None and dim is not None:
              if spec_dim != dim:
                raise ValueError('Input ' + str(input_index) +
                                 ' is incompatible with layer ' + self.name +
                                 ': expected shape=' + str(spec.shape) +
                                 ', found shape=' + str(shape))


class InputSpec(object):
  """Specifies the ndim, dtype and shape of every input to a layer.

  Every layer should expose (if appropriate) an `input_spec` attribute:
  a list of instances of InputSpec (one per input tensor).

  A None entry in a shape is compatible with any dimension,
  a None shape is compatible with any shape.

  Arguments:
      dtype: Expected DataType of the input.
      shape: Shape tuple, expected shape of the input
          (may include None for unchecked axes).
      ndim: Integer, expected rank of the input.
      max_ndim: Integer, maximum rank of the input.
      min_ndim: Integer, minimum rank of the input.
      axes: Dictionary mapping integer axes to
          a specific dimension value.
  """

  def __init__(self,
               dtype=None,
               shape=None,
               ndim=None,
               max_ndim=None,
               min_ndim=None,
               axes=None):
    self.dtype = dtype
    self.shape = shape
    if shape is not None:
      self.ndim = len(shape)
    else:
      self.ndim = ndim
    self.max_ndim = max_ndim
    self.min_ndim = min_ndim
    self.axes = axes or {}


class Node(object):
  """A `Node` describes the connectivity between two layers.

  Each time a layer is connected to some new input,
  a node is added to `layer.inbound_nodes`.
  Each time the output of a layer is used by another layer,
  a node is added to `layer.outbound_nodes`.

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
    - A.outbound_nodes
    - B.inbound_nodes
  """

  def __init__(self,
               outbound_layer,
               inbound_layers,
               node_indices,
               tensor_indices,
               input_tensors,
               output_tensors,
               arguments=None):
    # Layer instance (NOT a list).
    if isinstance(outbound_layer, list):
      raise ValueError(
          '`outbound_layer` should be a layer instance, not a list.')
    # this is the layer that takes a list of input tensors
    # and turns them into a list of output tensors.
    # the current node will be added to
    # the inbound_nodes of outbound_layer.
    self.outbound_layer = outbound_layer

    # The following 3 properties describe where
    # the input tensors come from: which layers,
    # and for each layer, which node and which
    # tensor output of each node.

    # List of layer instances.
    self.inbound_layers = inbound_layers
    # List of integers, 1:1 mapping with inbound_layers.
    self.node_indices = node_indices
    # List of integers, 1:1 mapping with inbound_layers.
    self.tensor_indices = tensor_indices

    # Following 2 properties:
    # tensor inputs and outputs of outbound_layer.

    # List of tensors. 1:1 mapping with inbound_layers.
    self.input_tensors = input_tensors
    # List of tensors, created by outbound_layer.call().
    self.output_tensors = output_tensors

    # Following 2 properties: input and output shapes.

    # List of shape tuples, shapes of input_tensors.
    self.input_shapes = [_static_shape(x) for x in input_tensors]
    # List of shape tuples, shapes of output_tensors.
    self.output_shapes = [_static_shape(x) for x in output_tensors]

    # Optional keyword arguments to layer's `call`.
    self.arguments = arguments

    # Add nodes to all layers involved.
    for layer in inbound_layers:
      if layer is not None:
        layer.outbound_nodes.append(self)
    outbound_layer.inbound_nodes.append(self)

  def get_config(self):
    inbound_names = []
    for layer in self.inbound_layers:
      if layer:
        inbound_names.append(layer.name)
      else:
        inbound_names.append(None)
    return {
        'outbound_layer': self.outbound_layer.name,
        'inbound_layers': inbound_names,
        'node_indices': self.node_indices,
        'tensor_indices': self.tensor_indices
    }


class InputLayer(Layer):
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
      self.batch_input_shape = batch_input_shape
    else:
      # For compatibility with Keras API.
      self.is_placeholder = False
      self.batch_input_shape = tuple(input_tensor.get_shape().as_list())

    # Create an input node to add to self.outbound_node
    # and set output_tensors' _keras_history.
    input_tensor._keras_history = (self, 0, 0)  # pylint: disable=protected-access
    Node(
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
  outputs = input_layer.inbound_nodes[0].output_tensors
  if len(outputs) == 1:
    return outputs[0]
  else:
    return outputs


class Network(Layer):
  """A Network is a directed acyclic graph of layers.

  It is the topological form of a "model".
  A Model is simply a Network with added training/evaluation routines.

  A Network instance implements the full Layer API. In particular, a network
  can be called on new inputs.

  Example:

      ```python
      # This is a logistic regression
      x = tf.layers.Input(shape=(32,))
      y = tf.layers.Dense(16, activation='softmax')(x)
      network = tf.layers.Network(x, y)

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
    Network has the same attributes as Layer. On top of it, it also has:
      - layers: a list of the children layers of the network,
        a list of layer instances, ordered from "earlier in the graph"
        to "later in the graph".

  Methods:
    Network has the same methods as Layer. On top of it, it also has:
      - get_layer: retrieves a child layer by name or index in the graph.
  """

  def __init__(self, inputs, outputs, name=None):  # pylint: disable=super-init-not-called
    # Set layer name and scope
    if isinstance(name, vs.VariableScope):
      base_name = name.name
    else:
      base_name = name
      self.name = name
    if not name:
      base_name = _to_snake_case(self.__class__.__name__)
      self.name = _unique_layer_name(base_name)
    self._scope = next(vs.variable_scope(None, default_name=base_name).gen)
    self._base_name = base_name

    # This acts just like the `trainable` attribute of any layer instance.
    # It does not affect users of the underlying layers, only users of the
    # Network instance.
    self.trainable = True
    # A Network does not create weights of its own, thus it is already built.
    self.built = True
    # A Network does not create weights of its own, thus dtype is not settable.
    self.dtype = None
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

    # Network-specific properties.
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

    # This is for performance optimization
    # when calling the Network on new inputs.
    # every time the Network is called on a set on input tensors,
    # we compute the output tensors,
    # output masks and output shapes in one pass,
    # then cache them here. When any of these outputs is queried later,
    # we retrieve it from there instead of recomputing it.
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
      layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      if len(layer.inbound_nodes) > 1 or (
          layer.inbound_nodes and layer.inbound_nodes[0].inbound_layers):
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
    network_nodes = set()  # ids of all nodes relevant to the Network
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
      node = layer.inbound_nodes[node_index]

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
        inbound_node = inbound_layer.inbound_nodes[node_index]
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
    self.outbound_nodes = []  # Will be appended to by future calls to __call__
    self.inbound_nodes = [
    ]  # Will be appended to below, and by future calls to __call__
    # Create the node linking internal inputs to internal outputs.
    Node(
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
    # It would be unreliable to build a dictionary
    # based on layer names, because names can potentially
    # be changed at any point by the user
    # without the network being notified of it.
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
        for node_index, node in enumerate(layer.inbound_nodes):
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
        for node_index, node in enumerate(layer.inbound_nodes):
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
    inputs = _to_list(inputs)
    if mask is None:
      masks = [None for _ in range(len(inputs))]
    else:
      masks = _to_list(mask)
    # Try to retrieve cached outputs if the layer has already been called
    # on these exact inputs.
    cache_key = _object_list_uid(inputs) + '_' + _object_list_uid(masks)
    if cache_key in self._output_tensor_cache:
      # Cache hit.
      return self._output_tensor_cache[cache_key]
    else:
      # Cache miss: actually apply the network graph to the new inputs.
      output_tensors, _, _ = self._run_internal_graph(inputs, masks)
      return output_tensors

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

    cache_key = _object_list_uid(input_shapes)
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

            node_index = layer.inbound_nodes.index(node)
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
    # It cannot be factored out without having the fully reimplement the
    # network calling logic on the Keras side. We choose to incorporate it
    # in Network because 1) it may be useful to fully support in tf.layers in
    # the future and 2) Keras is a major user of Network.
    # If you don't use masking, it does not interfere with regular behavior
    # at all and you can ignore it.
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
              output_tensors = _to_list(layer.call(computed_tensor, **kwargs))
              if hasattr(layer, 'compute_mask'):
                output_masks = _to_list(
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
              output_tensors = _to_list(layer.call(computed_tensors, **kwargs))
              if hasattr(layer, 'compute_mask'):
                output_masks = _to_list(
                    layer.compute_mask(computed_tensors, computed_masks))
              else:
                output_masks = [None for _ in range(len(output_tensors))]

            # Apply activity regularizer if any:
            if hasattr(layer, 'activity_regularizer'
                      ) and layer.activity_regularizer is not None:
              regularization_losses = [
                  layer.activity_regularizer(x) for x in computed_tensors
              ]
              layer.add_loss(regularization_losses, computed_tensors)

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
      output_shapes.append(_static_shape(x))
      output_tensors.append(tensor)
      output_masks.append(mask)

    # Update cache;
    # keys are based on ids on input tensors and inputs masks.
    cache_key = _object_list_uid(inputs) + '_' + _object_list_uid(masks)

    if len(output_tensors) == 1:
      output_tensors = output_tensors[0]
      self._output_tensor_cache[cache_key] = output_tensors
    else:
      self._output_tensor_cache[cache_key] = output_tensors

    if len(output_masks) == 1:
      output_masks = output_masks[0]
      self._output_mask_cache[cache_key] = output_masks
    else:
      self._output_mask_cache[cache_key] = output_masks

    if output_shapes is not None:
      input_shapes = [_static_shape(x) for x in inputs]
      cache_key = _object_list_uid(input_shapes)
      if len(output_shapes) == 1:
        output_shapes = output_shapes[0]
        self._output_shape_cache[cache_key] = output_shapes
      else:
        self._output_shape_cache[cache_key] = output_shapes
    return output_tensors, output_masks, output_shapes


def _is_tensor_or_tensor_list(v):
  v = nest.flatten(v)
  if v and isinstance(v[0], ops.Tensor):
    return True
  else:
    return False


def _to_snake_case(name):
  intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
  insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
  # If the class is private the name starts with "_" which is not secure
  # for creating scopes. We prefix the name with "private" in this case.
  if insecure[0] != '_':
    return insecure
  return 'private' + insecure


def _to_list(x):
  """This normalizes a list/tuple or single element into a list.

  If a single element is passed, we return
  a list of size 1 containing the element.

  Arguments:
    x: list or tuple or single element.

  Returns:
    A list.
  """
  if isinstance(x, (list, tuple)):
    return list(x)
  return [x]


def _add_elements_to_collection(elements, collection_list):
  elements = _to_list(elements)
  collection_list = _to_list(collection_list)
  for name in collection_list:
    collection = ops.get_collection_ref(name)
    collection_set = set(collection)
    for element in elements:
      if element not in collection_set:
        collection.append(element)


def _object_list_uid(object_list):
  object_list = _to_list(object_list)
  return ', '.join([str(abs(id(x))) for x in object_list])


def _make_node_key(layer_name, node_index):
  return layer_name + '_ib-' + str(node_index)


def _static_shape(x):
  if x is None:
    return None
  try:
    return tuple(x.get_shape().as_list())
  except ValueError:
    return None


def _is_all_none(iterable_or_element):
  if not isinstance(iterable_or_element, (list, tuple)):
    iterable = [iterable_or_element]
  else:
    iterable = iterable_or_element
  # We cannot use Python's `any` because the iterable may return Tensors.
  for element in iterable:
    if element is not None:
      return False
  return True


def _have_all_keras_metadata(iterable_or_element):
  if not isinstance(iterable_or_element, (list, tuple)):
    iterable = [iterable_or_element]
  else:
    iterable = iterable_or_element
  return all([hasattr(x, '_keras_history') for x in iterable])


def _collect_previous_mask(input_tensors):
  """Retrieves the output mask(s) of the previous node.

  Arguments:
      input_tensors: A tensor or list of tensors.

  Returns:
      A mask tensor or list of mask tensors.
  """
  input_tensors = _to_list(input_tensors)
  masks = []
  for x in input_tensors:
    if hasattr(x, '_keras_mask'):
      mask = x._keras_mask  # pylint: disable=protected-access
      masks.append(mask)
    else:
      masks.append(None)
  if len(masks) == 1:
    return masks[0]
  return masks

# A global dictionary mapping graph objects to an index of counters used
# for various layer names in each graph.
# Allows to give unique autogenerated names to layers, in a graph-specific way.
PER_GRAPH_LAYER_NAME_UIDS = weakref.WeakKeyDictionary()


def _unique_layer_name(name):
  """Makes a layer name (or arbitrary string) unique within a TensorFlow graph.

  Arguments:
    name: String name to make unique.

  Returns:
    Unique string name.

  Example:

  ```python
  _unique_layer_name('dense')  # dense_1
  _unique_layer_name('dense')  # dense_2
  ```
  """
  graph = ops.get_default_graph()
  if graph not in PER_GRAPH_LAYER_NAME_UIDS:
    PER_GRAPH_LAYER_NAME_UIDS[graph] = collections.defaultdict(int)
  layer_name_uids = PER_GRAPH_LAYER_NAME_UIDS[graph]
  layer_name_uids[name] += 1
  return name + '_' + str(layer_name_uids[name])
