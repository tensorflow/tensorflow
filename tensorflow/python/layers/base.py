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

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect


def _is_tensor_or_tensor_list(v):
  v = nest.flatten(v)
  if v and isinstance(v[0], ops.Tensor):
    return True
  else:
    return False


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

  def call(self, inputs, **kwargs):
    """The logic of the layer lives here.

    Arguments:
      inputs: input tensor(s).
     **kwargs: additional keyword arguments.

    Returns:
      Output tensor(s).
    """
    raise NotImplementedError

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
                   initializer=None, regularizer=None, trainable=True):
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
    """
    self._set_scope(kwargs.pop('scope', None))

    # Ensure the Layer, if being reused, is working with inputs from
    # the same graph as where it was created.
    try:
      ops._get_graph_from_inputs(nest.flatten(inputs), graph=self.graph)  # pylint: disable=protected-access
    except ValueError as e:
      raise ValueError('Input graph and Layer graph are not the same: %s' % e)

    with vs.variable_scope(self._scope,
                           reuse=self.built or self._reuse) as scope:
      with ops.name_scope(scope.original_name_scope):
        if not self.built:
          # Check input assumptions set before layer building, e.g. input rank.
          self._assert_input_compatibility(inputs)
          input_list = [
              ops.convert_to_tensor(x, name='input')
              for x in nest.flatten(inputs)]
          input_shapes = [x.get_shape() for x in input_list]
          if len(input_shapes) == 1:
            self.build(input_shapes[0])
          else:
            self.build(input_shapes)
        if 'scope' in tf_inspect.getargspec(self.call).args:
          kwargs['scope'] = scope
        # Check input assumptions set after layer building, e.g. input shape.
        self._assert_input_compatibility(inputs)
        outputs = self.call(inputs, *args, **kwargs)

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
                           'its rank is undefined, by the layer requires a '
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

  ```
    >>> _unique_layer_name('dense')
    dense_1
    >>> _unique_layer_name('dense')
    dense_2
  ```
  """
  graph = ops.get_default_graph()
  if graph not in PER_GRAPH_LAYER_NAME_UIDS:
    PER_GRAPH_LAYER_NAME_UIDS[graph] = collections.defaultdict(int)
  layer_name_uids = PER_GRAPH_LAYER_NAME_UIDS[graph]
  layer_name_uids[name] += 1
  return name + '_' + str(layer_name_uids[name])
