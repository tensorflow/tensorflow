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

import copy
import functools
import re
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import six

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect


class _Layer(object):
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

    self._trainable = trainable
    self._built = False
    self._trainable_variables = []
    self._non_trainable_variables = []
    self._updates = []
    self._losses = []
    self._reuse = kwargs.get('_reuse')
    self._graph = ops.get_default_graph()
    self.dtype = dtype

    # Determine base name (non-unique).
    if isinstance(name, vs.VariableScope):
      base_name = name.name
    else:
      base_name = name
    if not name:
      base_name = _to_snake_case(self.__class__.__name__)
    self._base_name = base_name

    # Determine variable scope.
    scope = kwargs.get('_scope')
    if scope:
      self._scope = next(vs.variable_scope(scope).gen)
    else:
      self._scope = None

    # Unique name is borrowed from scope to match variable names.
    if self._scope is not None:
      self._name = self._scope.name
    else:
      # No name available until we see a scope
      self._name = None

  def __setattr__(self, name, value):
    if hasattr(self, name):
      # Only allow private attributes to be set more than once, under the
      # convention that private attributes should only be set from inside
      # the class.
      # All attributes meant to be set several times should be set to private.
      if name[0] != '_':
        raise AttributeError('Read-only property cannot be set: %s' % name)
    super(_Layer, self).__setattr__(name, value)

  @property
  def name(self):
    if self._name is None:
      raise ValueError(
          'No name available for layer because it has not been used yet.')
    return self._name

  @property
  def trainable_variables(self):
    return self._trainable_variables if self.trainable else []

  @property
  def non_trainable_variables(self):
    return self._non_trainable_variables if self.trainable else self.variables

  @property
  def trainable_weights(self):
    return self.trainable_variables

  @property
  def non_trainable_weights(self):
    return self.non_trainable_variables

  @property
  def variables(self):
    """Returns the list of all layer variables/weights.

    Returns:
      A list of variables.
    """
    return self._trainable_variables + self._non_trainable_variables

  @property
  def updates(self):
    return self._updates

  @property
  def losses(self):
    return self._losses

  @property
  def built(self):
    return self._built

  @property
  def trainable(self):
    return self._trainable

  @property
  def weights(self):
    """Returns the list of all layer variables/weights.

    Returns:
      A list of variables.
    """
    return self.variables

  def build(self, _):
    """Creates the variables of the layer.
    """
    self._built = True

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
    raise NotImplementedError

  def _add_variable(self, name, shape, dtype=None,
                    initializer=None, regularizer=None, trainable=True,
                    variable_getter=vs.get_variable):
    """Adds a new variable to the layer.

    Arguments:
      name: variable name.
      shape: variable shape.
      dtype: The type of the variable. Defaults to `self.dtype`.
      initializer: initializer instance (callable).
      regularizer: regularizer instance (callable).
      trainable: whether the variable should be part of the layer's
        "trainable_variables" (e.g. variables, biases)
        or "non_trainable_variables" (e.g. BatchNorm mean, stddev).
      variable_getter: The getter to use for TensorFlow variables.

    Returns:
      The created variable.
    """
    if dtype is None:
      dtype = self.dtype
    existing_variables = set(tf_variables.global_variables())
    variable = variable_getter(name,
                               shape=shape,
                               initializer=initializer,
                               dtype=dtype,
                               trainable=trainable and self.trainable)
    # TODO(sguada) fix name = variable.op.name
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
            self._losses.append(regularization)
            _add_elements_to_collection(
                regularization, ops.GraphKeys.REGULARIZATION_LOSSES)
      else:
        with ops.colocate_with(variable.op):
          with ops.name_scope(name + '/Regularizer'):
            regularization = regularizer(variable)
        if regularization is not None:
          self._losses.append(regularization)
          _add_elements_to_collection(
              regularization, ops.GraphKeys.REGULARIZATION_LOSSES)
    if trainable:
      self._trainable_variables.append(variable)
    else:
      self._non_trainable_variables.append(variable)
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
    scope = kwargs.pop('scope', None)

    # Define a custom getter to override tf.get_variable when creating layer
    # variables. The current custom getter is nested by the variable scope.
    def variable_getter(getter, name, shape, dtype=None, initializer=None,
                        regularizer=None, trainable=True, **getter_kwargs):
      return self._add_variable(
          name, shape, initializer=initializer, regularizer=regularizer,
          dtype=dtype, trainable=trainable,
          variable_getter=functools.partial(getter, **getter_kwargs))

    if not self._built and self._scope is None:
      # If constructed with _scope=None, lazy setting of scope.
      if self._reuse:
        self._scope = next(vs.variable_scope(
            scope if scope is not None else self._base_name).gen)
      else:
        self._scope = next(vs.variable_scope(
            scope, default_name=self._base_name).gen)
      self._name = self._scope.name

    # Build (if necessary) and call the layer, inside a variable
    # scope.
    with vs.variable_scope(self._scope,
                           reuse=True if self._built else self._reuse,
                           custom_getter=variable_getter) as scope:
      # Ensure the Layer, if being reused, is working with inputs from
      # the same graph as where it was created.
      try:
        ops._get_graph_from_inputs(nest.flatten(inputs), graph=self.graph)  # pylint: disable=protected-access
      except ValueError as e:
        raise ValueError("Inputs' and Layer's graphs are not the same: %s" % e)

      with ops.name_scope(scope.original_name_scope):
        if not self.built:
          input_list = [
              ops.convert_to_tensor(x, name='input')
              for x in nest.flatten(inputs)]
          input_shapes = [x.get_shape() for x in input_list]
          if len(input_shapes) == 1:
            self.build(input_shapes[0])
          else:
            self.build(input_shapes)
          self._built = True
        if 'scope' in tf_inspect.getargspec(self.call).args:
          kwargs['scope'] = scope
        outputs = self.call(inputs, *args, **kwargs)

        # Apply activity regularization.
        # Note that it should be applied every time the layer creates a new
        # output, since it is output-specific.
        if hasattr(self, 'activity_regularizer') and self.activity_regularizer:
          output_list = _to_list(outputs)
          for output in output_list:
            with ops.name_scope('ActivityRegularizer'):
              activity_regularization = self.activity_regularizer(output)
            self._losses.append(activity_regularization)
            _add_elements_to_collection(
                activity_regularization, ops.GraphKeys.REGULARIZATION_LOSSES)

    # Update global default collections.
    _add_elements_to_collection(self.updates, ops.GraphKeys.UPDATE_OPS)
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


def _add_elements_to_collection(elements, collections):
  elements = _to_list(elements)
  collections = _to_list(collections)
  for name in collections:
    collection = ops.get_collection_ref(name)
    collection_set = set(collection)
    for element in elements:
      if element not in collection_set:
        collection.append(element)
