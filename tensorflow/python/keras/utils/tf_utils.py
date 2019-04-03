# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow-related utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import nest


def smart_cond(pred, true_fn=None, false_fn=None, name=None):
  """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

  If `pred` is a bool or has a constant value, we return either `true_fn()`
  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

  Arguments:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`.

  Raises:
    TypeError: If `true_fn` or `false_fn` is not callable.
  """
  if isinstance(pred, variables.Variable):
    return control_flow_ops.cond(
        pred, true_fn=true_fn, false_fn=false_fn, name=name)
  return smart_module.smart_cond(
      pred, true_fn=true_fn, false_fn=false_fn, name=name)


def constant_value(pred):
  """Return the bool value for `pred`, or None if `pred` had a dynamic value.

  Arguments:
    pred: A scalar, either a Python bool or a TensorFlow boolean variable
      or tensor, or the Python integer 1 or 0.

  Returns:
    True or False if `pred` has a constant boolean value, None otherwise.

  Raises:
    TypeError: If `pred` is not a Variable, Tensor or bool, or Python
      integer 1 or 0.
  """
  # Allow integer booleans.
  if isinstance(pred, int):
    if pred == 1:
      pred = True
    elif pred == 0:
      pred = False

  if isinstance(pred, variables.Variable):
    return None
  return smart_module.smart_constant_value(pred)


def is_tensor_or_tensor_list(v):
  v = nest.flatten(v)
  if v and isinstance(v[0], ops.Tensor):
    return True
  else:
    return False


def get_reachable_from_inputs(inputs, targets=None):
  """Returns the set of tensors/ops reachable from `inputs`.

  Stops if all targets have been found (target is optional).

  Only valid in Symbolic mode, not Eager mode.

  Args:
    inputs: List of tensors.
    targets: List of tensors.

  Returns:
    A set of tensors reachable from the inputs (includes the inputs themselves).
  """
  inputs = nest.flatten(inputs)
  reachable = set(inputs)
  if targets:
    targets = set(targets)
  queue = inputs[:]

  while queue:
    x = queue.pop()
    if isinstance(x, tuple(_user_convertible_tensor_types)):
      # Can't find consumers of user-specific types.
      continue

    if isinstance(x, ops.Operation):
      outputs = x.outputs[:] or []
      outputs += x._control_outputs  # pylint: disable=protected-access
    elif isinstance(x, variables.Variable):
      try:
        outputs = [x.op]
      except AttributeError:
        # Variables can be created in an Eager context.
        outputs = []
    elif tensor_util.is_tensor(x):
      outputs = x.consumers()
    else:
      raise TypeError('Expected Operation, Variable, or Tensor, got ' + str(x))

    for y in outputs:
      if y not in reachable:
        reachable.add(y)
        queue.insert(0, y)

    if targets and targets.issubset(reachable):
      return reachable
  return reachable


# This function needs access to private functions of `nest`.
#  pylint: disable=protected-access
def map_structure_with_atomic(is_atomic_fn, map_fn, nested):
  """Maps the atomic elements of a nested structure.

  Arguments:
    is_atomic_fn: A function that determines if an element of `nested` is
      atomic.
    map_fn: The function to apply to atomic elements of `nested`.
    nested: A nested structure.

  Returns:
    The nested structure, with atomic elements mapped according to `map_fn`.

  Raises:
    ValueError: If an element that is neither atomic nor a sequence is
      encountered.
  """
  if is_atomic_fn(nested):
    return map_fn(nested)

  # Recursively convert.
  if not nest.is_sequence(nested):
    raise ValueError(
        'Received non-atomic and non-sequence element: {}'.format(nested))
  if nest._is_mapping(nested):
    values = [nested[k] for k in nest._sorted(nested)]
  else:
    values = nested
  mapped_values = [
      map_structure_with_atomic(is_atomic_fn, map_fn, ele) for ele in values
  ]
  return nest._sequence_like(nested, mapped_values)


#  pylint: enable=protected-access


def convert_shapes(input_shape, to_tuples=True):
  """Converts nested shape representations to desired format.

  Performs:

  TensorShapes -> tuples if `to_tuples=True`.
  tuples of int or None -> TensorShapes if `to_tuples=False`.

  Valid objects to be converted are:
  - TensorShapes
  - tuples with elements of type int or None.
  - ints
  - None

  Arguments:
    input_shape: A nested structure of objects to be converted to TensorShapes.
    to_tuples: If `True`, converts all TensorShape to tuples. Otherwise converts
      all tuples representing shapes to TensorShapes.

  Returns:
    Nested structure of shapes in desired format.
  """

  def _is_shape_component(value):
    return value is None or isinstance(value, (int, tensor_shape.Dimension))

  def _is_atomic_shape(input_shape):
    # Ex: TensorShape or (None, 10, 32) or 5 or `None`
    if _is_shape_component(input_shape):
      return True
    if isinstance(input_shape, tensor_shape.TensorShape):
      return True
    if (isinstance(input_shape, (tuple, list)) and
        all(_is_shape_component(ele) for ele in input_shape)):
      return True
    return False

  def _convert_shape(input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if to_tuples:
      input_shape = tuple(input_shape.as_list())
    return input_shape

  return map_structure_with_atomic(_is_atomic_shape, _convert_shape,
                                   input_shape)


class ListWrapper(object):
  """A wrapper for lists to be treated as elements for `nest`."""

  def __init__(self, list_to_wrap):
    self._list = list_to_wrap

  def as_list(self):
    return self._list


def convert_inner_node_data(nested, wrap=False):
  """Either wraps or unwraps innermost node data lists in `ListWrapper` objects.

  Arguments:
    nested: A nested data structure.
    wrap: If `True`, wrap innermost lists in `ListWrapper` objects. If `False`,
      unwraps `ListWrapper` objects into lists.

  Returns:
    Strucutre of same type as nested, with lists wrapped/unwrapped.
  """

  def _is_atomic_nested(nested):
    """Returns `True` if `nested` is a list representing node data."""
    if isinstance(nested, ListWrapper):
      return True
    # Node data can be of form `[layer_name, node_id, tensor_id]` or
    # `[layer_name, node_id, tensor_id, kwargs]`.
    if (isinstance(nested, list) and (len(nested) in [3, 4]) and
        isinstance(nested[0], six.string_types)):
      return True
    return False

  def _convert_object_or_list(nested):
    """Convert b/t `ListWrapper` object and list representations."""
    if wrap:
      if isinstance(nested, ListWrapper):
        return nested
      return ListWrapper(nested)
    else:
      if isinstance(nested, ListWrapper):
        return nested.as_list()
      return nested

  return map_structure_with_atomic(_is_atomic_nested, _convert_object_or_list,
                                   nested)


def shape_type_conversion(fn):
  """Decorator that handles tuple/TensorShape conversion.

  Used in `compute_output_shape` and `build`.

  Arguments:
    fn: function to wrap.

  Returns:
    Wrapped function.
  """

  def wrapper(instance, input_shape):
    # Pass shapes as tuples to `fn`
    # This preserves compatibility with external Keras.
    if input_shape is not None:
      input_shape = convert_shapes(input_shape, to_tuples=True)
    output_shape = fn(instance, input_shape)
    # Return shapes from `fn` as TensorShapes.
    if output_shape is not None:
      output_shape = convert_shapes(output_shape, to_tuples=False)
    return output_shape

  return wrapper


def are_all_symbolic_tensors(tensors):
  return all(is_symbolic_tensor(tensor) for tensor in tensors)


_user_convertible_tensor_types = set()


def is_symbolic_tensor(tensor):
  """Returns whether a tensor is symbolic (from a TF graph) or an eager tensor.

  A Variable can be seen as either: it is considered symbolic
  when we are in a graph scope, and eager when we are in an eager scope.

  Arguments:
    tensor: A tensor instance to test.

  Returns:
    True for symbolic tensors, False for eager tensors.
  """
  if isinstance(tensor, tuple(_user_convertible_tensor_types)):
    tensor = ops.convert_to_tensor_or_composite(tensor)
  if isinstance(tensor, variables.Variable):
    # Variables that are output of a Keras Layer in Functional API mode
    # should be considered symbolic.
    # TODO(omalleyt): We need a better way to check this in order to
    # enable `run_eagerly=True` for Models containing Layers that
    # return Variables as outputs.
    return (getattr(tensor, '_keras_history', False) or
            not context.executing_eagerly())
  if isinstance(tensor, composite_tensor.CompositeTensor):
    return tensor._is_graph_tensor  # pylint: disable=protected-access
  if isinstance(tensor, ops.Tensor):
    return hasattr(tensor, 'graph')
  return False


def register_symbolic_tensor_type(cls):
  """Allows users to specify types regarded as symbolic `Tensor`s.

  Used in conjunction with `tf.register_tensor_conversion_function`, calling
  `tf.keras.utils.register_symbolic_tensor_type(cls)` allows non-`Tensor`
  objects to be plumbed through Keras layers.

  Example:

  ```python
  # One-time setup.
  class Foo(object):
    def __init__(self, input_):
      self._input = input_
    def value(self):
      return tf.constant(42.)

  tf.register_tensor_conversion_function(
      Foo, lambda x, *args, **kwargs: x.value())

  tf.keras.utils.register_symbolic_tensor_type(Foo)

  # User-land.
  layer = tf.keras.layers.Lambda(lambda input_: Foo(input_))
  ```

  Arguments:
    cls: A `class` type which shall be regarded as a symbolic `Tensor`.
  """
  global _user_convertible_tensor_types
  _user_convertible_tensor_types.add(cls)


def is_tensor_or_variable(x):
  return tensor_util.is_tensor(x) or isinstance(x, variables.Variable)
