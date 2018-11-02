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

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.framework import sparse_tensor
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
  reachable = set(inputs)
  if targets:
    targets = set(targets)
  queue = inputs[:]

  while queue:
    x = queue.pop()
    if isinstance(x, ops.Operation):
      outputs = x.outputs[:] or []
      outputs += x._control_outputs  # pylint: disable=protected-access
    elif isinstance(x, variables.Variable):
      outputs = [x.op]
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


def are_all_symbolic_tensors(tensors):
  return all(is_symbolic_tensor(tensor) for tensor in tensors)


def is_symbolic_tensor(tensor):
  """Returns whether a tensor is symbolic (from a TF graph) or an eager tensor.

  A Variable can be seen as either: it is considered symbolic
  when we are in a graph scope, and eager when we are in an eager scope.

  Arguments:
    tensor: A tensor instance to test.

  Returns:
    True for symbolic tensors, False for eager tensors.
  """
  if isinstance(tensor, variables.Variable):
    return not context.executing_eagerly()
  if isinstance(tensor, (ops.Tensor, sparse_tensor.SparseTensor)):
    try:
      _ = tensor.graph
      return True
    except AttributeError:
      return False
  return False
