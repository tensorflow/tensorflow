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
"""Common util functions used by layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from collections import OrderedDict
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables

__all__ = ['collect_named_outputs',
           'constant_value',
           'static_cond',
           'smart_cond',
           'get_variable_collections',
           'two_element_tuple',
           'n_positive_integers',
           'last_dimension',
           'first_dimension']

NamedOutputs = namedtuple('NamedOutputs', ['name', 'outputs'])


def collect_named_outputs(collections, alias, outputs):
  """Add `Tensor` outputs tagged with alias to collections.

  It is useful to collect end-points or tags for summaries. Example of usage:

  logits = collect_named_outputs('end_points', 'inception_v3/logits', logits)
  assert logits.alias == 'inception_v3/logits'

  Args:
    collections: A collection or list of collections. If None skip collection.
    alias: String, alias to name the outputs, ex. 'inception_v3/conv1'
    outputs: Tensor, an output tensor to collect

  Returns:
    The outputs Tensor to allow inline call.
  """
  # Remove ending '/' if present.
  if alias[-1] == '/':
    alias = alias[:-1]
  outputs.alias = alias
  if collections:
    ops.add_to_collections(collections, outputs)
  return outputs


def gather_tensors_alias(tensors):
  """Given a list of tensors, gather their aliases.

  If the tensor does not have an alias it would default to its name.

  Args:
    tensors: A list of `Tensors`.

  Returns:
    A list of strings with the alias of each tensor.
  """
  return [get_tensor_alias(tensor) for tensor in tensors]


def get_tensor_alias(tensor):
  """Given a tensor gather its alias, its op.name or its name.

  If the tensor does not have an alias it would default to its name.

  Args:
    tensor: A `Tensor`.

  Returns:
    A string with the alias of the tensor.
  """
  if hasattr(tensor, 'alias'):
    alias = tensor.alias
  else:
    if tensor.name[-2:] == ':0':
      # Use op.name for tensor ending in :0
      alias = tensor.op.name
    else:
      alias = tensor.name
  return alias


def convert_collection_to_dict(collection):
  """Returns an OrderedDict of Tensors using get_tensor_alias as key.

  Args:
    collection: A collection.

  Returns:
    An OrderedDict of {get_tensor_alias(tensor): tensor}
  """
  return OrderedDict((get_tensor_alias(t), t) for t in ops.get_collection(collection))


def constant_value(value_or_tensor_or_var, dtype=None):
  """Returns value if value_or_tensor_or_var has a constant value.

  Args:
    value_or_tensor_or_var: A value, a `Tensor` or a `Variable`.
    dtype: Optional `tf.dtype`, if set it would check it has the right
      dtype.

  Returns:
    The constant value or None if it not constant.

  Raises:
    ValueError: if value_or_tensor_or_var is None or the tensor_variable has the
    wrong dtype.
  """
  if value_or_tensor_or_var is None:
    raise ValueError('value_or_tensor_or_var cannot be None')
  value = value_or_tensor_or_var
  if isinstance(value_or_tensor_or_var, (ops.Tensor, variables.Variable)):
    if dtype and value_or_tensor_or_var.dtype != dtype:
      raise ValueError('It has the wrong type %s instead of %s' % (
          value_or_tensor_or_var.dtype, dtype))
    if isinstance(value_or_tensor_or_var, variables.Variable):
      value = None
    else:
      value = tensor_util.constant_value(value_or_tensor_or_var)
  return value


def static_cond(pred, fn1, fn2):
  """Return either fn1() or fn2() based on the boolean value of `pred`.

  Same signature as `control_flow_ops.cond()` but requires pred to be a bool.

  Args:
    pred: A value determining whether to return the result of `fn1` or `fn2`.
    fn1: The callable to be performed if pred is true.
    fn2: The callable to be performed if pred is false.

  Returns:
    Tensors returned by the call to either `fn1` or `fn2`.

  Raises:
    TypeError: if `fn1` or `fn2` is not callable.
  """
  if not callable(fn1):
    raise TypeError('fn1 must be callable.')
  if not callable(fn2):
    raise TypeError('fn2 must be callable.')
  if pred:
    return fn1()
  else:
    return fn2()


def smart_cond(pred, fn1, fn2, name=None):
  """Return either fn1() or fn2() based on the boolean predicate/value `pred`.

  If `pred` is bool or has a constant value it would use `static_cond`,
  otherwise it would use `tf.cond`.

  Args:
    pred: A scalar determining whether to return the result of `fn1` or `fn2`.
    fn1: The callable to be performed if pred is true.
    fn2: The callable to be performed if pred is false.
    name: Optional name prefix when using tf.cond
  Returns:
    Tensors returned by the call to either `fn1` or `fn2`.
  """
  pred_value = constant_value(pred)
  if pred_value is not None:
    # Use static_cond if pred has a constant value.
    return static_cond(pred_value, fn1, fn2)
  else:
    # Use dynamic cond otherwise.
    return control_flow_ops.cond(pred, fn1, fn2, name)


def get_variable_collections(variables_collections, name):
  if isinstance(variables_collections, dict):
    variable_collections = variables_collections.get(name, None)
  else:
    variable_collections = variables_collections
  return variable_collections


def first_dimension(shape, min_rank=1):
  """Returns the first dimension of shape while checking it has min_rank.

  Args:
    shape: A `TensorShape`.
    min_rank: Integer, minimum rank of shape.

  Returns:
    The value of the first dimension.

  Raises:
    ValueError: if inputs don't have at least min_rank dimensions, or if the
      first dimension value is not defined.
  """
  dims = shape.dims
  if dims is None:
    raise ValueError('dims of shape must be known but is None')
  if len(dims) < min_rank:
    raise ValueError('rank of shape must be at least %d not: %d' % (min_rank,
                                                                    len(dims)))
  value = dims[0].value
  if value is None:
    raise ValueError('first dimension shape must be known but is None')
  return value


def last_dimension(shape, min_rank=1):
  """Returns the last dimension of shape while checking it has min_rank.

  Args:
    shape: A `TensorShape`.
    min_rank: Integer, minimum rank of shape.

  Returns:
    The value of the last dimension.

  Raises:
    ValueError: if inputs don't have at least min_rank dimensions, or if the
      last dimension value is not defined.
  """
  dims = shape.dims
  if dims is None:
    raise ValueError('dims of shape must be known but is None')
  if len(dims) < min_rank:
    raise ValueError('rank of shape must be at least %d not: %d' % (min_rank,
                                                                    len(dims)))
  value = dims[-1].value
  if value is None:
    raise ValueError('last dimension shape must be known but is None')
  return value


def two_element_tuple(int_or_tuple):
  """Converts `int_or_tuple` to height, width.

  Several of the functions that follow accept arguments as either
  a tuple of 2 integers or a single integer.  A single integer
  indicates that the 2 values of the tuple are the same.

  This functions normalizes the input value by always returning a tuple.

  Args:
    int_or_tuple: A list of 2 ints, a single int or a `TensorShape`.

  Returns:
    A tuple with 2 values.

  Raises:
    ValueError: If `int_or_tuple` it not well formed.
  """
  if isinstance(int_or_tuple, (list, tuple)):
    if len(int_or_tuple) != 2:
      raise ValueError('Must be a list with 2 elements: %s' % int_or_tuple)
    return int(int_or_tuple[0]), int(int_or_tuple[1])
  if isinstance(int_or_tuple, int):
    return int(int_or_tuple), int(int_or_tuple)
  if isinstance(int_or_tuple, tensor_shape.TensorShape):
    if len(int_or_tuple) == 2:
      return int_or_tuple[0], int_or_tuple[1]
  raise ValueError('Must be an int, a list with 2 elements or a TensorShape of '
                   'length 2')


def n_positive_integers(n, value):
  """Converts `value` to a sequence of `n` positive integers.

  `value` may be either be a sequence of values convertible to `int`, or a
  single value convertible to `int`, in which case the the resultant integer is
  duplicated `n` times.  It may also be a TensorShape of rank `n`.

  Args:
    n: Length of sequence to return.
    value: Either a single value convertible to a positive `int` or an
      `n`-element sequence of values convertible to a positive `int`.

  Returns:
    A tuple of `n` positive integers.

  Raises:
    TypeError: If `n` is not convertible to an integer.
    ValueError: If `n` or `value` are invalid.
  """

  n_orig = n
  n = int(n)
  if n < 1 or n != n_orig:
    raise ValueError('n must be a positive integer')

  try:
    value = int(value)
  except (TypeError, ValueError):
    sequence_len = len(value)
    if sequence_len != n:
      raise ValueError(
          'Expected sequence of %d positive integers, but received %r' %
          (n, value))
    try:
      values = tuple(int(x) for x in value)
    except:
      raise ValueError(
          'Expected sequence of %d positive integers, but received %r' %
          (n, value))
    for x in values:
      if x < 1:
        raise ValueError('expected positive integer, but received %d' % x)
    return values

  if value < 1:
    raise ValueError('expected positive integer, but received %d' % value)
  return (value,) * n
