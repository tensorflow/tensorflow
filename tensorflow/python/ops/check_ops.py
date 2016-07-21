# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-short-docstring-punctuation
"""## Asserts and Boolean Checks

@@assert_negative
@@assert_positive
@@assert_proper_iterable
@@assert_non_negative
@@assert_non_positive
@@assert_equal
@@assert_integer
@@assert_less
@@assert_less_equal
@@assert_rank
@@assert_rank_at_least
@@assert_type
@@is_non_decreasing
@@is_numeric_tensor
@@is_strictly_increasing
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat

NUMERIC_TYPES = frozenset(
    [dtypes.float32, dtypes.float64, dtypes.int8, dtypes.int16, dtypes.int32,
     dtypes.int64, dtypes.uint8, dtypes.qint8, dtypes.qint32, dtypes.quint8,
     dtypes.complex64])

__all__ = [
    'assert_negative',
    'assert_positive',
    'assert_proper_iterable',
    'assert_non_negative',
    'assert_non_positive',
    'assert_equal',
    'assert_integer',
    'assert_less',
    'assert_less_equal',
    'assert_rank',
    'assert_rank_at_least',
    'assert_type',
    'is_non_decreasing',
    'is_numeric_tensor',
    'is_strictly_increasing',
]


def assert_proper_iterable(values):
  """Static assert that values is a "proper" iterable.

  `Ops` that expect iterables of `Tensor` can call this to validate input.
  Useful since `Tensor`, `ndarray`, byte/text type are all iterables themselves.

  Args:
    values:  Object to be checked.

  Raises:
    TypeError:  If `values` is not iterable or is one of
      `Tensor`, `SparseTensor`, `np.array`, `tf.compat.bytes_or_text_types`.
  """
  unintentional_iterables = (
      (ops.Tensor, ops.SparseTensor, np.ndarray)
      + compat.bytes_or_text_types
  )
  if isinstance(values, unintentional_iterables):
    raise TypeError(
        'Expected argument "values" to be a "proper" iterable.  Found: %s' %
        type(values))

  if not hasattr(values, '__iter__'):
    raise TypeError(
        'Expected argument "values" to be iterable.  Found: %s' % type(values))


def assert_negative(x, data=None, summarize=None, name=None):
  """Assert the condition `x < 0` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_negative(x)]):
    output = tf.reduce_sum(x)
  ```

  Example of adding dependency to the tensor being checked:

  ```python
  x = tf.with_dependencies([tf.assert_negative(x)], x)
  ```

  Negative means, for every element `x[i]` of `x`, we have `x[i] < 0`.
  If `x` is empty this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to "assert_negative".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all negative.
  """
  with ops.op_scope([x, data], name, 'assert_negative'):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      data = ['Condition x < 0 did not hold element-wise: x = ', x.name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less(x, zero, data=data, summarize=summarize)


def assert_positive(x, data=None, summarize=None, name=None):
  """Assert the condition `x > 0` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_positive(x)]):
    output = tf.reduce_sum(x)
  ```

  Example of adding dependency to the tensor being checked:

  ```python
  x = tf.with_dependencies([tf.assert_positive(x)], x)
  ```

  Positive means, for every element `x[i]` of `x`, we have `x[i] > 0`.
  If `x` is empty this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to "assert_positive".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all positive.
  """
  with ops.op_scope([x, data], name, 'assert_positive'):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      data = ['Condition x > 0 did not hold element-wise: x = ', x.name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less(zero, x, data=data, summarize=summarize)


def assert_non_negative(x, data=None, summarize=None, name=None):
  """Assert the condition `x >= 0` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_non_negative(x)]):
    output = tf.reduce_sum(x)
  ```

  Example of adding dependency to the tensor being checked:

  ```python
  x = tf.with_dependencies([tf.assert_non_negative(x)], x)
  ```

  Non-negative means, for every element `x[i]` of `x`, we have `x[i] >= 0`.
  If `x` is empty this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).
      Defaults to "assert_non_negative".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all non-negative.
  """
  with ops.op_scope([x, data], name, 'assert_non_negative'):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      data = ['Condition x >= 0 did not hold element-wise: x = ', x.name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less_equal(zero, x, data=data, summarize=summarize)


def assert_non_positive(x, data=None, summarize=None, name=None):
  """Assert the condition `x <= 0` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_non_positive(x)]):
    output = tf.reduce_sum(x)
  ```

  Example of adding dependency to the tensor being checked:

  ```python
  x = tf.with_dependencies([tf.assert_non_positive(x)], x)
  ```

  Non-positive means, for every element `x[i]` of `x`, we have `x[i] <= 0`.
  If `x` is empty this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).
      Defaults to "assert_non_positive".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all non-positive.
  """
  with ops.op_scope([x, data], name, 'assert_non_positive'):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      data = ['Condition x <= 0 did not hold element-wise: x = ', x.name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less_equal(x, zero, data=data, summarize=summarize)


def assert_equal(x, y, data=None, summarize=None, name=None):
  """Assert the condition `x == y` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_equal(x, y)]):
    output = tf.reduce_sum(x)
  ```

  Example of adding dependency to the tensor being checked:

  ```python
  x = tf.with_dependencies([tf.assert_equal(x, y)], x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have `x[i] == y[i]`.
  If both `x` and `y` are empty, this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to "assert_equal".

  Returns:
    Op that raises `InvalidArgumentError` if `x == y` is False.
  """
  with ops.op_scope([x, y, data], name, 'assert_equal'):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')
    if data is None:
      data = [
          'Condition x == y did not hold element-wise: x = ', x.name, x, 'y = ',
          y.name, y
      ]
    condition = math_ops.reduce_all(math_ops.equal(x, y))
    return logging_ops.Assert(condition, data, summarize=summarize)


def assert_less(x, y, data=None, summarize=None, name=None):
  """Assert the condition `x < y` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_less(x, y)]):
    output = tf.reduce_sum(x)
  ```

  Example of adding dependency to the tensor being checked:

  ```python
  x = tf.with_dependencies([tf.assert_less(x, y)], x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have `x[i] < y[i]`.
  If both `x` and `y` are empty, this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to "assert_less".

  Returns:
    Op that raises `InvalidArgumentError` if `x < y` is False.
  """
  with ops.op_scope([x, y, data], name, 'assert_less'):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')
    if data is None:
      data = [
          'Condition x < y did not hold element-wise: x = ', x.name, x, 'y = ',
          y.name, y
      ]
    condition = math_ops.reduce_all(math_ops.less(x, y))
    return logging_ops.Assert(condition, data, summarize=summarize)


def assert_less_equal(x, y, data=None, summarize=None, name=None):
  """Assert the condition `x <= y` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_less_equal(x, y)]):
    output = tf.reduce_sum(x)
  ```

  Example of adding dependency to the tensor being checked:

  ```python
  x = tf.with_dependencies([tf.assert_less_equal(x, y)], x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have `x[i] <= y[i]`.
  If both `x` and `y` are empty, this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to "assert_less_equal"

  Returns:
    Op that raises `InvalidArgumentError` if `x <= y` is False.
  """
  with ops.op_scope([x, y, data], name, 'assert_less_equal'):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')
    if data is None:
      data = [
          'Condition x <= y did not hold element-wise: x = ', x.name, x, 'y = ',
          y.name, y
      ]
    condition = math_ops.reduce_all(math_ops.less_equal(x, y))
    return logging_ops.Assert(condition, data, summarize=summarize)


def _assert_rank_condition(x, rank, static_condition, dynamic_condition, data,
                           summarize, name):
  """Assert `x` has a rank that satisfies a given condition.

  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar `Tensor`.
    static_condition:   A python function that takes `[actual_rank, given_rank]`
      and returns `True` if the condition is satisfied, `False` otherwise.
    dynamic_condition:  An `op` that takes [actual_rank, given_rank]
      and return `True` if the condition is satisfied, `False` otherwise.
    data:  The tensors to print out if the condition is false.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).
      Defaults to "assert_rank_at_least".

  Returns:
    Op raising `InvalidArgumentError` if `x` fails dynamic_condition.

  Raises:
    ValueError:  If static checks determine `x` fails static_condition.
  """
  with ops.op_scope([x], name, 'assert_rank'):
    x = ops.convert_to_tensor(x, name='x')
    rank = ops.convert_to_tensor(rank, name='rank')

    # Attempt to statically defined rank.
    x_rank_static = x.get_shape().ndims
    rank_static = tensor_util.constant_value(rank)

    assert_type(rank, dtypes.int32)

    if rank_static is not None:
      if rank_static.ndim != 0:
        raise ValueError('Rank must be a scalar')

      if x_rank_static is not None:
        if not static_condition(x_rank_static, rank_static):
          raise ValueError('Static rank condition failed', x_rank_static,
                           rank_static)
        return control_flow_ops.no_op(name='static_checks_determined_all_ok')

    condition = dynamic_condition(array_ops.rank(x), rank)

    if rank_static is None:
      rank_check = assert_rank(rank,
                               0,
                               data=[['Rank must be a scalar.'
                                      'Received rank:'], rank])

      condition = control_flow_ops.with_dependencies([rank_check], condition)

  return logging_ops.Assert(condition, data, summarize=summarize)


def assert_rank(x, rank, data=None, summarize=None, name=None):
  """Assert `x` has rank equal to `rank`.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_rank(x, 2)]):
    output = tf.reduce_sum(x)
  ```

  Example of adding dependency to the tensor being checked:

  ```python
  x = tf.with_dependencies([tf.assert_rank(x, 2)], x)
  ```

  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar integer `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to "assert_rank".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank.

  Raises:
    ValueError:  If static checks determine `x` has wrong rank.
  """

  static_condition = lambda actual_rank, given_rank: actual_rank == given_rank
  dynamic_condition = math_ops.equal

  if data is None:
    data = [
        'Tensor %s must have rank' % x.name, rank, 'Received shape: ',
        array_ops.shape(x)
    ]

  try:
    assert_op = _assert_rank_condition(x, rank, static_condition,
                                       dynamic_condition, data, summarize, name)

  except ValueError as e:
    if e.args[0] == 'Static rank condition failed':
      raise ValueError(
          'Tensor %s must have rank %d.  Received rank %d, shape %s' %
          (x.name, e.args[2], e.args[1], x.get_shape()))
    else:
      raise

  return assert_op


def assert_rank_at_least(x, rank, data=None, summarize=None, name=None):
  """Assert `x` has rank equal to `rank` or higher.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_rank_at_least(x, 2)]):
    output = tf.reduce_sum(x)
  ```

  Example of adding dependency to the tensor being checked:

  ```python
  x = tf.with_dependencies([tf.assert_rank_at_least(x, 2)], x)
  ```

  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).
      Defaults to "assert_rank_at_least".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank or higher.

  Raises:
    ValueError:  If static checks determine `x` has wrong rank.
  """

  static_condition = lambda actual_rank, given_rank: actual_rank >= given_rank
  dynamic_condition = math_ops.greater_equal
  if data is None:
    data = [
        'Tensor %s must have rank at least' % x.name, rank,
        'Received shape: ', array_ops.shape(x)
    ]

  try:
    assert_op = _assert_rank_condition(x, rank, static_condition,
                                       dynamic_condition, data, summarize, name)

  except ValueError as e:
    if e.args[0] == 'Static rank condition failed':
      raise ValueError(
          'Tensor %s must have rank at least %d.  Received rank %d, shape %s' %
          (x.name, e.args[2], e.args[1], x.get_shape()))
    else:
      raise

  return assert_op


def assert_integer(x, data=None, summarize=None, name=None):
  """Assert that `x` is of integer dtype.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_integer(x)]):
    output = tf.reduce_sum(x)
  ```

  Example of adding dependency to the tensor being checked:

  ```python
  x = tf.with_dependencies([tf.assert_integer(x)], x)
  ```

  Args:
    x: `Tensor` whose basetype is integer and is not quantized.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to "assert_integer".

  Returns:
    Op that raises `InvalidArgumentError` if `x == y` is False.
  """
  with ops.op_scope([x], name, 'assert_integer'):
    x = ops.convert_to_tensor(x, name='x')
    data = ['x is not of integer dtype: x = ', x.name, x]
    condition = x.dtype.is_integer
    return logging_ops.Assert(condition, data, summarize=summarize)


def assert_type(tensor, tf_type):
  """Asserts that the given `Tensor` is of the specified type.

  Args:
    tensor: A tensorflow `Tensor`.
    tf_type: A tensorflow type (dtypes.float32, tf.int64, dtypes.bool, etc).

  Raises:
    ValueError: If the tensors data type doesn't match tf_type.
  """
  if tensor.dtype != tf_type:
    raise ValueError('%s must be of type %s' % (tensor.op.name, tf_type))


def _get_diff_for_monotonic_comparison(x):
  """Gets the difference x[1:] - x[:-1]."""
  x = array_ops.reshape(x, [-1])
  if not is_numeric_tensor(x):
    raise TypeError('Expected x to be numeric, instead found: %s' % x)

  # If x has less than 2 elements, there is nothing to compare.  So return [].
  is_shorter_than_two = math_ops.less(array_ops.size(x), 2)
  short_result = lambda: ops.convert_to_tensor([], dtype=x.dtype)

  # With 2 or more elements, return x[1:] - x[:-1]
  s_len = array_ops.shape(x) - 1
  diff = lambda: array_ops.slice(x, [1], s_len) - array_ops.slice(x, [0], s_len)
  return control_flow_ops.cond(is_shorter_than_two, short_result, diff)


def is_numeric_tensor(tensor):
  return isinstance(tensor, ops.Tensor) and tensor.dtype in NUMERIC_TYPES


def is_non_decreasing(x, name=None):
  """Returns `True` if `x` is non-decreasing.

  Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`
  is non-decreasing if for every adjacent pair we have `x[i] <= x[i+1]`.
  If `x` has less than two elements, it is trivially non-decreasing.

  See also:  `is_strictly_increasing`

  Args:
    x: Numeric `Tensor`.
    name: A name for this operation (optional).  Defaults to "is_non_decreasing"

  Returns:
    Boolean `Tensor`, equal to `True` iff `x` is non-decreasing.

  Raises:
    TypeError: if `x` is not a numeric tensor.
  """
  with ops.op_scope([x], name, 'is_non_decreasing'):
    diff = _get_diff_for_monotonic_comparison(x)
    # When len(x) = 1, diff = [], less_equal = [], and reduce_all([]) = True.
    zero = ops.convert_to_tensor(0, dtype=diff.dtype)
    return math_ops.reduce_all(math_ops.less_equal(zero, diff))


def is_strictly_increasing(x, name=None):
  """Returns `True` if `x` is strictly increasing.

  Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`
  is strictly increasing if for every adjacent pair we have `x[i] < x[i+1]`.
  If `x` has less than two elements, it is trivially strictly increasing.

  See also:  `is_non_decreasing`

  Args:
    x: Numeric `Tensor`.
    name: A name for this operation (optional).
      Defaults to "is_strictly_increasing"

  Returns:
    Boolean `Tensor`, equal to `True` iff `x` is strictly increasing.

  Raises:
    TypeError: if `x` is not a numeric tensor.
  """
  with ops.op_scope([x], name, 'is_strictly_increasing'):
    diff = _get_diff_for_monotonic_comparison(x)
    # When len(x) = 1, diff = [], less = [], and reduce_all([]) = True.
    zero = ops.convert_to_tensor(0, dtype=diff.dtype)
    return math_ops.reduce_all(math_ops.less(zero, diff))
