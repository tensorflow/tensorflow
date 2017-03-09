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
"""Asserts and Boolean Checks.

See the @{$python/check_ops} guide.

@@assert_negative
@@assert_positive
@@assert_non_negative
@@assert_non_positive
@@assert_equal
@@assert_none_equal
@@assert_less
@@assert_less_equal
@@assert_greater
@@assert_greater_equal
@@assert_rank
@@assert_rank_at_least
@@assert_type
@@assert_integer
@@assert_proper_iterable
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
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
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
    'assert_none_equal',
    'assert_integer',
    'assert_less',
    'assert_less_equal',
    'assert_greater',
    'assert_greater_equal',
    'assert_rank',
    'assert_rank_at_least',
    'assert_rank_in',
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
      (ops.Tensor, sparse_tensor.SparseTensor, np.ndarray)
      + compat.bytes_or_text_types
  )
  if isinstance(values, unintentional_iterables):
    raise TypeError(
        'Expected argument "values" to be a "proper" iterable.  Found: %s' %
        type(values))

  if not hasattr(values, '__iter__'):
    raise TypeError(
        'Expected argument "values" to be iterable.  Found: %s' % type(values))


def assert_negative(x, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x < 0` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_negative(x)]):
    output = tf.reduce_sum(x)
  ```

  Negative means, for every element `x[i]` of `x`, we have `x[i] < 0`.
  If `x` is empty this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_negative".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all negative.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_negative', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      data = [
          message, 'Condition x < 0 did not hold element-wise: x = ', x.name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less(x, zero, data=data, summarize=summarize)


def assert_positive(x, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x > 0` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_positive(x)]):
    output = tf.reduce_sum(x)
  ```

  Positive means, for every element `x[i]` of `x`, we have `x[i] > 0`.
  If `x` is empty this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_positive".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all positive.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_positive', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      data = [
          message, 'Condition x > 0 did not hold element-wise: x = ', x.name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less(zero, x, data=data, summarize=summarize)


def assert_non_negative(x, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x >= 0` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_non_negative(x)]):
    output = tf.reduce_sum(x)
  ```

  Non-negative means, for every element `x[i]` of `x`, we have `x[i] >= 0`.
  If `x` is empty this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_non_negative".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all non-negative.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_non_negative', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      data = [
          message,
          'Condition x >= 0 did not hold element-wise: x = ', x.name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less_equal(zero, x, data=data, summarize=summarize)


def assert_non_positive(x, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x <= 0` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_non_positive(x)]):
    output = tf.reduce_sum(x)
  ```

  Non-positive means, for every element `x[i]` of `x`, we have `x[i] <= 0`.
  If `x` is empty this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_non_positive".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all non-positive.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_non_positive', [x, data]):
    x = ops.convert_to_tensor(x, name='x')
    if data is None:
      data = [
          message,
          'Condition x <= 0 did not hold element-wise: x = ', x.name, x]
    zero = ops.convert_to_tensor(0, dtype=x.dtype)
    return assert_less_equal(x, zero, data=data, summarize=summarize)


def assert_equal(x, y, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x == y` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_equal(x, y)]):
    output = tf.reduce_sum(x)
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
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_equal".

  Returns:
    Op that raises `InvalidArgumentError` if `x == y` is False.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_equal', [x, y, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')
    if data is None:
      data = [
          message,
          'Condition x == y did not hold element-wise: x = ', x.name, x, 'y = ',
          y.name, y
      ]
    condition = math_ops.reduce_all(math_ops.equal(x, y))
    return control_flow_ops.Assert(condition, data, summarize=summarize)


def assert_none_equal(
    x, y, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x != y` holds for all elements.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_none_equal(x, y)]):
    output = tf.reduce_sum(x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have `x[i] != y[i]`.
  If both `x` and `y` are empty, this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_none_equal".

  Returns:
    Op that raises `InvalidArgumentError` if `x != y` is ever False.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_none_equal', [x, y, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')
    if data is None:
      data = [
          message,
          'Condition x != y did not hold for every single element: x = ',
          x.name, x,
          'y = ', y.name, y
      ]
    condition = math_ops.reduce_all(math_ops.not_equal(x, y))
    return control_flow_ops.Assert(condition, data, summarize=summarize)


def assert_less(x, y, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x < y` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_less(x, y)]):
    output = tf.reduce_sum(x)
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
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_less".

  Returns:
    Op that raises `InvalidArgumentError` if `x < y` is False.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_less', [x, y, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')
    if data is None:
      data = [
          message,
          'Condition x < y did not hold element-wise: x = ', x.name, x, 'y = ',
          y.name, y
      ]
    condition = math_ops.reduce_all(math_ops.less(x, y))
    return control_flow_ops.Assert(condition, data, summarize=summarize)


def assert_less_equal(x, y, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x <= y` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_less_equal(x, y)]):
    output = tf.reduce_sum(x)
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
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_less_equal"

  Returns:
    Op that raises `InvalidArgumentError` if `x <= y` is False.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_less_equal', [x, y, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')
    if data is None:
      data = [
          message,
          'Condition x <= y did not hold element-wise: x = ', x.name, x, 'y = ',
          y.name, y
      ]
    condition = math_ops.reduce_all(math_ops.less_equal(x, y))
    return control_flow_ops.Assert(condition, data, summarize=summarize)


def assert_greater(x, y, data=None, summarize=None, message=None, name=None):
  """Assert the condition `x > y` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_greater(x, y)]):
    output = tf.reduce_sum(x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have `x[i] > y[i]`.
  If both `x` and `y` are empty, this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_greater".

  Returns:
    Op that raises `InvalidArgumentError` if `x > y` is False.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_greater', [x, y, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')
    if data is None:
      data = [
          message,
          'Condition x > y did not hold element-wise: x = ', x.name, x, 'y = ',
          y.name, y
      ]
    condition = math_ops.reduce_all(math_ops.greater(x, y))
    return control_flow_ops.Assert(condition, data, summarize=summarize)


def assert_greater_equal(x, y, data=None, summarize=None, message=None,
                         name=None):
  """Assert the condition `x >= y` holds element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_greater_equal(x, y)]):
    output = tf.reduce_sum(x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have `x[i] >= y[i]`.
  If both `x` and `y` are empty, this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to
      "assert_greater_equal"

  Returns:
    Op that raises `InvalidArgumentError` if `x >= y` is False.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_greater_equal', [x, y, data]):
    x = ops.convert_to_tensor(x, name='x')
    y = ops.convert_to_tensor(y, name='y')
    if data is None:
      data = [
          message,
          'Condition x >= y did not hold element-wise: x = ', x.name, x, 'y = ',
          y.name, y
      ]
    condition = math_ops.reduce_all(math_ops.greater_equal(x, y))
    return control_flow_ops.Assert(condition, data, summarize=summarize)


def _assert_rank_condition(
    x, rank, static_condition, dynamic_condition, data, summarize):
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

  Returns:
    Op raising `InvalidArgumentError` if `x` fails dynamic_condition.

  Raises:
    ValueError:  If static checks determine `x` fails static_condition.
  """
  assert_type(rank, dtypes.int32)

  # Attempt to statically defined rank.
  rank_static = tensor_util.constant_value(rank)
  if rank_static is not None:
    if rank_static.ndim != 0:
      raise ValueError('Rank must be a scalar.')

    x_rank_static = x.get_shape().ndims
    if x_rank_static is not None:
      if not static_condition(x_rank_static, rank_static):
        raise ValueError(
            'Static rank condition failed', x_rank_static, rank_static)
      return control_flow_ops.no_op(name='static_checks_determined_all_ok')

  condition = dynamic_condition(array_ops.rank(x), rank)

  # Add the condition that `rank` must have rank zero.  Prevents the bug where
  # someone does assert_rank(x, [n]), rather than assert_rank(x, n).
  if rank_static is None:
    this_data = ['Rank must be a scalar. Received rank: ', rank]
    rank_check = assert_rank(rank, 0, data=this_data)
    condition = control_flow_ops.with_dependencies([rank_check], condition)

  return control_flow_ops.Assert(condition, data, summarize=summarize)


def assert_rank(x, rank, data=None, summarize=None, message=None, name=None):
  """Assert `x` has rank equal to `rank`.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_rank(x, 2)]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar integer `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_rank".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank.
    If static checks determine `x` has correct rank, a `no_op` is returned.

  Raises:
    ValueError:  If static checks determine `x` has wrong rank.
  """
  with ops.name_scope(name, 'assert_rank', (x, rank) + tuple(data or [])):
    x = ops.convert_to_tensor(x, name='x')
    rank = ops.convert_to_tensor(rank, name='rank')
    message = message or ''

    static_condition = lambda actual_rank, given_rank: actual_rank == given_rank
    dynamic_condition = math_ops.equal

    if data is None:
      data = [
          message,
          'Tensor %s must have rank' % x.name, rank, 'Received shape: ',
          array_ops.shape(x)
      ]

    try:
      assert_op = _assert_rank_condition(x, rank, static_condition,
                                         dynamic_condition, data, summarize)

    except ValueError as e:
      if e.args[0] == 'Static rank condition failed':
        raise ValueError(
            '%s.  Tensor %s must have rank %d.  Received rank %d, shape %s' %
            (message, x.name, e.args[2], e.args[1], x.get_shape()))
      else:
        raise

  return assert_op


def assert_rank_at_least(
    x, rank, data=None, summarize=None, message=None, name=None):
  """Assert `x` has rank equal to `rank` or higher.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_rank_at_least(x, 2)]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_rank_at_least".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank or higher.
    If static checks determine `x` has correct rank, a `no_op` is returned.

  Raises:
    ValueError:  If static checks determine `x` has wrong rank.
  """
  with ops.name_scope(
      name, 'assert_rank_at_least', (x, rank) + tuple(data or [])):
    x = ops.convert_to_tensor(x, name='x')
    rank = ops.convert_to_tensor(rank, name='rank')
    message = message or ''

    static_condition = lambda actual_rank, given_rank: actual_rank >= given_rank
    dynamic_condition = math_ops.greater_equal
    if data is None:
      data = [
          message,
          'Tensor %s must have rank at least' % x.name, rank,
          'Received shape: ', array_ops.shape(x)
      ]

    try:
      assert_op = _assert_rank_condition(x, rank, static_condition,
                                         dynamic_condition, data, summarize)

    except ValueError as e:
      if e.args[0] == 'Static rank condition failed':
        raise ValueError(
            '%s.  Tensor %s must have rank at least %d.  Received rank %d, '
            'shape %s' % (message, x.name, e.args[2], e.args[1], x.get_shape()))
      else:
        raise

  return assert_op


def _static_rank_in(actual_rank, given_ranks):
  return actual_rank in given_ranks


def _dynamic_rank_in(actual_rank, given_ranks):
  if len(given_ranks) < 1:
    return ops.convert_to_tensor(False)
  result = math_ops.equal(given_ranks[0], actual_rank)
  for given_rank in given_ranks[1:]:
    result = math_ops.logical_or(
        result, math_ops.equal(given_rank, actual_rank))
  return result


def _assert_ranks_condition(
    x, ranks, static_condition, dynamic_condition, data, summarize):
  """Assert `x` has a rank that satisfies a given condition.

  Args:
    x:  Numeric `Tensor`.
    ranks:  Scalar `Tensor`.
    static_condition:   A python function that takes
      `[actual_rank, given_ranks]` and returns `True` if the condition is
      satisfied, `False` otherwise.
    dynamic_condition:  An `op` that takes [actual_rank, given_ranks]
      and return `True` if the condition is satisfied, `False` otherwise.
    data:  The tensors to print out if the condition is false.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.

  Returns:
    Op raising `InvalidArgumentError` if `x` fails dynamic_condition.

  Raises:
    ValueError:  If static checks determine `x` fails static_condition.
  """
  for rank in ranks:
    assert_type(rank, dtypes.int32)

  # Attempt to statically defined rank.
  ranks_static = tuple([tensor_util.constant_value(rank) for rank in ranks])
  if None not in ranks_static:
    for rank_static in ranks_static:
      if rank_static.ndim != 0:
        raise ValueError('Rank must be a scalar.')

    x_rank_static = x.get_shape().ndims
    if x_rank_static is not None:
      if not static_condition(x_rank_static, ranks_static):
        raise ValueError(
            'Static rank condition failed', x_rank_static, ranks_static)
      return control_flow_ops.no_op(name='static_checks_determined_all_ok')

  condition = dynamic_condition(array_ops.rank(x), ranks)

  # Add the condition that `rank` must have rank zero.  Prevents the bug where
  # someone does assert_rank(x, [n]), rather than assert_rank(x, n).
  for rank, rank_static in zip(ranks, ranks_static):
    if rank_static is None:
      this_data = ['Rank must be a scalar. Received rank: ', rank]
      rank_check = assert_rank(rank, 0, data=this_data)
      condition = control_flow_ops.with_dependencies([rank_check], condition)

  return control_flow_ops.Assert(condition, data, summarize=summarize)


def assert_rank_in(
    x, ranks, data=None, summarize=None, message=None, name=None):
  """Assert `x` has rank in `ranks`.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_rank_in(x, (2, 4))]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x:  Numeric `Tensor`.
    ranks:  Iterable of scalar `Tensor` objects.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_rank_in".

  Returns:
    Op raising `InvalidArgumentError` unless rank of `x` is in `ranks`.
    If static checks determine `x` has matching rank, a `no_op` is returned.

  Raises:
    ValueError:  If static checks determine `x` has mismatched rank.
  """
  with ops.name_scope(
      name, 'assert_rank_in', (x,) + tuple(ranks) + tuple(data or [])):
    x = ops.convert_to_tensor(x, name='x')
    ranks = tuple([ops.convert_to_tensor(rank, name='rank') for rank in ranks])
    message = message or ''

    if data is None:
      data = [
          message, 'Tensor %s must have rank in' % x.name
      ] + list(ranks) + [
          'Received shape: ', array_ops.shape(x)
      ]

    try:
      assert_op = _assert_ranks_condition(x, ranks, _static_rank_in,
                                          _dynamic_rank_in, data, summarize)

    except ValueError as e:
      if e.args[0] == 'Static rank condition failed':
        raise ValueError(
            '%s.  Tensor %s must have rank in %s.  Received rank %d, '
            'shape %s' % (message, x.name, e.args[2], e.args[1], x.get_shape()))
      else:
        raise

  return assert_op


def assert_integer(x, message=None, name=None):
  """Assert that `x` is of integer dtype.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_integer(x)]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x: `Tensor` whose basetype is integer and is not quantized.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_integer".

  Raises:
    TypeError:  If `x.dtype` is anything other than non-quantized integer.

  Returns:
    A `no_op` that does nothing.  Type can be determined statically.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_integer', [x]):
    x = ops.convert_to_tensor(x, name='x')
    if not x.dtype.is_integer:
      err_msg = (
          '%s  Expected "x" to be integer type.  Found: %s of dtype %s'
          % (message, x.name, x.dtype))
      raise TypeError(err_msg)

    return control_flow_ops.no_op('statically_determined_was_integer')


def assert_type(tensor, tf_type, message=None, name=None):
  """Statically asserts that the given `Tensor` is of the specified type.

  Args:
    tensor: A tensorflow `Tensor`.
    tf_type: A tensorflow type (`dtypes.float32`, `tf.int64`, `dtypes.bool`,
      etc).
    message: A string to prefix to the default message.
    name:  A name to give this `Op`.  Defaults to "assert_type"

  Raises:
    TypeError: If the tensors data type doesn't match `tf_type`.

  Returns:
    A `no_op` that does nothing.  Type can be determined statically.
  """
  message = message or ''
  with ops.name_scope(name, 'assert_type', [tensor]):
    tensor = ops.convert_to_tensor(tensor, name='tensor')
    if tensor.dtype != tf_type:
      raise TypeError(
          '%s  %s must be of type %s' % (message, tensor.op.name, tf_type))

    return control_flow_ops.no_op('statically_determined_correct_type')


# pylint: disable=line-too-long
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
  diff = lambda: array_ops.strided_slice(x, [1], [1] + s_len)- array_ops.strided_slice(x, [0], s_len)
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
  with ops.name_scope(name, 'is_non_decreasing', [x]):
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
  with ops.name_scope(name, 'is_strictly_increasing', [x]):
    diff = _get_diff_for_monotonic_comparison(x)
    # When len(x) = 1, diff = [], less = [], and reduce_all([]) = True.
    zero = ops.convert_to_tensor(0, dtype=diff.dtype)
    return math_ops.reduce_all(math_ops.less(zero, diff))
