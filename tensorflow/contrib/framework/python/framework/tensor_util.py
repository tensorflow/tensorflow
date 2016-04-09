# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tensor utility functions.

@@assert_negative
@@assert_positive
@@assert_non_negative
@@assert_non_positive
@@assert_less
@@assert_less_equal
@@assert_rank
@@assert_rank_at_least
@@assert_same_float_dtype
@@assert_scalar_int
@@is_numeric_tensor
@@is_non_decreasing
@@is_strictly_increasing
@@local_variable
@@reduce_sum_n
@@with_shape
@@with_same_shape
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
from tensorflow.python.ops import variables

__all__ = [
    'assert_same_float_dtype', 'is_numeric_tensor', 'assert_scalar_int',
    'local_variable', 'reduce_sum_n', 'with_shape', 'with_same_shape',
    'assert_positive', 'assert_negative', 'assert_non_positive',
    'assert_non_negative', 'assert_less', 'assert_less_equal',
    'assert_rank', 'assert_rank_at_least',
    'is_strictly_increasing', 'is_non_decreasing',
]


NUMERIC_TYPES = frozenset([dtypes.float32, dtypes.float64, dtypes.int8,
                           dtypes.int16, dtypes.int32, dtypes.int64,
                           dtypes.uint8, dtypes.qint8, dtypes.qint32,
                           dtypes.quint8, dtypes.complex64])


def assert_negative(x, data=None, summarize=None, name=None):
  """Assert the condition `x < 0` holds element-wise.

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

  Positive means, for every element `x[i]` of `x`, we have `x[i] > 0`.
  If `x` is empty this is trivially satisfied.

  Args:
    x:  Numeric `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to "assert_negative".

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


def assert_less(x, y, data=None, summarize=None, name=None):
  """Assert the condition `x < y` holds element-wise.

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
          'Condition x < y did not hold element-wise: x = ',
          x.name, x, 'y = ', y.name, y]
    condition = math_ops.reduce_all(math_ops.less(x, y))
    return logging_ops.Assert(condition, data, summarize=summarize)


def assert_less_equal(x, y, data=None, summarize=None, name=None):
  """Assert the condition `x <= y` holds element-wise.

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
          'Condition x <= y did not hold element-wise: x = ',
          x.name, x, 'y = ', y.name, y]
    condition = math_ops.reduce_all(math_ops.less_equal(x, y))
    return logging_ops.Assert(condition, data, summarize=summarize)


def assert_rank(x, rank, data=None, summarize=None, name=None):
  """Assert `x` has rank equal to `rank`.

  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to "assert_rank".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank.

  Raises:
    ValueError:  If static checks determine `x` has wrong rank.
  """
  with ops.op_scope([x], name, 'assert_rank'):
    x = ops.convert_to_tensor(x, name='x')
    rank = ops.convert_to_tensor(rank, name='rank')

    # Attempt to statically defined rank.
    x_rank_static = x.get_shape().ndims
    rank_static = tensor_util.constant_value(rank)
    if x_rank_static is not None and rank_static is not None:
      if x_rank_static != rank_static:
        raise ValueError(
            'Tensor %s must have rank %d.  Received rank %d, shape %s'
            % (x.name, rank_static, x_rank_static, x.get_shape()))
      return control_flow_ops.no_op(name='static_checks_determined_all_ok')

    # Assert dynamically.
    if data is None:
      data = [
          'Tensor %s must have rank' % x.name,
          rank,
          'Received shape: ',
          array_ops.shape(x)]
    condition = math_ops.equal(array_ops.rank(x), rank)
    return logging_ops.Assert(condition, data, summarize=summarize)


def assert_rank_at_least(x, rank, data=None, summarize=None, name=None):
  """Assert `x` has rank equal to `rank` or higher.

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
  with ops.op_scope([x], name, 'assert_rank_at_least'):
    x = ops.convert_to_tensor(x, name='x')
    rank = ops.convert_to_tensor(rank, name='rank')

    # Attempt to statically defined rank.
    x_rank_static = x.get_shape().ndims
    rank_static = tensor_util.constant_value(rank)
    if x_rank_static is not None and rank_static is not None:
      if x_rank_static < rank_static:
        raise ValueError(
            'Tensor %s must have rank %d.  Received rank %d, shape %s'
            % (x.name, rank_static, x_rank_static, x.get_shape()))
      return control_flow_ops.no_op(name='static_checks_determined_all_ok')

    if data is None:
      data = [
          'Tensor %s must have rank at least' % x.name,
          rank,
          'Received shape: ',
          array_ops.shape(x)]
    condition = math_ops.greater_equal(array_ops.rank(x), rank)
    return logging_ops.Assert(condition, data, summarize=summarize)


def _get_diff_for_monotonic_comparison(x):
  """Gets the difference x[1:] - x[:-1]."""
  x = array_ops.reshape(x, [-1])
  if not is_numeric_tensor(x):
    raise ValueError('Expected x to be numeric, instead found: %s' % x)

  # If x has less than 2 elements, there is nothing to compare.  So return [].
  is_shorter_than_two = math_ops.less(array_ops.size(x), 2)
  short_result = lambda: ops.convert_to_tensor([], dtype=x.dtype)

  # With 2 or more elements, return x[1:] - x[:-1]
  s_len = array_ops.shape(x) - 1
  diff = lambda: array_ops.slice(x, [1], s_len) - array_ops.slice(x, [0], s_len)
  return control_flow_ops.cond(is_shorter_than_two, short_result, diff)


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
    ValueError: if `x` is not a numeric tensor.
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
    ValueError: if `x` is not a numeric tensor.
  """
  with ops.op_scope([x], name, 'is_strictly_increasing'):
    diff = _get_diff_for_monotonic_comparison(x)
    # When len(x) = 1, diff = [], less = [], and reduce_all([]) = True.
    zero = ops.convert_to_tensor(0, dtype=diff.dtype)
    return math_ops.reduce_all(math_ops.less(zero, diff))


def _assert_same_base_type(items, expected_type=None):
  """Asserts all items are of the same base type.

  Args:
    items: List of graph items (e.g., `Variable`, `Tensor`, `SparseTensor`,
        `Operation`, or `IndexedSlices`). Can include `None` elements, which
        will be ignored.
    expected_type: Expected type. If not specified, assert all items are
        of the same base type.
  Returns:
    Validated type, or none if neither expected_type nor items provided.

  Raises:
    ValueError: If any types do not match.
  """
  original_item_str = None
  for item in items:
    if item is not None:
      item_type = item.dtype.base_dtype
      if not expected_type:
        expected_type = item_type
        original_item_str = item.name if hasattr(item, 'name') else str(item)
      elif expected_type != item_type:
        raise ValueError('%s, type=%s, must be of the same type (%s)%s.' % (
            item.name if hasattr(item, 'name') else str(item),
            item_type, expected_type,
            (' as %s' % original_item_str) if original_item_str else ''))
  return expected_type


def assert_same_float_dtype(tensors=None, dtype=None):
  """Validate and return float type based on `tensors` and `dtype`.

  For ops such as matrix multiplication, inputs and weights must be of the
  same float type. This function validates that all `tensors` are the same type,
  validates that type is `dtype` (if supplied), and returns the type. Type must
  be `dtypes.float32` or `dtypes.float64`. If neither `tensors` nor
  `dtype` is supplied, default to `dtypes.float32`.

  Args:
    tensors: Tensors of input values. Can include `None` elements, which will be
        ignored.
    dtype: Expected type.
  Returns:
    Validated type.
  Raises:
    ValueError: if neither `tensors` nor `dtype` is supplied, or result is not
        float.
  """
  if tensors:
    dtype = _assert_same_base_type(tensors, dtype)
  if not dtype:
    dtype = dtypes.float32
  elif not dtype.is_floating:
    raise ValueError('Expected float, got %s.' % dtype)
  return dtype


def assert_scalar_int(tensor):
  """Assert `tensor` is 0-D, of type `tf.int32` or `tf.int64`.

  Args:
    tensor: Tensor to test.
  Returns:
    `tensor`, for chaining.
  Raises:
    ValueError: if `tensor` is not 0-D, of type `tf.int32` or `tf.int64`.
  """
  data_type = tensor.dtype
  if data_type.base_dtype not in [dtypes.int32, dtypes.int64]:
    raise ValueError('Unexpected type %s for %s.' % (data_type, tensor.name))
  shape = tensor.get_shape()
  if shape.ndims != 0:
    raise ValueError('Unexpected shape %s for %s.' % (shape, tensor.name))
  return tensor


def is_numeric_tensor(tensor):
  return isinstance(tensor, ops.Tensor) and tensor.dtype in NUMERIC_TYPES


# TODO(ptucker): Move to tf.variables?
def local_variable(initial_value, validate_shape=True, name=None):
  """Create variable and add it to `GraphKeys.LOCAL_VARIABLES` collection.

  Args:
    initial_value: See variables.Variable.__init__.
    validate_shape: See variables.Variable.__init__.
    name: See variables.Variable.__init__.
  Returns:
    New variable.
  """
  return variables.Variable(
      initial_value, trainable=False,
      collections=[ops.GraphKeys.LOCAL_VARIABLES],
      validate_shape=validate_shape, name=name)


def reduce_sum_n(tensors, name=None):
  """Reduce tensors to a scalar sum.

  This reduces each tensor in `tensors` to a scalar via `tf.reduce_sum`, then
  adds them via `tf.add_n`.

  Args:
    tensors: List of tensors, all of the same numeric type.
    name: Tensor name, and scope for all other ops.

  Returns:
    Total loss tensor, or None if no losses have been configured.

  Raises:
    ValueError: if `losses` is missing or empty.
  """
  if not tensors:
    raise ValueError('No tensors provided.')
  tensors = [math_ops.reduce_sum(t, name='%s/sum' % t.op.name) for t in tensors]
  if len(tensors) == 1:
    return tensors[0]
  with ops.op_scope(tensors, name, 'reduce_sum_n') as scope:
    return math_ops.add_n(tensors, name=scope)


def _all_equal(tensor0, tensor1):
  with ops.op_scope([tensor0, tensor1], 'all_equal') as scope:
    return math_ops.reduce_all(
        math_ops.equal(tensor0, tensor1, name='equal'), name=scope)


def _is_rank(expected_rank, actual_tensor):
  """Returns whether actual_tensor's rank is expected_rank.

  Args:
    expected_rank: Integer defining the expected rank, or tensor of same.
    actual_tensor: Tensor to test.
  Returns:
    New tensor.
  """
  with ops.op_scope([actual_tensor], 'is_rank') as scope:
    expected = ops.convert_to_tensor(expected_rank, name='expected')
    actual = array_ops.rank(actual_tensor, name='actual')
    return math_ops.equal(expected, actual, name=scope)


def _is_shape(expected_shape, actual_tensor, actual_shape=None):
  """Returns whether actual_tensor's shape is expected_shape.

  Args:
    expected_shape: Integer list defining the expected shape, or tensor of same.
    actual_tensor: Tensor to test.
    actual_shape: Shape of actual_tensor, if we already have it.
  Returns:
    New tensor.
  """
  with ops.op_scope([actual_tensor], 'is_shape') as scope:
    is_rank = _is_rank(array_ops.size(expected_shape), actual_tensor)
    if actual_shape is None:
      actual_shape = array_ops.shape(actual_tensor, name='actual')
    shape_equal = _all_equal(
        ops.convert_to_tensor(expected_shape, name='expected'),
        actual_shape)
    return math_ops.logical_and(is_rank, shape_equal, name=scope)


def _assert_shape_op(expected_shape, actual_tensor):
  """Asserts actual_tensor's shape is expected_shape.

  Args:
    expected_shape: List of integers defining the expected shape, or tensor of
        same.
    actual_tensor: Tensor to test.
  Returns:
    New assert tensor.
  """
  with ops.op_scope([actual_tensor], 'assert_shape') as scope:
    actual_shape = array_ops.shape(actual_tensor, name='actual')
    is_shape = _is_shape(expected_shape, actual_tensor, actual_shape)
    return logging_ops.Assert(
        is_shape, [
            'Wrong shape for %s [expected] [actual].' % actual_tensor.name,
            expected_shape,
            actual_shape
        ], name=scope)


def with_same_shape(expected_tensor, tensor):
  """Assert tensors are the same shape, from the same graph.

  Args:
    expected_tensor: Tensor with expected shape.
    tensor: Tensor of actual values.
  Returns:
    Tuple of (actual_tensor, label_tensor), possibly with assert ops added.
  """
  with ops.op_scope([expected_tensor, tensor], '%s/' % tensor.op.name):
    tensor_shape = expected_tensor.get_shape()
    expected_shape = (
        tensor_shape.as_list() if tensor_shape.is_fully_defined()
        else array_ops.shape(expected_tensor, name='expected_shape'))
    return with_shape(expected_shape, tensor)


def _is_tensor(t):
  return isinstance(t, (ops.Tensor, ops.SparseTensor, variables.Variable))


def with_shape(expected_shape, tensor):
  """Asserts tensor has expected shape.

  If tensor shape and expected_shape, are fully defined, assert they match.
  Otherwise, add assert op that will validate the shape when tensor is
  evaluated, and set shape on tensor.

  Args:
    expected_shape: Expected shape to assert, as a 1D array of ints, or tensor
        of same.
    tensor: Tensor whose shape we're validating.
  Returns:
    tensor, perhaps with a dependent assert operation.
  Raises:
    ValueError: if tensor has an invalid shape.
  """
  if isinstance(tensor, ops.SparseTensor):
    raise ValueError('SparseTensor not supported.')

  # Shape type must be 1D int32.
  if _is_tensor(expected_shape):
    if expected_shape.dtype.base_dtype != dtypes.int32:
      raise ValueError(
          'Invalid dtype %s for shape %s expected of tensor %s.' % (
              expected_shape.dtype, expected_shape, tensor.name))
  if isinstance(expected_shape, (list, tuple)):
    if not expected_shape:
      expected_shape = np.asarray([], dtype=np.int32)
    else:
      np_expected_shape = np.asarray(expected_shape)
      expected_shape = (
          np.asarray(expected_shape, dtype=np.int32)
          if np_expected_shape.dtype == np.int64 else np_expected_shape)
  if isinstance(expected_shape, np.ndarray):
    if expected_shape.ndim > 1:
      raise ValueError(
          'Invalid rank %s for shape %s expected of tensor %s.' % (
              expected_shape.ndim, expected_shape, tensor.name))
    if expected_shape.dtype != np.int32:
      raise ValueError(
          'Invalid dtype %s for shape %s expected of tensor %s.' % (
              expected_shape.dtype, expected_shape, tensor.name))

  actual_shape = tensor.get_shape()

  if not actual_shape.is_fully_defined() or _is_tensor(expected_shape):
    with ops.op_scope([tensor], '%s/' % tensor.op.name):
      if not _is_tensor(expected_shape) and (len(expected_shape) < 1):
        # TODO(irving): Remove scalar special case
        return array_ops.reshape(tensor, [])
      with ops.control_dependencies([_assert_shape_op(expected_shape, tensor)]):
        result = array_ops.identity(tensor)
      if not _is_tensor(expected_shape):
        result.set_shape(expected_shape)
      return result

  if (not _is_tensor(expected_shape) and
      not actual_shape.is_compatible_with(expected_shape)):
    if (len(expected_shape) < 1) and actual_shape.is_compatible_with([1]):
      # TODO(irving): Remove scalar special case.
      with ops.op_scope([tensor], '%s/' % tensor.op.name):
        return array_ops.reshape(tensor, [])
    raise ValueError('Invalid shape for tensor %s, expected %s, got %s.' % (
        tensor.name, expected_shape, actual_shape))

  return tensor
