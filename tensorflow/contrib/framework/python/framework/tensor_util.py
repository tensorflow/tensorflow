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

"""Tensor utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables


__all__ = [
    'assert_same_float_dtype',
    'assert_scalar_int',
    'convert_to_tensor_or_sparse_tensor',
    'is_tensor',
    'reduce_sum_n',
    'remove_squeezable_dimensions',
    'with_shape',
    'with_same_shape']


def _assert_same_base_type(items, expected_type=None):
  r"""Asserts all items are of the same base type.

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
  with ops.name_scope(name, 'reduce_sum_n', tensors) as scope:
    return math_ops.add_n(tensors, name=scope)


def remove_squeezable_dimensions(predictions, labels):
  """Squeeze last dim if ranks of `predictions` and `labels` differ by 1.

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    predictions: Predicted values, a `Tensor` of arbitrary dimensions.
    labels: Label values, a `Tensor` whose dimensions match `predictions`.

  Returns:
    Tuple of `predictions` and `labels`, possibly with last dim squeezed.
  """
  predictions = ops.convert_to_tensor(predictions)
  labels = ops.convert_to_tensor(labels)
  predictions_shape = predictions.get_shape()
  predictions_rank = predictions_shape.ndims
  labels_shape = labels.get_shape()
  labels_rank = labels_shape.ndims
  if (labels_rank is not None) and (predictions_rank is not None):
    # Use static rank.
    rank_diff = predictions_rank - labels_rank
    if rank_diff == -1:
      labels = array_ops.squeeze(labels, [-1])
    elif rank_diff == 1:
      predictions = array_ops.squeeze(predictions, [-1])
    return predictions, labels

  # Use dynamic rank.
  rank_diff = array_ops.rank(predictions) - array_ops.rank(labels)
  if (predictions_rank is None) or (
      predictions_shape.dims[-1].is_compatible_with(1)):
    predictions = control_flow_ops.cond(
        math_ops.equal(1, rank_diff),
        lambda: array_ops.squeeze(predictions, [-1]),
        lambda: predictions)
  if (labels_rank is None) or (
      labels_shape.dims[-1].is_compatible_with(1)):
    labels = control_flow_ops.cond(
        math_ops.equal(-1, rank_diff),
        lambda: array_ops.squeeze(labels, [-1]),
        lambda: labels)
  return predictions, labels


def _all_equal(tensor0, tensor1):
  with ops.name_scope('all_equal', values=[tensor0, tensor1]) as scope:
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
  with ops.name_scope('is_rank', values=[actual_tensor]) as scope:
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
  with ops.name_scope('is_shape', values=[actual_tensor]) as scope:
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
  with ops.name_scope('assert_shape', values=[actual_tensor]) as scope:
    actual_shape = array_ops.shape(actual_tensor, name='actual')
    is_shape = _is_shape(expected_shape, actual_tensor, actual_shape)
    return control_flow_ops.Assert(
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
  with ops.name_scope('%s/' % tensor.op.name, values=[expected_tensor, tensor]):
    tensor_shape = expected_tensor.get_shape()
    expected_shape = (
        tensor_shape.as_list() if tensor_shape.is_fully_defined()
        else array_ops.shape(expected_tensor, name='expected_shape'))
    return with_shape(expected_shape, tensor)


def is_tensor(x):
  """Check for tensor types.

  Check whether an object is a tensor. Equivalent to
  `isinstance(x, [tf.Tensor, tf.SparseTensor, tf.Variable])`.

  Args:
    x: An python object to check.

  Returns:
    `True` if `x` is a tensor, `False` if not.
  """
  tensor_types = (ops.Tensor, ops.SparseTensor, variables.Variable)
  return isinstance(x, tensor_types)


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
  if is_tensor(expected_shape):
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

  if not actual_shape.is_fully_defined() or is_tensor(expected_shape):
    with ops.name_scope('%s/' % tensor.op.name, values=[tensor]):
      if not is_tensor(expected_shape) and (len(expected_shape) < 1):
        # TODO(irving): Remove scalar special case
        return array_ops.reshape(tensor, [])
      with ops.control_dependencies([_assert_shape_op(expected_shape, tensor)]):
        result = array_ops.identity(tensor)
      if not is_tensor(expected_shape):
        result.set_shape(expected_shape)
      return result

  if (not is_tensor(expected_shape) and
      not actual_shape.is_compatible_with(expected_shape)):
    if (len(expected_shape) < 1) and actual_shape.is_compatible_with([1]):
      # TODO(irving): Remove scalar special case.
      with ops.name_scope('%s/' % tensor.op.name, values=[tensor]):
        return array_ops.reshape(tensor, [])
    raise ValueError('Invalid shape for tensor %s, expected %s, got %s.' % (
        tensor.name, expected_shape, actual_shape))

  return tensor


def convert_to_tensor_or_sparse_tensor(
    value, dtype=None, name=None, as_ref=False):
  """Converts value to a `SparseTensor` or `Tensor`.

  Args:
    value: A `SparseTensor`, `SparseTensorValue`, or an object whose type has a
      registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the
      type is inferred from the type of `value`.
    name: Optional name to use if a new `Tensor` is created.
    as_ref: True if we want the result as a ref tensor. Only used if a new
      `Tensor` is created.

  Returns:
    A `SparseTensor` or `Tensor` based on `value`.

  Raises:
    RuntimeError: If result type is incompatible with `dtype`.
  """
  if dtype is not None:
    dtype = dtypes.as_dtype(dtype)
  if isinstance(value, ops.SparseTensorValue):
    value = ops.SparseTensor.from_value(value)
  if isinstance(value, ops.SparseTensor):
    if dtype and not dtype.is_compatible_with(value.dtype):
      raise RuntimeError(
          'Sparse dtype: requested = %s, actual = %s' % (
              dtype.name, value.dtype.name))
    return value
  return ops.convert_to_tensor(value, dtype=dtype, name=name, as_ref=as_ref)
