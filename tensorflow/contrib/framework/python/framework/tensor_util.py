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
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.deprecation import deprecated


__all__ = [
    'assert_same_float_dtype',
    'assert_scalar',
    'assert_scalar_int',
    'convert_to_tensor_or_sparse_tensor',
    'is_tensor',
    'reduce_sum_n',
    'remove_squeezable_dimensions',
    'with_shape',
    'with_same_shape']


# Temporary for backwards compatibility
is_tensor = tensor_util.is_tensor
assert_same_float_dtype = check_ops.assert_same_float_dtype
assert_scalar = check_ops.assert_scalar

convert_to_tensor_or_sparse_tensor = (
    sparse_tensor.convert_to_tensor_or_sparse_tensor)


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
  with ops.name_scope(name, 'reduce_sum_n', tensors) as name_scope:
    tensors = [
        math_ops.reduce_sum(t, name='%s/sum' % t.op.name) for t in tensors]
    if len(tensors) == 1:
      return tensors[0]
    return math_ops.add_n(tensors, name=name_scope)

@deprecated(
    None, "Please switch to remove_squeezable_dimensions from "
    "tf.confusion_matrix. Note that the order of the inputs and outputs of "
    "labels and predictions have also been switched.")
def remove_squeezable_dimensions(predictions, labels, name=None):
  """Squeeze last dim if ranks of `predictions` and `labels` differ by 1.

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    predictions: Predicted values, a `Tensor` of arbitrary dimensions.
    labels: Label values, a `Tensor` whose dimensions match `predictions`.
    name: Name of the op.

  Returns:
    Tuple of `predictions` and `labels`, possibly with last dim squeezed.
  """
  with ops.name_scope(name, 'remove_squeezable_dimensions',
                      [predictions, labels]):
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


def _shape_tensor_compatible(expected_shape, actual_shape):
  """Returns whether actual_shape is compatible with expected_shape.

  Note that -1 in `expected_shape` is recognized as unknown dimension.

  Args:
    expected_shape: Integer list defining the expected shape, or tensor of same.
    actual_shape: Shape of the tensor to test.
  Returns:
    New tensor.
  """
  with ops.name_scope('shape_tensor_equal',
                      values=[expected_shape, actual_shape]) as scope:
    return math_ops.reduce_all(
        math_ops.logical_or(
            math_ops.equal(expected_shape, -1),
            math_ops.equal(expected_shape, actual_shape, 'equal'),
            name='exclude_partial_shape'),
        name=scope)


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

  Note that -1 in `expected_shape` is recognized as unknown dimension.

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
    shape_equal = _shape_tensor_compatible(expected_shape, actual_shape)
    return math_ops.logical_and(is_rank, shape_equal, name=scope)


def _assert_shape_op(expected_shape, actual_tensor):
  """Asserts actual_tensor's shape is expected_shape.

  Note that unknown dimension in `expected_shape` will be ignored.

  Args:
    expected_shape: List of integers defining the expected shape, or tensor of
        same.
    actual_tensor: Tensor to test.
  Returns:
    New assert tensor.
  """
  with ops.name_scope('assert_shape', values=[actual_tensor]) as scope:
    actual_shape = array_ops.shape(actual_tensor, name='actual')
    if (isinstance(expected_shape, tensor_shape.TensorShape)
        and not expected_shape.is_fully_defined()):
      expected_shape = [d if d else -1 for d in expected_shape.as_list()]
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
    The original tensor argument, possibly with assert ops added.
  """
  with ops.name_scope('%s/' % tensor.op.name, values=[expected_tensor, tensor]):
    tensor_shape = expected_tensor.get_shape()
    expected_shape = (
        tensor_shape.as_list() if tensor_shape.is_fully_defined()
        else array_ops.shape(expected_tensor, name='expected_shape'))
    return with_shape(expected_shape, tensor)


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
  if isinstance(tensor, sparse_tensor.SparseTensor):
    raise ValueError('SparseTensor not supported.')

  # Shape type must be 1D int32.
  if tensor_util.is_tensor(expected_shape):
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

  if (not actual_shape.is_fully_defined()
      or tensor_util.is_tensor(expected_shape)):
    with ops.name_scope('%s/' % tensor.op.name, values=[tensor]):
      if (not tensor_util.is_tensor(expected_shape)
          and (len(expected_shape) < 1)):
        # TODO(irving): Remove scalar special case
        return array_ops.reshape(tensor, [])
      with ops.control_dependencies([_assert_shape_op(expected_shape, tensor)]):
        result = array_ops.identity(tensor)
      if not tensor_util.is_tensor(expected_shape):
        result.set_shape(expected_shape)
      return result

  if (not tensor_util.is_tensor(expected_shape) and
      not actual_shape.is_compatible_with(expected_shape)):
    if (len(expected_shape) < 1) and actual_shape.is_compatible_with([1]):
      # TODO(irving): Remove scalar special case.
      with ops.name_scope('%s/' % tensor.op.name, values=[tensor]):
        return array_ops.reshape(tensor, [])
    raise ValueError('Invalid shape for tensor %s, expected %s, got %s.' % (
        tensor.name, expected_shape, actual_shape))

  return tensor


def assert_scalar_int(tensor, name=None):
  """Assert `tensor` is 0-D, of type `tf.int32` or `tf.int64`.

  Args:
    tensor: `Tensor` to test.
    name: Name of the op and of the new `Tensor` if one is created.
  Returns:
    `tensor`, for chaining.
  Raises:
    ValueError: if `tensor` is not 0-D, of integer type.
  """
  with ops.name_scope(name, 'assert_scalar_int', [tensor]) as name_scope:
    tensor = ops.convert_to_tensor(tensor)
    data_type = tensor.dtype
    if not data_type.base_dtype.is_integer:
      raise ValueError('Expected integer type for %s, received type: %s.'
                       % (tensor.name, data_type))
    return check_ops.assert_scalar(tensor, name=name_scope)
