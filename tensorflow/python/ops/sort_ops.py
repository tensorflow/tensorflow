# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Support for sorting tensors."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


@tf_export('sort')
@dispatch.add_dispatch_support
def sort(values, axis=-1, direction='ASCENDING', name=None):
  """Sorts a tensor.

  Usage:

  >>> a = [1, 10, 26.9, 2.8, 166.32, 62.3]
  >>> tf.sort(a).numpy()
  array([  1.  ,   2.8 ,  10.  ,  26.9 ,  62.3 , 166.32], dtype=float32)

  >>> tf.sort(a, direction='DESCENDING').numpy()
  array([166.32,  62.3 ,  26.9 ,  10.  ,   2.8 ,   1.  ], dtype=float32)

  For multidimensional inputs you can control which axis the sort is applied
  along. The default `axis=-1` sorts the innermost axis.

  >>> mat = [[3,2,1],
  ...        [2,1,3],
  ...        [1,3,2]]
  >>> tf.sort(mat, axis=-1).numpy()
  array([[1, 2, 3],
         [1, 2, 3],
         [1, 2, 3]], dtype=int32)
  >>> tf.sort(mat, axis=0).numpy()
  array([[1, 1, 1],
         [2, 2, 2],
         [3, 3, 3]], dtype=int32)

  See also:

    * `tf.argsort`: Like sort, but it returns the sort indices.
    * `tf.math.top_k`: A partial sort that returns a fixed number of top values
      and corresponding indices.


  Args:
    values: 1-D or higher **numeric** `Tensor`.
    axis: The axis along which to sort. The default is -1, which sorts the last
      axis.
    direction: The direction in which to sort the values (`'ASCENDING'` or
      `'DESCENDING'`).
    name: Optional name for the operation.

  Returns:
    A `Tensor` with the same dtype and shape as `values`, with the elements
        sorted along the given `axis`.

  Raises:
    tf.errors.InvalidArgumentError: If the `values.dtype` is not a `float` or
        `int` type.
    ValueError: If axis is not a constant scalar, or the direction is invalid.
  """
  with framework_ops.name_scope(name, 'sort'):
    return _sort_or_argsort(values, axis, direction, return_argsort=False)


@tf_export('argsort')
@dispatch.add_dispatch_support
def argsort(values, axis=-1, direction='ASCENDING', stable=False, name=None):
  """Returns the indices of a tensor that give its sorted order along an axis.

  >>> values = [1, 10, 26.9, 2.8, 166.32, 62.3]
  >>> sort_order = tf.argsort(values)
  >>> sort_order.numpy()
  array([0, 3, 1, 2, 5, 4], dtype=int32)

  For a 1D tensor:

  >>> sorted = tf.gather(values, sort_order)
  >>> assert tf.reduce_all(sorted == tf.sort(values))

  For higher dimensions, the output has the same shape as
  `values`, but along the given axis, values represent the index of the sorted
  element in that slice of the tensor at the given position.

  >>> mat = [[30,20,10],
  ...        [20,10,30],
  ...        [10,30,20]]
  >>> indices = tf.argsort(mat)
  >>> indices.numpy()
  array([[2, 1, 0],
         [1, 0, 2],
         [0, 2, 1]], dtype=int32)

  If `axis=-1` these indices can be used to apply a sort using `tf.gather`:

  >>> tf.gather(mat, indices, batch_dims=-1).numpy()
  array([[10, 20, 30],
         [10, 20, 30],
         [10, 20, 30]], dtype=int32)

  See also:

    * `tf.sort`: Sort along an axis.
    * `tf.math.top_k`: A partial sort that returns a fixed number of top values
      and corresponding indices.

  Args:
    values: 1-D or higher **numeric** `Tensor`.
    axis: The axis along which to sort. The default is -1, which sorts the last
      axis.
    direction: The direction in which to sort the values (`'ASCENDING'` or
      `'DESCENDING'`).
    stable: If True, equal elements in the original tensor will not be
      re-ordered in the returned order. Unstable sort is not yet implemented,
      but will eventually be the default for performance reasons. If you require
      a stable order, pass `stable=True` for forwards compatibility.
    name: Optional name for the operation.

  Returns:
    An int32 `Tensor` with the same shape as `values`. The indices that would
        sort each slice of the given `values` along the given `axis`.

  Raises:
    ValueError: If axis is not a constant scalar, or the direction is invalid.
    tf.errors.InvalidArgumentError: If the `values.dtype` is not a `float` or
        `int` type.
  """
  del stable  # Unused.
  with framework_ops.name_scope(name, 'argsort'):
    return _sort_or_argsort(values, axis, direction, return_argsort=True)


def _sort_or_argsort(values, axis, direction, return_argsort):
  """Internal sort/argsort implementation.

  Args:
    values: The input values.
    axis: The axis along which to sort.
    direction: 'ASCENDING' or 'DESCENDING'.
    return_argsort: Whether to return the argsort result.

  Returns:
    Either the sorted values, or the indices of the sorted values in the
        original tensor. See the `sort` and `argsort` docstrings.

  Raises:
    ValueError: If axis is not a constant scalar, or the direction is invalid.
  """
  if direction not in _SORT_IMPL:
    valid_directions = ', '.join(sorted(_SORT_IMPL.keys()))
    raise ValueError(f'Argument `direction` should be one of {valid_directions}'
                     f'. Received: direction={direction}')
  # Axis must be an integer, not a Tensor.
  axis = framework_ops.convert_to_tensor(axis, name='axis')
  axis_static = tensor_util.constant_value(axis)
  if axis.shape.ndims not in (None, 0) or axis_static is None:
    raise ValueError(
        f'Argument `axis` must be a constant scalar. Received: axis={axis}.')
  axis_static = int(axis_static)  # Avoids NumPy casting error

  values = framework_ops.convert_to_tensor(values, name='values')

  return _SORT_IMPL[direction](values, axis_static, return_argsort)


def _descending_sort(values, axis, return_argsort=False):
  """Sorts values in reverse using `top_k`.

  Args:
    values: Tensor of numeric values.
    axis: Index of the axis which values should be sorted along.
    return_argsort: If False, return the sorted values. If True, return the
      indices that would sort the values.

  Returns:
    The sorted values.
  """
  # TODO(b/190410105): replace with a proper sort kernel.
  k = array_ops.shape(values)[axis]
  rank = array_ops.rank(values)
  static_rank = values.shape.ndims
  # Fast path: sorting the last axis.
  if axis == -1 or axis + 1 == values.get_shape().ndims:
    top_k_input = values
    transposition = None
  else:
    # Otherwise, transpose the array. Swap axes `axis` and `rank - 1`.
    if axis < 0:
      # Calculate the actual axis index if counting from the end. Use the static
      # rank if available, or else make the axis back into a tensor.
      axis += static_rank or rank
    if static_rank is not None:
      # Prefer to calculate the transposition array in NumPy and make it a
      # constant.
      transposition = constant_op.constant(
          np.r_[
              # Axes up to axis are unchanged.
              np.arange(axis),
              # Swap axis and rank - 1.
              [static_rank - 1],
              # Axes in [axis + 1, rank - 1) are unchanged.
              np.arange(axis + 1, static_rank - 1),
              # Swap axis and rank - 1.
              [axis]],
          name='transposition')
    else:
      # Generate the transposition array from the tensors.
      transposition = array_ops.tensor_scatter_update(
          math_ops.range(rank), [[axis], [rank-1]], [rank-1, axis])
    top_k_input = array_ops.transpose(values, transposition)

  values, indices = nn_ops.top_k(top_k_input, k)
  return_value = indices if return_argsort else values
  if transposition is not None:
    # transposition contains a single cycle of length 2 (swapping 2 elements),
    # so it is an involution (it is its own inverse).
    return_value = array_ops.transpose(return_value, transposition)
  return return_value


def _ascending_sort(values, axis, return_argsort=False):
  """Sorts values in ascending order.

  Args:
    values: Tensor of numeric values.
    axis: Index of the axis which values should be sorted along.
    return_argsort: If False, return the sorted values. If True, return the
      indices that would sort the values.

  Returns:
    The sorted values.
  """
  # TODO(b/190410105): replace with a proper sort kernel.
  # If values are integers, we need special handling.
  dtype = values.dtype
  if dtype.is_unsigned:
    # Subtract values from dtype.max to reverse sort order.
    offset = dtype.max
    values_or_indices = _descending_sort(offset - values, axis, return_argsort)
    return values_or_indices if return_argsort else offset - values_or_indices

  elif dtype.is_integer:
    # Negate and subtract 1 to map dtype.min to dtype.max.  Technically this
    # will result in signed-integer-overflow UB for dtype.min, though
    # practically should produce correct results on all systems.
    #
    # Casting to unsigned would be better, but uint* subtraction is not
    # supported on all devices.
    #
    # Although more complex and slightly slower than descend+reverse, this
    # approach preserves sort stability.
    values_or_indices = _descending_sort(-values - 1, axis, return_argsort)
    return values_or_indices if return_argsort else -values_or_indices - 1

  else:
    # Otherwise, negate the values and use descending sort.
    values_or_indices = _descending_sort(-values, axis, return_argsort)
    # If not argsort, negate the values again.
    return values_or_indices if return_argsort else -values_or_indices


_SORT_IMPL = {
    'ASCENDING': _ascending_sort,
    'DESCENDING': _descending_sort,
}
