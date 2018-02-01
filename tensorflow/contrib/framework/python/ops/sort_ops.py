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
"""Support for sorting tensors.

@@sort
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


def sort(values, axis=-1, direction='ASCENDING', name=None):
  """Sorts a tensor.

  Args:
    values: 1-D or higher numeric `Tensor`.
    axis: The axis along which to sort. The default is -1, which sorts the last
        axis.
    direction: The direction in which to sort the values (`'ASCENDING'` or
        `'DESCENDING'`).
    name: Optional name for the operation.

  Returns:
    A `Tensor` with the same dtype and shape as `values`, with the elements
        sorted along the given `axis`.

  Raises:
    ValueError: If axis is not a constant scalar, or the direction is invalid.
  """
  with framework_ops.name_scope(name, 'sort'):
    if direction not in _SORT_IMPL:
      raise ValueError('%s should be one of %s' %
                       (direction, ', '.join(sorted(_SORT_IMPL.keys()))))
    # Axis must be an integer, not a Tensor.
    axis = framework_ops.convert_to_tensor(axis, name='axis')
    axis_static = tensor_util.constant_value(axis)
    if axis.shape.ndims != 0 or axis_static is None:
      raise ValueError('axis must be a constant scalar')
    axis_static = int(axis_static)  # Avoids NumPy casting error

    values = framework_ops.convert_to_tensor(values, name='values')

    return _SORT_IMPL[direction](values, axis_static)


def _descending_sort(values, axis):
  """Sorts values in reverse using `top_k`.

  Args:
    values: Tensor of numeric values.
    axis: Index of the axis which values should be sorted along.

  Returns:
    The sorted values.
  """
  k = array_ops.shape(values)[axis]
  rank = array_ops.rank(values)
  # Fast path: sorting the last axis.
  if axis == -1 or axis + 1 == values.get_shape().ndims:
    return nn_ops.top_k(values, k)[0]

  # Otherwise, transpose the array. Swap axes `axis` and `rank - 1`.
  if axis < 0:
    # Make axis a Tensor with the real axis index if needed.
    axis += rank
  transposition = array_ops.concat(
      [
          # Axes up to axis are unchanged.
          math_ops.range(axis),
          # Swap axis and rank - 1.
          [rank - 1],
          # Axes in [axis + 1, rank - 1) are unchanged.
          math_ops.range(axis + 1, rank - 1),
          # Swap axis and rank - 1.
          [axis]
      ],
      axis=0)
  top_k_input = array_ops.transpose(values, transposition)
  values, unused_indices = nn_ops.top_k(top_k_input, k)
  # transposition contains a single cycle of length 2 (swapping 2 elements),
  # so it is an involution (it is its own inverse).
  return array_ops.transpose(values, transposition)


def _ascending_sort(values, axis):
  # Negate the values to get the ascending order from descending sort.
  values_or_indices = _descending_sort(-values, axis)
  return -values_or_indices


_SORT_IMPL = {
    'ASCENDING': _ascending_sort,
    'DESCENDING': _descending_sort,
}
