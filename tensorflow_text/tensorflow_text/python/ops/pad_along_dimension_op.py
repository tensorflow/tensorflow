# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Pad-along-dimension op.

Pads the beginning and end of a given dimension.
"""

from __future__ import absolute_import
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor


def pad_along_dimension(data, axis=-1, left_pad=None, right_pad=None,
                        name=None):
  """Add padding to the beginning and end of data in a specific dimension.

  Returns a tensor constructed from `data`, where each row in dimension `axis`
  is replaced by the concatenation of the left padding followed by the row
  followed by the right padding.  I.e., if `L=left_pad.shape[0]` and
  `R=right_pad.shape[0]`, then:

  ```python
  result[i1...iaxis, 0:L] = left_pad
  result[i1...iaxis, L:-R] = data[i0...iaxis]
  result[i1...iaxis, -R:] = right_pad
  ```

  Args:
    data: `<dtype>[O1...ON, A, I1...IM]` A potentially ragged `K` dimensional
      tensor with outer dimensions of size `O1...ON`; axis dimension of size
      `A`; and inner dimensions of size `I1...IM`.  I.e. `K = N + 1 + M`, where
      `N>=0` and `M>=0`.
    axis: An integer constant specifying the axis along which padding is added.
      Negative axis values from `-K` to `-1` are supported.
    left_pad: `<dtype>[L, I1...IM]` An `M+1` dimensional tensor that should be
      prepended to each row along dimension `axis`; or `None` if no padding
      should be added to the left side.
    right_pad: `<dtype>[R, I1...IM]` An `M+1` dimensional tensor that should be
      appended to each row along dimension `axis`; or `None` if no padding
      should be added to the right side.
    name: The name of this op (optional).

  Returns:
    `<dtype>[O1...ON, L + A + R, I1...IM]`
      A potentially ragged `K` dimensional tensor with outer dimensions of size
      `O1...ON`; padded axis dimension size `L+A+R`; and inner dimensions of
      size `I1...IM`.  If `data` is a `RaggedTensor`, then the returned tensor
      is a `RaggedTensor` with the same `ragged_rank`.
  """
  data = ragged_tensor.convert_to_tensor_or_ragged_tensor(data, name="data")

  if not isinstance(axis, int):
    raise TypeError("axis must be an int; got %s" % type(axis).__name__)

  if left_pad is None and right_pad is None:
    return data

  with ops.name_scope(name, "PadAlongDimension", [data]):
    if data.shape.ndims is not None and (axis < -data.shape.ndims or
                                         axis >= data.shape.ndims):
      raise errors.InvalidArgumentError(
          None, None, "axis must be between -k <= axis <= -1 OR 0 <= axis < k")
    if isinstance(data, ragged_tensor.RaggedTensor):
      axis = _get_positive_axis(axis, data.shape.ndims)

    if left_pad is not None:
      left_pad = ragged_tensor.convert_to_tensor_or_ragged_tensor(
          left_pad, dtype=data.dtype, name="left_pad")
    if right_pad is not None:
      right_pad = ragged_tensor.convert_to_tensor_or_ragged_tensor(
          right_pad, dtype=data.dtype, name="left_pad")

    left_padding = _padding_for_dimension(data, axis, left_pad)
    right_padding = _padding_for_dimension(data, axis, right_pad)

    pieces = [left_padding, data, right_padding]
    if isinstance(data, ragged_tensor.RaggedTensor):
      return array_ops.concat([p for p in pieces if p is not None], axis)
    else:
      return array_ops.concat([p for p in pieces if p is not None], axis)


def _get_positive_axis(axis, ndims):
  """Normalize axis` to be positive."""
  if axis >= 0:
    return axis
  elif ndims is None:
    raise ValueError("axis may not be negative if data is ragged and "
                     "data.ndims is not statically known.")
  else:
    return axis + ndims


def _padding_for_dimension(data, axis, pad_value):
  """Tile `pad_value` so it can be used to pad `data` at the given axis.

  Returns a tensor `result` that has the same shape as `data` up to dimension
  `axis`, but where each value `data[i0...iaxis]` is replaced by `pad_value`.
  I.e., returns `result[i0...iaxis, j0...jN] = pad_value[j0...jN]`
  (where `N=rank(pad_value)`).

  Args:
    data: The potentially ragged tensor that will be padded.
    axis: The axis along which padding will be added.
    pad_value: The padding value that should be used, or None if no padding will
      be added.  `rank(pad_value)` must be `rank(data) - axis`, and
      `pad_value.shape[1:]` must be compatible with `data.shape[axis + 1:]`.

  Returns:
    A padding tensor with the same rank as `data`, which can be concatenated
    to `data` to add padding.
  """
  if pad_value is None:
    return None

  # Verify shape compatibility.
  pad_value.shape[1:].assert_is_compatible_with(data.shape[axis:][1:])

  if not isinstance(data, ragged_tensor.RaggedTensor):
    data_shape = array_ops.shape(data)
    pad_shape = array_ops.shape(pad_value)
    outer_dimensions = data_shape[:axis]
    expanded_pad_shape = array_ops.concat(
        [array_ops.ones_like(outer_dimensions), pad_shape], axis=0)
    tile_multiples = array_ops.concat(
        [outer_dimensions, array_ops.ones_like(pad_shape)], axis=0)
    tiled_padding = array_ops.tile(
        array_ops.reshape(pad_value, expanded_pad_shape), tile_multiples)
    tiled_padding.set_shape(data.shape[:axis].concatenate(pad_value.shape))
    return tiled_padding

  assert axis >= 0
  # Return the padding as-is if we're padding the outermost dimension.
  if axis == 0:
    return pad_value

  elif axis == 1:
    if isinstance(pad_value, ragged_tensor.RaggedTensor):
      pad_rank = array_ops.rank(pad_value.flat_values) + pad_value.ragged_rank
      pad_nrows = pad_value.nrows()
    else:
      pad_rank = array_ops.rank(pad_value)
      pad_nrows = array_ops.shape(pad_value, out_type=dtypes.int64)[0]

    # Return a RaggedTensor that has the same number of rows as `data`, where
    # each row contains a single copy of `pad_value`.
    data_nrows = data.nrows()
    pad_repeats = array_ops.concat(
        [[math_ops.cast(data_nrows, dtypes.int32)],
         array_ops.ones([pad_rank - 1], dtypes.int32)],
        axis=0)
    result_values = array_ops.tile(pad_value, pad_repeats)
    return ragged_tensor.RaggedTensor.from_row_splits(
        result_values,
        math_ops.range(0, data_nrows + 1) * pad_nrows)

  else:  # Recurse if axis>1.
    return ragged_tensor.RaggedTensor.from_row_splits(
        _padding_for_dimension(data.values, axis - 1, pad_value),
        data.row_splits)
