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
"""Ops to work with `SparseTensor`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat


def _multiplier_helper(shape):
  """Returns moving offset for each dimension given shape."""
  multipliers = []
  for dim in reversed(shape):
    if multipliers:
      multipliers.append(dim * multipliers[-1])
    else:
      multipliers.append(dim)
  multipliers.reverse()
  return multipliers


def dense_to_sparse_tensor(dense_tensor, ignore_value=None):
  """Converts a dense Tensor to a SparseTensor, dropping ignore_value cells.

  Args:
    dense_tensor: A `Tensor`.
    ignore_value: Entries in `dense_tensor` equal to this value will be
      absent from the return `SparseTensor`. If `None`, default value of
      dense_tensor's dtype will be used (e.g. '' for `str`, 0 for `int`).

  Returns:
    A `SparseTensor` with the same shape as `dense_tensor`.

  Raises:
    ValueError: when `dense_tensor`'s rank is `None`.
  """
  with ops.name_scope("DenseToSparseTensor"):
    dense_t = ops.convert_to_tensor(dense_tensor)
    if dense_t.get_shape().ndims is None:
      # TODO(b/32318825): Implement dense_to_sparse_tensor for undefined rank.
      raise ValueError("dense_tensor.get_shape() should be defined, got None.")
    if ignore_value is None:
      if dense_t.dtype == dtypes.string:
        # Exception due to TF strings are converted to numpy objects by default.
        ignore_value = ""
      else:
        ignore_value = dense_t.dtype.as_numpy_dtype()
    dense_shape = math_ops.cast(array_ops.shape(dense_t), dtypes.int64)
    indices = array_ops.where(
        math_ops.not_equal(dense_t, math_ops.cast(ignore_value, dense_t.dtype)))
    index_dims = len(dense_t.get_shape())
    # Flattens the tensor and indices for use with gather.
    flat_tensor = array_ops.reshape(dense_t, [-1])
    flat_indices = indices[:, index_dims - 1]
    # Computes the correct flattened indices for 2d (or higher) tensors.
    if index_dims > 1:
      higher_dims = indices[:, :index_dims - 1]
      shape_multipliers = array_ops.stack(
          _multiplier_helper(array_ops.unstack(dense_shape)[1:]))
      offsets = math_ops.reduce_sum(
          math_ops.multiply(higher_dims, shape_multipliers),
          reduction_indices=[1])
      flat_indices = math_ops.add(flat_indices, offsets)
    values = array_ops.gather(flat_tensor, flat_indices)
    return sparse_tensor.SparseTensor(indices, values, dense_shape)


def sparse_row_envelope(sparse_input, row_axis=0, col_axis=1, name=None):
  """Returns the length of each 'row' in a `SparseTensor`.

  For example, if `sparse_input` has indices `[[0,0], [2, 0], [2, 1], [2, 2]]`
  and shape `[3, 3]`, this function will return `[1, 0, 3]`.

  Args:
    sparse_input: a `SparseTensor` of rank at least 2.
    row_axis: An integer. The axis for the row of the envelope matrix. Default
      is 0.
    col_axis: An integer. The axis for the col of the envelope matrix. Default
      is 1.
    name: A name for the operation (optional).

  Returns:
    A one-dimensional `Tensor` whose entries correspond to the length of each
    row of `SparseTensor`.

  Raises:
    ValueError: If row_axis and col_axis are the same axis or they are not
      integers.
  """
  if not (isinstance(row_axis, compat.integral_types) and
          isinstance(col_axis, compat.integral_types)):
    raise ValueError("`row_axis` and `col_axis` must be integers.")

  if row_axis == col_axis:
    raise ValueError("Row and column can not be the same axis.")

  with ops.name_scope(name, "sparse_row_envelope", [sparse_input]):
    indices = sparse_input.indices
    row_indices = indices[:, row_axis]
    col_indices = indices[:, col_axis]
    num_rows = math_ops.cast(sparse_input.dense_shape[row_axis], dtypes.int32)
    row_envelope = math_ops.unsorted_segment_max(
        col_indices + 1, row_indices, num_rows, name=name)
    zeros = array_ops.zeros_like(row_envelope)
    return array_ops.where(row_envelope > zeros, row_envelope, zeros)
