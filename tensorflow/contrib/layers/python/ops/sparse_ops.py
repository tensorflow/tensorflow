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
      shape_multipliers = array_ops.pack(
          _multiplier_helper(array_ops.unpack(dense_shape)[1:]))
      offsets = math_ops.reduce_sum(
          math_ops.mul(higher_dims, shape_multipliers), reduction_indices=[1])
      flat_indices = math_ops.add(flat_indices, offsets)
    values = array_ops.gather(flat_tensor, flat_indices)
    return sparse_tensor.SparseTensor(indices, values, dense_shape)
