# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Ops for converting between row_splits and segment_ids."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


# For background on "segments" and "segment ids", see:
# https://www.tensorflow.org/api_docs/python/tf/math#Segmentation
@tf_export("ragged.row_splits_to_segment_ids")
@dispatch.add_dispatch_support
def row_splits_to_segment_ids(splits, name=None, out_type=None):
  """Generates the segmentation corresponding to a RaggedTensor `row_splits`.

  Returns an integer vector `segment_ids`, where `segment_ids[i] == j` if
  `splits[j] <= i < splits[j+1]`.  Example:

  >>> print(tf.ragged.row_splits_to_segment_ids([0, 3, 3, 5, 6, 9]))
   tf.Tensor([0 0 0 2 2 3 4 4 4], shape=(9,), dtype=int64)

  Args:
    splits: A sorted 1-D integer Tensor.  `splits[0]` must be zero.
    name: A name prefix for the returned tensor (optional).
    out_type: The dtype for the return value.  Defaults to `splits.dtype`,
      or `tf.int64` if `splits` does not have a dtype.

  Returns:
    A sorted 1-D integer Tensor, with `shape=[splits[-1]]`

  Raises:
    ValueError: If `splits` is invalid.
  """
  with ops.name_scope(name, "RaggedSplitsToSegmentIds", [splits]) as name:
    splits = ops.convert_to_tensor(
        splits, name="splits",
        preferred_dtype=dtypes.int64)
    if splits.dtype not in (dtypes.int32, dtypes.int64):
      raise ValueError("splits must have dtype int32 or int64")
    splits.shape.assert_has_rank(1)
    if tensor_shape.dimension_value(splits.shape[0]) == 0:
      raise ValueError("Invalid row_splits: []")
    if out_type is None:
      out_type = splits.dtype
    else:
      out_type = dtypes.as_dtype(out_type)
    row_lengths = splits[1:] - splits[:-1]
    nrows = array_ops.shape(splits, out_type=out_type)[-1] - 1
    indices = math_ops.range(nrows)
    return ragged_util.repeat(indices, repeats=row_lengths, axis=0)


# For background on "segments" and "segment ids", see:
# https://www.tensorflow.org/api_docs/python/tf/math#Segmentation
@tf_export("ragged.segment_ids_to_row_splits")
@dispatch.add_dispatch_support
def segment_ids_to_row_splits(segment_ids, num_segments=None,
                              out_type=None, name=None):
  """Generates the RaggedTensor `row_splits` corresponding to a segmentation.

  Returns an integer vector `splits`, where `splits[0] = 0` and
  `splits[i] = splits[i-1] + count(segment_ids==i)`.  Example:

  >>> print(tf.ragged.segment_ids_to_row_splits([0, 0, 0, 2, 2, 3, 4, 4, 4]))
  tf.Tensor([0 3 3 5 6 9], shape=(6,), dtype=int64)

  Args:
    segment_ids: A 1-D integer Tensor.
    num_segments: A scalar integer indicating the number of segments.  Defaults
      to `max(segment_ids) + 1` (or zero if `segment_ids` is empty).
    out_type: The dtype for the return value.  Defaults to `segment_ids.dtype`,
      or `tf.int64` if `segment_ids` does not have a dtype.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A sorted 1-D integer Tensor, with `shape=[num_segments + 1]`.
  """
  # Local import bincount_ops to avoid import-cycle.
  from tensorflow.python.ops import bincount_ops  # pylint: disable=g-import-not-at-top
  if out_type is None:
    if isinstance(segment_ids, ops.Tensor):
      out_type = segment_ids.dtype
    elif isinstance(num_segments, ops.Tensor):
      out_type = num_segments.dtype
    else:
      out_type = dtypes.int64
  else:
    out_type = dtypes.as_dtype(out_type)
  with ops.name_scope(name, "SegmentIdsToRaggedSplits", [segment_ids]) as name:
    # Note: we cast int64 tensors to int32, since bincount currently only
    # supports int32 inputs.
    segment_ids = ragged_util.convert_to_int_tensor(segment_ids, "segment_ids",
                                                    dtype=dtypes.int32)
    segment_ids.shape.assert_has_rank(1)
    if num_segments is not None:
      num_segments = ragged_util.convert_to_int_tensor(num_segments,
                                                       "num_segments",
                                                       dtype=dtypes.int32)
      num_segments.shape.assert_has_rank(0)

    row_lengths = bincount_ops.bincount(
        segment_ids,
        minlength=num_segments,
        maxlength=num_segments,
        dtype=out_type)
    splits = array_ops.concat([[0], math_ops.cumsum(row_lengths)], axis=0)

    # Update shape information, if possible.
    if num_segments is not None:
      const_num_segments = tensor_util.constant_value(num_segments)
      if const_num_segments is not None:
        splits.set_shape(tensor_shape.TensorShape([const_num_segments + 1]))

    return splits
