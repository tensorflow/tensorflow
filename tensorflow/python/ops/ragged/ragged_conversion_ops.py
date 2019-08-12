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
"""Ops to convert between RaggedTensors and other tensor types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor


def from_tensor(tensor,
                lengths=None,
                padding=None,
                ragged_rank=1,
                row_splits_dtype=dtypes.int64,
                name=None):
  if ragged_tensor.is_ragged(tensor):
    return tensor
  else:
    return ragged_tensor.RaggedTensor.from_tensor(
        tensor,
        lengths=lengths,
        padding=padding,
        ragged_rank=ragged_rank,
        row_splits_dtype=row_splits_dtype,
        name=name)


def to_tensor(rt_input, default_value=None, name=None):
  if ragged_tensor.is_ragged(rt_input):
    return rt_input.to_tensor(default_value, name)
  else:
    return rt_input


def _get_row_partition_type_tensor_pairs_tail(rt_value):
  """Gets a list of the row partitions for rt_value.

  If parent_indices are defined, then they are used. Otherwise, row_splits
  are used.

  This assumes that rt_input is nested inside another RaggedTensor. If it is
  a tensor, then return an empty list.

  Args:
    rt_value: a ragged tensor value. May be a tensor.

  Returns:
    A list of (row_partition_type, row_partition_tensor) pairs.
  """
  if isinstance(rt_value, ragged_tensor.RaggedTensor):
    tail = _get_row_partition_type_tensor_pairs_tail(rt_value.values)
    if rt_value._cached_value_rowids is not None:  # pylint: disable=protected-access
      return [("VALUE_ROWIDS", rt_value.value_rowids())] + tail
    else:
      return [("ROW_SPLITS", rt_value.row_splits)] + tail
  return []


def _get_row_partition_type_tensor_pairs(rt_input):
  """Gets a list of the row partitions for rt_input.

  If value_rowids are defined, then they are used. Otherwise, row_splits
  are used. If the outermost level has value_rowids defind, then nrows is
  also added.

  Args:
    rt_input: a ragged tensor.

  Returns:
    A list of (row_partition_type, row_partition_tensor) pairs.
  """
  tail = _get_row_partition_type_tensor_pairs_tail(rt_input.values)
  if rt_input._cached_value_rowids is not None:  # pylint: disable=protected-access
    return [("FIRST_DIM_SIZE", rt_input.nrows()),
            ("VALUE_ROWIDS", rt_input.value_rowids())] + tail
  else:
    return [("ROW_SPLITS", rt_input.row_splits)] + tail


def _shape_as_tensor(shape, dtype):
  """Takes shape and coerces it to a shape as a tensor.

  If the object is already a tensor, simply passes it on (result is guaranteed
  to be int64 or int32, but not necessarily dtype).
  If not, creates a tensor of type dtype.

  Result is either a scalar equal to -1 if the shape is unknown_rank.
  Otherwise, it is a vector, where unknown dimensions are represented with a
  value of -1.

  In C++, see TensorShapeFromTensor for parsing shapes in kernels, and
  InferenceContext::MakeShapeFromShapeTensorTreatScalarAsUnknownShape, for
  use in the shape inference function.

  Args:
    shape: input to coerce from TensorShape, Tensor, None, List[Optional[Int]],
      Tuple[Optional[Int]].
    dtype: tf.int64 or tf.int32

  Returns:
    a scalar or vector tensor of dtype tf.int32 or tf.int64.
  """
  if dtype != dtypes.int64 and dtype != dtypes.int32:
    raise ValueError("Expected int64 or int32 for dtype: got {}".format(dtype))

  if isinstance(shape, ops.Tensor):
    if shape.dtype != dtypes.int64 and shape.dtype != dtypes.int32:
      return math_ops.cast(shape, dtype)
    return shape
  shape = tensor_shape.as_shape(shape)
  if not shape:
    # Imply rank is unknown using a -1 scalar.
    return constant_op.constant(-1, dtype=dtype)
  shape = [(-1 if x is None else x) for x in shape.as_list()]
  # At this point, shape is List[Int].
  return constant_op.constant(shape, dtype=dtype)


# TODO(martinz): add a gradient for this op.
# TODO(martinz): this is a replacement for RaggedTensor.to_tensor. Move this
# after there is a chance for the kernels to propagate.
def ragged_to_dense(rt_input, default_value=None, shape=None):
  """Create a dense tensor from a ragged tensor.

  If the shape is None, then the resulting dense tensor is the same size as
  the maximum length of the ragged tensor in each dimension.

  If the shape is not None, then it must be the same number of dimensions
  as the ragged tensor. For dimension i, if shape[i] is None, then the maximum
  length of the ragged tensor in that dimension is the size of the output in
  that dimension. If shape[i] is an integer, then that is the size of the output
  in that dimension.

  Args:
    rt_input: the tensor to densify.
    default_value: used when a value is missing.
    shape: the shape of the resulting tensor.

  Returns:
    a dense tensor.
  """

  type_tensor_pairs = _get_row_partition_type_tensor_pairs(rt_input)
  row_partition_types = [x[0] for x in type_tensor_pairs]
  row_partition_tensors = [x[1] for x in type_tensor_pairs]
  values = rt_input.flat_values
  if default_value is None:
    default_value = array_ops.zeros((), values.dtype)

  shape_tensor = _shape_as_tensor(shape, row_partition_tensors[0].dtype)
  return gen_ragged_conversion_ops.ragged_tensor_to_tensor(
      shape=shape_tensor,
      values=values,
      default_value=default_value,
      row_partition_types=row_partition_types,
      row_partition_tensors=row_partition_tensors)


def to_sparse(rt_input, name=None):
  return rt_input.to_sparse(name)


def from_sparse(st_input, name=None):
  return ragged_tensor.RaggedTensor.from_sparse(st_input, name)
