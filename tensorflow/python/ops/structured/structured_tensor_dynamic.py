# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Dynamic shape for structured Tensors."""

from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.structured.structured_tensor import _find_shape_dtype


# pylint:disable=protected-access
def _dynamic_ragged_shape_init(fields, shape, nrows, row_partitions):
  """Produce a DynamicRaggedShape for StructuredTensor."""
  assert isinstance(fields, dict), fields
  assert isinstance(shape, tensor_shape.TensorShape), shape
  assert nrows is None or isinstance(nrows, tensor.Tensor), nrows
  assert isinstance(row_partitions, tuple), row_partitions

  rank = shape.rank
  if rank is None:
    raise TypeError("StructuredTensor's shape must have known rank.")

  # TODO(martinz): figure out whether to validate.
  dtype = _find_shape_dtype(fields, nrows, row_partitions)
  if rank == 0:
    return dynamic_ragged_shape.DynamicRaggedShape._from_inner_shape(
        array_ops.zeros((0,), dtype=dtype))

  if rank == 1:
    alt_value = shape[0]
    if isinstance(alt_value, tensor_shape.Dimension):
      alt_value = alt_value.value
    if alt_value is not None:
      nrows = alt_value
    return dynamic_ragged_shape.DynamicRaggedShape._from_inner_shape(
        [nrows], dtype=dtype)

  return dynamic_ragged_shape.DynamicRaggedShape.from_row_partitions(
      row_partitions, dtype=dtype)
