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
"""TensorSpec factory for ragged tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops.ragged import ragged_tensor


def ragged_tensor_spec(shape=None, dtype=dtypes.float32,
                       ragged_rank=None, row_splits_dtype=dtypes.int64,
                       name=None):
  """Returns a tensor specification for a RaggedTensor.

  Returns an object which can be passed to `tf.function` (or other
  functions that expect `TensorSpec`s) to specify shape constraints
  for a `RaggedTensor` argument.

  Args:
    shape: The shape of the RaggedTensor, or `None` to allow any shape.
    dtype: Data type of values in the RaggedTensor.
    ragged_rank: Python integer, the ragged rank of the RaggedTensor
      to be described.  Defaults to `shape.ndims - 1`.
    row_splits_dtype: `dtype` for the RaggedTensor's `row_splits` tensor.
      One of `tf.int32` or `tf.int64`.
    name: Optional name prefix for the `TensorSpec`s.

  Returns:
    An object describing the `flat_values` and `nested_row_splits` tensors
    that comprise the `RaggedTensor`.
  """
  dtype = dtypes.as_dtype(dtype)
  shape = tensor_shape.TensorShape(shape)
  if ragged_rank is None:
    if shape.ndims is None:
      raise ValueError("Must specify ragged_rank or a shape with known rank.")
    ragged_rank = shape.ndims - 1
  elif not isinstance(ragged_rank, int):
    raise TypeError("ragged_rank must be an int")
  if ragged_rank == 0:
    return tensor_spec.TensorSpec(shape=shape, dtype=dtype, name=name)

  result = tensor_spec.TensorSpec(
      tensor_shape.TensorShape([None]).concatenate(shape[ragged_rank + 1:]),
      dtype, name)

  for i in range(ragged_rank - 1, 0, -1):
    splits = tensor_spec.TensorSpec(
        [None], row_splits_dtype,
        "%s.row_splits_%d" % (name, i) if name else None)
    result = ragged_tensor.RaggedTensor.from_row_splits(result, splits)

  outer_dim = tensor_shape.dimension_at_index(shape, 0)
  splits_shape = [None if outer_dim is None else outer_dim + 1]
  splits = tensor_spec.TensorSpec(
      splits_shape, row_splits_dtype,
      "%s.row_splits_0" % name if name else None)
  result = ragged_tensor.RaggedTensor.from_row_splits(result, splits)

  return result
