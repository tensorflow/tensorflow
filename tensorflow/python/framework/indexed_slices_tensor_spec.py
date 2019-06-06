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
"""TensorSpec factory for sparse tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec


def indexed_slices_tensor_spec(shape=None,
                               dtype=dtypes.float32,
                               num_slices=None,
                               has_dense_shape=True,
                               name=None):
  """Returns a tensor specification for a IndexedSlices.

  Returns an object which can be passed to `tf.function` (or other
  functions that expect `TensorSpec`s) to specify shape constraints
  for a `IndexedSlices` argument.

  Args:
    shape: The shape of the IndexedSlices, or `None` to allow any shape.
      The returned specification object depends only on `shape[1:]`.
    dtype: Data type of values in the IndexedSlices.
    num_slices: Number of slices.  Default allows for any number of slices.
    has_dense_shape: Indicates whether the IndexedSlices is expected to have a
      `dense_shape` component.
    name: Optional name prefix for the `TensorSpec`s.

  Returns:
    An object describing the `values`, `indices` and `dense_shape` tensors
    that comprise the `IndexedSlices`.
  """
  dtype = dtypes.as_dtype(dtype)
  shape = tensor_shape.TensorShape(shape)
  num_slices = tensor_shape.Shape([num_slices])

  values = tensor_spec.TensorSpec(
      num_slices.concatenate(shape[1:]), dtype, name)
  indices = tensor_spec.TensorSpec(num_slices, dtypes.int64,
                                   ("%s.indices" % name) if name else None)
  if has_dense_shape:
    dense_shape = tensor_spec.TensorSpec([shape.ndims], dtypes.int64,
                                         ("%s.dense_shape" %
                                          name) if name else None)
  else:
    dense_shape = None
  return ops.IndexedSlices(values, indices, dense_shape)
