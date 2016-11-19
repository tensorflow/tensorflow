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
"""Wrappers for sparse cross operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.contrib.framework import deprecated_arg_values
from tensorflow.contrib.util import loader
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import resource_loader

_sparse_feature_cross_op = loader.load_op_library(
    resource_loader.get_path_to_datafile("_sparse_feature_cross_op.so"))

# Default hash key for the FingerprintCat64.
SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY = 0xDECAFCAFFE


@deprecated_arg_values(
    "2016-11-20",
    "The default behavior of sparse_feature_cross is changing, the default\n"
    "value for hash_key will change to SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY.\n"
    "From that point on sparse_feature_cross will always use FingerprintCat64\n"
    "to concatenate the feature fingerprints. And the underlying\n"
    "_sparse_feature_cross_op.sparse_feature_cross operation will be marked\n"
    "as deprecated.",
    hash_key=None)
def sparse_feature_cross(inputs, hashed_output=False, num_buckets=0,
                         name=None, hash_key=None):
  """Crosses a list of Tensor or SparseTensor objects.

  See sparse_feature_cross_kernel.cc for more details.

  Args:
    inputs: List of `SparseTensor` or `Tensor` to be crossed.
    hashed_output: If true, returns the hash of the cross instead of the string.
      This will allow us avoiding string manipulations.
    num_buckets: It is used if hashed_output is true.
      output = hashed_value%num_buckets if num_buckets > 0 else hashed_value.
    name: A name prefix for the returned tensors (optional).
    hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
      function to combine the crosses fingerprints on SparseFeatureCrossOp.
      The default value is None, but will become
      SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY after 2016-11-20 (optional).

  Returns:
    A `SparseTensor` with the crossed features.
    Return type is string if hashed_output=False, int64 otherwise.

  Raises:
    TypeError: If the inputs aren't either SparseTensor or Tensor.
  """
  if not isinstance(inputs, list):
    raise TypeError("Inputs must be a list")
  if not all(isinstance(i, sparse_tensor.SparseTensor) or
             isinstance(i, ops.Tensor) for i in inputs):
    raise TypeError("All inputs must be SparseTensors")

  sparse_inputs = [i for i in inputs
                   if isinstance(i, sparse_tensor.SparseTensor)]
  dense_inputs = [i for i in inputs
                  if not isinstance(i, sparse_tensor.SparseTensor)]

  indices = [sp_input.indices for sp_input in sparse_inputs]
  values = [sp_input.values for sp_input in sparse_inputs]
  shapes = [sp_input.shape for sp_input in sparse_inputs]
  out_type = dtypes.int64 if hashed_output else dtypes.string

  internal_type = dtypes.string
  for i in range(len(values)):
    if values[i].dtype != dtypes.string:
      values[i] = math_ops.to_int64(values[i])
      internal_type = dtypes.int64
  for i in range(len(dense_inputs)):
    if dense_inputs[i].dtype != dtypes.string:
      dense_inputs[i] = math_ops.to_int64(dense_inputs[i])
      internal_type = dtypes.int64

  if hash_key:
    indices_out, values_out, shape_out = (
        _sparse_feature_cross_op.sparse_feature_cross_v2(
            indices,
            values,
            shapes,
            dense_inputs,
            hashed_output,
            num_buckets,
            hash_key=hash_key,
            out_type=out_type,
            internal_type=internal_type,
            name=name))
  else:
    indices_out, values_out, shape_out = (
        _sparse_feature_cross_op.sparse_feature_cross(
            indices,
            values,
            shapes,
            dense_inputs,
            hashed_output,
            num_buckets,
            out_type=out_type,
            internal_type=internal_type,
            name=name))

  return sparse_tensor.SparseTensor(indices_out, values_out, shape_out)


ops.NotDifferentiable("SparseFeatureCross")


ops.NotDifferentiable("SparseFeatureCrossV2")
