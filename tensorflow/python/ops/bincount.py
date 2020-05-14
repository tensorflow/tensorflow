# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# maxlengthations under the License.
# ==============================================================================
"""tf.sparse.bincount ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util.tf_export import tf_export


@tf_export("sparse.bincount")
def sparse_bincount(values,
                    weights=None,
                    axis=0,
                    minlength=None,
                    maxlength=None,
                    binary_count=False,
                    name=None):
  """Count the number of times an integer value appears in a tensor.

  This op takes an N-dimensional `Tensor`, `RaggedTensor`, or `SparseTensor`,
  and returns an N-dimensional int64 SparseTensor where element
  `[i0...i[axis], j]` contains the number of times the value `j` appears in
  slice `[i0...i[axis], :]` of the input tensor.  Currently, only N=0 and
  N=-1 are supported.

  Args:
    values: A Tensor, RaggedTensor, or SparseTensor whose values should be
      counted. These tensors must have a rank of 1 or 2.
    weights: A 1-dimensional Tensor of weights. If specified, the input array is
      weighted by the weight array, i.e. if a value `n` is found at position
      `i`, `out[n]`  will be increased by `weight[i]` instead of 1.
    axis: The axis to slice over. Axes at and below `axis` will be flattened
      before bin counting. Currently, only `0`, and `-1` are supported. If None,
      all axes will be flattened (identical to passing `0`).
    minlength: If given, skips `values` that are less than `minlength`, and
      ensures that the output has a `dense_shape` of at least `minlength` in the
      inner dimension.
    maxlength: If given, skips `values` that are greater than or equal to
      `maxlength`, and ensures that the output has a `dense_shape` of at most
      `maxlength` in the inner dimension.
    binary_count: Whether to do a binary count. When True, this op will return 1
      for any value that exists instead of counting the number of occurrences.
    name: A name for this op.

  Returns:
    A SparseTensor with `output.shape = values.shape[:axis] + [N]`, where `N` is
      * `maxlength` (if set);
      * `minlength` (if set, and `minlength > reduce_max(values)`);
      * `0` (if `values` is empty);
      * `reduce_max(values) + 1` otherwise.


  Examples:

  **Bin-counting every item in individual batches**

  This example takes an input (which could be a Tensor, RaggedTensor, or
  SparseTensor) and returns a SparseTensor where the value of (i,j) is the
  number of times value j appears in batch i.

  >>> data = [[10, 20, 30, 20], [11, 101, 11, 10001]]
  >>> output = tf.sparse.bincount(data, axis=-1)
  >>> print(output)
  SparseTensor(indices=tf.Tensor(
  [[    0    10]
   [    0    20]
   [    0    30]
   [    1    11]
   [    1   101]
   [    1 10001]], shape=(6, 2), dtype=int64),
   values=tf.Tensor([1 2 1 2 1 1], shape=(6,), dtype=int64),
   dense_shape=tf.Tensor([    2 10002], shape=(2,), dtype=int64))

  **Bin-counting with defined output shape**

  This example takes an input (which could be a Tensor, RaggedTensor, or
  SparseTensor) and returns a SparseTensor where the value of (i,j) is the
  number of times value j appears in batch i. However, all values of j
  above 'maxlength' are ignored. The dense_shape of the output sparse tensor
  is set to 'minlength'. Note that, while the input is identical to the
  example above, the value '10001' in batch item 2 is dropped, and the
  dense shape is [2, 500] instead of [2,10002] or [2, 102].

  >>> minlength = maxlength = 500
  >>> data = [[10, 20, 30, 20], [11, 101, 11, 10001]]
  >>> output = tf.sparse.bincount(
  ...    data, axis=-1, minlength=minlength, maxlength=maxlength)
  >>> print(output)
  SparseTensor(indices=tf.Tensor(
  [[  0  10]
   [  0  20]
   [  0  30]
   [  1  11]
   [  1 101]], shape=(5, 2), dtype=int64),
   values=tf.Tensor([1 2 1 2 1], shape=(5,), dtype=int64),
   dense_shape=tf.Tensor([  2 500], shape=(2,), dtype=int64))

  **Binary bin-counting**

  This example takes an input (which could be a Tensor, RaggedTensor, or
  SparseTensor) and returns a SparseTensor where (i,j) is 1 if the value j
  appears in batch i at least once and is 0 otherwise. Note that, even though
  some values (like 20 in batch 1 and 11 in batch 2) appear more than once,
  the 'values' tensor is all 1s.

  >>> dense = [[10, 20, 30, 20], [11, 101, 11, 10001]]
  >>> output = tf.sparse.bincount(dense, binary_count=True, axis=-1)
  >>> print(output)
  SparseTensor(indices=tf.Tensor(
  [[    0    10]
   [    0    20]
   [    0    30]
   [    1    11]
   [    1   101]
   [    1 10001]], shape=(6, 2), dtype=int64),
   values=tf.Tensor([1 1 1 1 1 1], shape=(6,), dtype=int64),
   dense_shape=tf.Tensor([    2 10002], shape=(2,), dtype=int64))

  """
  with ops.name_scope(name, "count", [values, weights]):
    if not isinstance(values, sparse_tensor.SparseTensor):
      values = ragged_tensor.convert_to_tensor_or_ragged_tensor(
          values, name="values")

    if weights is not None and binary_count:
      raise ValueError("binary_count and weights are mutually exclusive.")

    if weights is None:
      weights = []
      output_type = dtypes.int64
    else:
      output_type = dtypes.float32

    if axis is None:
      axis = 0

    if axis not in [0, -1]:
      raise ValueError("Unsupported axis value %s. Only 0 and -1 are currently "
                       "supported." % axis)

    minlength_value = minlength if minlength is not None else -1
    maxlength_value = maxlength if maxlength is not None else -1

    if axis == 0:
      if isinstance(values,
                    (sparse_tensor.SparseTensor, ragged_tensor.RaggedTensor)):
        values = values.values
      else:
        values = array_ops.reshape(values, [-1])

    if isinstance(values, sparse_tensor.SparseTensor):
      c_ind, c_val, c_shape = gen_count_ops.sparse_count_sparse_output(
          values.indices,
          values.values,
          values.dense_shape,
          weights=weights,
          minlength=minlength_value,
          maxlength=maxlength_value,
          binary_count=binary_count,
          output_type=output_type)
    elif isinstance(values, ragged_tensor.RaggedTensor):
      c_ind, c_val, c_shape = gen_count_ops.ragged_count_sparse_output(
          values.row_splits,
          values.values,
          weights=weights,
          minlength=minlength_value,
          maxlength=maxlength_value,
          binary_count=binary_count,
          output_type=output_type)
    else:
      c_ind, c_val, c_shape = gen_count_ops.dense_count_sparse_output(
          values,
          weights=weights,
          minlength=minlength_value,
          maxlength=maxlength_value,
          binary_count=binary_count,
          output_type=output_type)

    return sparse_tensor.SparseTensor(c_ind, c_val, c_shape)
