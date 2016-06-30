# pylint: disable=g-bad-file-header
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
"""Embedding functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import embedding_ops as contrib_embedding_ops
from tensorflow.contrib.layers.python.ops import sparse_feature_cross_op
from tensorflow.python.framework import dtypes

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops

__all__ = ["safe_embedding_lookup_sparse", "hashed_embedding_lookup",
           "hashed_embedding_lookup_sparse"]


# TODO(chapelle): move the safe_embedding_lookup_sparse code here (b/29826543)
safe_embedding_lookup_sparse = contrib_embedding_ops.safe_embedding_lookup_sparse  # pylint: disable=line-too-long


def hashed_embedding_lookup(params, values, dimension, name=None):
  """Looks up embeddings using parameter hashing for each value in `values`.

  The i-th embedding component of a value v in `values` is found by retrieving
  the weight whose index is a fingerprint of the pair (v,i).
  The concept is explored as "feature hashing" for model compression in this
  paper: http://arxiv.org/pdf/1504.04788.pdf

  Feature hashing has the pleasant effect of allowing us to compute an embedding
  without needing a pre-determined vocabulary, relieving some amount of process
  complexity. It also allows for us to maintain embeddings for possibly
  trillions of features with a fixed amount of memory.

  Note that this is superior to out-of-vocabulary shared "hash buckets" in that
  the embedding is extremely likely to be unique for each token as opposed to
  being shared across probably-colliding tokens. The price is that we must
  compute a hash once for each scalar in the token's embedding as opposed to
  once per token.

  If `params` is a list, it represents a partition of the embedding parameters.
  Each tensor in the list should have the same length, except for the first ones
  which may have an additional element. For instance 10 parameters can be
  partitioned in 4 tensors with length `[3, 3, 2, 2]`.

  Args:
    params: A `Tensor` or `list` of `Tensors`.
      Each tensor must be of rank 1 with fully-defined shape.
    values: `Tensor` of values to be embedded.
    dimension: Embedding dimension
    name: An optional name for this op.

  Returns:
    A tensor with shape [d0, ..., dn, dimension]
      with shape(values) = [d0, ..., dn]

  Raises:
    ValueError: if dimension is not positive or the partition size is invalid.
  """
  if not isinstance(params, list):
    params = [params]

  with ops.op_scope(params + [dimension, values], name,
                    "hashed_embedding_lookup"):
    if dimension <= 0:
      raise ValueError("Dimension should be >0 not %d" % dimension)

    num_partitions = len(params)
    partition_sizes = []
    for p in range(num_partitions):
      shape = params[p].get_shape()
      shape.assert_has_rank(1)
      shape.assert_is_fully_defined()
      partition_sizes.append(shape[0].value)
    num_params = sum(partition_sizes)  # Total number of parameters.

    # Assert the size of each partition.
    for p in range(num_partitions):
      expected_size = (num_params - p - 1) // num_partitions + 1
      if partition_sizes[p] != expected_size:
        raise ValueError("Tensor %d in params has size %d, expected %d." %
                         (p, partition_sizes[p], expected_size))

    # Flatten the values
    values_shape = array_ops.shape(values)
    values = array_ops.reshape(values, [-1, 1])

    # With two values v1 and v2 and 3 dimensions, we will cross
    # [[0, 1, 2], [0, 1, 2]] with [[v1], [v2]].
    tensors_to_cross = [array_ops.tile(array_ops.expand_dims(
        math_ops.range(0, dimension), 0), array_ops.shape(values)), values]
    ids = sparse_feature_cross_op.sparse_feature_cross(
        tensors_to_cross, hashed_output=True, num_buckets=num_params)
    ids = sparse_ops.sparse_tensor_to_dense(ids)

    # No need to validate the indices since we have checked the params
    # dimensions and we know the largest id.
    result = embedding_ops.embedding_lookup(
        params, ids, partition_strategy="div", validate_indices=False)

    return array_ops.reshape(result, array_ops.concat(
        0, [values_shape, [dimension]]))


def hashed_embedding_lookup_sparse(params,
                                   sparse_values,
                                   dimension,
                                   combiner="mean",
                                   default_value=None,
                                   name=None):
  """Looks up embeddings of a sparse feature using parameter hashing.

  See `tf.contrib.layers.hashed_embedding_lookup` for embedding with hashing.

  Args:
    params: A `Tensor` or `list` of `Tensors`.
      Each tensor must be of rank 1 with fully-defined shape.
    sparse_values: A 2-D `SparseTensor` containing the values to be embedded.
      Some rows may be empty.
    dimension: Embedding dimension
    combiner: A string specifying how to combine embedding results for each
        entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean"
        the default.
    default_value: The value to use for an entry with no features.
    name: An optional name for this op.

  Returns:
     Dense tensor with shape [N, dimension] with N the number of rows in
       sparse_values.

  Raises:
    TypeError: If sparse_values is not a SparseTensor.
    ValueError: If combiner is not one of {"mean", "sqrtn", "sum"}.
  """

  if not isinstance(params, list):
    params = [params]
  if not isinstance(sparse_values, ops.SparseTensor):
    raise TypeError("sparse_values must be SparseTensor")

  with ops.op_scope(params + [sparse_values], name,
                    "hashed_sparse_embedding_lookup") as scope:
    # Fill in the empty rows.
    if default_value is None:
      # Random default values to reduce the risk of collision.
      if sparse_values.dtype == dtypes.string:
        default_value = "6ZxWzWOHxZ"
      else:
        default_value = 1288896567
    sparse_values, _ = sparse_ops.sparse_fill_empty_rows(
        sparse_values, default_value)

    segment_ids = sparse_values.indices[:, 0]
    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)

    values = sparse_values.values
    values, idx = array_ops.unique(values)

    embeddings = hashed_embedding_lookup(params, values, dimension)

    if combiner == "sum":
      embeddings = math_ops.sparse_segment_sum(embeddings, idx, segment_ids,
                                               name=scope)
    elif combiner == "mean":
      embeddings = math_ops.sparse_segment_mean(embeddings, idx, segment_ids,
                                                name=scope)
    elif combiner == "sqrtn":
      embeddings = math_ops.sparse_segment_sqrt_n(embeddings, idx, segment_ids,
                                                  name=scope)
    else:
      raise ValueError("Combiner must be one of 'mean', 'sqrtn' or 'sum'.")

    return embeddings
