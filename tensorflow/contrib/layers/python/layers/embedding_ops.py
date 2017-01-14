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

from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.contrib.layers.python.ops import sparse_feature_cross_op

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops

__all__ = ["safe_embedding_lookup_sparse", "hashed_embedding_lookup",
           "hashed_embedding_lookup_sparse"]


def safe_embedding_lookup_sparse(embedding_weights,
                                 sparse_ids,
                                 sparse_weights=None,
                                 combiner="mean",
                                 default_id=None,
                                 name=None,
                                 partition_strategy="div"):
  """Lookup embedding results, accounting for invalid IDs and empty features.

  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of `P`.

  Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
  with non-positive weight. For an entry with no features, the embedding vector
  for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

  The ids and weights may be multi-dimensional. Embeddings are always aggregated
  along the last dimension.

  Args:
    embedding_weights:  A list of `P` float tensors or values representing
        partitioned embedding tensors.  The total unpartitioned shape should be
        `[e_0, e_1, ..., e_m]`, where `e_0` represents the vocab size and
        `e_1, ..., e_m` are the embedding dimensions.
    sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
        ids. `d_0` is typically batch size.
    sparse_weights: `SparseTensor` of same shape as `sparse_ids`, containing
        float weights corresponding to `sparse_ids`, or `None` if all weights
        are be assumed to be 1.0.
    combiner: A string specifying how to combine embedding results for each
        entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean"
        the default.
    default_id: The id to use for an entry with no features.
    name: A name for this operation (optional).
    partition_strategy: A string specifying the partitioning strategy.
        Currently `"div"` and `"mod"` are supported. Default is `"div"`.


  Returns:
    Dense tensor of shape `[d_0, d_1, ..., d_{n-1}, e_1, ..., e_m]`.

  Raises:
    ValueError: if `embedding_weights` is empty.
  """
  if embedding_weights is None or len(embedding_weights) < 1:
    raise ValueError("Missing embedding_weights %s." % embedding_weights)

  dtype = sparse_weights.dtype if sparse_weights is not None else None
  embedding_weights = [
      ops.convert_to_tensor(w, dtype=dtype) for w in embedding_weights
  ]

  contrib_tensor_util.assert_same_float_dtype(embedding_weights +
                                              [sparse_weights])

  with ops.op_scope(embedding_weights + [sparse_ids, sparse_weights], name,
                    "embedding_lookup") as scope:
    # Reshape higher-rank sparse ids and weights to linear segment ids.
    original_shape = sparse_ids.shape
    original_rank_dim = sparse_ids.shape.get_shape()[0]
    original_rank = (
        array_ops.size(original_shape)
        if original_rank_dim.value is None
        else original_rank_dim.value)
    sparse_ids = sparse_ops.sparse_reshape(sparse_ids, [
        math_ops.reduce_prod(
            array_ops.slice(original_shape, [0], [original_rank - 1])),
        array_ops.gather(original_shape, original_rank - 1)])
    if sparse_weights is not None:
      sparse_weights = ops.SparseTensor(sparse_ids.indices,
                                        sparse_weights.values, sparse_ids.shape)

    # Prune invalid ids and weights.
    sparse_ids, sparse_weights = _prune_invalid_ids(sparse_ids, sparse_weights)

    # Fill in dummy values for empty features, if necessary.
    sparse_ids, is_row_empty = sparse_ops.sparse_fill_empty_rows(sparse_ids,
                                                                 default_id or
                                                                 0)
    if sparse_weights is not None:
      sparse_weights, _ = sparse_ops.sparse_fill_empty_rows(sparse_weights, 1.0)

    result = embedding_ops.embedding_lookup_sparse(
        embedding_weights,
        sparse_ids,
        sparse_weights,
        combiner=combiner,
        partition_strategy=partition_strategy,
        name=None if default_id is None else scope)

    if default_id is None:
      # Broadcast is_row_empty to the same shape as embedding_lookup_result,
      # for use in Select.
      is_row_empty = array_ops.tile(
          array_ops.reshape(is_row_empty, [-1, 1]),
          array_ops.pack([1, array_ops.shape(result)[1]]))

      result = math_ops.select(is_row_empty,
                               array_ops.zeros_like(result),
                               result,
                               name=scope)

    # Reshape back from linear ids back into higher-dimensional dense result.
    final_result = array_ops.reshape(result, array_ops.concat(0, [
        array_ops.slice(
            math_ops.cast(original_shape, dtypes.int32),
            [0], [original_rank - 1]),
        array_ops.slice(array_ops.shape(result), [1], [-1])]))
    final_result.set_shape(tensor_shape.unknown_shape(
        (original_rank_dim - 1).value).concatenate(result.get_shape()[1:]))
    return final_result


def _prune_invalid_ids(sparse_ids, sparse_weights):
  """Prune invalid IDs (< 0) from the input ids and weights."""
  is_id_valid = math_ops.greater_equal(sparse_ids.values, 0)
  if sparse_weights is not None:
    is_id_valid = math_ops.logical_and(
        is_id_valid, math_ops.greater(sparse_weights.values, 0))
  sparse_ids = sparse_ops.sparse_retain(sparse_ids, is_id_valid)
  if sparse_weights is not None:
    sparse_weights = sparse_ops.sparse_retain(sparse_weights, is_id_valid)
  return sparse_ids, sparse_weights


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
