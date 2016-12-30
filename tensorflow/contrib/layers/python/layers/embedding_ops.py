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
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging

__all__ = [
    "safe_embedding_lookup_sparse", "scattered_embedding_lookup",
    "scattered_embedding_lookup_sparse", "embedding_lookup_unique"
]


def safe_embedding_lookup_sparse(embedding_weights,
                                 sparse_ids,
                                 sparse_weights=None,
                                 combiner=None,
                                 default_id=None,
                                 name=None,
                                 partition_strategy="div",
                                 max_norm=None):
  """Lookup embedding results, accounting for invalid IDs and empty features.

  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of `P`.  `embedding_weights`
  may be a `PartitionedVariable` as returned by using `tf.get_variable()` with a
  partitioner.

  Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
  with non-positive weight. For an entry with no features, the embedding vector
  for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

  The ids and weights may be multi-dimensional. Embeddings are always aggregated
  along the last dimension.

  Args:
    embedding_weights:  A list of `P` float tensors or values representing
        partitioned embedding tensors.  Alternatively, a `PartitionedVariable`,
        created by partitioning along dimension 0.  The total unpartitioned
        shape should be `[e_0, e_1, ..., e_m]`, where `e_0` represents the
        vocab size and `e_1, ..., e_m` are the embedding dimensions.
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
    max_norm: If not None, all embeddings are l2-normalized to max_norm before
        combining.


  Returns:
    Dense tensor of shape `[d_0, d_1, ..., d_{n-1}, e_1, ..., e_m]`.

  Raises:
    ValueError: if `embedding_weights` is empty.
  """
  if combiner is None:
    logging.warn("The default value of combiner will change from \"mean\" "
                 "to \"sqrtn\" after 2016/11/01.")
    combiner = "mean"
  if embedding_weights is None or len(embedding_weights) < 1:
    raise ValueError("Missing embedding_weights %s." % embedding_weights)

  dtype = sparse_weights.dtype if sparse_weights is not None else None
  if isinstance(embedding_weights, variables.PartitionedVariable):
    embedding_weights = list(embedding_weights)
  embedding_weights = [
      ops.convert_to_tensor(w, dtype=dtype) for w in embedding_weights
  ]

  contrib_tensor_util.assert_same_float_dtype(embedding_weights +
                                              [sparse_weights])

  with ops.name_scope(name, "embedding_lookup",
                      embedding_weights + [sparse_ids,
                                           sparse_weights]) as scope:
    # Reshape higher-rank sparse ids and weights to linear segment ids.
    original_shape = sparse_ids.dense_shape
    original_rank_dim = sparse_ids.dense_shape.get_shape()[0]
    original_rank = (
        array_ops.size(original_shape)
        if original_rank_dim.value is None
        else original_rank_dim.value)
    sparse_ids = sparse_ops.sparse_reshape(sparse_ids, [
        math_ops.reduce_prod(
            array_ops.slice(original_shape, [0], [original_rank - 1])),
        array_ops.gather(original_shape, original_rank - 1)])
    if sparse_weights is not None:
      sparse_weights = sparse_tensor.SparseTensor(
          sparse_ids.indices,
          sparse_weights.values, sparse_ids.dense_shape)

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
        name=None if default_id is None else scope,
        max_norm=max_norm)

    if default_id is None:
      # Broadcast is_row_empty to the same shape as embedding_lookup_result,
      # for use in Select.
      is_row_empty = array_ops.tile(
          array_ops.reshape(is_row_empty, [-1, 1]),
          array_ops.stack([1, array_ops.shape(result)[1]]))

      result = array_ops.where(is_row_empty,
                               array_ops.zeros_like(result),
                               result,
                               name=scope)

    # Reshape back from linear ids back into higher-dimensional dense result.
    final_result = array_ops.reshape(
        result,
        array_ops.concat_v2([
            array_ops.slice(
                math_ops.cast(original_shape, dtypes.int32), [0],
                [original_rank - 1]),
            array_ops.slice(array_ops.shape(result), [1], [-1])
        ], 0))
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


def scattered_embedding_lookup(params,
                               values,
                               dimension,
                               name=None,
                               hash_key=None):
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
    params: A `Tensor`, `list` of `Tensors`, or `PartitionedVariable`.
      Each tensor must be of rank 1 with fully-defined shape.
    values: `Tensor` of values to be embedded with shape `[d0, ..., dn]`.
    dimension: Embedding dimension.
    name: An optional name for this op.
    hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
      function to combine the crosses fingerprints on SparseFeatureCrossOp
      (optional).

  Returns:
    A `Tensor` with shape `[d0, ..., dn, dimension]`.

  Raises:
    ValueError: if dimension is not positive or the partition size is invalid.
  """
  if dimension is None:
    raise ValueError("You must specify dimension.")
  return _sampled_scattered_embedding_lookup(
      params, values, dimension=dimension, sampled_candidates=None,
      hash_key=hash_key, name=name)


def _sampled_scattered_embedding_lookup(
    params, values, dimension=None, sampled_candidates=None, hash_key=None,
    name=None):
  """Looks up embeddings using parameter hashing for each value in `values`.

  This method looks up selected embedding dimensions if `sampled_candidates` is
  given, otherwise looks up all dimensions.

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
    params: A `Tensor`, `list` of `Tensors`, or `PartitionedVariable`.
      Each tensor must be of rank 1 with fully-defined shape.
    values: `Tensor` of values to be embedded with shape `[d0, ..., dn]`.
    dimension: Embedding dimension. The user must specify either `dimension` or
      `sampled_candidates`.
    sampled_candidates: An optional `Tensor` of slice indices to keep along the
      final dimension with shape `[d0, ..., dn, N]`. If given, `dimension` is
      ignored. If `None`, looks up all candidates.
    hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
      function to combine the crosses fingerprints on SparseFeatureCrossOp
      (optional).
    name: An optional name for this op.

  Returns:
    A `Tensor` with shape `[d0, ..., dn, dimension]`.
    If `sampled_candidates` is given, the output shape is `[d0, ..., dn, N]`

  Raises:
    ValueError: if dimension is not positive or the partition size is invalid.
  """
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)
  if not isinstance(params, list):
    params = [params]

  with ops.name_scope(name, "scattered_embedding_lookup",
                      params + [dimension, values]):
    # Flatten the values
    values_shape = array_ops.shape(values)
    values = array_ops.reshape(values, [-1, 1])

    if sampled_candidates is None:
      if dimension is None:
        raise ValueError(
            "You must specify either dimension or sampled_candidates.")
      if dimension <= 0:
        raise ValueError("Dimension must be >0. Given is %d" % dimension)
      sampled_candidates = array_ops.tile(array_ops.expand_dims(
          math_ops.range(0, dimension), 0), array_ops.shape(values))
    else:
      dimension = array_ops.shape(sampled_candidates)[
          math_ops.subtract(array_ops.rank(sampled_candidates), 1)]
      sampled_candidates_shape = array_ops.shape(sampled_candidates)
      dimension_tensor = array_ops.reshape(dimension, shape=[1,])
      expected_shape = array_ops.concat_v2([values_shape, dimension_tensor], 0)
      with ops.control_dependencies([control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(sampled_candidates_shape,
                                             expected_shape)),
          ["The shape of sampled_candidates: ", sampled_candidates_shape,
           " does not match the shape of values: ", values_shape])]):
        # Flatten sampled_candidates, same way as values are flattened.
        sampled_candidates = array_ops.reshape(sampled_candidates,
                                               [-1, dimension])

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

    # With two values v1 and v2 and 3 dimensions, we will cross
    # [[0, 1, 2], [0, 1, 2]] with [[v1], [v2]].
    tensors_to_cross = [sampled_candidates, values]
    ids = sparse_feature_cross_op.sparse_feature_cross(
        tensors_to_cross, hashed_output=True, num_buckets=num_params,
        hash_key=hash_key)
    ids = sparse_ops.sparse_tensor_to_dense(ids)

    # No need to validate the indices since we have checked the params
    # dimensions and we know the largest id.
    result = embedding_ops.embedding_lookup(
        params, ids, partition_strategy="div", validate_indices=False)

    return array_ops.reshape(
        result, array_ops.concat_v2([values_shape, [dimension]], 0))


def scattered_embedding_lookup_sparse(params,
                                      sparse_values,
                                      dimension,
                                      combiner=None,
                                      default_value=None,
                                      name=None,
                                      hash_key=None):
  """Looks up embeddings of a sparse feature using parameter hashing.

  See `tf.contrib.layers.scattered_embedding_lookup` for embedding with hashing.

  Args:
    params: A `Tensor`, `list` of `Tensors`, or `PartitionedVariable`.
      Each tensor must be of rank 1 with fully-defined shape.
    sparse_values: A 2-D `SparseTensor` containing the values to be embedded.
      Some rows may be empty.
    dimension: Embedding dimension
    combiner: A string specifying how to combine embedding results for each
        entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean"
        the default.
    default_value: The value to use for an entry with no features.
    name: An optional name for this op.
    hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
      function to combine the crosses fingerprints on SparseFeatureCrossOp
      (optional).

  Returns:
     Dense tensor with shape [N, dimension] with N the number of rows in
       sparse_values.

  Raises:
    TypeError: If sparse_values is not a SparseTensor.
    ValueError: If combiner is not one of {"mean", "sqrtn", "sum"}.
  """
  if combiner is None:
    logging.warn("The default value of combiner will change from \"mean\" "
                 "to \"sqrtn\" after 2016/11/01.")
    combiner = "mean"
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)
  if not isinstance(params, list):
    params = [params]
  if not isinstance(sparse_values, sparse_tensor.SparseTensor):
    raise TypeError("sparse_values must be SparseTensor")

  with ops.name_scope(name, "scattered_embedding_lookup_sparse",
                      params + [sparse_values]) as scope:
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

    embeddings = scattered_embedding_lookup(
        params, values, dimension, hash_key=hash_key)

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


def embedding_lookup_unique(params, ids, name=None):
  """Version of embedding_lookup that avoids duplicate lookups.

  This can save communication in the case of repeated ids.
  Same interface as embedding_lookup. Except it supports multi-dimensional `ids`
  which allows to not reshape input/output to fit gather.

  Args:
    params: A list of tensors with the same shape and type, or a
      `PartitionedVariable`. Shape `[index, d1, d2, ...]`.
    ids: A one-dimensional `Tensor` with type `int32` or `int64` containing
      the ids to be looked up in `params`. Shape `[ids1, ids2, ...]`.
    name: A name for this operation (optional).

  Returns:
    A `Tensor` with the same type as the tensors in `params` and dimension of
    `[ids1, ids2, d1, d2, ...]`.

  Raises:
    ValueError: If `params` is empty.
  """
  with ops.name_scope(name, "EmbeddingLookupUnique", [params, ids]):
    ids = ops.convert_to_tensor(ids)
    shape = array_ops.shape(ids)
    ids_flat = array_ops.reshape(
        ids, math_ops.reduce_prod(shape, keep_dims=True))
    unique_ids, idx = array_ops.unique(ids_flat)
    unique_embeddings = embedding_ops.embedding_lookup(params, unique_ids)
    embeds_flat = array_ops.gather(unique_embeddings, idx)
    embed_shape = array_ops.concat_v2(
        [shape, array_ops.shape(unique_embeddings)[1:]], 0)
    embeds = array_ops.reshape(embeds_flat, embed_shape)
    embeds.set_shape(ids.get_shape().concatenate(
        unique_embeddings.get_shape()[1:]))
    return embeds


def _sampled_scattered_embedding_lookup_sparse(params,
                                               sp_values,
                                               dimension=None,
                                               sampled_candidates=None,
                                               hash_key=None,
                                               with_sign_hash=False,
                                               name=None):
  """Looks up embeddings using parameter hashing for sparse values.

  This method looks up selected embedding dimensions if `sampled_candidates` is
  given, otherwise looks up all dimensions.

  The i-th embedding component of a value v in `values` is found by retrieving
  the weight whose index is a fingerprint of the pair (v,i).
  The concept is explored as "feature hashing" for model compression in this
  paper: http://arxiv.org/pdf/1504.04788.pdf

  This is logically equivalent to:
  * Transforming `sp_values` (which has shape `[d0, d1]`) into a one-hot
    `Tensor` of shape `[d0, N]`.
  * Multiplying with a `Tensor` `h` of shape `[N, dimension]`, where
    `h(i, j) = params[hash(i, j)]`.

  Args:
    params: A float `Tensor` with rank 1 and fully-defined shape.
    sp_values: A 2D `SparseTensor` to be embedded with shape `[d0, d1]`.
    dimension: An int `Tensor` of the final dimension. The user needs to provide
      either `dimension` or `sampled_candidates`.
    sampled_candidates: An optional `Tensor` of column indices to keep along
      the final dimension with shape `[d0, N]`. If given, `dimension` is
      ignored. If `None`, looks up all candidates.
    hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
      function to combine the crosses fingerprints on SparseFeatureCrossOp
      (optional).
    with_sign_hash:  A `bool` indicating whether `h(i, j)` should be multiplied
      by `+1` or `-1`, where the value selected is determined by hashing
      `(i, j)`. This is often necessary to remove bias resulting from hash
      collisions.
    name: An optional name for this op.

  Returns:
    A `Tensor` of shape `[d0, dimension]`.
    If `sampled_candidates` is given, the output shape is `[d0, N]`.

  Raises:
    TypeError: If sp_values is not `SparseTensor`.
    ValueError: If both `dimension` and `sampled_candidates` are `None`.
  """
  if not isinstance(sp_values, sparse_tensor.SparseTensor):
    raise TypeError("sp_values must be SparseTensor")

  with ops.name_scope(
      name=name,
      default_name="sampled_scattered_embedding_lookup_sparse",
      values=[sp_values, params, dimension, sampled_candidates]) as name_scope:
    segment_ids = sp_values.indices[:, 0]
    if sampled_candidates is not None:
      # Tile sampled_candidates so there is one line corresponding to each
      # element in sp_values.values
      sampled_candidates = array_ops.gather(sampled_candidates, segment_ids)

    embeddings = _sampled_scattered_embedding_lookup(
        params, sp_values.values, dimension=dimension,
        sampled_candidates=sampled_candidates,
        hash_key=hash_key, name="values_lookup")
    if with_sign_hash:
      signs = _sampled_scattered_embedding_lookup(
          array_ops.constant([-1., 1.]), sp_values.values, dimension=dimension,
          sampled_candidates=sampled_candidates, hash_key=hash_key,
          name="signs_lookup")
      embeddings = math_ops.multiply(signs, embeddings, name="signs_hash")

    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)
    num_segments = array_ops.shape(sp_values)[0]

    return math_ops.unsorted_segment_sum(embeddings, segment_ids,
                                         num_segments=num_segments,
                                         name=name_scope)
