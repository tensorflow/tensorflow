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

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.contrib.layers.python.ops import sparse_feature_cross_op

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging

__all__ = [
    "safe_embedding_lookup_sparse", "scattered_embedding_lookup",
    "scattered_embedding_lookup_sparse", "embedding_lookup_unique",
    "embedding_lookup_sparse_with_distributed_aggregation"
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
  if embedding_weights is None:
    raise ValueError("Missing embedding_weights %s." % embedding_weights)
  if isinstance(embedding_weights, variables.PartitionedVariable):
    embedding_weights = list(embedding_weights)  # get underlying Variables.
  if not isinstance(embedding_weights, list):
    embedding_weights = [embedding_weights]
  if len(embedding_weights) < 1:
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
        array_ops.concat([
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
      expected_shape = array_ops.concat([values_shape, dimension_tensor], 0)
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
        params, ids, partition_strategy="div")

    return array_ops.reshape(result,
                             array_ops.concat([values_shape, [dimension]], 0))


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
    embed_shape = array_ops.concat(
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


def embedding_lookup_sparse_with_distributed_aggregation(
    params,
    sp_ids,
    sp_weights,
    partition_strategy="mod",
    name=None,
    combiner=None,
    max_norm=None):
  """Computes embeddings for the given ids and weights.

  Embeddings belonging to same param are aggregated on that device first. This
  op is intended to decrease data transmission and improve parallelism. See
  `tf.nn.embedding_lookup_sparse` for the functionality and example of this op.

  Args:
    params: A single tensor representing the complete embedding tensor,
      or a list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.  Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    sp_ids: N x M SparseTensor of int64 ids (typically from FeatureValueToId),
      where N is typically batch size and M is arbitrary.
    sp_weights: either a SparseTensor of float / double weights, or None to
      indicate all weights should be taken to be 1. If specified, sp_weights
      must have exactly the same shape and indices as sp_ids.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: Optional name for the op.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
      and "sum" are supported.
      "sum" computes the weighted sum of the embedding results for each row.
      "mean" is the weighted sum divided by the total weight.
      "sqrtn" is the weighted sum divided by the square root of the sum of the
      squares of the weights.
    max_norm: If not None, each embedding is normalized to have l2 norm equal
      to max_norm before combining.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by sp_ids, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

  Raises:
    TypeError: If sp_ids is not a SparseTensor, or if sp_weights is neither
      None nor SparseTensor.
    ValueError: If combiner is not one of {"mean", "sqrtn", "sum"}.
  """
  if combiner is None:
    logging.warn("The default value of combiner will change from \"mean\" "
                 "to \"sqrtn\" after 2016/11/01.")
    combiner = "mean"
  if combiner not in ("mean", "sqrtn", "sum"):
    raise ValueError("combiner must be one of 'mean', 'sqrtn' or 'sum'")
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  if not isinstance(params, list):
    params = [params]
  if not isinstance(sp_ids, sparse_tensor.SparseTensor):
    raise TypeError("sp_ids must be SparseTensor")
  ignore_weights = sp_weights is None
  if not ignore_weights:
    if not isinstance(sp_weights, sparse_tensor.SparseTensor):
      raise TypeError("sp_weights must be either None or SparseTensor")
    sp_ids.values.get_shape().assert_is_compatible_with(
        sp_weights.values.get_shape())
    sp_ids.indices.get_shape().assert_is_compatible_with(
        sp_weights.indices.get_shape())
    sp_ids.dense_shape.get_shape().assert_is_compatible_with(
        sp_weights.dense_shape.get_shape())
    # TODO(yleon): Add enhanced node assertions to verify that sp_ids and
    # sp_weights have equal indices and shapes.

  with ops.name_scope(name, "embedding_lookup_sparse",
                      params + [sp_ids]) as name:
    segment_ids = sp_ids.indices[:, 0]
    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)

    ids = sp_ids.values
    if ignore_weights:
      ids, idx = array_ops.unique(ids)
    else:
      idx = None

    weights = None if ignore_weights else sp_weights.values
    embeddings = _embedding_lookup_with_distributed_aggregation(
        params,
        ids,
        partition_strategy=partition_strategy,
        max_norm=max_norm,
        weights=weights,
        idx=idx,
        segment_ids=segment_ids)
    # Set weights to all one if ignore weights.
    if ignore_weights:
      weights = array_ops.fill([array_ops.shape(segment_ids)[0]], 1)
    if weights.dtype != embeddings.dtype:
      weights = math_ops.cast(weights, embeddings.dtype)
    # Reshape weights.
    ones = array_ops.fill(
        array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0), 1)
    bcast_weights_shape = array_ops.concat([array_ops.shape(weights), ones], 0)
    orig_weights_shape = weights.get_shape()
    weights = array_ops.reshape(weights, bcast_weights_shape)
    if embeddings.get_shape().ndims is not None:
      weights.set_shape(
          orig_weights_shape.concatenate(
              [1 for _ in range(embeddings.get_shape().ndims - 1)]))

    if combiner == "mean":
      weight_sum = math_ops.segment_sum(weights, segment_ids)
      embeddings = math_ops.div(embeddings, weight_sum)
    elif combiner == "sqrtn":
      weights_squared = math_ops.pow(weights, 2)
      weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
      weight_sum_sqrt = math_ops.sqrt(weight_sum)
      embeddings = math_ops.div(embeddings, weight_sum_sqrt)
    elif combiner != "sum":
      assert False, "Unrecognized combiner"
    return embeddings


def _do_gather(params, ids, name=None):
  """Deals with doing gather differently for resource variables."""
  if isinstance(params, resource_variable_ops.ResourceVariable):
    return params.sparse_read(ids, name=name)
  return array_ops.gather(params, ids, name=name)


def _embedding_lookup_with_distributed_aggregation(params,
                                                   ids,
                                                   partition_strategy="mod",
                                                   name=None,
                                                   max_norm=None,
                                                   weights=None,
                                                   idx=None,
                                                   segment_ids=None):
  """Lookup helper for embedding_lookup_sparse_with_distributed_aggregation."""
  if params is None or params == []:  # pylint: disable=g-explicit-bool-comparison
    raise ValueError("Need at least one param")
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  if not isinstance(params, list):
    params = [params]

  def maybe_normalize(x):
    if max_norm is not None:
      if x.get_shape().ndims is not None:
        ndims = x.get_shape().ndims
      else:
        ndims = array_ops.size(array_ops.shape(x))
      return clip_ops.clip_by_norm(x, max_norm, axes=list(range(1, ndims)))
    return x

  with ops.name_scope(name, "embedding_lookup_with_distributed_aggregation",
                      params + [ids]) as name:
    np = len(params)  # Number of partitions
    # Preserve the resource variable status to avoid accidental dense reads.
    if not any(
        isinstance(p, resource_variable_ops.ResourceVariable) for p in params):
      params = ops.convert_n_to_tensor_or_indexed_slices(params, name="params")
    if np == 1:
      with ops.colocate_with(params[0]):
        ret = maybe_normalize(_do_gather(params[0], ids))
        ignore_weights = weights is None
        if not ignore_weights:
          if weights.dtype != ret.dtype:
            weights = math_ops.cast(weights, ret.dtype)
          # Reshape to allow broadcast
          ones = array_ops.fill(
              array_ops.expand_dims(array_ops.rank(ret) - 1, 0), 1)
          bcast_weights_shape = array_ops.concat(
              [array_ops.shape(weights), ones], 0)
          orig_weights_shape = weights.get_shape()
          weights = array_ops.reshape(weights, bcast_weights_shape)
          # Set weights shape after reshape
          if ret.get_shape().ndims is not None:
            weights.set_shape(
                orig_weights_shape.concatenate(
                    [1 for _ in range(ret.get_shape().ndims - 1)]))
          ret *= weights
          return math_ops.segment_sum(ret, segment_ids, name=name)
        else:
          return math_ops.sparse_segment_sum(ret, idx, segment_ids, name=name)
    else:
      ids = ops.convert_to_tensor(ids, name="ids")
      flat_ids = array_ops.reshape(ids, [-1])
      original_indices = math_ops.range(array_ops.size(flat_ids))

      # Create p_assignments and set new_ids depending on the strategy.
      if partition_strategy == "mod":
        p_assignments = flat_ids % np
        new_ids = flat_ids // np
      elif partition_strategy == "div":
        # Compute num_total_ids as the sum of dim-0 of params, then assign to
        # partitions based on a constant number of ids per partition. Optimize
        # if we already know the full shape statically.
        dim_0_size = params[0].get_shape()[0]
        for p in xrange(1, np):
          dim_0_size += params[p].get_shape()[0]
        if dim_0_size.value:
          num_total_ids = constant_op.constant(dim_0_size.value, flat_ids.dtype)
        else:
          dim_0_sizes = []
          for p in xrange(np):
            if params[p].get_shape()[0].value is not None:
              dim_0_sizes.append(params[p].get_shape()[0].value)
            else:
              with ops.colocate_with(params[p]):
                dim_0_sizes.append(array_ops.shape(params[p])[0])
          num_total_ids = math_ops.reduce_sum(
              math_ops.cast(array_ops.stack(dim_0_sizes), flat_ids.dtype))
        ids_per_partition = num_total_ids // np
        extras = num_total_ids % np

        p_assignments = math_ops.maximum(flat_ids // (ids_per_partition + 1), (
            flat_ids - extras) // ids_per_partition)

        # Emulate a conditional using a boolean indicator tensor
        is_in_first_extras_partitions = math_ops.cast(p_assignments < extras,
                                                      flat_ids.dtype)
        new_ids = (is_in_first_extras_partitions * (flat_ids %
                                                    (ids_per_partition + 1)) +
                   (1 - is_in_first_extras_partitions) * (
                       (flat_ids - extras) % ids_per_partition))
      else:
        raise ValueError("Unrecognized partition strategy: " +
                         partition_strategy)

      # Cast partition assignments to int32 for use in dynamic_partition.
      # There really should not be more than 2^32 partitions.
      p_assignments = math_ops.cast(p_assignments, dtypes.int32)
      # Partition list of ids based on assignments into np separate lists
      gather_ids = data_flow_ops.dynamic_partition(new_ids, p_assignments, np)
      # Similarly, partition the original indices.
      pindices = data_flow_ops.dynamic_partition(original_indices,
                                                 p_assignments, np)
      # Do np separate lookups, finding embeddings for plist[p] in params[p]
      partitioned_result = []
      for p in xrange(np):
        with ops.colocate_with(params[p]):
          partitioned_result.append(_do_gather(params[p], gather_ids[p]))

      ignore_weights = weights is None
      if not ignore_weights:
        # Partition weights according to pindices.
        partitioned_weight = []
        for p in xrange(np):
          partitioned_weight.append(array_ops.gather(weights, pindices[p]))
      # Reshape each partition result.
      element_shape = params[0].get_shape()[1:]
      for p in params[1:]:
        element_shape = element_shape.merge_with(p.get_shape()[1:])
      if element_shape.is_fully_defined():
        for p in xrange(np):
          with ops.colocate_with(params[p]):
            partitioned_result[p] = array_ops.reshape(
                partitioned_result[p],
                array_ops.concat([array_ops.shape(pindices[p]), element_shape],
                                 0))
      else:
        with ops.colocate_with(params[0]):
          params_shape = array_ops.shape(params[0])
        for p in xrange(np):
          with ops.colocate_with(params[p]):
            partitioned_result[p] = array_ops.reshape(
                partitioned_result[p],
                array_ops.concat([
                    array_ops.shape(pindices[p]), array_ops.slice(
                        params_shape, [1], [-1])
                ], 0))
      # Normalize each partition result.
      for p in xrange(np):
        with ops.colocate_with(params[p]):
          partitioned_result[p] = maybe_normalize(partitioned_result[p])
      if not ignore_weights:
        # Multiply each partition result with partition weights.
        for p in xrange(np):
          with ops.colocate_with(params[p]):
            if partitioned_weight[p].dtype != partitioned_result[p].dtype:
              partitioned_weight[p] = math_ops.cast(partitioned_weight[p],
                                                    partitioned_result[p].dtype)
            # Reshape partition weights.
            ones = array_ops.fill(
                array_ops.expand_dims(
                    array_ops.rank(partitioned_result[p]) - 1, 0), 1)
            bcast_weights_shape = array_ops.concat(
                [array_ops.shape(partitioned_weight[p]), ones], 0)
            orig_weights_shape = partitioned_weight[p].get_shape()
            partitioned_weight[p] = array_ops.reshape(partitioned_weight[p],
                                                      bcast_weights_shape)
            if partitioned_result[p].get_shape().ndims is not None:
              partitioned_weight[p].set_shape(
                  orig_weights_shape.concatenate([
                      1
                      for _ in range(partitioned_result[p].get_shape().ndims -
                                     1)
                  ]))
            partitioned_result[p] *= partitioned_weight[p]
      partitioned_segment_ids = []
      for p in xrange(np):
        if not ignore_weights:
          # Partition segment_ids according to pindices.
          p_segment_ids = array_ops.gather(segment_ids, pindices[p])
          # Number the p_segment_ids to meet segment_sum's requirements. Note
          # that unique_p_segment_ids contains unique segment ids of this
          # partition and these ids' order is unchanged.
          unique_p_segment_ids, unique_p_segment_idx = array_ops.unique(
              p_segment_ids)
          partitioned_segment_ids.append(unique_p_segment_ids)
          # segment_sum this partition's result.
          with ops.colocate_with(params[p]):
            partitioned_result[p] = math_ops.segment_sum(
                partitioned_result[p], unique_p_segment_idx)
        else:
          # When ignore weights, we need to get indexs of elements in idx and
          # segment_ids.
          _, exclude_idx = array_ops.setdiff1d(idx, pindices[p])
          all_idx = math_ops.range(array_ops.shape(idx)[0])
          _, include_idx = array_ops.setdiff1d(all_idx, exclude_idx)
          # Gather segment_ids and idx according to indexs.
          p_segment_ids = array_ops.gather(segment_ids, include_idx)
          p_idx = array_ops.gather(idx, include_idx)
          # Number the p_segment_ids, same as ignore_weights case above.
          unique_p_segment_ids, unique_p_segment_idx = array_ops.unique(
              p_segment_ids)
          _, unique_p_idx_idx = array_ops.unique(p_idx)
          partitioned_segment_ids.append(unique_p_segment_ids)
          with ops.colocate_with(params[p]):
            partitioned_result[p] = math_ops.sparse_segment_sum(
                partitioned_result[p], unique_p_idx_idx, unique_p_segment_idx)
      # Concat each partition's segment_ids and result for final segment_sum.
      concat_segment_ids = array_ops.concat(partitioned_segment_ids, 0)
      concat_partitioned_result = array_ops.concat(partitioned_result, 0)
      return math_ops.unsorted_segment_sum(
          concat_partitioned_result,
          concat_segment_ids,
          math_ops.reduce_max(concat_segment_ids) + 1,
          name=name)
