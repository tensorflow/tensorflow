# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Embedding operations."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch


@dispatch.dispatch_for_api(embedding_ops.embedding_lookup)
def embedding_lookup(
    params,
    ids: ragged_tensor.Ragged,
    partition_strategy="mod",
    name=None,
    validate_indices=True,  # pylint: disable=unused-argument
    max_norm=None):
  """Look up the ragged ids in a list of embedding tensors.

  Args:
    params: A tensor representing the complete embedding tensor having the shape
      [e1, ...eM]
    ragged_ids: A 'RaggedTensor' with type 'int32' or 'int64' containing the ids
      to be looked up in 'params' of shape [r0, ..rN]. Values must be
      in the range '[0, params.shape[0]]'.
    partition_strategy: A string specifying the partitioning strategy.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.
    name: A name for the operation (optional)

  Returns:
    A ragged tensor of shape [r0, r1, ...rN, e1, ...eM].

  Raises:
    ValueError: When params is empty or the type of the ids is not int32 or
      int64.
  """
  if params is None:
    raise ValueError("params must be specified.")
  if isinstance(params, (list, tuple)) and not params:
    raise ValueError("params should not be empty.")
  if ids.dtype != dtypes.int32 and ids.dtype != dtypes.int64:
    raise ValueError("The values contained by the inputs have type "
                     f"{str(ids.dtype)}"
                     " and cannot be processed. All values"
                     " should be indices, either of type `int32` or `int64`.")

  with ops.name_scope(name, "embedding_lookup_ragged") as name:
    looked_up_ragged = ragged_functional_ops.map_flat_values(
        embedding_ops.embedding_lookup,
        params=params,
        ids=ids,
        partition_strategy=partition_strategy,
        max_norm=max_norm)

    return looked_up_ragged

@dispatch.dispatch_for_api(embedding_ops.embedding_lookup_sparse)
def embedding_lookup_sparse(params,
                            sp_ids: ragged_tensor.Ragged,
                            sp_weights,
                            partition_strategy="mod",
                            name=None,
                            combiner=None,
                            max_norm=None,
                            allow_fast_lookup=False):
  """Looks up embeddings for the given ids and weights from a list of tensors.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.

  `sp_ids` and `sp_weights` (if not None) are `RaggedTensor`s with rank of 2.
  Embeddings are always aggregated along the last dimension.

  It also assumes that all id values lie in the range [0, p0), where p0
  is the sum of the size of params along dimension 0.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list tensors all of same shape except for the first dimension,
      representing sharded embedding tensors. Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    sp_ids: `RaggedTensor` with rank 2. The rank is not verified for
      performance reasons.
    sparse_weights: `RaggedTensor` of same type and shape as `sparse_ids`,
      containing float / double weights corresponding to `sparse_ids`, or `None`
      if all weights are assumed to be 1.0.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: Optional name for the op.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
      and "sum" are supported. "sum" computes the weighted sum of the embedding
      results for each row. "mean" is the weighted sum divided by the total
      weight. "sqrtn" is the weighted sum divided by the square root of the sum
      of the squares of the weights. Defaults to `mean`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    allow_fast_lookup: An optional boolean specifying whether to allow
      simplified embedding lookups when `params` is a single tensor and
      `max_norm` is `None`. Setting this flag to `True` during training can
      cause the use of dense gradients with increased memory footprint.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if

      `shape(combined params) = [p0, p1, ..., pm]`

    and

      `shape(sp_ids) = shape(sp_weights) = [d0, d1]`

    then

      `shape(output) = [d0, p1, ..., pm]`.

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      ```python
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id 0, weight 1.0
      [2, 3]: id 1, weight 3.0
      ```

    with `combiner`="mean", then the output will be a 3x20 matrix where

      ```python
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = (params[0, :] * 1.0) / 1.0
      output[2, :] = (params[1, :] * 3.0) / 3.0
      ```

  Raises:
    TypeError: If `sp_weights` is neither `None` nor of the same type as
      `sp_ids`.
    ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
  """
  rt_ids = sp_ids
  rt_weights = sp_weights
  if combiner is None:
    combiner = "mean"
  if combiner not in ("mean", "sqrtn", "sum"):
    raise ValueError(
        f"combiner must be one of 'mean', 'sqrtn' or 'sum', got {combiner}")
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  if not isinstance(params, list):
    params = [params]
  ignore_weights = rt_weights is None
  if not ignore_weights:
    if not isinstance(rt_weights, ragged_tensor.RaggedTensor):
      raise TypeError(f"sp_ids must be of the same type as sp_weights, "
         "received {type(sp_ids).__name__!r} for sp_ids and "
         "{type(sp_weights).__name__!r} for sp_weights.")
    rt_ids.values.get_shape().assert_is_compatible_with(
        rt_weights.values.get_shape())
    rt_ids.get_shape().assert_is_compatible_with(
        rt_weights.get_shape())

  with ops.name_scope(name, "embedding_lookup_sparse",
                      params + [rt_ids]) as name:

    segment_ids = rt_ids.value_rowids()
    ids = rt_ids.flat_values

    return embedding_ops.embedding_lookup_sparse_impl(params, segment_ids,
                                                      sp_weights, ids, combiner,
                                                      ignore_weights, max_norm,
                                                      allow_fast_lookup,
                                                      partition_strategy, name)


@dispatch.dispatch_for_api(embedding_ops.safe_embedding_lookup_sparse)
def safe_embedding_lookup_sparse(embedding_weights,
                                 sparse_ids: ragged_tensor.Ragged,
                                 sparse_weights=None,
                                 combiner="mean",
                                 default_id=None,
                                 name=None,
                                 partition_strategy="div",
                                 max_norm=None,
                                 allow_fast_lookup=False):
  """Lookup embedding results, accounting for invalid IDs and empty features.

  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of `P`.  `embedding_weights`
  may be a `PartitionedVariable` as returned by using
  `tf.compat.v1.get_variable()` with a
  partitioner.

  Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
  with non-positive weight. For an entry with no features, the embedding vector
  for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

  The ids and weights may be multi-dimensional `SparseTensor`s or
  `RaggedTensor`s with rank of 2. For `SpareTensor`s with left-aligned non-zero
  entries which can be described as `RaggedTensor`s, use of `RaggedTensor`s can
  yield higher performance. Embeddings are always aggregated along the last
  dimension.

  Args:
    embedding_weights: A single tensor representing the complete embedding
      tensor, or a list tensors all of same shape except for the first
      dimension, representing sharded embedding tensors. Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    sp_ids: `RaggedTensor` with rank 2. The rank is not verified for
      performance reasons.
    sparse_weights: `RaggedTensor` of same type and shape as `sparse_ids`,
      containing float weights corresponding to `sparse_ids`, or `None` if all
      weights are assumed to be 1.0.
    combiner: A string specifying how to combine embedding results for each
      entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean" the
      default.
    default_id: The id to use for an entry with no features.
    name: A name for this operation (optional).
    partition_strategy: A string specifying the partitioning strategy. Currently
      `"div"` and `"mod"` are supported. Default is `"div"`.
    max_norm: If not `None`, all embeddings are l2-normalized to max_norm before
      combining.
    allow_fast_lookup: An optional boolean specifying whether to allow
      simplified embedding lookups when `params` is a single tensor and
      `max_norm` is `None`. Setting this flag to `True` during training can
      cause the use of dense gradients with increased memory footprint.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if

      `shape(combined embedding_weights) = [p0, p1, ..., pm]`

    and

      `shape(sparse_ids) = shape(sparse_weights) = [d0, d1, ..., dn]`

    then

      `shape(output) = [d0, d1, ... dn-1, p1, ..., pm]`.

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      ```python
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id -1, weight 1.0
      [2, 3]: id 1, weight 3.0
      ```

    `default_id` is 0.

    with `combiner`="mean", then the output will be a 3x20 matrix where

      ```python
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = (params[0, :] * 1.0) / 1.0
      output[2, :] = (params[1, :] * 3.0) / 3.0
      ```

  Raises:
    ValueError: if `embedding_weights` is empty.
  """
  ragged_ids = sparse_ids
  ragged_weights = sparse_weights
  if embedding_weights is None:
    raise ValueError(f"Missing embedding_weights {embedding_weights}.")
  if isinstance(embedding_weights, variables.PartitionedVariable):
    embedding_weights = list(embedding_weights)  # get underlying Variables.
  if not isinstance(embedding_weights, list):
    embedding_weights = [embedding_weights]
  if len(embedding_weights) < 1:
    raise ValueError(f"Missing embedding_weights {embedding_weights}.")

  dtype = ragged_weights.dtype if ragged_weights is not None else None
  embedding_weights = [
      w if (isinstance(w, resource_variable_ops.ResourceVariable)
            and dtype in (None, w.dtype))
      else ops.convert_to_tensor(w, dtype=dtype)
      for w in embedding_weights
  ]

  with ops.name_scope(name, "embedding_lookup", embedding_weights +
                      [ragged_ids, ragged_weights]) as scope:

    # Prune invalid ids and weights.
    ragged_ids, ragged_weights = _prune_invalid_ids_ragged(ragged_ids,
                                                            ragged_weights)
    if combiner != "sum":
      ragged_ids, ragged_weights = _prune_invalid_weights_ragged(
          ragged_ids, ragged_weights)
    ragged_ids, is_row_empty = ragged_array_ops.fill_empty_rows(
      ragged_ids, default_id or 0)
    if ragged_weights is not None:
      ragged_weights, _ = ragged_array_ops.fill_empty_rows(
          ragged_weights, 1.0)

    result = embedding_lookup_sparse(
        embedding_weights,
        ragged_ids,
        ragged_weights,
        combiner=combiner,
        partition_strategy=partition_strategy,
        name=None if default_id is None else scope,
        max_norm=max_norm,
        allow_fast_lookup=allow_fast_lookup)

    if default_id is None:
      # Broadcast is_row_empty to the same shape as embedding_lookup_result,
      # for use in Select.
      is_row_empty = array_ops.tile(
          array_ops.reshape(is_row_empty, [-1, 1]),
          array_ops.stack([1, array_ops.shape(result)[1]]))

      result = array_ops.where(
          is_row_empty, array_ops.zeros_like(result), result, name=scope)

    return result


def _prune_invalid_ids_ragged(ids, weights):
  """Prune invalid IDs (< 0) from the input ids and weights."""
  is_id_valid = math_ops.greater_equal(ids.values, 0)
  nrows = ids.nrows()
  # TODO(philipphack): Consider calling ragged_array_ops.boolean_mask once the
  # resulting performance is comparable to array_ops.boolean_mask. Currently,
  # ragged_array_ops.boolean_mask constructs the returned RaggedTensor by
  # calling its from_row_splits method which does not set value_row_ids and
  # requires it to be computed on demand.
  pruned_values = array_ops.boolean_mask_v2(ids.values, is_id_valid)
  pruned_value_rowids = array_ops.boolean_mask_v2(ids.value_rowids(),
                                                  is_id_valid)
  ids = ragged_tensor.RaggedTensor.from_value_rowids(pruned_values,
                                                     pruned_value_rowids,
                                                     nrows=nrows,
                                                     validate=False)
  if weights is not None:
    pruned_weights_values = array_ops.boolean_mask_v2(weights.values,
                                                      is_id_valid)
    weights = ragged_tensor.RaggedTensor.from_value_rowids(
                                                    pruned_weights_values,
                                                    pruned_value_rowids,
                                                    nrows=nrows,
                                                    validate=False)

  return ids, weights


def _prune_invalid_weights_ragged(ids, weights):
  """Prune invalid weights (< 0) from the input ids and weights."""
  if weights is not None:
    is_weights_valid = math_ops.greater(weights.values, 0)
    nrows = ids.nrows()
  # TODO(philipphack): Consider calling ragged_array_ops.boolean_mask once the
  # resulting performance is comparable to array_ops.boolean_mask. Currently,
  # ragged_array_ops.boolean_mask constructs the returned RaggedTensor by
  # calling its from_row_splits method which does not set value_row_ids and
  # requires it to be computed on demand.
    pruned_values = array_ops.boolean_mask_v2(ids.values, is_weights_valid)
    pruned_value_rowids = array_ops.boolean_mask_v2(ids.value_rowids(),
                                                    is_weights_valid)
    ids = ragged_tensor.RaggedTensor.from_value_rowids(pruned_values,
                                                       pruned_value_rowids,
                                                       nrows=nrows,
                                                       validate=False)

    pruned_weights_values = array_ops.boolean_mask_v2(weights.values,
                                                      is_weights_valid)
    weights = ragged_tensor.RaggedTensor.from_value_rowids(
                                                  pruned_weights_values,
                                                  pruned_value_rowids,
                                                  nrows=nrows,
                                                  validate=False)

  return ids, weights
