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
    embedding_weights: A tensor representing the complete embedding tensor
      having the shape [e1, ...eM]
    ragged_ids: A 'RaggedTensor' with type 'int32' or 'int64' containing the ids
      to be looked up in 'embedding_weights' of shape [r0, ..rN]. Values must be
      in the range '[0, embedding_weights.shape[0]]'.
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
    raise ValueError("The embedding weights must be specified.")
  if isinstance(params, (list, tuple)) and not params:
    raise ValueError("The embedding weights should not be empty.")
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
                            allow_dense_grads=False):
  if combiner is None:
    combiner = "mean"
  if combiner not in ("mean", "sqrtn", "sum"):
    raise ValueError(
        f"combiner must be one of 'mean', 'sqrtn' or 'sum', got {combiner}")
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  if not isinstance(params, list):
    params = [params]
  ignore_weights = sp_weights is None
  if not ignore_weights:
    if not isinstance(sp_weights, ragged_tensor.RaggedTensor):
      raise TypeError(f"sp_ids must be of the same type as sp_weights.")
    sp_ids.values.get_shape().assert_is_compatible_with(
        sp_weights.values.get_shape())
    sp_ids.get_shape().assert_is_compatible_with(
        sp_weights.get_shape())
    # TODO(yleon): Add enhanced node assertions to verify that sp_ids and
    # sp_weights have equal indices and shapes.

  with ops.name_scope(name, "embedding_lookup_sparse",
                      params + [sp_ids]) as name:

    segment_ids = sp_ids.value_rowids()
    ids = sp_ids.flat_values

    if len(params) == 1 and max_norm is None and allow_dense_grads:
      idx = ids
      embeddings = params[0]
    else:
      ids, idx = array_ops.unique(ids)
      embeddings = embedding_ops.embedding_lookup(
          params, ids, partition_strategy=partition_strategy, max_norm=max_norm)

    return embedding_ops.embedding_lookup_sparse_impl(embeddings, segment_ids,
                                                      sp_weights, idx, combiner,
                                                      ignore_weights, name)


@dispatch.dispatch_for_api(embedding_ops.safe_embedding_lookup_sparse)
def safe_embedding_lookup_sparse(embedding_weights,
                                 sparse_ids: ragged_tensor.Ragged,
                                 sparse_weights=None,
                                 combiner="mean",
                                 default_id=None,
                                 name=None,
                                 partition_strategy="div",
                                 max_norm=None,
                                 allow_dense_grads=False):
  if embedding_weights is None:
    raise ValueError(f"Missing embedding_weights {embedding_weights}.")
  if isinstance(embedding_weights, variables.PartitionedVariable):
    embedding_weights = list(embedding_weights)  # get underlying Variables.
  if not isinstance(embedding_weights, list):
    embedding_weights = [embedding_weights]
  if len(embedding_weights) < 1:
    raise ValueError(f"Missing embedding_weights {embedding_weights}.")

  dtype = sparse_weights.dtype if sparse_weights is not None else None
  embedding_weights = [
      w if (isinstance(w, resource_variable_ops.ResourceVariable)
            and dtype in (None, w.dtype))
      else ops.convert_to_tensor(w, dtype=dtype)
      for w in embedding_weights
  ]

  with ops.name_scope(name, "embedding_lookup", embedding_weights +
                      [sparse_ids, sparse_weights]) as scope:

    # Prune invalid ids and weights.
    sparse_ids, sparse_weights = _prune_invalid_ids_ragged(sparse_ids,
                                                            sparse_weights)
    if combiner != "sum":
      sparse_ids, sparse_weights = _prune_invalid_weights_ragged(
          sparse_ids, sparse_weights)
    sparse_ids, is_row_empty = ragged_array_ops.fill_empty_rows(
      sparse_ids, default_id or 0)
    if sparse_weights is not None:
      sparse_weights, _ = ragged_array_ops.fill_empty_rows(
          sparse_weights, 1.0)

    result = embedding_lookup_sparse(
        embedding_weights,
        sparse_ids,
        sparse_weights,
        combiner=combiner,
        partition_strategy=partition_strategy,
        name=None if default_id is None else scope,
        max_norm=max_norm,
        allow_dense_grads=allow_dense_grads)

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
  if weights is not None:
    is_id_valid = math_ops.logical_and(
        is_id_valid,
        array_ops.ones_like(weights.values, dtype=dtypes.bool))
  nrows = ids.nrows()
  pruned_values = array_ops.boolean_mask_v2(ids.values, is_id_valid)
  pruned_value_rowids = array_ops.boolean_mask_v2(ids.value_rowids(),
                                                  is_id_valid)
  ids = ragged_tensor.RaggedTensor.from_value_rowids(pruned_values,
                                                     pruned_value_rowids,
                                                     nrows=nrows)
  if weights is not None:
    pruned_weights_values = array_ops.boolean_mask_v2(weights.values,
                                                      is_id_valid)
    pruned_weights_value_rowids = array_ops.boolean_mask_v2(
                                                        weights.value_rowids(),
                                                        is_id_valid)
    weights = ragged_tensor.RaggedTensor.from_value_rowids(
                                                    pruned_weights_values,
                                                    pruned_weights_value_rowids,
                                                    nrows=nrows)

  return ids, weights


def _prune_invalid_weights_ragged(ids, weights):
  """Prune invalid weights (< 0) from the input ids and weights."""
  if weights is not None:
    is_weights_valid = math_ops.greater(weights.values, 0)
    nrows = ids.nrows()
    pruned_values = array_ops.boolean_mask_v2(ids.values, is_weights_valid)
    pruned_value_rowids = array_ops.boolean_mask_v2(ids.value_rowids(),
                                                    is_weights_valid)
    ids = ragged_tensor.RaggedTensor.from_value_rowids(pruned_values,
                                                       pruned_value_rowids,
                                                       nrows=nrows)

    pruned_weights_values = array_ops.boolean_mask_v2(weights.values,
                                                      is_weights_valid)
    pruned_weights_value_rowids = array_ops.boolean_mask_v2(
                                                weights.value_rowids(),
                                                is_weights_valid)
    weights = ragged_tensor.RaggedTensor.from_value_rowids(
                                                  pruned_weights_values,
                                                  pruned_weights_value_rowids,
                                                  nrows=nrows)

  return ids, weights
