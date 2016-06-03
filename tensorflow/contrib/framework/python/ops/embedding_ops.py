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
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops as tf_embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops


__all__ = ["safe_embedding_lookup_sparse",]


def safe_embedding_lookup_sparse(
    embedding_weights, sparse_ids, sparse_weights=None, combiner="mean",
    default_id=None, name=None, partition_strategy="div"):
  """Lookup embedding results, accounting for invalid IDs and empty features.

  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of `P`.

  Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
  with non-positive weight. For an entry with no features, the embedding vector
  for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

  Args:
    embedding_weights:  A list of `P` float tensors or values representing
        partitioned embedding tensors.
    sparse_ids: `SparseTensor` of shape `[batch_size, ?]` containing the ids.
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
    Dense tensor of shape `[batch_size, embed_dim]`.

  Raises:
    ValueError: if `embedding_weights` is empty.
  """
  if embedding_weights is None or len(embedding_weights) < 1:
    raise ValueError("Missing embedding_weights %s." % embedding_weights)

  dtype = sparse_weights.dtype if sparse_weights else None
  embedding_weights = [
      ops.convert_to_tensor(w, dtype=dtype) for w in embedding_weights]

  contrib_tensor_util.assert_same_float_dtype(
      embedding_weights + [sparse_weights])

  with ops.op_scope(
      embedding_weights + [sparse_ids, sparse_weights], name,
      "embedding_lookup") as scope:
    # Prune invalid ids and weights.
    sparse_ids, sparse_weights = _prune_invalid_ids(sparse_ids, sparse_weights)

    # Fill in dummy values for empty features, if necessary.
    sparse_ids, is_row_empty = sparse_ops.sparse_fill_empty_rows(
        sparse_ids, default_id or 0)
    if sparse_weights:
      sparse_weights, _ = sparse_ops.sparse_fill_empty_rows(
          sparse_weights, 1.0)

    result = tf_embedding_ops.embedding_lookup_sparse(
        embedding_weights, sparse_ids, sparse_weights, combiner=combiner,
        partition_strategy=partition_strategy,
        name=None if default_id is None else scope)

    if default_id is None:
      # Broadcast is_row_empty to the same shape as embedding_lookup_result,
      # for use in Select.
      is_row_empty = array_ops.tile(
          array_ops.reshape(is_row_empty, [-1, 1]),
          array_ops.pack([1, array_ops.shape(result)[1]]))

      result = math_ops.select(
          is_row_empty, array_ops.zeros_like(result), result, name=scope)

    return result


def _prune_invalid_ids(sparse_ids, sparse_weights=None,
                       filter_invalid_weights=True):
  """Prune invalid IDs (< 0) from the input ids and weights."""
  is_id_valid = math_ops.greater_equal(sparse_ids.values, 0)
  if sparse_weights and filter_invalid_weights:
    is_id_valid = math_ops.logical_and(
        is_id_valid, math_ops.greater(sparse_weights.values, 0))
  sparse_ids = sparse_ops.sparse_retain(sparse_ids, is_id_valid)
  if sparse_weights:
    sparse_weights = sparse_ops.sparse_retain(sparse_weights, is_id_valid)
  return sparse_ids, sparse_weights
