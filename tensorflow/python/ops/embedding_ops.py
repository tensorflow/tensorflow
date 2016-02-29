# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Operations for embeddings."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops


def embedding_lookup(params, ids, partition_strategy="mod", name=None,
                     validate_indices=True):
  """Looks up `ids` in a list of embedding tensors.

  This function is used to perform parallel lookups on the list of
  tensors in `params`.  It is a generalization of
  [`tf.gather()`](../../api_docs/python/array_ops.md#gather), where `params` is
  interpreted as a partition of a larger embedding tensor.

  If `len(params) > 1`, each element `id` of `ids` is partitioned between
  the elements of `params` according to the `partition_strategy`.
  In all strategies, if the id space does not evenly divide the number of
  partitions, each of the first `(max_id + 1) % len(params)` partitions will
  be assigned one more id.

  If `partition_strategy` is `"mod"`, we assign each id to partition
  `p = id % len(params)`. For instance,
  13 ids are split across 5 partitions as:
  `[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]]`

  If `partition_strategy` is `"div"`, we assign ids to partitions in a
  contiguous manner. In this case, 13 ids are split across 5 partitions as:
  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`

  The results of the lookup are concatenated into a dense
  tensor. The returned tensor has shape `shape(ids) + shape(params)[1:]`.

  Args:
    params: A list of tensors with the same type and which can be concatenated
      along dimension 0. Each `Tensor` must be appropriately sized for the given
      `partition_strategy`.
    ids: A `Tensor` with type `int32` or `int64` containing the ids to be looked
      up in `params`.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`.
    name: A name for the operation (optional).
    validate_indices: Whether or not to validate gather indices.

  Returns:
    A `Tensor` with the same type as the tensors in `params`.

  Raises:
    ValueError: If `params` is empty.
  """
  if not isinstance(params, list):
    params = [params]
  with ops.op_scope(params + [ids], name, "embedding_lookup") as name:
    if not params:
      raise ValueError("Need at least one param")
    np = len(params)  # Number of partitions
    params = ops.convert_n_to_tensor_or_indexed_slices(params, name="params")
    if np == 1:
      with ops.device(params[0].device):
        return array_ops.gather(params[0], ids, name=name,
                                validate_indices=validate_indices)
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
            with ops.device(params[p].device):
              dim_0_sizes.append(array_ops.shape(params[p])[0])
          num_total_ids = math_ops.reduce_sum(
              math_ops.cast(array_ops.pack(dim_0_sizes), flat_ids.dtype))
        ids_per_partition = num_total_ids // np
        extras = num_total_ids % np

        p_assignments = math_ops.maximum(
            flat_ids // (ids_per_partition + 1),
            (flat_ids - extras) // ids_per_partition)

        # Emulate a conditional using a boolean indicator tensor
        is_in_first_extras_partitions = math_ops.cast(
            p_assignments < extras, flat_ids.dtype)
        new_ids = (
            is_in_first_extras_partitions * (
                flat_ids % (ids_per_partition + 1)) +
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
        with ops.device(params[p].device):
          partitioned_result.append(array_ops.gather(
              params[p], gather_ids[p],
              validate_indices=validate_indices))
      # Stitch these back together
      ret = data_flow_ops.dynamic_stitch(pindices, partitioned_result,
                                         name=name)
      # Reshape to reverse the flattening of ids.
      # It's important that we compute params[0].shape on the right device
      # to avoid data motion.
      with ops.device(params[0].device):
        params_shape = array_ops.shape(params[0])
      ret = array_ops.reshape(ret, array_ops.concat(0, [
          array_ops.shape(ids), array_ops.slice(params_shape, [1], [-1])]))
      # output shape = ids.shape + params[*].shape[1:]
      # Normally the reshape is sufficient, but setting shape explicitly
      # teaches shape inference that params[1:].get_shape() matters.
      element_shape = params[0].get_shape()[1:]
      for p in params[1:]:
        element_shape = element_shape.merge_with(p.get_shape()[1:])
      ret.set_shape(ids.get_shape().concatenate(element_shape))
      return ret


# TODO(lif): Add support for higher-rank SparseTensors
def embedding_lookup_sparse(params, sp_ids, sp_weights,
                            partition_strategy="mod",
                            name=None,
                            combiner="mean"):
  """Computes embeddings for the given ids and weights.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.

  It also assumes that all id values lie in the range [0, p0), where p0
  is the sum of the size of params along dimension 0.

  Args:
    params: A single tensor representing the complete embedding tensor,
      or a list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.
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

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by sp_ids, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if
      shape(combined params) = [p0, p1, ..., pm]
    and
      shape(sp_ids) = shape(sp_weights) = [d0, d1, ..., dn]
    then
      shape(output) = [d0, d1, ..., dn-1, p1, ..., pm].

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id 0, weight 1.0
      [2, 3]: id 1, weight 3.0

    with combiner="mean", then the output will be a 3x20 matrix where
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = params[0, :] * 1.0
      output[2, :] = params[1, :] * 3.0

  Raises:
    TypeError: If sp_ids is not a SparseTensor, or if sp_weights is neither
      None nor SparseTensor.
    ValueError: If combiner is not one of {"mean", "sqrtn", "sum"}.
  """
  if combiner not in ("mean", "sqrtn", "sum"):
    raise ValueError("combiner must be one of 'mean', 'sqrtn' or 'sum'")
  if not isinstance(params, list):
    params = [params]
  if not isinstance(sp_ids, ops.SparseTensor):
    raise TypeError("sp_ids must be SparseTensor")
  ignore_weights = sp_weights is None
  if not ignore_weights:
    if not isinstance(sp_weights, ops.SparseTensor):
      raise TypeError("sp_weights must be either None or SparseTensor")
    sp_ids.values.get_shape().assert_is_compatible_with(
        sp_weights.values.get_shape())
    sp_ids.indices.get_shape().assert_is_compatible_with(
        sp_weights.indices.get_shape())
    sp_ids.shape.get_shape().assert_is_compatible_with(
        sp_weights.shape.get_shape())
    # TODO(yleon): Add enhanced node assertions to verify that sp_ids and
    # sp_weights have equal indices and shapes.

  with ops.op_scope(params + [sp_ids], name, "embedding_lookup_sparse") as name:
    segment_ids = sp_ids.indices[:, 0]
    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)

    ids = sp_ids.values
    if ignore_weights:
      ids, idx = array_ops.unique(ids)
    else:
      idx = None

    embeddings = embedding_lookup(
        params, ids, partition_strategy=partition_strategy)
    if not ignore_weights:
      weights = sp_weights.values
      if weights.dtype != embeddings.dtype:
        weights = math_ops.cast(weights, embeddings.dtype)

      # Reshape weights to allow broadcast
      ones = array_ops.fill(
          array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0), 1)
      bcast_weights_shape = array_ops.concat(0, [
          array_ops.shape(weights), ones])

      orig_weights_shape = weights.get_shape()
      weights = array_ops.reshape(weights, bcast_weights_shape)

      # Set the weight shape, since after reshaping to bcast_weights_shape,
      # the shape becomes None.
      if embeddings.get_shape().ndims is not None:
        weights.set_shape(orig_weights_shape.concatenate(
            [1 for _ in range(embeddings.get_shape().ndims - 1)]))

      embeddings *= weights

      if combiner == "sum":
        embeddings = math_ops.segment_sum(embeddings, segment_ids, name=name)
      elif combiner == "mean":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weight_sum = math_ops.segment_sum(weights, segment_ids)
        embeddings = math_ops.div(embeddings, weight_sum, name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weights_squared = math_ops.pow(weights, 2)
        weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
        weight_sum_sqrt = math_ops.sqrt(weight_sum)
        embeddings = math_ops.div(embeddings, weight_sum_sqrt, name=name)
      else:
        assert False, "Unrecognized combiner"
    else:
      assert idx is not None
      if combiner == "sum":
        embeddings = math_ops.sparse_segment_sum(embeddings, idx, segment_ids,
                                                 name=name)
      elif combiner == "mean":
        embeddings = math_ops.sparse_segment_mean(embeddings, idx, segment_ids,
                                                  name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.sparse_segment_sqrt_n(embeddings, idx,
                                                    segment_ids, name=name)
      else:
        assert False, "Unrecognized combiner"

    return embeddings
