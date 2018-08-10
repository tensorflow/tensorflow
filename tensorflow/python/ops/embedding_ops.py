# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import clip_ops
# Imports gradient definitions.
from tensorflow.python.ops import data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import partitioned_lookup_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

def _clip(params, ids, max_norm):
  """Helper function for _embedding_lookup_and_transform.

  This function optionally clips embeddings to an l2-norm of max_norm.

  Args:
    params: A `Tensor` of embeddings retrieved by `gather`.
    ids: The `ids` argument that was passed to `gather`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is
      larger than this value.

  Returns:
    A `Tensor` with the same type as `params`.
  """

  def _rank(x):
    """Helper function to retrieve the rank of a tensor.

    Args:
      x: Something convertible to `Tensor`.

    Returns:
      Either a pair `(rank, True)` where `rank` is an integer or a pair
      `(rank, False)` where `rank` is an integer `Tensor`. In either case,
      `rank` is the rank of `x`.
    """
    rank = ops.convert_to_tensor(x).get_shape().ndims
    if rank:
      return rank, True
    else:
      return array_ops.rank(x), False

  if max_norm is None:
    return params
  ids_rank, ids_static = _rank(ids)
  params_rank, params_static = _rank(params)
  return clip_ops.clip_by_norm(
      params,
      max_norm,
      axes=(list(range(ids_rank, params_rank))
            if ids_static and params_static
            else math_ops.range(ids_rank, params_rank)))


def _embedding_lookup_and_transform(params,
                                    ids,
                                    partition_strategy="mod",
                                    name=None,
                                    max_norm=None,
                                    transform_fn=None):
  """Helper function for embedding_lookup and _compute_sampled_logits.

  This function is a generalization of embedding_lookup that optionally
  applies a caller-specified transformation to each embedding. This is
  done through the `transform_fn` argument. If provided, the function is
  applied to each partitioned tensor of retrieved embeddings, colocated
  with the embeddings. This function will be called with a single `Tensor`
  argument of the same type as the `params` tensor and should return a
  `Tensor`. The shape of the argument will be the same as `params` except
  for the size of the first dimension. The first dimension of the result's
  shape must be the same size as the argument's.

  Args:
    params: See embedding_lookup.
    ids: See embedding_lookup.
    partition_strategy: See embedding_lookup.
    name: See embedding_lookup.
    max_norm: See embedding_lookup.
    transform_fn: An optional function to apply to each retrieved embedding.
      If max_norm is provided, transform_fn is applied to the norm-limited
      embeddings.

  Returns:
    See embedding_lookup for details.
  Raises:
    ValueError: If `params` is empty.
  """
  if params is None or params in ((), []):
    raise ValueError("Need at least one param")
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  if not isinstance(params, list):
    params = [params]

  with ops.name_scope(name, "embedding_lookup", params + [ids]) as name:
    np = len(params)  # Number of partitions
    # Preserve the resource variable status to avoid accidental dense reads.
    if not any(
        isinstance(p, resource_variable_ops.ResourceVariable) for p in params):
      params = ops.convert_n_to_tensor_or_indexed_slices(params, name="params")
    ids = ops.convert_to_tensor(ids, name="ids")
    if np == 1 and (not transform_fn or ids.get_shape().ndims == 1):
      with ops.colocate_with(params[0]):
        result = _clip(array_ops.gather(params[0], ids, name=name),
                       ids, max_norm)
        if transform_fn:
          result = transform_fn(result)
        return result
    else:
      # Flatten the ids. There are two cases where we need to do this.
      # - There is more than one params tensor.
      # - There is a transform_fn and ids is not statically known to be 1-D.
      #   We must flatten in this case because transform_fn expects a flat
      #   tensor of embeddings.
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

        p_assignments = math_ops.maximum(
            flat_ids // (ids_per_partition + 1),
            (flat_ids - extras) // ids_per_partition)

        # Emulate a conditional using a boolean indicator tensor
        new_ids = array_ops.where(p_assignments < extras,
                                  flat_ids % (ids_per_partition + 1),
                                  (flat_ids - extras) % ids_per_partition)
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
        pids = gather_ids[p]
        with ops.colocate_with(params[p]):
          result = array_ops.gather(params[p], pids)
          if transform_fn:
            # If transform_fn is provided, the clip_by_norm precedes
            # the transform and hence must be co-located. See below
            # for the counterpart if transform_fn is not proveded.
            result = transform_fn(_clip(result, pids, max_norm))
        partitioned_result.append(result)
      # Stitch these back together
      ret = data_flow_ops.parallel_dynamic_stitch(
          pindices, partitioned_result, name=name)

      # Determine the static element shape.
      if transform_fn is None:
        element_shape_s = params[0].get_shape()[1:]
        for p in params[1:]:
          element_shape_s = element_shape_s.merge_with(p.get_shape()[1:])
      else:
        element_shape_s = ret.get_shape()[1:]

      # Compute the dynamic element shape.
      if element_shape_s.is_fully_defined():
        element_shape_d = element_shape_s
      elif transform_fn is None:
        # It's important that we compute params[0].shape on the right device
        # to avoid data motion.
        with ops.colocate_with(params[0]):
          params_shape = array_ops.shape(params[0])
        element_shape_d = params_shape[1:]
      else:
        element_shape_d = array_ops.shape(ret)[1:]

      # Reshape to reverse the flattening of ids.
      ret = array_ops.reshape(ret,
                              array_ops.concat(
                                  [array_ops.shape(ids), element_shape_d], 0))

      # Normally the reshape is sufficient, but setting shape explicitly
      # teaches shape inference that params[1:].get_shape() matters
      # (in the case that transform_fn is None).
      ret.set_shape(ids.get_shape().concatenate(element_shape_s))
      if not transform_fn:
        # If transform_fn was provided, the clip_by_norm was done above.
        ret = _clip(ret, ids, max_norm)
      return ret


@tf_export("nn.embedding_lookup")
def embedding_lookup(
    params,
    ids,
    partition_strategy="mod",
    name=None,
    validate_indices=True,  # pylint: disable=unused-argument
    max_norm=None):
  """Looks up `ids` in a list of embedding tensors.

  This function is used to perform parallel lookups on the list of
  tensors in `params`.  It is a generalization of
  @{tf.gather}, where `params` is
  interpreted as a partitioning of a large embedding tensor.  `params` may be
  a `PartitionedVariable` as returned by using `tf.get_variable()` with a
  partitioner.

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
    params: A single tensor representing the complete embedding tensor,
      or a list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.  Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    ids: A `Tensor` with type `int32` or `int64` containing the ids to be looked
      up in `params`.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`.
    name: A name for the operation (optional).
    validate_indices: DEPRECATED. If this operation is assigned to CPU, values
      in `indices` are always validated to be within range.  If assigned to GPU,
      out-of-bound indices result in safe but unspecified behavior, which may
      include raising an error.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is
      larger than this value.

  Returns:
    A `Tensor` with the same type as the tensors in `params`.

  Raises:
    ValueError: If `params` is empty.
  """
  return _embedding_lookup_and_transform(
      params=params,
      ids=ids,
      partition_strategy=partition_strategy,
      name=name,
      max_norm=max_norm,
      transform_fn=None)


@tf_export("nn.embedding_lookup_sparse")
def embedding_lookup_sparse(params,
                            sp_ids,
                            sp_weights,
                            partition_strategy="mod",
                            name=None,
                            combiner=None,
                            max_norm=None):
  """Computes embeddings for the given ids and weights.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.

  It also assumes that all id values lie in the range [0, p0), where p0
  is the sum of the size of params along dimension 0.

  Args:
    params: A single tensor representing the complete embedding tensor,
      or a list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.  Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    sp_ids: N x M `SparseTensor` of int64 ids where N is typically batch size
      and M is arbitrary.
    sp_weights: either a `SparseTensor` of float / double weights, or `None` to
      indicate all weights should be taken to be 1. If specified, `sp_weights`
      must have exactly the same shape and indices as `sp_ids`.
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
    max_norm: If not `None`, each embedding is clipped if its l2-norm is
      larger than this value, before combining.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if

      `shape(combined params) = [p0, p1, ..., pm]`

    and

      `shape(sp_ids) = shape(sp_weights) = [d0, d1, ..., dn]`

    then

      `shape(output) = [d0, d1, ..., dn-1, p1, ..., pm]`.

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
    TypeError: If `sp_ids` is not a `SparseTensor`, or if `sp_weights` is
      neither `None` nor `SparseTensor`.
    ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
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
    ids, idx = array_ops.unique(ids)

    embeddings = embedding_lookup(
        params, ids, partition_strategy=partition_strategy, max_norm=max_norm)
    if not ignore_weights:
      weights = sp_weights.values
      if weights.dtype != embeddings.dtype:
        weights = math_ops.cast(weights, embeddings.dtype)

      embeddings = array_ops.gather(embeddings, idx)

      # Reshape weights to allow broadcast
      ones = array_ops.fill(
          array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0), 1)
      bcast_weights_shape = array_ops.concat([array_ops.shape(weights), ones],
                                             0)

      orig_weights_shape = weights.get_shape()
      weights = array_ops.reshape(weights, bcast_weights_shape)

      # Set the weight shape, since after reshaping to bcast_weights_shape,
      # the shape becomes None.
      if embeddings.get_shape().ndims is not None:
        weights.set_shape(
            orig_weights_shape.concatenate(
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
        embeddings = math_ops.sparse_segment_sum(
            embeddings, idx, segment_ids, name=name)
      elif combiner == "mean":
        embeddings = math_ops.sparse_segment_mean(
            embeddings, idx, segment_ids, name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.sparse_segment_sqrt_n(
            embeddings, idx, segment_ids, name=name)
      else:
        assert False, "Unrecognized combiner"

    return embeddings

def embedding_lookup_sparse_with_hash_table(
    emb_table,
    sp_ids,
    sp_weights=None,
    name=None,
    combiner=None,
    is_training=True,
    count_table=None,
    count_filter_thr=0):
  """Computes embeddings for the given ids and weights.

  Args:
    emb_table: A PartitionedMutableHashTable representing the complete embeddings.
    sp_ids: N x M SparseTensor of int64 ids (typically from FeatureValueToId),
      where N is typically batch size and M is arbitrary.
    sp_weights: either a SparseTensor of float / double weights, or None to
      indicate all weights should be taken to be 1. If specified, sp_weights
      must have exactly the same shape and indices as sp_ids.
    name: Optional name for the op.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
      and "sum" are supported.
      "sum" computes the weighted sum of the embedding results for each row.
      "mean" is the weighted sum divided by the total weight.
      "sqrtn" is the weighted sum divided by the square root of the sum of the
      squares of the weights.
    count_table: A PartitionedMutableHashTable representing the count number of id in sp_ids
      used for sparse filter.
    count_filter_thr: A scalar representing the minimum number of id. During training, only `id`
      which count number is greater than the `thr` will be used to get or update its embedding.
      Otherwise, `default` embedding will be used, mostly zero.

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
    logging.warn("The default value of combiner is sum.")
    combiner = "sum"
  if combiner not in ("mean", "sqrtn", "sum"):
    raise ValueError("combiner must be one of 'mean', 'sqrtn' or 'sum'")
  if not isinstance(emb_table, partitioned_lookup_ops.PartitionedMutableHashTable):
    raise TypeError("emb_table must be PartitionedMutableHashTable")
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

  with ops.name_scope(name, "embedding_lookup_sparse",
                      [sp_ids]) as name:
    segment_ids = sp_ids.indices[:, 0]
    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)

    ids = sp_ids.values
    weights = None if ignore_weights else sp_weights.values
    embeddings = _embedding_lookup_with_hash_table(emb_table,
                                                   ids,
                                                   weights,
                                                   segment_ids,
                                                   is_training=is_training,
                                                   count_table=count_table,
                                                   count_filter_thr=count_filter_thr)

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


def _embedding_lookup_with_hash_table(emb_table,
                                      ids,
                                      weights,
                                      segment_ids,
                                      name=None,
                                      is_training=True,
                                      count_table=None,
                                      count_filter_thr=0,
                                      initializer="normal"):
  """Looks up embedding for `ids` from a hash table.

  emb_list = []
  if is_training:
    if use_count_filter:
      for id in ids:
        count_table.insert(count_table.lookup(id)+1)
        if count_table.lookup(id) > count_filter_thr:
          if !emb_table.contain(id):
            emb = init()
            emb_table.insert(id, emb)
        emb_list.append(emb_table.lookup(id)) # emb_table查不到的用默认值
    else:
      for id in ids:
        if !emb_table.contain(id):
          emb = init()
          emb_table.insert(id, emb)
        emb_list.append(emb_table.lookup(id))
  else:
    for id in ids:
        emb_list.append(emb_table.lookup(id))

  Args:
    keys: Keys to look up. Can be a tensor of list shape. Must match the
      table's key_dtype.
    name: A name for the operation (optional).

  Returns:
    A tensor containing the bool values in the same shape as `keys`.

  Raises:
    TypeError: when `keys` do not match the table data types.
  """
  orig_ids = ids
  if orig_ids.dtype != emb_table._key_dtype:
    orig_ids = math_ops.cast(orig_ids, dtype=emb_table._key_dtype)
  ids, idx, counter_sum = array_ops.unique_with_counts(orig_ids)
  if is_training:
    if count_table != None:
      # counter update
      counter = count_table.lookup(ids)
      if counter_sum.dtype != counter.dtype:
        counter_sum = math_ops.cast(counter_sum, dtype=counter.dtype)
      counter_new = counter + counter_sum
      with ops.control_dependencies(count_table.insert(ids, counter_new)):
        counter_pass_thr = math_ops.greater(counter_new, count_filter_thr)
        pass_ids = array_ops.boolean_mask(ids, counter_pass_thr)
        table_contain_mask = emb_table.contain(pass_ids)
        ids_not_contain = array_ops.boolean_mask(gen_array_ops.reshape(pass_ids, shape=(-1, 1)), gen_math_ops.logical_not(table_contain_mask))
        emb_val_shape = array_ops.shape(emb_table._default_value)[1:]
        table_insert_op = control_flow_ops.cond(array_ops.size(ids_not_contain) > 0,
          lambda: emb_table.insert(ids_not_contain,
                                   random_ops.random_normal([array_ops.size(ids_not_contain), 3], dtype=dtypes.float32)),
          lambda: [gen_control_flow_ops.no_op()] * emb_table._shard_num) # TODO. init
        with ops.control_dependencies(table_insert_op):
          emb = emb_table.lookup(orig_ids)
    else:
      table_contain_mask = emb_table.contain(ids)
      ids_not_contain = array_ops.boolean_mask(gen_array_ops.reshape(ids, shape=(-1, 1)), gen_math_ops.logical_not(table_contain_mask))
      emb_val_shape = array_ops.shape(emb_table._default_value)[1:]
      table_insert_op = control_flow_ops.cond(array_ops.size(ids_not_contain) > 0,
        lambda: emb_table.insert(ids_not_contain,
                               random_ops.random_normal([array_ops.size(ids_not_contain), 3], dtype=dtypes.float32)),
        lambda: [gen_control_flow_ops.no_op()] * emb_table._shard_num)
      with ops.control_dependencies(table_insert_op):
        emb = emb_table.lookup(orig_ids)
  else:
    emb = emb_table.lookup(orig_ids)

  ignore_weights = weights is None
  if not ignore_weights:
    if weights.dtype != emb.dtype:
      weights = math_ops.cast(weights, emb.dtype)
      # Reshape to allow broadcast
      ones = array_ops.fill(
              array_ops.expand_dims(array_ops.rank(emb) - 1, 0), 1)
      bcast_weights_shape = array_ops.concat(
              [array_ops.shape(weights), ones], 0)
      orig_weights_shape = weights.get_shape()
      weights = array_ops.reshape(weights, bcast_weights_shape)
      # Set weights shape after reshape
      if emb.get_shape().ndims is not None:
        weights.set_shape(
                orig_weights_shape.concatenate(
                    [1 for _ in range(emb.get_shape().ndims - 1)]))
      emb *= weights

  return math_ops.segment_sum(emb, segment_ids, name=name)
