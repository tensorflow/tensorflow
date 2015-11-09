"""Operations for embeddings."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import types
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops


def embedding_lookup(params, ids, name=None):
  """Looks up `ids` in a list of embedding tensors.

  This function is used to perform parallel lookups on the list of
  tensors in `params`.  It is a generalization of
  [`tf.gather()`](../../api_docs/python/array_ops.md#gather), where `params` is
  interpreted as a partition of a larger embedding tensor.

  If `len(params) > 1`, each element `id` of `ids` is partitioned between
  the elements of `params` by computing `p = id % len(params)`, and is
  then used to look up the slice `params[p][id // len(params), ...]`.

  The results of the lookup are then concatenated into a dense
  tensor. The returned tensor has shape `shape(ids) + shape(params)[1:]`.

  Args:
    params: A list of tensors with the same shape and type.
    ids: A `Tensor` with type `int32` containing the ids to be looked
      up in `params`.
    name: A name for the operation (optional).

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
        return array_ops.gather(params[0], ids, name=name)
    else:
      ids = ops.convert_to_tensor(ids, name="ids")
      flat_ids = array_ops.reshape(ids, [-1])
      original_indices = math_ops.range(0, array_ops.size(flat_ids))
      # Compute flat_ids % partitions for each id
      ids_mod_p = flat_ids % np
      if ids_mod_p.dtype != types.int32:
        ids_mod_p = math_ops.cast(ids_mod_p, types.int32)
      # Partition single list of ids based on ids % np into np separate lists
      plist = data_flow_ops.dynamic_partition(flat_ids, ids_mod_p, np)
      # Similarly, partition the original indices.
      pindices = data_flow_ops.dynamic_partition(original_indices, ids_mod_p,
                                                 np)
      # Do np separate lookups, finding embeddings for plist[p] in params[p]
      partitioned_result = []
      for p in range(np):
        # TODO(agarwal): handle device allocations here and later in the
        # colocate code.
        gather_ids = plist[p] / np
        with ops.device(params[p].device):
          partitioned_result.append(array_ops.gather(params[p], gather_ids))
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
      representing sharded embedding tensors. In the latter case, the ids are
      partitioned by id % P, and we do separate lookups in params[p] for
      0 <= p < P, and then stitch the results back together into a single
      result tensor. The first dimension is allowed to vary as the vocab
      size is not necessarily a multiple of P.
    sp_ids: N x M SparseTensor of int64 ids (typically from FeatureValueToId),
      where N is typically batch size and M is arbitrary.
    sp_weights: either a SparseTensor of float / double weights, or None to
      indicate all weights should be taken to be 1. If specified, sp_weights
      must have exactly the same shape and indices as sp_ids.
    name: Optional name for the op.
    combiner: A string specifying the reduction op. Currently "mean" and "sum"
      are supported.
      "sum" computes the weighted sum of the embedding results for each row.
      "mean" is the weighted sum divided by the total weight.

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
    ValueError: If combiner is not one of {"mean", "sum"}.
  """
  if combiner not in ("mean", "sum"):
    raise ValueError("combiner must be one of 'mean' or 'sum'")
  if not isinstance(params, list):
    params = [params]
  if not isinstance(sp_ids, ops.SparseTensor):
    raise TypeError("sp_ids must be SparseTensor")
  ignore_weights = sp_weights is None
  if not ignore_weights and not isinstance(sp_weights, ops.SparseTensor):
    raise TypeError("sp_weights must be either None or SparseTensor")

  with ops.op_scope(params + [sp_ids], name, "embedding_lookup_sparse") as name:
    segment_ids = sp_ids.indices[:, 0]
    if segment_ids.dtype != types.int32:
      segment_ids = math_ops.cast(segment_ids, types.int32)

    ids = sp_ids.values
    if ignore_weights:
      ids, idx = array_ops.unique(ids)
    else:
      idx = None

    embeddings = embedding_lookup(params, ids)
    if not ignore_weights:
      weights = sp_weights.values
      if weights.dtype != embeddings.dtype:
        weights = math_ops.cast(weights, embeddings.dtype)

      # Reshape weights to allow broadcast
      ones = array_ops.fill(
          array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0), 1)
      bcast_weights_shape = array_ops.concat(0, [
          array_ops.shape(weights), ones])
      weights = array_ops.reshape(weights, bcast_weights_shape)
      embeddings *= weights

      if combiner == "sum":
        embeddings = math_ops.segment_sum(embeddings, segment_ids, name=name)
      elif combiner == "mean":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weight_sum = math_ops.segment_sum(weights, segment_ids)
        embeddings = math_ops.div(embeddings, weight_sum, name=name)
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
      else:
        assert False, "Unrecognized combiner"

    return embeddings
