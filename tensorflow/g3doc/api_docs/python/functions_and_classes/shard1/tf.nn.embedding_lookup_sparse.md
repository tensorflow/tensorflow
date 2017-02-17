### `tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights, partition_strategy='mod', name=None, combiner=None, max_norm=None)` {#embedding_lookup_sparse}

Computes embeddings for the given ids and weights.

This op assumes that there is at least one id for each row in the dense tensor
represented by sp_ids (i.e. there are no rows with empty features), and that
all the indices of sp_ids are in canonical row-major order.

It also assumes that all id values lie in the range [0, p0), where p0
is the sum of the size of params along dimension 0.

##### Args:


*  <b>`params`</b>: A single tensor representing the complete embedding tensor,
    or a list of P tensors all of same shape except for the first dimension,
    representing sharded embedding tensors.  Alternatively, a
    `PartitionedVariable`, created by partitioning along dimension 0. Each
    element must be appropriately sized for the given `partition_strategy`.
*  <b>`sp_ids`</b>: N x M SparseTensor of int64 ids (typically from FeatureValueToId),
    where N is typically batch size and M is arbitrary.
*  <b>`sp_weights`</b>: either a SparseTensor of float / double weights, or None to
    indicate all weights should be taken to be 1. If specified, sp_weights
    must have exactly the same shape and indices as sp_ids.
*  <b>`partition_strategy`</b>: A string specifying the partitioning strategy, relevant
    if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
    is `"mod"`. See `tf.nn.embedding_lookup` for more details.
*  <b>`name`</b>: Optional name for the op.
*  <b>`combiner`</b>: A string specifying the reduction op. Currently "mean", "sqrtn"
    and "sum" are supported.
    "sum" computes the weighted sum of the embedding results for each row.
    "mean" is the weighted sum divided by the total weight.
    "sqrtn" is the weighted sum divided by the square root of the sum of the
    squares of the weights.
*  <b>`max_norm`</b>: If not None, each embedding is normalized to have l2 norm equal
    to max_norm before combining.

##### Returns:

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

  with `combiner`="mean", then the output will be a 3x20 matrix where

    output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
    output[1, :] = params[0, :] * 1.0
    output[2, :] = params[1, :] * 3.0

##### Raises:


*  <b>`TypeError`</b>: If sp_ids is not a SparseTensor, or if sp_weights is neither
    None nor SparseTensor.
*  <b>`ValueError`</b>: If combiner is not one of {"mean", "sqrtn", "sum"}.

