### `tf.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True)` {#embedding_lookup}

Looks up `ids` in a list of embedding tensors.

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

##### Args:


*  <b>`params`</b>: A list of tensors with the same type and which can be concatenated
    along dimension 0. Each `Tensor` must be appropriately sized for the given
    `partition_strategy`.
*  <b>`ids`</b>: A `Tensor` with type `int32` or `int64` containing the ids to be looked
    up in `params`.
*  <b>`partition_strategy`</b>: A string specifying the partitioning strategy, relevant
    if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
    is `"mod"`.
*  <b>`name`</b>: A name for the operation (optional).
*  <b>`validate_indices`</b>: Whether or not to validate gather indices.

##### Returns:

  A `Tensor` with the same type as the tensors in `params`.

##### Raises:


*  <b>`ValueError`</b>: If `params` is empty.

