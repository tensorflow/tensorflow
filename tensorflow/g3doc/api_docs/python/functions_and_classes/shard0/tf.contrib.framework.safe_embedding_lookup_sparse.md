### `tf.contrib.framework.safe_embedding_lookup_sparse(embedding_weights, sparse_ids, sparse_weights=None, combiner='mean', default_id=None, name=None, partition_strategy='div')` {#safe_embedding_lookup_sparse}

Lookup embedding results, accounting for invalid IDs and empty features.

The partitioned embedding in `embedding_weights` must all be the same shape
except for the first dimension. The first dimension is allowed to vary as the
vocabulary size is not necessarily a multiple of `P`.

Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
with non-positive weight. For an entry with no features, the embedding vector
for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

##### Args:


*  <b>`embedding_weights`</b>: A list of `P` float tensors or values representing
      partitioned embedding tensors.
*  <b>`sparse_ids`</b>: `SparseTensor` of shape `[batch_size, ?]` containing the ids.
*  <b>`sparse_weights`</b>: `SparseTensor` of same shape as `sparse_ids`, containing
      float weights corresponding to `sparse_ids`, or `None` if all weights
      are be assumed to be 1.0.
*  <b>`combiner`</b>: A string specifying how to combine embedding results for each
      entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean"
      the default.
*  <b>`default_id`</b>: The id to use for an entry with no features.
*  <b>`name`</b>: A name for this operation (optional).
*  <b>`partition_strategy`</b>: A string specifying the partitioning strategy.
      Currently `"div"` and `"mod"` are supported. Default is `"div"`.


##### Returns:

  Dense tensor of shape `[batch_size, embed_dim]`.

##### Raises:


*  <b>`ValueError`</b>: if `embedding_weights` is empty.

