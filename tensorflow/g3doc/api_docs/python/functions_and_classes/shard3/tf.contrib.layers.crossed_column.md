### `tf.contrib.layers.crossed_column(columns, hash_bucket_size, combiner=None, ckpt_to_load_from=None, tensor_name_in_ckpt=None, hash_key=None)` {#crossed_column}

Creates a _CrossedColumn for performing feature crosses.

##### Args:


*  <b>`columns`</b>: An iterable of _FeatureColumn. Items can be an instance of
    _SparseColumn, _CrossedColumn, or _BucketizedColumn.
*  <b>`hash_bucket_size`</b>: An int that is > 1. The number of buckets.
*  <b>`combiner`</b>: A combiner string, supports sum, mean, sqrtn.
*  <b>`ckpt_to_load_from`</b>: (Optional). String representing checkpoint name/pattern
    to restore the column weights. Required if `tensor_name_in_ckpt` is not
    None.
*  <b>`tensor_name_in_ckpt`</b>: (Optional). Name of the `Tensor` in the provided
    checkpoint from which to restore the column weights. Required if
    `ckpt_to_load_from` is not None.
*  <b>`hash_key`</b>: Specify the hash_key that will be used by the `FingerprintCat64`
    function to combine the crosses fingerprints on SparseFeatureCrossOp
    (optional).

##### Returns:

  A _CrossedColumn.

##### Raises:


*  <b>`TypeError`</b>: if any item in columns is not an instance of _SparseColumn,
    _CrossedColumn, or _BucketizedColumn, or
    hash_bucket_size is not an int.
*  <b>`ValueError`</b>: if hash_bucket_size is not > 1 or
    len(columns) is not > 1.

