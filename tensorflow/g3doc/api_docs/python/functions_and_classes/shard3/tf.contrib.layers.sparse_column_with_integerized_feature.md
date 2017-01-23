### `tf.contrib.layers.sparse_column_with_integerized_feature(column_name, bucket_size, combiner=None, dtype=tf.int64)` {#sparse_column_with_integerized_feature}

Creates an integerized _SparseColumn.

Use this when your features are already pre-integerized into int64 IDs.
output_id = input_feature

##### Args:


*  <b>`column_name`</b>: A string defining sparse column name.
*  <b>`bucket_size`</b>: An int that is > 1. The number of buckets. It should be bigger
    than maximum feature. In other words features in this column should be an
    int64 in range [0, bucket_size)
*  <b>`combiner`</b>: A string specifying how to reduce if the sparse column is
    multivalent. Currently "mean", "sqrtn" and "sum" are supported, with
    "sum" the default:
      * "sum": do not normalize features in the column
      * "mean": do l1 normalization on features in the column
      * "sqrtn": do l2 normalization on features in the column
    For more information: `tf.embedding_lookup_sparse`.
*  <b>`dtype`</b>: Type of features. It should be an integer type. Default value is
    dtypes.int64.

##### Returns:

  An integerized _SparseColumn definition.

##### Raises:


*  <b>`ValueError`</b>: bucket_size is not greater than 1.
*  <b>`ValueError`</b>: dtype is not integer.

