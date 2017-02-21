### `tf.contrib.layers.weighted_sparse_column(sparse_id_column, weight_column_name, dtype=tf.float32)` {#weighted_sparse_column}

Creates a _SparseColumn by combining sparse_id_column with a weight column.

Example:

  ```python
  sparse_feature = sparse_column_with_hash_bucket(column_name="sparse_col",
                                                  hash_bucket_size=1000)
  weighted_feature = weighted_sparse_column(sparse_id_column=sparse_feature,
                                            weight_column_name="weights_col")
  ```

  This configuration assumes that input dictionary of model contains the
  following two items:
    * (key="sparse_col", value=sparse_tensor) where sparse_tensor is
      a SparseTensor.
    * (key="weights_col", value=weights_tensor) where weights_tensor
      is a SparseTensor.
   Following are assumed to be true:
     * sparse_tensor.indices = weights_tensor.indices
     * sparse_tensor.dense_shape = weights_tensor.dense_shape

##### Args:


*  <b>`sparse_id_column`</b>: A `_SparseColumn` which is created by
    `sparse_column_with_*` functions.
*  <b>`weight_column_name`</b>: A string defining a sparse column name which represents
    weight or value of the corresponding sparse id feature.
*  <b>`dtype`</b>: Type of weights, such as `tf.float32`. Only floating and integer
    weights are supported.

##### Returns:

  A _WeightedSparseColumn composed of two sparse features: one represents id,
  the other represents weight (value) of the id feature in that example.

##### Raises:


*  <b>`ValueError`</b>: if dtype is not convertible to float.

