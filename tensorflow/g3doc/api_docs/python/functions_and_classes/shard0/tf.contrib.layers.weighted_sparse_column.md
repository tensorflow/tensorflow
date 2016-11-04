### `tf.contrib.layers.weighted_sparse_column(sparse_id_column, weight_column_name, dtype=tf.float32)` {#weighted_sparse_column}

Creates a _SparseColumn by combining sparse_id_column with a weight column.

##### Args:


*  <b>`sparse_id_column`</b>: A `_SparseColumn` which is created by
    `sparse_column_with_*` functions.
*  <b>`weight_column_name`</b>: A string defining a sparse column name which represents
    weight or value of the corresponding sparse id feature.
*  <b>`dtype`</b>: Type of weights, such as `tf.float32`

##### Returns:

  A _WeightedSparseColumn composed of two sparse features: one represents id,
  the other represents weight (value) of the id feature in that example.

##### Raises:


*  <b>`ValueError`</b>: if dtype is not convertible to float.

##### An example usage:

  ```python
  words = sparse_column_with_hash_bucket("words", 1000)
  tfidf_weighted_words = weighted_sparse_column(words, "tfidf_score")
  ```

  This configuration assumes that input dictionary of model contains the
  following two items:
    * (key="words", value=word_tensor) where word_tensor is a SparseTensor.
    * (key="tfidf_score", value=tfidf_score_tensor) where tfidf_score_tensor
      is a SparseTensor.
   Following are assumed to be true:
     * word_tensor.indices = tfidf_score_tensor.indices
     * word_tensor.shape = tfidf_score_tensor.shape

