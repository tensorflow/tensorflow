### `tf.contrib.layers.shared_embedding_columns(sparse_id_columns, dimension, combiner='mean', shared_embedding_name=None, initializer=None, ckpt_to_load_from=None, tensor_name_in_ckpt=None, max_norm=None)` {#shared_embedding_columns}

Creates a list of `_EmbeddingColumn` sharing the same embedding.

##### Args:


*  <b>`sparse_id_columns`</b>: An iterable of `_SparseColumn`, such as those created by
    `sparse_column_with_*` or crossed_column functions. Note that `combiner`
    defined in each sparse_id_column is ignored.
*  <b>`dimension`</b>: An integer specifying dimension of the embedding.
*  <b>`combiner`</b>: A string specifying how to reduce if there are multiple entries
    in a single row. Currently "mean", "sqrtn" and "sum" are supported, with
    "mean" the default. "sqrtn" often achieves good accuracy, in particular
    with bag-of-words columns. Each of this can be thought as example level
    normalizations on the column:
      * "sum": do not normalize
      * "mean": do l1 normalization
      * "sqrtn": do l2 normalization
    For more information: `tf.embedding_lookup_sparse`.
*  <b>`shared_embedding_name`</b>: (Optional). A string specifying the name of shared
    embedding weights. This will be needed if you want to reference the shared
    embedding separately from the generated `_EmbeddingColumn`.
*  <b>`initializer`</b>: A variable initializer function to be used in embedding
    variable initialization. If not specified, defaults to
    `tf.truncated_normal_initializer` with mean 0.0 and standard deviation
    1/sqrt(sparse_id_columns[0].length).
*  <b>`ckpt_to_load_from`</b>: (Optional). String representing checkpoint name/pattern
    to restore the column weights. Required if `tensor_name_in_ckpt` is not
    None.
*  <b>`tensor_name_in_ckpt`</b>: (Optional). Name of the `Tensor` in the provided
    checkpoint from which to restore the column weights. Required if
    `ckpt_to_load_from` is not None.
*  <b>`max_norm`</b>: (Optional). If not None, embedding values are l2-normalized to
    the value of max_norm.

##### Returns:

  A tuple of `_EmbeddingColumn` with shared embedding space.

##### Raises:


*  <b>`ValueError`</b>: if sparse_id_columns is empty, or its elements are not
    compatible with each other.
*  <b>`TypeError`</b>: if `sparse_id_columns` is not a sequence or is a string. If at
    least one element of `sparse_id_columns` is not a `SparseTensor`.

