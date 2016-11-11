### `tf.contrib.layers.one_hot_column(sparse_id_column)` {#one_hot_column}

Creates a _OneHotColumn.

##### Args:


*  <b>`sparse_id_column`</b>: A _SparseColumn which is created by
      `sparse_column_with_*`
      or crossed_column functions. Note that `combiner` defined in
      `sparse_id_column` is ignored.

##### Returns:

  An _OneHotColumn.

