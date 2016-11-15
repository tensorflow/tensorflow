### `tf.contrib.layers.bucketized_column(source_column, boundaries)` {#bucketized_column}

Creates a _BucketizedColumn for discretizing dense input.

##### Args:


*  <b>`source_column`</b>: A _RealValuedColumn defining dense column.
*  <b>`boundaries`</b>: A list of floats specifying the boundaries. It has to be sorted.

##### Returns:

  A _BucketizedColumn.

##### Raises:


*  <b>`ValueError`</b>: if 'boundaries' is empty or not sorted.

