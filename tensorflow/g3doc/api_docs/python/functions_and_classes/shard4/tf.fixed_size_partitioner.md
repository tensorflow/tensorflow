### `tf.fixed_size_partitioner(num_shards, axis=0)` {#fixed_size_partitioner}

Partitioner to specify a fixed number of shards along given axis.

##### Args:


*  <b>`num_shards`</b>: `int`, number of shards to partition variable.
*  <b>`axis`</b>: `int`, axis to partition on.

##### Returns:

  A partition function usable as the `partitioner` argument to
  `variable_scope`, `get_variable`, and `get_partitioned_variable_list`.

