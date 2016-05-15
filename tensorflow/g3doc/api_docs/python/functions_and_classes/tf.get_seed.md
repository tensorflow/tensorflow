### `tf.get_seed(op_seed)` {#get_seed}

Returns the local seeds an operation should use given an op-specific seed.

Given operation-specific seed, `op_seed`, this helper function returns two
seeds derived from graph-level and op-level seeds. Many random operations
internally use the two seeds to allow user to change the seed globally for a
graph, or for only specific operations.

For details on how the graph-level seed interacts with op seeds, see
[`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed).

##### Args:


*  <b>`op_seed`</b>: integer.

##### Returns:

  A tuple of two integers that should be used for the local seed of this
  operation.

