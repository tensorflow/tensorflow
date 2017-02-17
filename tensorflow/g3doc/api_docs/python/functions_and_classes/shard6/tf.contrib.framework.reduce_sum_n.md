### `tf.contrib.framework.reduce_sum_n(tensors, name=None)` {#reduce_sum_n}

Reduce tensors to a scalar sum.

This reduces each tensor in `tensors` to a scalar via `tf.reduce_sum`, then
adds them via `tf.add_n`.

##### Args:


*  <b>`tensors`</b>: List of tensors, all of the same numeric type.
*  <b>`name`</b>: Tensor name, and scope for all other ops.

##### Returns:

  Total loss tensor, or None if no losses have been configured.

##### Raises:


*  <b>`ValueError`</b>: if `losses` is missing or empty.

