### `tf.contrib.framework.assert_scalar_int(tensor, name=None)` {#assert_scalar_int}

Assert `tensor` is 0-D, of type `tf.int32` or `tf.int64`.

##### Args:


*  <b>`tensor`</b>: `Tensor` to test.
*  <b>`name`</b>: Name of the op and of the new `Tensor` if one is created.

##### Returns:

  `tensor`, for chaining.

##### Raises:


*  <b>`ValueError`</b>: if `tensor` is not 0-D, of type `tf.int32` or `tf.int64`.

