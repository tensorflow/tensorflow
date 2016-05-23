### `tf.assert_type(tensor, tf_type)` {#assert_type}

Asserts that the given `Tensor` is of the specified type.

##### Args:


*  <b>`tensor`</b>: A tensorflow `Tensor`.
*  <b>`tf_type`</b>: A tensorflow type (dtypes.float32, tf.int64, dtypes.bool, etc).

##### Raises:


*  <b>`ValueError`</b>: If the tensors data type doesn't match tf_type.

