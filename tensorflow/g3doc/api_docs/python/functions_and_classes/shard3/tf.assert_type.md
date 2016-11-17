### `tf.assert_type(tensor, tf_type, message=None, name=None)` {#assert_type}

Statically asserts that the given `Output` is of the specified type.

##### Args:


*  <b>`tensor`</b>: A tensorflow `Output`.
*  <b>`tf_type`</b>: A tensorflow type (`dtypes.float32`, `tf.int64`, `dtypes.bool`,
    etc).
*  <b>`message`</b>: A string to prefix to the default message.
*  <b>`name`</b>: A name to give this `Op`.  Defaults to "assert_type"

##### Raises:


*  <b>`TypeError`</b>: If the tensors data type doesn't match `tf_type`.

##### Returns:

  A `no_op` that does nothing.  Type can be determined statically.

