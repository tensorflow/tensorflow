### `tf.contrib.framework.is_tensor(x)` {#is_tensor}

Check for tensor types.
Check whether an object is a tensor. Equivalent to
`isinstance(x, [tf.Tensor, tf.SparseTensor, tf.Variable])`.

##### Args:


*  <b>`x`</b>: An python object to check.

##### Returns:

  `True` if `x` is a tensor, `False` if not.

