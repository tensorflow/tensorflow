### `tf.random_uniform_initializer(minval=0, maxval=None, seed=None, dtype=tf.float32)` {#random_uniform_initializer}

Returns an initializer that generates tensors with a uniform distribution.

##### Args:


*  <b>`minval`</b>: A python scalar or a scalar tensor. Lower bound of the range
    of random values to generate.
*  <b>`maxval`</b>: A python scalar or a scalar tensor. Upper bound of the range
    of random values to generate.  Defaults to 1 for float types.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`dtype`</b>: The data type.

##### Returns:

  An initializer that generates tensors with a uniform distribution.

