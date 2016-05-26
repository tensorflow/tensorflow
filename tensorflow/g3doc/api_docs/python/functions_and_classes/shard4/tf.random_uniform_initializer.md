### `tf.random_uniform_initializer(minval=0.0, maxval=1.0, seed=None, dtype=tf.float32)` {#random_uniform_initializer}

Returns an initializer that generates tensors with a uniform distribution.

##### Args:


*  <b>`minval`</b>: a python scalar or a scalar tensor. lower bound of the range
    of random values to generate.
*  <b>`maxval`</b>: a python scalar or a scalar tensor. upper bound of the range
    of random values to generate.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`dtype`</b>: The data type. Only floating point types are supported.

##### Returns:

  An initializer that generates tensors with a uniform distribution.

##### Raises:


*  <b>`ValueError`</b>: if `dtype` is not a floating point type.

