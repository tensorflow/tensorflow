Initializer that generates tensors with a uniform distribution.

Args:
  minval: A python scalar or a scalar tensor. Lower bound of the range
    of random values to generate.
  maxval: A python scalar or a scalar tensor. Upper bound of the range
    of random values to generate.  Defaults to 1 for float types.
  seed: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
  dtype: The data type.
- - -

#### `tf.random_uniform_initializer.__call__(shape, dtype=None, partition_info=None)` {#random_uniform_initializer.__call__}




- - -

#### `tf.random_uniform_initializer.__init__(minval=0, maxval=None, seed=None, dtype=tf.float32)` {#random_uniform_initializer.__init__}




