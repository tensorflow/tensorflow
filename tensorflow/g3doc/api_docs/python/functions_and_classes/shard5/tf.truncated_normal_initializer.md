Initializer that generates a truncated normal distribution.

These values are similar to values from a `random_normal_initializer`
except that values more than two standard deviations from the mean
are discarded and re-drawn. This is the recommended initializer for
neural network weights and filters.

Args:
  mean: a python scalar or a scalar tensor. Mean of the random values
    to generate.
  stddev: a python scalar or a scalar tensor. Standard deviation of the
    random values to generate.
  seed: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
  dtype: The data type. Only floating point types are supported.
- - -

#### `tf.truncated_normal_initializer.__call__(shape, dtype=None, partition_info=None)` {#truncated_normal_initializer.__call__}




- - -

#### `tf.truncated_normal_initializer.__init__(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)` {#truncated_normal_initializer.__init__}




