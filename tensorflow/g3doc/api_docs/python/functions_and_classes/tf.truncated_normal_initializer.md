### `tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)` {#truncated_normal_initializer}

Returns an initializer that generates a truncated normal distribution.

These values are similar to values from a `random_normal_initializer`
except that values more than two standard deviations from the mean
are discarded and re-drawn. This is the recommended initializer for
neural network weights and filters.

##### Args:


*  <b>`mean`</b>: a python scalar or a scalar tensor. Mean of the random values
    to generate.
*  <b>`stddev`</b>: a python scalar or a scalar tensor. Standard deviation of the
    random values to generate.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`dtype`</b>: The data type. Only floating point types are supported.

##### Returns:

  An initializer that generates tensors with a truncated normal
  distribution.

##### Raises:


*  <b>`ValueError`</b>: if `dtype` is not a floating point type.

