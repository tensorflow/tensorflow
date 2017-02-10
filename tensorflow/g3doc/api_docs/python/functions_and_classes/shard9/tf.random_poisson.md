### `tf.random_poisson(lam, shape, dtype=tf.float32, seed=None, name=None)` {#random_poisson}

Draws `shape` samples from each of the given Poisson distribution(s).

`lam` is the rate parameter describing the distribution(s).

Example:

  samples = tf.random_poisson([0.5, 1.5], [10])
  # samples has shape [10, 2], where each slice [:, 0] and [:, 1] represents
  # the samples drawn from each distribution

  samples = tf.random_poisson([12.2, 3.3], [7, 5])
  # samples has shape [7, 5, 2], where each slice [:, :, 0] and [:, :, 1]
  # represents the 7x5 samples drawn from each of the two distributions

##### Args:


*  <b>`lam`</b>: A Tensor or Python value or N-D array of type `dtype`.
    `lam` provides the rate parameter(s) describing the poisson
    distribution(s) to sample.
*  <b>`shape`</b>: A 1-D integer Tensor or Python array. The shape of the output samples
    to be drawn per "rate"-parameterized distribution.
*  <b>`dtype`</b>: The type of `lam` and the output: `float16`, `float32`, or
    `float64`.
*  <b>`seed`</b>: A Python integer. Used to create a random seed for the distributions.
    See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`name`</b>: Optional name for the operation.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of shape `tf.concat(shape, tf.shape(lam))` with
    values of type `dtype`.

