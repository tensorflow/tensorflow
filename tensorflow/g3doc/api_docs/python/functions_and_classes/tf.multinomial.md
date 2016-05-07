### `tf.multinomial(logits, num_samples, seed=None, name=None)` {#multinomial}

Draws samples from a multinomial distribution.

Example:

  samples = tf.multinomial(tf.log([[0.5, 0.5]]), 10)
  # samples has shape [1, 10], where each value is either 0 or 1.

  samples = tf.multinomial([[1, -1, -1]], 10)
  # samples is equivalent to tf.zeros([1, 10], dtype=tf.int64).

##### Args:


*  <b>`logits`</b>: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice
    `[i, :]` represents the unnormalized log probabilities for all classes.
*  <b>`num_samples`</b>: 0-D.  Number of independent samples to draw for each row slice.
*  <b>`seed`</b>: A Python integer. Used to create a random seed for the distribution.
    See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  The drawn samples of shape `[batch_size, num_samples]`.

