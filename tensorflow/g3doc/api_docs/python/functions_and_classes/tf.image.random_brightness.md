### `tf.image.random_brightness(image, max_delta, seed=None)` {#random_brightness}

Adjust the brightness of images by a random factor.

Equivalent to `adjust_brightness()` using a `delta` randomly picked in the
interval `[-max_delta, max_delta)`.

##### Args:


*  <b>`image`</b>: An image.
*  <b>`max_delta`</b>: float, must be non-negative.
*  <b>`seed`</b>: A Python integer. Used to create a random seed. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.

##### Returns:

  The brightness-adjusted image.

##### Raises:


*  <b>`ValueError`</b>: if `max_delta` is negative.

