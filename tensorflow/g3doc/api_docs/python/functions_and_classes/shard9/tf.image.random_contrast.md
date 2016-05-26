### `tf.image.random_contrast(image, lower, upper, seed=None)` {#random_contrast}

Adjust the contrast of an image by a random factor.

Equivalent to `adjust_contrast()` but uses a `contrast_factor` randomly
picked in the interval `[lower, upper]`.

##### Args:


*  <b>`image`</b>: An image tensor with 3 or more dimensions.
*  <b>`lower`</b>: float.  Lower bound for the random contrast factor.
*  <b>`upper`</b>: float.  Upper bound for the random contrast factor.
*  <b>`seed`</b>: A Python integer. Used to create a random seed. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.

##### Returns:

  The contrast-adjusted tensor.

##### Raises:


*  <b>`ValueError`</b>: if `upper <= lower` or if `lower < 0`.

