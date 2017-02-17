### `tf.image.random_saturation(image, lower, upper, seed=None)` {#random_saturation}

Adjust the saturation of an RGB image by a random factor.

Equivalent to `adjust_saturation()` but uses a `saturation_factor` randomly
picked in the interval `[lower, upper]`.

##### Args:


*  <b>`image`</b>: RGB image or images. Size of the last dimension must be 3.
*  <b>`lower`</b>: float.  Lower bound for the random saturation factor.
*  <b>`upper`</b>: float.  Upper bound for the random saturation factor.
*  <b>`seed`</b>: An operation-specific seed. It will be used in conjunction
    with the graph-level seed to determine the real seeds that will be
    used in this operation. Please see the documentation of
    set_random_seed for its interaction with the graph-level random seed.

##### Returns:

  Adjusted image(s), same shape and DType as `image`.

##### Raises:


*  <b>`ValueError`</b>: if `upper <= lower` or if `lower < 0`.

