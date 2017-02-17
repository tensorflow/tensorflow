### `tf.image.random_hue(image, max_delta, seed=None)` {#random_hue}

Adjust the hue of an RGB image by a random factor.

Equivalent to `adjust_hue()` but uses a `delta` randomly
picked in the interval `[-max_delta, max_delta]`.

`max_delta` must be in the interval `[0, 0.5]`.

##### Args:


*  <b>`image`</b>: RGB image or images. Size of the last dimension must be 3.
*  <b>`max_delta`</b>: float.  Maximum value for the random delta.
*  <b>`seed`</b>: An operation-specific seed. It will be used in conjunction
    with the graph-level seed to determine the real seeds that will be
    used in this operation. Please see the documentation of
    set_random_seed for its interaction with the graph-level random seed.

##### Returns:

  3-D float tensor of shape `[height, width, channels]`.

##### Raises:


*  <b>`ValueError`</b>: if `max_delta` is invalid.

