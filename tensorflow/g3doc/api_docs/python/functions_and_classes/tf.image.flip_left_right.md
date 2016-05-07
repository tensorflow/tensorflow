### `tf.image.flip_left_right(image)` {#flip_left_right}

Flip an image horizontally (left to right).

Outputs the contents of `image` flipped along the second dimension, which is
`width`.

See also `reverse()`.

##### Args:


*  <b>`image`</b>: A 3-D tensor of shape `[height, width, channels].`

##### Returns:

  A 3-D tensor of the same type and shape as `image`.

##### Raises:


*  <b>`ValueError`</b>: if the shape of `image` not supported.

