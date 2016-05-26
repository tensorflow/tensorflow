### `tf.image.flip_up_down(image)` {#flip_up_down}

Flip an image horizontally (upside down).

Outputs the contents of `image` flipped along the first dimension, which is
`height`.

See also `reverse()`.

##### Args:


*  <b>`image`</b>: A 3-D tensor of shape `[height, width, channels].`

##### Returns:

  A 3-D tensor of the same type and shape as `image`.

##### Raises:


*  <b>`ValueError`</b>: if the shape of `image` not supported.

