### `tf.image.transpose_image(image)` {#transpose_image}

Transpose an image by swapping the first and second dimension.

See also `transpose()`.

##### Args:


*  <b>`image`</b>: 3-D tensor of shape `[height, width, channels]`

##### Returns:

  A 3-D tensor of shape `[width, height, channels]`

##### Raises:


*  <b>`ValueError`</b>: if the shape of `image` not supported.

