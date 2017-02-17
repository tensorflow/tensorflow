### `tf.image.resize_nearest_neighbor(images, size, align_corners=None, name=None)` {#resize_nearest_neighbor}

Resize `images` to `size` using nearest neighbor interpolation.

##### Args:


*  <b>`images`</b>: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `half`, `float32`, `float64`.
    4-D with shape `[batch, height, width, channels]`.
*  <b>`size`</b>: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
    new size for the images.
*  <b>`align_corners`</b>: An optional `bool`. Defaults to `False`.
    If true, rescale input by (new_height - 1) / (height - 1), which
    exactly aligns the 4 corners of images and resized images. If false, rescale
    by new_height / height. Treat similarly the width dimension.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `images`. 4-D with shape
  `[batch, new_height, new_width, channels]`.

