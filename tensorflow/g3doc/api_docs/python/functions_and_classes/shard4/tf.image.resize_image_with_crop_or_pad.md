### `tf.image.resize_image_with_crop_or_pad(image, target_height, target_width)` {#resize_image_with_crop_or_pad}

Crops and/or pads an image to a target width and height.

Resizes an image to a target width and height by either centrally
cropping the image or padding it evenly with zeros.

If `width` or `height` is greater than the specified `target_width` or
`target_height` respectively, this op centrally crops along that dimension.
If `width` or `height` is smaller than the specified `target_width` or
`target_height` respectively, this op centrally pads with 0 along that
dimension.

##### Args:


*  <b>`image`</b>: 3-D tensor of shape [height, width, channels]
*  <b>`target_height`</b>: Target height.
*  <b>`target_width`</b>: Target width.

##### Raises:


*  <b>`ValueError`</b>: if `target_height` or `target_width` are zero or negative.

##### Returns:

  Cropped and/or padded image of shape
  `[target_height, target_width, channels]`

