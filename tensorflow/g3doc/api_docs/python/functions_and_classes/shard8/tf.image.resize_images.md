### `tf.image.resize_images(images, new_height, new_width, method=0, align_corners=False)` {#resize_images}

Resize `images` to `new_width`, `new_height` using the specified `method`.

Resized images will be distorted if their original aspect ratio is not
the same as `new_width`, `new_height`.  To avoid distortions see
[`resize_image_with_crop_or_pad`](#resize_image_with_crop_or_pad).

`method` can be one of:

*   <b>`ResizeMethod.BILINEAR`</b>: [Bilinear interpolation.]
    (https://en.wikipedia.org/wiki/Bilinear_interpolation)
*   <b>`ResizeMethod.NEAREST_NEIGHBOR`</b>: [Nearest neighbor interpolation.]
    (https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
*   <b>`ResizeMethod.BICUBIC`</b>: [Bicubic interpolation.]
    (https://en.wikipedia.org/wiki/Bicubic_interpolation)
*   <b>`ResizeMethod.AREA`</b>: Area interpolation.

##### Args:


*  <b>`images`</b>: 4-D Tensor of shape `[batch, height, width, channels]` or
          3-D Tensor of shape `[height, width, channels]`.
*  <b>`new_height`</b>: integer.
*  <b>`new_width`</b>: integer.
*  <b>`method`</b>: ResizeMethod.  Defaults to `ResizeMethod.BILINEAR`.
*  <b>`align_corners`</b>: bool. If true, exactly align all 4 corners of the input and
                 output. Defaults to `false`.

##### Raises:


*  <b>`ValueError`</b>: if the shape of `images` is incompatible with the
    shape arguments to this function
*  <b>`ValueError`</b>: if an unsupported resize method is specified.

##### Returns:

  If `images` was 4-D, a 4-D float Tensor of shape
  `[batch, new_height, new_width, channels]`.
  If `images` was 3-D, a 3-D float Tensor of shape
  `[new_height, new_width, channels]`.

