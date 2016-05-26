### `tf.image.hsv_to_rgb(images, name=None)` {#hsv_to_rgb}

Convert one or more images from HSV to RGB.

Outputs a tensor of the same shape as the `images` tensor, containing the RGB
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.

See `rgb_to_hsv` for a description of the HSV encoding.

##### Args:


*  <b>`images`</b>: A `Tensor` of type `float32`.
    1-D or higher rank. HSV data to convert. Last dimension must be size 3.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`. `images` converted to RGB.

