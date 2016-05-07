### `tf.image.rgb_to_hsv(images, name=None)` {#rgb_to_hsv}

Converts one or more images from RGB to HSV.

Outputs a tensor of the same shape as the `images` tensor, containing the HSV
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.

`output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
`output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.

##### Args:


*  <b>`images`</b>: A `Tensor` of type `float32`.
    1-D or higher rank. RGB data to convert. Last dimension must be size 3.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`. `images` converted to HSV.

