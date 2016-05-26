### `tf.image.adjust_hue(image, delta, name=None)` {#adjust_hue}

Adjust hue of an RGB image.

This is a convenience method that converts an RGB image to float
representation, converts it to HSV, add an offset to the hue channel, converts
back to RGB and then back to the original data type. If several adjustments
are chained it is advisable to minimize the number of redundant conversions.

`image` is an RGB image.  The image hue is adjusted by converting the
image to HSV and rotating the hue channel (H) by
`delta`.  The image is then converted back to RGB.

`delta` must be in the interval `[-1, 1]`.

##### Args:


*  <b>`image`</b>: RGB image or images. Size of the last dimension must be 3.
*  <b>`delta`</b>: float.  How much to add to the hue channel.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  Adjusted image(s), same shape and DType as `image`.

