### `tf.image.adjust_saturation(image, saturation_factor, name=None)` {#adjust_saturation}

Adjust saturation of an RGB image.

This is a convenience method that converts an RGB image to float
representation, converts it to HSV, add an offset to the saturation channel,
converts back to RGB and then back to the original data type. If several
adjustments are chained it is advisable to minimize the number of redundant
conversions.

`image` is an RGB image.  The image saturation is adjusted by converting the
image to HSV and multiplying the saturation (S) channel by
`saturation_factor` and clipping. The image is then converted back to RGB.

##### Args:


*  <b>`image`</b>: RGB image or images. Size of the last dimension must be 3.
*  <b>`saturation_factor`</b>: float. Factor to multiply the saturation by.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  Adjusted image(s), same shape and DType as `image`.

