### `tf.image.adjust_contrast(images, contrast_factor)` {#adjust_contrast}

Adjust contrast of RGB or grayscale images.

This is a convenience method that converts an RGB image to float
representation, adjusts its contrast, and then converts it back to the
original data type. If several adjustments are chained it is advisable to
minimize the number of redundant conversions.

`images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
interpreted as `[height, width, channels]`.  The other dimensions only
represent a collection of images, such as `[batch, height, width, channels].`

Contrast is adjusted independently for each channel of each image.

For each channel, this Op computes the mean of the image pixels in the
channel and then adjusts each component `x` of each pixel to
`(x - mean) * contrast_factor + mean`.

##### Args:


*  <b>`images`</b>: Images to adjust.  At least 3-D.
*  <b>`contrast_factor`</b>: A float multiplier for adjusting contrast.

##### Returns:

  The contrast-adjusted image or images.

