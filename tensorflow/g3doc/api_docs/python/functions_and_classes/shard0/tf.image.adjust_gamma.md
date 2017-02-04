### `tf.image.adjust_gamma(image, gamma=1, gain=1)` {#adjust_gamma}

Performs Gamma Correction on the input image.
  Also known as Power Law Transform. This function transforms the
  input image pixelwise according to the equation Out = In**gamma
  after scaling each pixel to the range 0 to 1.

##### Args:

  image : A Tensor.
  gamma : A scalar. Non negative real number.
  gain  : A scalar. The constant multiplier.

##### Returns:

  A Tensor. Gamma corrected output image.

##### Notes:

  For gamma greater than 1, the histogram will shift towards left and
  the output image will be darker than the input image.
  For gamma less than 1, the histogram will shift towards right and
  the output image will be brighter than the input image.

##### References:

  [1] http://en.wikipedia.org/wiki/Gamma_correction

