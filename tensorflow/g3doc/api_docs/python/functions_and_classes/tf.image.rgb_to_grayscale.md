### `tf.image.rgb_to_grayscale(images, name=None)` {#rgb_to_grayscale}

Converts one or more images from RGB to Grayscale.

Outputs a tensor of the same `DType` and rank as `images`.  The size of the
last dimension of the output is 1, containing the Grayscale value of the
pixels.

##### Args:


*  <b>`images`</b>: The RGB tensor to convert. Last dimension must have size 3 and
    should contain RGB values.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The converted grayscale image(s).

