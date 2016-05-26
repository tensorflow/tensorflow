### `tf.image.grayscale_to_rgb(images, name=None)` {#grayscale_to_rgb}

Converts one or more images from Grayscale to RGB.

Outputs a tensor of the same `DType` and rank as `images`.  The size of the
last dimension of the output is 3, containing the RGB value of the pixels.

##### Args:


*  <b>`images`</b>: The Grayscale tensor to convert. Last dimension must be size 1.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The converted grayscale image(s).

