### `tf.image.central_crop(image, central_fraction)` {#central_crop}

Crop the central region of the image.

Remove the outer parts of an image but retain the central region of the image
along each dimension. If we specify central_fraction = 0.5, this function
returns the region marked with "X" in the below diagram.

     --------
    |        |
    |  XXXX  |
    |  XXXX  |
    |        |   where "X" is the central 50% of the image.
     --------

##### Args:


*  <b>`image`</b>: 3-D float Tensor of shape [height, width, depth]
*  <b>`central_fraction`</b>: float (0, 1], fraction of size to crop

##### Raises:


*  <b>`ValueError`</b>: if central_crop_fraction is not within (0, 1].

##### Returns:

  3-D float Tensor

