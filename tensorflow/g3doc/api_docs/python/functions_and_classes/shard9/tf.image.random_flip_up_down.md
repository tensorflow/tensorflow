### `tf.image.random_flip_up_down(image, seed=None)` {#random_flip_up_down}

Randomly flips an image vertically (upside down).

With a 1 in 2 chance, outputs the contents of `image` flipped along the first
dimension, which is `height`.  Otherwise output the image as-is.

##### Args:


*  <b>`image`</b>: A 3-D tensor of shape `[height, width, channels].`
*  <b>`seed`</b>: A Python integer. Used to create a random seed. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.

##### Returns:

  A 3-D tensor of the same type and shape as `image`.

##### Raises:


*  <b>`ValueError`</b>: if the shape of `image` not supported.

