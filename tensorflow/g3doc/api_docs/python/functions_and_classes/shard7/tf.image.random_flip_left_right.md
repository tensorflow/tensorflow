### `tf.image.random_flip_left_right(image, seed=None)` {#random_flip_left_right}

Randomly flip an image horizontally (left to right).

With a 1 in 2 chance, outputs the contents of `image` flipped along the
second dimension, which is `width`.  Otherwise output the image as-is.

##### Args:


*  <b>`image`</b>: A 3-D tensor of shape `[height, width, channels].`
*  <b>`seed`</b>: A Python integer. Used to create a random seed. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.

##### Returns:

  A 3-D tensor of the same type and shape as `image`.

##### Raises:


*  <b>`ValueError`</b>: if the shape of `image` not supported.

