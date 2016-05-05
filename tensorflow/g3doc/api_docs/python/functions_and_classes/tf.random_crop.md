### `tf.random_crop(value, size, seed=None, name=None)` {#random_crop}

Randomly crops a tensor to a given size.

Slices a shape `size` portion out of `value` at a uniformly chosen offset.
Requires `value.shape >= size`.

If a dimension should not be cropped, pass the full size of that dimension.
For example, RGB images can be cropped with
`size = [crop_height, crop_width, 3]`.

##### Args:


*  <b>`value`</b>: Input tensor to crop.
*  <b>`size`</b>: 1-D tensor with size the rank of `value`.
*  <b>`seed`</b>: Python integer. Used to create a random seed. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  A cropped tensor of the same rank as `value` and shape `size`.

