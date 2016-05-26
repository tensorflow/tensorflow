### `tf.space_to_batch(input, paddings, block_size, name=None)` {#space_to_batch}

SpaceToBatch for 4-D tensors of type T.

Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
More specifically, this op outputs a copy of the input tensor where values from
the `height` and `width` dimensions are moved to the `batch` dimension. After
the zero-padding, both `height` and `width` of the input must be divisible by the
block size.

##### Args:


*  <b>`input`</b>: A `Tensor`. 4-D with shape `[batch, height, width, depth]`.
*  <b>`paddings`</b>: A `Tensor` of type `int32`.
    2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
      the padding of the input with zeros across the spatial dimensions as follows:

          paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]

      The effective spatial dimensions of the zero-padded input tensor will be:

          height_pad = pad_top + height + pad_bottom
          width_pad = pad_left + width + pad_right

    The attr `block_size` must be greater than one. It indicates the block size.

      * Non-overlapping blocks of size `block_size x block size` in the height and
        width dimensions are rearranged into the batch dimension at each location.
      * The batch of the output tensor is `batch * block_size * block_size`.
      * Both height_pad and width_pad must be divisible by block_size.

    The shape of the output will be:

        [batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
         depth]

*  <b>`block_size`</b>: An `int`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.

