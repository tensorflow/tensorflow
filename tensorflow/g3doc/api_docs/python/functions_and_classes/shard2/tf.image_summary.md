### `tf.image_summary(tag, tensor, max_images=3, collections=None, name=None)` {#image_summary}

Outputs a `Summary` protocol buffer with images.

The summary has up to `max_images` summary values containing images. The
images are built from `tensor` which must be 4-D with shape `[batch_size,
height, width, channels]` and where `channels` can be:

*  1: `tensor` is interpreted as Grayscale.
*  3: `tensor` is interpreted as RGB.
*  4: `tensor` is interpreted as RGBA.

The images have the same number of channels as the input tensor. For float
input, the values are normalized one image at a time to fit in the range
`[0, 255]`.  `uint8` values are unchanged.  The op uses two different
normalization algorithms:

*  If the input values are all positive, they are rescaled so the largest one
   is 255.

*  If any input value is negative, the values are shifted so input value 0.0
   is at 127.  They are then rescaled so that either the smallest value is 0,
   or the largest one is 255.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

*  If `max_images` is 1, the summary value tag is '*tag*/image'.
*  If `max_images` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.

##### Args:


*  <b>`tag`</b>: A scalar `Tensor` of type `string`. Used to build the `tag`
    of the summary values.
*  <b>`tensor`</b>: A 4-D `uint8` or `float32` `Tensor` of shape `[batch_size, height,
    width, channels]` where `channels` is 1, 3, or 4.
*  <b>`max_images`</b>: Max number of batch elements to generate images for.
*  <b>`collections`</b>: Optional list of ops.GraphKeys.  The collections to add the
    summary to.  Defaults to [ops.GraphKeys.SUMMARIES]
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.

