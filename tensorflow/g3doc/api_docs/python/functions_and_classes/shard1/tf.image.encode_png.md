### `tf.image.encode_png(image, compression=None, name=None)` {#encode_png}

PNG-encode an image.

`image` is a 3-D uint8 or uint16 Tensor of shape `[height, width, channels]`
where `channels` is:

*   1: for grayscale.
*   2: for grayscale + alpha.
*   3: for RGB.
*   4: for RGBA.

The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
default or a value from 0 to 9.  9 is the highest compression level, generating
the smallest output, but is slower.

##### Args:


*  <b>`image`</b>: A `Tensor`. Must be one of the following types: `uint8`, `uint16`.
    3-D with shape `[height, width, channels]`.
*  <b>`compression`</b>: An optional `int`. Defaults to `-1`. Compression level.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`. 0-D. PNG-encoded image.

