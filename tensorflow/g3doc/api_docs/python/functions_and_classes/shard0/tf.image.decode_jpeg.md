### `tf.image.decode_jpeg(contents, channels=None, ratio=None, fancy_upscaling=None, try_recover_truncated=None, acceptable_fraction=None, name=None)` {#decode_jpeg}

Decode a JPEG-encoded image to a uint8 tensor.

The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

*   0: Use the number of channels in the JPEG-encoded image.
*   1: output a grayscale image.
*   3: output an RGB image.

If needed, the JPEG-encoded image is transformed to match the requested number
of color channels.

The attr `ratio` allows downscaling the image by an integer factor during
decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
downscaling the image later.

##### Args:


*  <b>`contents`</b>: A `Tensor` of type `string`. 0-D.  The JPEG-encoded image.
*  <b>`channels`</b>: An optional `int`. Defaults to `0`.
    Number of color channels for the decoded image.
*  <b>`ratio`</b>: An optional `int`. Defaults to `1`. Downscaling ratio.
*  <b>`fancy_upscaling`</b>: An optional `bool`. Defaults to `True`.
    If true use a slower but nicer upscaling of the
    chroma planes (yuv420/422 only).
*  <b>`try_recover_truncated`</b>: An optional `bool`. Defaults to `False`.
    If true try to recover an image from truncated input.
*  <b>`acceptable_fraction`</b>: An optional `float`. Defaults to `1`.
    The minimum required fraction of lines before a truncated
    input is accepted.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `uint8`. 3-D with shape `[height, width, channels]`..

