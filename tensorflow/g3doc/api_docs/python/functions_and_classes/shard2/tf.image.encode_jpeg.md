### `tf.image.encode_jpeg(image, format=None, quality=None, progressive=None, optimize_size=None, chroma_downsampling=None, density_unit=None, x_density=None, y_density=None, xmp_metadata=None, name=None)` {#encode_jpeg}

JPEG-encode an image.

`image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.

The attr `format` can be used to override the color format of the encoded
output.  Values can be:

*   `''`: Use a default format based on the number of channels in the image.
*   `grayscale`: Output a grayscale JPEG image.  The `channels` dimension
    of `image` must be 1.
*   `rgb`: Output an RGB JPEG image. The `channels` dimension
    of `image` must be 3.

If `format` is not specified or is the empty string, a default format is picked
in function of the number of channels in `image`:

*   1: Output a grayscale image.
*   3: Output an RGB image.

##### Args:


*  <b>`image`</b>: A `Tensor` of type `uint8`.
    3-D with shape `[height, width, channels]`.
*  <b>`format`</b>: An optional `string` from: `"", "grayscale", "rgb"`. Defaults to `""`.
    Per pixel image format.
*  <b>`quality`</b>: An optional `int`. Defaults to `95`.
    Quality of the compression from 0 to 100 (higher is better and slower).
*  <b>`progressive`</b>: An optional `bool`. Defaults to `False`.
    If True, create a JPEG that loads progressively (coarse to fine).
*  <b>`optimize_size`</b>: An optional `bool`. Defaults to `False`.
    If True, spend CPU/RAM to reduce size with no quality change.
*  <b>`chroma_downsampling`</b>: An optional `bool`. Defaults to `True`.
    See http://en.wikipedia.org/wiki/Chroma_subsampling.
*  <b>`density_unit`</b>: An optional `string` from: `"in", "cm"`. Defaults to `"in"`.
    Unit used to specify `x_density` and `y_density`:
    pixels per inch (`'in'`) or centimeter (`'cm'`).
*  <b>`x_density`</b>: An optional `int`. Defaults to `300`.
    Horizontal pixels per density unit.
*  <b>`y_density`</b>: An optional `int`. Defaults to `300`.
    Vertical pixels per density unit.
*  <b>`xmp_metadata`</b>: An optional `string`. Defaults to `""`.
    If not empty, embed this XMP metadata in the image header.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`. 0-D. JPEG-encoded image.

