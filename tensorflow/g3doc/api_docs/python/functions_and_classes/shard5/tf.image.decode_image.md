### `tf.image.decode_image(contents, channels=None, name=None)` {#decode_image}

Convenience function for `decode_gif`, `decode_jpeg`, and `decode_png`.
Detects whether an image is a GIF, JPEG, or PNG, and performs the appropriate
operation to convert the input bytes `string` into a `Tensor` of type `uint8`.

Note: `decode_gif` returns a 4-D array `[num_frames, height, width, 3]`, as
opposed to `decode_jpeg` and `decode_png`, which return 3-D arrays
`[height, width, num_channels]`. Make sure to take this into account when
constructing your graph if you are intermixing GIF files with JPEG and/or PNG
files.

##### Args:


*  <b>`contents`</b>: 0-D `string`. The encoded image bytes.
*  <b>`channels`</b>: An optional `int`. Defaults to `0`. Number of color channels for
    the decoded image.
*  <b>`name`</b>: A name for the operation (optional)

##### Returns:

  `Tensor` with type `uint8` with shape `[height, width, num_channels]` for
    JPEG and PNG images and shape `[num_frames, height, width, 3]` for GIF
    images.

