### `tf.extract_image_patches(images, padding, ksizes=None, strides=None, rates=None, name=None)` {#extract_image_patches}

Extract `patches` from `images` and puth them in the "depth" output dimension.

##### Args:


*  <b>`images`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    4-D Tensor with shape `[batch, in_rows, in_cols, depth]`.
*  <b>`padding`</b>: A `string` from: `"SAME", "VALID"`.
    The type of padding algorithm to use.

    We specify the size-related attributes as:

          ksizes = [1, ksize_rows, ksize_cols, 1]
          strides = [1, strides_rows, strides_cols, 1]
          rates = [1, rates_rows, rates_cols, 1]

*  <b>`ksizes`</b>: An optional list of `ints`. Defaults to `[]`.
    The size of the sliding window for each dimension of `images`.
*  <b>`strides`</b>: An optional list of `ints`. Defaults to `[]`.
    1-D of length 4. How far the centers of two consecutive patches are in
    the images. Must be: `[1, stride_rows, stride_cols, 1]`.
*  <b>`rates`</b>: An optional list of `ints`. Defaults to `[]`.
    1-D of length 4. Must be: `[1, rate_rows, rate_cols, 1]`. This is the
    input stride, specifying how far two consecutive patch samples are in the
    input. Equivalent to extracting patches with
    `patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1), followed by
    subsampling them spatially by a factor of `rates`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `images`.
  4-D Tensor with shape `[batch, out_rows, out_cols, ksize_rows *
  ksize_cols * depth]` containing image patches with size
  `ksize_rows x ksize_cols x depth` vectorized in the "depth" dimension.

