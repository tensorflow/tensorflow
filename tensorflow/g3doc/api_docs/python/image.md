<!-- This file is machine generated: DO NOT EDIT! -->

# Images

Note: Functions taking `Tensor` arguments can also take anything accepted by
[`tf.convert_to_tensor`](../../api_docs/python/framework.md#convert_to_tensor).

[TOC]

## Encoding and Decoding

TensorFlow provides Ops to decode and encode JPEG and PNG formats.  Encoded
images are represented by scalar string Tensors, decoded images by 3-D uint8
tensors of shape `[height, width, channels]`.

The encode and decode Ops apply to one image at a time.  Their input and output
are all of variable size.  If you need fixed size images, pass the output of
the decode Ops to one of the cropping and resizing Ops.

Note: The PNG encode and decode Ops support RGBA, but the conversions Ops
presently only support RGB, HSV, and GrayScale. Presently, the alpha channel has
to be stripped from the image and re-attached using slicing ops.

- - -

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


- - -

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



- - -

### `tf.image.decode_png(contents, channels=None, name=None)` {#decode_png}

Decode a PNG-encoded image to a uint8 tensor.

The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

*   0: Use the number of channels in the PNG-encoded image.
*   1: output a grayscale image.
*   3: output an RGB image.
*   4: output an RGBA image.

If needed, the PNG-encoded image is transformed to match the requested number
of color channels.

##### Args:


*  <b>`contents`</b>: A `Tensor` of type `string`. 0-D.  The PNG-encoded image.
*  <b>`channels`</b>: An optional `int`. Defaults to `0`.
    Number of color channels for the decoded image.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `uint8`. 3-D with shape `[height, width, channels]`.


- - -

### `tf.image.encode_png(image, compression=None, name=None)` {#encode_png}

PNG-encode an image.

`image` is a 3-D uint8 Tensor of shape `[height, width, channels]` where
`channels` is:

*   1: for grayscale.
*   3: for RGB.
*   4: for RGBA.

The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
default or a value from 0 to 9.  9 is the highest compression level, generating
the smallest output, but is slower.

##### Args:


*  <b>`image`</b>: A `Tensor` of type `uint8`.
    3-D with shape `[height, width, channels]`.
*  <b>`compression`</b>: An optional `int`. Defaults to `-1`. Compression level.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`. 0-D. PNG-encoded image.



## Resizing

The resizing Ops accept input images as tensors of several types.  They always
output resized images as float32 tensors.

The convenience function [`resize_images()`](#resize_images) supports both 4-D
and 3-D tensors as input and output.  4-D tensors are for batches of images,
3-D tensors for individual images.

Other resizing Ops only support 4-D batches of images as input:
[`resize_area`](#resize_area), [`resize_bicubic`](#resize_bicubic),
[`resize_bilinear`](#resize_bilinear),
[`resize_nearest_neighbor`](#resize_nearest_neighbor).

Example:

```python
# Decode a JPG image and resize it to 299 by 299 using default method.
image = tf.image.decode_jpeg(...)
resized_image = tf.image.resize_images(image, 299, 299)
```

- - -

### `tf.image.resize_images(images, new_height, new_width, method=0)` {#resize_images}

Resize `images` to `new_width`, `new_height` using the specified `method`.

Resized images will be distorted if their original aspect ratio is not
the same as `new_width`, `new_height`.  To avoid distortions see
[`resize_image_with_crop_or_pad`](#resize_image_with_crop_or_pad).

`method` can be one of:

*   <b>`ResizeMethod.BILINEAR`</b>: [Bilinear interpolation.]
    (https://en.wikipedia.org/wiki/Bilinear_interpolation)
*   <b>`ResizeMethod.NEAREST_NEIGHBOR`</b>: [Nearest neighbor interpolation.]
    (https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
*   <b>`ResizeMethod.BICUBIC`</b>: [Bicubic interpolation.]
    (https://en.wikipedia.org/wiki/Bicubic_interpolation)
*   <b>`ResizeMethod.AREA`</b>: Area interpolation.

##### Args:


*  <b>`images`</b>: 4-D Tensor of shape `[batch, height, width, channels]` or
          3-D Tensor of shape `[height, width, channels]`.
*  <b>`new_height`</b>: integer.
*  <b>`new_width`</b>: integer.
*  <b>`method`</b>: ResizeMethod.  Defaults to `ResizeMethod.BILINEAR`.

##### Raises:


*  <b>`ValueError`</b>: if the shape of `images` is incompatible with the
    shape arguments to this function
*  <b>`ValueError`</b>: if an unsupported resize method is specified.

##### Returns:

  If `images` was 4-D, a 4-D float Tensor of shape
  `[batch, new_height, new_width, channels]`.
  If `images` was 3-D, a 3-D float Tensor of shape
  `[new_height, new_width, channels]`.



- - -

### `tf.image.resize_area(images, size, name=None)` {#resize_area}

Resize `images` to `size` using area interpolation.

Input images can be of different types but output images are always float.

##### Args:


*  <b>`images`</b>: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `float32`, `float64`.
    4-D with shape `[batch, height, width, channels]`.
*  <b>`size`</b>: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
    new size for the images.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`. 4-D with shape
  `[batch, new_height, new_width, channels]`.


- - -

### `tf.image.resize_bicubic(images, size, name=None)` {#resize_bicubic}

Resize `images` to `size` using bicubic interpolation.

Input images can be of different types but output images are always float.

##### Args:


*  <b>`images`</b>: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `float32`, `float64`.
    4-D with shape `[batch, height, width, channels]`.
*  <b>`size`</b>: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
    new size for the images.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`. 4-D with shape
  `[batch, new_height, new_width, channels]`.


- - -

### `tf.image.resize_bilinear(images, size, name=None)` {#resize_bilinear}

Resize `images` to `size` using bilinear interpolation.

Input images can be of different types but output images are always float.

##### Args:


*  <b>`images`</b>: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `float32`, `float64`.
    4-D with shape `[batch, height, width, channels]`.
*  <b>`size`</b>: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
    new size for the images.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`. 4-D with shape
  `[batch, new_height, new_width, channels]`.


- - -

### `tf.image.resize_nearest_neighbor(images, size, name=None)` {#resize_nearest_neighbor}

Resize `images` to `size` using nearest neighbor interpolation.

Input images can be of different types but output images are always float.

##### Args:


*  <b>`images`</b>: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `float32`, `float64`.
    4-D with shape `[batch, height, width, channels]`.
*  <b>`size`</b>: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
    new size for the images.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `images`. 4-D with shape
  `[batch, new_height, new_width, channels]`.




## Cropping

- - -

### `tf.image.resize_image_with_crop_or_pad(image, target_height, target_width)` {#resize_image_with_crop_or_pad}

Crops and/or pads an image to a target width and height.

Resizes an image to a target width and height by either centrally
cropping the image or padding it evenly with zeros.

If `width` or `height` is greater than the specified `target_width` or
`target_height` respectively, this op centrally crops along that dimension.
If `width` or `height` is smaller than the specified `target_width` or
`target_height` respectively, this op centrally pads with 0 along that
dimension.

##### Args:


*  <b>`image`</b>: 3-D tensor of shape [height, width, channels]
*  <b>`target_height`</b>: Target height.
*  <b>`target_width`</b>: Target width.

##### Raises:


*  <b>`ValueError`</b>: if `target_height` or `target_width` are zero or negative.

##### Returns:

  Cropped and/or padded image of shape
  `[target_height, target_width, channels]`



- - -

### `tf.image.pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width)` {#pad_to_bounding_box}

Pad `image` with zeros to the specified `height` and `width`.

Adds `offset_height` rows of zeros on top, `offset_width` columns of
zeros on the left, and then pads the image on the bottom and right
with zeros until it has dimensions `target_height`, `target_width`.

This op does nothing if `offset_*` is zero and the image already has size
`target_height` by `target_width`.

##### Args:


*  <b>`image`</b>: 3-D tensor with shape `[height, width, channels]`
*  <b>`offset_height`</b>: Number of rows of zeros to add on top.
*  <b>`offset_width`</b>: Number of columns of zeros to add on the left.
*  <b>`target_height`</b>: Height of output image.
*  <b>`target_width`</b>: Width of output image.

##### Returns:

  3-D tensor of shape `[target_height, target_width, channels]`

##### Raises:


*  <b>`ValueError`</b>: If the shape of `image` is incompatible with the `offset_*` or
    `target_*` arguments


- - -

### `tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)` {#crop_to_bounding_box}

Crops an image to a specified bounding box.

This op cuts a rectangular part out of `image`. The top-left corner of the
returned image is at `offset_height, offset_width` in `image`, and its
lower-right corner is at
`offset_height + target_height, offset_width + target_width`.

##### Args:


*  <b>`image`</b>: 3-D tensor with shape `[height, width, channels]`
*  <b>`offset_height`</b>: Vertical coordinate of the top-left corner of the result in
                 the input.
*  <b>`offset_width`</b>: Horizontal coordinate of the top-left corner of the result in
                the input.
*  <b>`target_height`</b>: Height of the result.
*  <b>`target_width`</b>: Width of the result.

##### Returns:

  3-D tensor of image with shape `[target_height, target_width, channels]`

##### Raises:


*  <b>`ValueError`</b>: If the shape of `image` is incompatible with the `offset_*` or
  `target_*` arguments


- - -

### `tf.image.random_crop(image, size, seed=None, name=None)` {#random_crop}

Randomly crops `image` to size `[target_height, target_width]`.

The offset of the output within `image` is uniformly random. `image` always
fully contains the result.

##### Args:


*  <b>`image`</b>: 3-D tensor of shape `[height, width, channels]`
*  <b>`size`</b>: 1-D tensor with two elements, specifying target `[height, width]`
*  <b>`seed`</b>: A Python integer. Used to create a random seed. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  A cropped 3-D tensor of shape `[target_height, target_width, channels]`.


- - -

### `tf.image.extract_glimpse(input, size, offsets, centered=None, normalized=None, uniform_noise=None, name=None)` {#extract_glimpse}

Extracts a glimpse from the input tensor.

Returns a set of windows called glimpses extracted at location `offsets`
from the input tensor. If the windows only partially overlaps the inputs, the
non overlapping areas will be filled with random noise.

The result is a 4-D tensor of shape `[batch_size, glimpse_height,
glimpse_width, channels]`. The channels and batch dimensions are the same as that
of the input tensor. The height and width of the output windows are
specified in the `size` parameter.

The argument `normalized` and `centered` controls how the windows are built:
* If the coordinates are normalized but not centered, 0.0 and 1.0
  correspond to the minimum and maximum of each height and width dimension.
* If the coordinates are both normalized and centered, they range from -1.0 to
  1.0. The coordinates (-1.0, -1.0) correspond to the upper left corner, the
  lower right corner is located at  (1.0, 1.0) and the center is at (0, 0).
* If the coordinates are not normalized they are interpreted as numbers of pixels.

##### Args:


*  <b>`input`</b>: A `Tensor` of type `float32`.
    A 4-D float tensor of shape `[batch_size, height, width, channels]`.
*  <b>`size`</b>: A `Tensor` of type `int32`.
    A 1-D tensor of 2 elements containing the size of the glimpses to extract.
    The glimpse height must be specified first, following by the glimpse width.
*  <b>`offsets`</b>: A `Tensor` of type `float32`.
    A 2-D integer tensor of shape `[batch_size, 2]` containing the x, y
    locations of the center of each window.
*  <b>`centered`</b>: An optional `bool`. Defaults to `True`.
    indicates if the offset coordinates are centered relative to
    the image, in which case the (0, 0) offset is relative to the center of the
    input images. If false, the (0,0) offset corresponds to the upper left corner
    of the input images.
*  <b>`normalized`</b>: An optional `bool`. Defaults to `True`.
    indicates if the offset coordinates are normalized.
*  <b>`uniform_noise`</b>: An optional `bool`. Defaults to `True`.
    indicates if the noise should be generated using a
    uniform distribution or a gaussian distribution.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`.
  A tensor representing the glimpses `[batch_size, glimpse_height,
  glimpse_width, channels]`.



## Flipping and Transposing

- - -

### `tf.image.flip_up_down(image)` {#flip_up_down}

Flip an image horizontally (upside down).

Outputs the contents of `image` flipped along the first dimension, which is
`height`.

See also `reverse()`.

##### Args:


*  <b>`image`</b>: A 3-D tensor of shape `[height, width, channels].`

##### Returns:

  A 3-D tensor of the same type and shape as `image`.

##### Raises:


*  <b>`ValueError`</b>: if the shape of `image` not supported.


- - -

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



- - -

### `tf.image.flip_left_right(image)` {#flip_left_right}

Flip an image horizontally (left to right).

Outputs the contents of `image` flipped along the second dimension, which is
`width`.

See also `reverse()`.

##### Args:


*  <b>`image`</b>: A 3-D tensor of shape `[height, width, channels].`

##### Returns:

  A 3-D tensor of the same type and shape as `image`.

##### Raises:


*  <b>`ValueError`</b>: if the shape of `image` not supported.


- - -

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



- - -

### `tf.image.transpose_image(image)` {#transpose_image}

Transpose an image by swapping the first and second dimension.

See also `transpose()`.

##### Args:


*  <b>`image`</b>: 3-D tensor of shape `[height, width, channels]`

##### Returns:

  A 3-D tensor of shape `[width, height, channels]`

##### Raises:


*  <b>`ValueError`</b>: if the shape of `image` not supported.



## Converting Between Colorspaces.

Image ops work either on individual images or on batches of images, depending on
the shape of their input Tensor.

If 3-D, the shape is `[height, width, channels]`, and the Tensor represents one
image. If 4-D, the shape is `[batch_size, height, width, channels]`, and the
Tensor represents `batch_size` images.

Currently, `channels` can usefully be 1, 2, 3, or 4. Single-channel images are
grayscale, images with 3 channels are encoded as either RGB or HSV. Images
with 2 or 4 channels include an alpha channel, which has to be stripped from the
image before passing the image to most image processing functions (and can be
re-attached later).

Internally, images are either stored in as one `float32` per channel per pixel
(implicitly, values are assumed to lie in `[0,1)`) or one `uint8` per channel
per pixel (values are assumed to lie in `[0,255]`).

Tensorflow can convert between images in RGB or HSV. The conversion functions
work only on float images, so you need to convert images in other formats using
[`convert_image_dtype`](#convert-image-dtype).

Example:

```python
# Decode an image and convert it to HSV.
rgb_image = tf.decode_png(...,  channels=3)
rgb_image_float = tf.convert_image_dtype(rgb_image, tf.float32)
hsv_image = tf.hsv_to_rgb(rgb_image)
```

- - -

### `tf.image.rgb_to_grayscale(images)` {#rgb_to_grayscale}

Converts one or more images from RGB to Grayscale.

Outputs a tensor of the same `DType` and rank as `images`.  The size of the
last dimension of the output is 1, containing the Grayscale value of the
pixels.

##### Args:


*  <b>`images`</b>: The RGB tensor to convert. Last dimension must have size 3 and
    should contain RGB values.

##### Returns:

  The converted grayscale image(s).


- - -

### `tf.image.grayscale_to_rgb(images)` {#grayscale_to_rgb}

Converts one or more images from Grayscale to RGB.

Outputs a tensor of the same `DType` and rank as `images`.  The size of the
last dimension of the output is 3, containing the RGB value of the pixels.

##### Args:


*  <b>`images`</b>: The Grayscale tensor to convert. Last dimension must be size 1.

##### Returns:

  The converted grayscale image(s).



- - -

### `tf.image.hsv_to_rgb(images, name=None)` {#hsv_to_rgb}

Convert one or more images from HSV to RGB.

Outputs a tensor of the same shape as the `images` tensor, containing the RGB
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.

See `rgb_to_hsv` for a description of the HSV encoding.

##### Args:


*  <b>`images`</b>: A `Tensor` of type `float32`.
    1-D or higher rank. HSV data to convert. Last dimension must be size 3.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`. `images` converted to RGB.


- - -

### `tf.image.rgb_to_hsv(images, name=None)` {#rgb_to_hsv}

Converts one or more images from RGB to HSV.

Outputs a tensor of the same shape as the `images` tensor, containing the HSV
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.

`output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
`output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.

##### Args:


*  <b>`images`</b>: A `Tensor` of type `float32`.
    1-D or higher rank. RGB data to convert. Last dimension must be size 3.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`. `images` converted to HSV.



- - -

### `tf.image.convert_image_dtype(image, dtype, name=None)` {#convert_image_dtype}

Convert `image` to `dtype`, scaling its values if needed.

Images that are represented using floating point values are expected to have
values in the range [0,1). Image data stored in integer data types are
expected to have values in the range `[0,MAX]`, wbere `MAX` is the largest
positive representable number for the data type.

This op converts between data types, scaling the values appropriately before
casting.

Note that for floating point inputs, this op expects values to lie in [0,1).
Conversion of an image containing values outside that range may lead to
overflow errors when converted to integer `Dtype`s.

##### Args:


*  <b>`image`</b>: An image.
*  <b>`dtype`</b>: A `DType` to convert `image` to.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  `image`, converted to `dtype`.



## Image Adjustments

TensorFlow provides functions to adjust images in various ways: brightness,
contrast, hue, and saturation.  Each adjustment can be done with predefined
parameters or with random parameters picked from predefined intervals. Random
adjustments are often useful to expand a training set and reduce overfitting.

If several adjustments are chained it is advisable to minimize the number of
redundant conversions by first converting the images to the most natural data
type and representation (RGB or HSV).

- - -

### `tf.image.adjust_brightness(image, delta, min_value=None, max_value=None)` {#adjust_brightness}

Adjust the brightness of RGB or Grayscale images.

The value `delta` is added to all components of the tensor `image`. `image`
and `delta` are cast to `float` before adding, and the resulting values are
clamped to `[min_value, max_value]`. Finally, the result is cast back to
`images.dtype`.

If `min_value` or `max_value` are not given, they are set to the minimum and
maximum allowed values for `image.dtype` respectively.

##### Args:


*  <b>`image`</b>: A tensor.
*  <b>`delta`</b>: A scalar. Amount to add to the pixel values.
*  <b>`min_value`</b>: Minimum value for output.
*  <b>`max_value`</b>: Maximum value for output.

##### Returns:

  A tensor of the same shape and type as `image`.


- - -

### `tf.image.random_brightness(image, max_delta, seed=None)` {#random_brightness}

Adjust the brightness of images by a random factor.

Equivalent to `adjust_brightness()` using a `delta` randomly picked in the
interval `[-max_delta, max_delta)`.

Note that `delta` is picked as a float. Because for integer type images,
the brightness adjusted result is rounded before casting, integer images may
have modifications in the range `[-max_delta,max_delta]`.

##### Args:


*  <b>`image`</b>: 3-D tensor of shape `[height, width, channels]`.
*  <b>`max_delta`</b>: float, must be non-negative.
*  <b>`seed`</b>: A Python integer. Used to create a random seed. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.

##### Returns:

  3-D tensor of images of shape `[height, width, channels]`

##### Raises:


*  <b>`ValueError`</b>: if `max_delta` is negative.



- - -

### `tf.image.adjust_contrast(images, contrast_factor, min_value=None, max_value=None)` {#adjust_contrast}

Adjust contrast of RGB or grayscale images.

`images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
interpreted as `[height, width, channels]`.  The other dimensions only
represent a collection of images, such as `[batch, height, width, channels].`

Contrast is adjusted independently for each channel of each image.

For each channel, this Op first computes the mean of the image pixels in the
channel and then adjusts each component `x` of each pixel to
`(x - mean) * contrast_factor + mean`.

The adjusted values are then clipped to fit in the `[min_value, max_value]`
interval. If `min_value` or `max_value` is not given, it is replaced with the
minimum and maximum values for the data type of `images` respectively.

The contrast-adjusted image is always computed as `float`, and it is
cast back to its original type after clipping.

##### Args:


*  <b>`images`</b>: Images to adjust.  At least 3-D.
*  <b>`contrast_factor`</b>: A float multiplier for adjusting contrast.
*  <b>`min_value`</b>: Minimum value for clipping the adjusted pixels.
*  <b>`max_value`</b>: Maximum value for clipping the adjusted pixels.

##### Returns:

  The constrast-adjusted image or images.

##### Raises:


*  <b>`ValueError`</b>: if the arguments are invalid.


- - -

### `tf.image.random_contrast(image, lower, upper, seed=None)` {#random_contrast}

Adjust the contrase of an image by a random factor.

Equivalent to `adjust_constrast()` but uses a `contrast_factor` randomly
picked in the interval `[lower, upper]`.

##### Args:


*  <b>`image`</b>: 3-D tensor of shape `[height, width, channels]`.
*  <b>`lower`</b>: float.  Lower bound for the random contrast factor.
*  <b>`upper`</b>: float.  Upper bound for the random contrast factor.
*  <b>`seed`</b>: A Python integer. Used to create a random seed. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.

##### Returns:

  3-D tensor of shape `[height, width, channels]`.

##### Raises:


*  <b>`ValueError`</b>: if `upper <= lower` or if `lower < 0`.



- - -

### `tf.image.adjust_hue(image, delta, name=None)` {#adjust_hue}

Adjust hue of an RGB image.

This is a convenience method that converts an RGB image to float
representation, converts it to HSV, add an offset to the hue channel, converts
back to RGB and then back to the original data type. If several adjustments
are chained it is advisable to minimize the number of redundant conversions.

`image` is an RGB image.  The image hue is adjusted by converting the
image to HSV and rotating the hue channel (H) by
`delta`.  The image is then converted back to RGB.

`delta` must be in the interval `[-1, 1]`.

##### Args:


*  <b>`image`</b>: RGB image or images. Size of the last dimension must be 3.
*  <b>`delta`</b>: float.  How much to add to the hue channel.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  Adjusted image(s), same shape and DType as `image`.


- - -

### `tf.image.random_hue(image, max_delta, seed=None)` {#random_hue}

Adjust the hue of an RGB image by a random factor.

Equivalent to `adjust_hue()` but uses a `delta` randomly
picked in the interval `[-max_delta, max_delta]`.

`max_delta` must be in the interval `[0, 0.5]`.

##### Args:


*  <b>`image`</b>: RGB image or images. Size of the last dimension must be 3.
*  <b>`max_delta`</b>: float.  Maximum value for the random delta.
*  <b>`seed`</b>: An operation-specific seed. It will be used in conjunction
    with the graph-level seed to determine the real seeds that will be
    used in this operation. Please see the documentation of
    set_random_seed for its interaction with the graph-level random seed.

##### Returns:

  3-D float tensor of shape `[height, width, channels]`.

##### Raises:


*  <b>`ValueError`</b>: if `max_delta` is invalid.



- - -

### `tf.image.adjust_saturation(image, saturation_factor, name=None)` {#adjust_saturation}

Adjust staturation of an RGB image.

This is a convenience method that converts an RGB image to float
representation, converts it to HSV, add an offset to the saturation channel,
converts back to RGB and then back to the original data type. If several
adjustments are chained it is advisable to minimize the number of redundant
conversions.

`image` is an RGB image.  The image saturation is adjusted by converting the
image to HSV and multiplying the saturation (S) channel by
`saturation_factor` and clipping. The image is then converted back to RGB.

##### Args:


*  <b>`image`</b>: RGB image or images. Size of the last dimension must be 3.
*  <b>`saturation_factor`</b>: float. Factor to multiply the saturation by.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  Adjusted image(s), same shape and DType as `image`.


- - -

### `tf.image.random_saturation(image, lower, upper, seed=None)` {#random_saturation}

Adjust the saturation of an RGB image by a random factor.

Equivalent to `adjust_saturation()` but uses a `saturation_factor` randomly
picked in the interval `[lower, upper]`.

##### Args:


*  <b>`image`</b>: RGB image or images. Size of the last dimension must be 3.
*  <b>`lower`</b>: float.  Lower bound for the random saturation factor.
*  <b>`upper`</b>: float.  Upper bound for the random saturation factor.
*  <b>`seed`</b>: An operation-specific seed. It will be used in conjunction
    with the graph-level seed to determine the real seeds that will be
    used in this operation. Please see the documentation of
    set_random_seed for its interaction with the graph-level random seed.

##### Returns:

  Adjusted image(s), same shape and DType as `image`.

##### Raises:


*  <b>`ValueError`</b>: if `upper <= lower` or if `lower < 0`.



- - -

### `tf.image.per_image_whitening(image)` {#per_image_whitening}

Linearly scales `image` to have zero mean and unit norm.

This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
of all values in image, and
`adjusted_stddev = max(stddev, 1.0/srqt(image.NumElements()))`.

`stddev` is the standard deviation of all values in `image`. It is capped
away from zero to protect against division by 0 when handling uniform images.

Note that this implementation is limited:
*  It only whitens based on the statistics of an individual image.
*  It does not take into account the covariance structure.

##### Args:


*  <b>`image`</b>: 3-D tensor of shape `[height, width, channels]`.

##### Returns:

  The whitened image with same shape as `image`.

##### Raises:


*  <b>`ValueError`</b>: if the shape of 'image' is incompatible with this function.



## Other Functions and Classes
- - -

### `tf.image.resize_nearest_neighbor_grad(grads, size, name=None)` {#resize_nearest_neighbor_grad}

Computes the gradient of nearest neighbor interpolation.

##### Args:


*  <b>`grads`</b>: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `float32`, `float64`.
    4-D with shape `[batch, height, width, channels]`.
*  <b>`size`</b>: A 1-D int32 Tensor of 2 elements: `orig_height, orig_width`. The
    original input size.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `grads`.
  4-D with shape `[batch, orig_height, orig_width, channels]`. Gradients
  with respect to the input image.


