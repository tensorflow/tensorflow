# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=g-short-docstring-punctuation
"""## Encoding and Decoding

TensorFlow provides Ops to decode and encode JPEG and PNG formats.  Encoded
images are represented by scalar string Tensors, decoded images by 3-D uint8
tensors of shape `[height, width, channels]`. (PNG also supports uint16.)

The encode and decode Ops apply to one image at a time.  Their input and output
are all of variable size.  If you need fixed size images, pass the output of
the decode Ops to one of the cropping and resizing Ops.

Note: The PNG encode and decode Ops support RGBA, but the conversions Ops
presently only support RGB, HSV, and GrayScale. Presently, the alpha channel has
to be stripped from the image and re-attached using slicing ops.

@@decode_jpeg
@@encode_jpeg

@@decode_png
@@encode_png

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

@@resize_images

@@resize_area
@@resize_bicubic
@@resize_bilinear
@@resize_nearest_neighbor


## Cropping

@@resize_image_with_crop_or_pad

@@central_crop
@@pad_to_bounding_box
@@crop_to_bounding_box
@@extract_glimpse

## Flipping and Transposing

@@flip_up_down
@@random_flip_up_down

@@flip_left_right
@@random_flip_left_right

@@transpose_image

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
rgb_image = tf.image.decode_png(...,  channels=3)
rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
hsv_image = tf.image.rgb_to_hsv(rgb_image)
```

@@rgb_to_grayscale
@@grayscale_to_rgb

@@hsv_to_rgb
@@rgb_to_hsv

@@convert_image_dtype

## Image Adjustments

TensorFlow provides functions to adjust images in various ways: brightness,
contrast, hue, and saturation.  Each adjustment can be done with predefined
parameters or with random parameters picked from predefined intervals. Random
adjustments are often useful to expand a training set and reduce overfitting.

If several adjustments are chained it is advisable to minimize the number of
redundant conversions by first converting the images to the most natural data
type and representation (RGB or HSV).

@@adjust_brightness
@@random_brightness

@@adjust_contrast
@@random_contrast

@@adjust_hue
@@random_hue

@@adjust_saturation
@@random_saturation

@@per_image_whitening

## Working with Bounding Boxes

@@draw_bounding_boxes
@@sample_distorted_bounding_box
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_image_ops import *
# pylint: enable=wildcard-import

from tensorflow.python.util.all_util import make_all



ops.NoGradient('RandomCrop')
ops.NoGradient('RGBToHSV')
ops.NoGradient('HSVToRGB')
ops.NoGradient('DrawBoundingBoxes')
ops.NoGradient('SampleDistortedBoundingBox')
# TODO(bsteiner): Implement the gradient function for extract_glimpse
ops.NoGradient("ExtractGlimpse")


def _ImageDimensions(images):
  """Returns the dimensions of an image tensor.

  Args:
    images: 4-D Tensor of shape [batch, height, width, channels]

  Returns:
    list of integers [batch, height, width, channels]
  """
  # A simple abstraction to provide names for each dimension. This abstraction
  # should make it simpler to switch dimensions in the future (e.g. if we ever
  # want to switch height and width.)
  return images.get_shape().as_list()


def _Check3DImage(image, require_static=True):
  """Assert that we are working with properly shaped image.

  Args:
    image: 3-D Tensor of shape [height, width, channels]
    require_static: If `True`, requires that all dimensions of `image` are
      known and non-zero.

  Raises:
    ValueError: if image.shape is not a [3] vector.
  """
  try:
    image_shape = image.get_shape().with_rank(3)
  except ValueError:
    raise ValueError('\'image\' must be three-dimensional.')
  if require_static and not image_shape.is_fully_defined():
    raise ValueError('\'image\' must be fully defined.')
  if any(x == 0 for x in image_shape):
    raise ValueError('all dims of \'image.shape\' must be > 0: %s' %
                     image_shape)


def _CheckAtLeast3DImage(image):
  """Assert that we are working with properly shaped image.

  Args:
    image: >= 3-D Tensor of size [*, height, width, depth]

  Raises:
    ValueError: if image.shape is not a [>= 3] vector.
  """
  if not image.get_shape().is_fully_defined():
    raise ValueError('\'image\' must be fully defined.')
  if image.get_shape().ndims < 3:
    raise ValueError('\'image\' must be at least three-dimensional.')
  if not all(x > 0 for x in image.get_shape()):
    raise ValueError('all dims of \'image.shape\' must be > 0: %s' %
                     image.get_shape())


def random_flip_up_down(image, seed=None):
  """Randomly flips an image vertically (upside down).

  With a 1 in 2 chance, outputs the contents of `image` flipped along the first
  dimension, which is `height`.  Otherwise output the image as-is.

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    seed: A Python integer. Used to create a random seed. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.

  Returns:
    A 3-D tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  image = ops.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=False)
  uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
  mirror = math_ops.less(array_ops.pack([uniform_random, 1.0, 1.0]), 0.5)
  return array_ops.reverse(image, mirror)


def random_flip_left_right(image, seed=None):
  """Randomly flip an image horizontally (left to right).

  With a 1 in 2 chance, outputs the contents of `image` flipped along the
  second dimension, which is `width`.  Otherwise output the image as-is.

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    seed: A Python integer. Used to create a random seed. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.

  Returns:
    A 3-D tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  image = ops.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=False)
  uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
  mirror = math_ops.less(array_ops.pack([1.0, uniform_random, 1.0]), 0.5)
  return array_ops.reverse(image, mirror)


def flip_left_right(image):
  """Flip an image horizontally (left to right).

  Outputs the contents of `image` flipped along the second dimension, which is
  `width`.

  See also `reverse()`.

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`

  Returns:
    A 3-D tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  image = ops.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=False)
  return array_ops.reverse(image, [False, True, False])


def flip_up_down(image):
  """Flip an image horizontally (upside down).

  Outputs the contents of `image` flipped along the first dimension, which is
  `height`.

  See also `reverse()`.

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`

  Returns:
    A 3-D tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  image = ops.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=False)
  return array_ops.reverse(image, [True, False, False])


def transpose_image(image):
  """Transpose an image by swapping the first and second dimension.

  See also `transpose()`.

  Args:
    image: 3-D tensor of shape `[height, width, channels]`

  Returns:
    A 3-D tensor of shape `[width, height, channels]`

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  image = ops.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=False)
  return array_ops.transpose(image, [1, 0, 2], name='transpose_image')


def central_crop(image, central_fraction):
  """Crop the central region of the image.

  Remove the outer parts of an image but retain the central region of the image
  along each dimension. If we specify central_fraction = 0.5, this function
  returns the region marked with "X" in the below diagram.

       --------
      |        |
      |  XXXX  |
      |  XXXX  |
      |        |   where "X" is the central 50% of the image.
       --------

  Args:
    image: 3-D float Tensor of shape [height, width, depth]
    central_fraction: float (0, 1], fraction of size to crop

  Raises:
    ValueError: if central_crop_fraction is not within (0, 1].

  Returns:
    3-D float Tensor
  """
  image = ops.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=False)
  if central_fraction <= 0.0 or central_fraction > 1.0:
    raise ValueError('central_fraction must be within (0, 1]')
  if central_fraction == 1.0:
    return image

  img_shape = array_ops.shape(image)
  depth = image.get_shape()[2]
  fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
  bbox_h_start = math_ops.div(img_shape[0], fraction_offset)
  bbox_w_start = math_ops.div(img_shape[1], fraction_offset)

  bbox_h_size = img_shape[0] - bbox_h_start * 2
  bbox_w_size = img_shape[1] - bbox_w_start * 2

  bbox_begin = array_ops.pack([bbox_h_start, bbox_w_start, 0])
  bbox_size = array_ops.pack([bbox_h_size, bbox_w_size, -1])
  image = array_ops.slice(image, bbox_begin, bbox_size)

  # The first two dimensions are dynamic and unknown.
  image.set_shape([None, None, depth])
  return image


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width):
  """Pad `image` with zeros to the specified `height` and `width`.

  Adds `offset_height` rows of zeros on top, `offset_width` columns of
  zeros on the left, and then pads the image on the bottom and right
  with zeros until it has dimensions `target_height`, `target_width`.

  This op does nothing if `offset_*` is zero and the image already has size
  `target_height` by `target_width`.

  Args:
    image: 3-D tensor with shape `[height, width, channels]`
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.

  Returns:
    3-D tensor of shape `[target_height, target_width, channels]`

  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
      `target_*` arguments
  """
  image = ops.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=True)
  height, width, depth = _ImageDimensions(image)

  if target_width < width:
    raise ValueError('target_width must be >= width')
  if target_height < height:
    raise ValueError('target_height must be >= height')

  after_padding_width = target_width - offset_width - width
  after_padding_height = target_height - offset_height - height

  if after_padding_width < 0:
    raise ValueError('target_width not possible given '
                     'offset_width and image width')
  if after_padding_height < 0:
    raise ValueError('target_height not possible given '
                     'offset_height and image height')

  # Do not pad on the depth dimensions.
  if (offset_width or offset_height or after_padding_width or
      after_padding_height):
    paddings = [[offset_height, after_padding_height],
                [offset_width, after_padding_width], [0, 0]]
    padded = array_ops.pad(image, paddings)
    padded.set_shape([target_height, target_width, depth])
  else:
    padded = image

  return padded


def crop_to_bounding_box(image, offset_height, offset_width, target_height,
                         target_width):
  """Crops an image to a specified bounding box.

  This op cuts a rectangular part out of `image`. The top-left corner of the
  returned image is at `offset_height, offset_width` in `image`, and its
  lower-right corner is at
  `offset_height + target_height, offset_width + target_width`.

  Args:
    image: 3-D tensor with shape `[height, width, channels]`
    offset_height: Vertical coordinate of the top-left corner of the result in
                   the input.
    offset_width: Horizontal coordinate of the top-left corner of the result in
                  the input.
    target_height: Height of the result.
    target_width: Width of the result.

  Returns:
    3-D tensor of image with shape `[target_height, target_width, channels]`

  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
    `target_*` arguments
  """
  image = ops.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=True)
  height, width, _ = _ImageDimensions(image)

  if offset_width < 0:
    raise ValueError('offset_width must be >= 0.')
  if offset_height < 0:
    raise ValueError('offset_height must be >= 0.')

  if width < (target_width + offset_width):
    raise ValueError('width must be >= target + offset.')
  if height < (target_height + offset_height):
    raise ValueError('height must be >= target + offset.')

  cropped = array_ops.slice(image, [offset_height, offset_width, 0],
                            [target_height, target_width, -1])

  return cropped


def resize_image_with_crop_or_pad(image, target_height, target_width):
  """Crops and/or pads an image to a target width and height.

  Resizes an image to a target width and height by either centrally
  cropping the image or padding it evenly with zeros.

  If `width` or `height` is greater than the specified `target_width` or
  `target_height` respectively, this op centrally crops along that dimension.
  If `width` or `height` is smaller than the specified `target_width` or
  `target_height` respectively, this op centrally pads with 0 along that
  dimension.

  Args:
    image: 3-D tensor of shape [height, width, channels]
    target_height: Target height.
    target_width: Target width.

  Raises:
    ValueError: if `target_height` or `target_width` are zero or negative.

  Returns:
    Cropped and/or padded image of shape
    `[target_height, target_width, channels]`
  """
  image = ops.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=True)
  original_height, original_width, _ = _ImageDimensions(image)

  if target_width <= 0:
    raise ValueError('target_width must be > 0.')
  if target_height <= 0:
    raise ValueError('target_height must be > 0.')

  offset_crop_width = 0
  offset_pad_width = 0
  if target_width < original_width:
    offset_crop_width = (original_width - target_width) // 2
  elif target_width > original_width:
    offset_pad_width = (target_width - original_width) // 2

  offset_crop_height = 0
  offset_pad_height = 0
  if target_height < original_height:
    offset_crop_height = (original_height - target_height) // 2
  elif target_height > original_height:
    offset_pad_height = (target_height - original_height) // 2

  # Maybe crop if needed.
  cropped = crop_to_bounding_box(image, offset_crop_height, offset_crop_width,
                                 min(target_height, original_height),
                                 min(target_width, original_width))

  # Maybe pad if needed.
  resized = pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                target_height, target_width)

  if resized.get_shape().ndims is None:
    raise ValueError('resized contains no shape.')
  if not resized.get_shape()[0].is_compatible_with(target_height):
    raise ValueError('resized height is not correct.')
  if not resized.get_shape()[1].is_compatible_with(target_width):
    raise ValueError('resized width is not correct.')
  return resized


class ResizeMethod(object):
  BILINEAR = 0
  NEAREST_NEIGHBOR = 1
  BICUBIC = 2
  AREA = 3


def resize_images(images,
                  new_height,
                  new_width,
                  method=ResizeMethod.BILINEAR,
                  align_corners=False):
  """Resize `images` to `new_width`, `new_height` using the specified `method`.

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

  Args:
    images: 4-D Tensor of shape `[batch, height, width, channels]` or
            3-D Tensor of shape `[height, width, channels]`.
    new_height: integer.
    new_width: integer.
    method: ResizeMethod.  Defaults to `ResizeMethod.BILINEAR`.
    align_corners: bool. If true, exactly align all 4 corners of the input and
                   output. Defaults to `false`.

  Raises:
    ValueError: if the shape of `images` is incompatible with the
      shape arguments to this function
    ValueError: if an unsupported resize method is specified.

  Returns:
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """
  images = ops.convert_to_tensor(images, name='images')
  if images.get_shape().ndims is None:
    raise ValueError('\'images\' contains no shape.')
  # TODO(shlens): Migrate this functionality to the underlying Op's.
  is_batch = True
  if len(images.get_shape()) == 3:
    is_batch = False
    images = array_ops.expand_dims(images, 0)

  _, height, width, depth = _ImageDimensions(images)

  # Handle tensor-valued sizes as well as Python integers.
  try:
    new_width = ops.convert_to_tensor(new_width, dtypes.int32,
                                      name='new_width')
    new_width.get_shape().assert_has_rank(0)
  except (TypeError, ValueError):
    raise ValueError('new_width must be a scalar integer')
  try:
    new_height = ops.convert_to_tensor(new_height, dtypes.int32,
                                       name='new_height')
    new_height.get_shape().assert_has_rank(0)
  except (TypeError, ValueError):
    raise ValueError('new_height must be a scalar integer')

  new_width_const = tensor_util.constant_value(new_width)
  new_height_const = tensor_util.constant_value(new_height)

  # If we can determine that the height and width will be unmodified by this
  # transformation, we avoid performing the resize.
  if all(x is not None
         for x in [new_width_const, width, new_height_const, height]) and (
             width == new_width_const and height == new_height_const):
    if not is_batch:
      images = array_ops.squeeze(images, squeeze_dims=[0])
    return images

  new_size = array_ops.pack([new_height, new_width])

  if method == ResizeMethod.BILINEAR:
    images = gen_image_ops.resize_bilinear(images,
                                           new_size,
                                           align_corners=align_corners)
  elif method == ResizeMethod.NEAREST_NEIGHBOR:
    images = gen_image_ops.resize_nearest_neighbor(images,
                                                   new_size,
                                                   align_corners=align_corners)
  elif method == ResizeMethod.BICUBIC:
    images = gen_image_ops.resize_bicubic(images,
                                          new_size,
                                          align_corners=align_corners)
  elif method == ResizeMethod.AREA:
    images = gen_image_ops.resize_area(images,
                                       new_size,
                                       align_corners=align_corners)
  else:
    raise ValueError('Resize method is not implemented.')

  # NOTE(mrry): The shape functions for the resize ops cannot unpack
  # the packed values in `new_size`, so set the shape here.
  images.set_shape([None, new_height_const, new_width_const, None])

  if not is_batch:
    images = array_ops.squeeze(images, squeeze_dims=[0])
  return images


def per_image_whitening(image):
  """Linearly scales `image` to have zero mean and unit norm.

  This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
  of all values in image, and
  `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.

  `stddev` is the standard deviation of all values in `image`. It is capped
  away from zero to protect against division by 0 when handling uniform images.

  Note that this implementation is limited:
  *  It only whitens based on the statistics of an individual image.
  *  It does not take into account the covariance structure.

  Args:
    image: 3-D tensor of shape `[height, width, channels]`.

  Returns:
    The whitened image with same shape as `image`.

  Raises:
    ValueError: if the shape of 'image' is incompatible with this function.
  """
  image = ops.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=False)
  num_pixels = math_ops.reduce_prod(array_ops.shape(image))

  image = math_ops.cast(image, dtype=dtypes.float32)
  image_mean = math_ops.reduce_mean(image)

  variance = (math_ops.reduce_mean(math_ops.square(image)) -
              math_ops.square(image_mean))
  variance = gen_nn_ops.relu(variance)
  stddev = math_ops.sqrt(variance)

  # Apply a minimum normalization that protects us against uniform images.
  min_stddev = math_ops.inv(
      math_ops.sqrt(math_ops.cast(num_pixels, dtypes.float32)))
  pixel_value_scale = math_ops.maximum(stddev, min_stddev)
  pixel_value_offset = image_mean

  image = math_ops.sub(image, pixel_value_offset)
  image = math_ops.div(image, pixel_value_scale)
  return image


def random_brightness(image, max_delta, seed=None):
  """Adjust the brightness of images by a random factor.

  Equivalent to `adjust_brightness()` using a `delta` randomly picked in the
  interval `[-max_delta, max_delta)`.

  Args:
    image: An image.
    max_delta: float, must be non-negative.
    seed: A Python integer. Used to create a random seed. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.

  Returns:
    The brightness-adjusted image.

  Raises:
    ValueError: if `max_delta` is negative.
  """
  if max_delta < 0:
    raise ValueError('max_delta must be non-negative.')

  delta = random_ops.random_uniform([], -max_delta, max_delta, seed=seed)
  return adjust_brightness(image, delta)


def random_contrast(image, lower, upper, seed=None):
  """Adjust the contrast of an image by a random factor.

  Equivalent to `adjust_contrast()` but uses a `contrast_factor` randomly
  picked in the interval `[lower, upper]`.

  Args:
    image: An image tensor with 3 or more dimensions.
    lower: float.  Lower bound for the random contrast factor.
    upper: float.  Upper bound for the random contrast factor.
    seed: A Python integer. Used to create a random seed. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.

  Returns:
    The contrast-adjusted tensor.

  Raises:
    ValueError: if `upper <= lower` or if `lower < 0`.
  """
  if upper <= lower:
    raise ValueError('upper must be > lower.')

  if lower < 0:
    raise ValueError('lower must be non-negative.')

  # Generate an a float in [lower, upper]
  contrast_factor = random_ops.random_uniform([], lower, upper, seed=seed)
  return adjust_contrast(image, contrast_factor)


def adjust_brightness(image, delta):
  """Adjust the brightness of RGB or Grayscale images.

  This is a convenience method that converts an RGB image to float
  representation, adjusts its brightness, and then converts it back to the
  original data type. If several adjustments are chained it is advisable to
  minimize the number of redundant conversions.

  The value `delta` is added to all components of the tensor `image`. Both
  `image` and `delta` are converted to `float` before adding (and `image` is
  scaled appropriately if it is in fixed-point representation). For regular
  images, `delta` should be in the range `[0,1)`, as it is added to the image in
  floating point representation, where pixel values are in the `[0,1)` range.

  Args:
    image: A tensor.
    delta: A scalar. Amount to add to the pixel values.

  Returns:
    A brightness-adjusted tensor of the same shape and type as `image`.
  """
  with ops.op_scope([image, delta], None, 'adjust_brightness') as name:
    image = ops.convert_to_tensor(image, name='image')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    flt_image = convert_image_dtype(image, dtypes.float32)

    adjusted = math_ops.add(flt_image,
                            math_ops.cast(delta, dtypes.float32),
                            name=name)

    return convert_image_dtype(adjusted, orig_dtype, saturate=True)


def adjust_contrast(images, contrast_factor):
  """Adjust contrast of RGB or grayscale images.

  This is a convenience method that converts an RGB image to float
  representation, adjusts its contrast, and then converts it back to the
  original data type. If several adjustments are chained it is advisable to
  minimize the number of redundant conversions.

  `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
  interpreted as `[height, width, channels]`.  The other dimensions only
  represent a collection of images, such as `[batch, height, width, channels].`

  Contrast is adjusted independently for each channel of each image.

  For each channel, this Op computes the mean of the image pixels in the
  channel and then adjusts each component `x` of each pixel to
  `(x - mean) * contrast_factor + mean`.

  Args:
    images: Images to adjust.  At least 3-D.
    contrast_factor: A float multiplier for adjusting contrast.

  Returns:
    The contrast-adjusted image or images.
  """
  with ops.op_scope([images, contrast_factor], None, 'adjust_contrast') as name:
    images = ops.convert_to_tensor(images, name='images')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = images.dtype
    flt_images = convert_image_dtype(images, dtypes.float32)

    # pylint: disable=protected-access
    adjusted = gen_image_ops._adjust_contrastv2(flt_images,
                                                contrast_factor=contrast_factor,
                                                name=name)
    # pylint: enable=protected-access

    return convert_image_dtype(adjusted, orig_dtype, saturate=True)


ops.RegisterShape('AdjustContrast')(
    common_shapes.unchanged_shape_with_rank_at_least(3))
ops.RegisterShape('AdjustContrastv2')(
    common_shapes.unchanged_shape_with_rank_at_least(3))
ops.RegisterShape('DrawBoundingBoxes')(
    common_shapes.unchanged_shape_with_rank_at_least(3))


@ops.RegisterShape('SampleDistortedBoundingBox')
def _SampleDistortedBoundingBoxShape(unused_op):  # pylint: disable=invalid-name
  """Shape function for the sample distorted bounding box."""
  return [tensor_shape.TensorShape([3]),
          tensor_shape.TensorShape([3]),
          tensor_shape.TensorShape([1, 1, 4])]


@ops.RegisterShape('ResizeBilinear')
@ops.RegisterShape('ResizeNearestNeighbor')
@ops.RegisterShape('ResizeBicubic')
@ops.RegisterShape('ResizeArea')
def _ResizeShape(op):
  """Shape function for the resize_bilinear and resize_nearest_neighbor ops."""
  input_shape = op.inputs[0].get_shape().with_rank(4)
  unused_size_shape = op.inputs[1].get_shape().merge_with([2])
  size = tensor_util.constant_value(op.inputs[1])
  if size is not None:
    height = size[0]
    width = size[1]
  else:
    height = None
    width = None
  return [tensor_shape.TensorShape(
      [input_shape[0], height, width, input_shape[3]])]


@ops.RegisterShape('DecodeJpeg')
@ops.RegisterShape('DecodePng')
def _ImageDecodeShape(op):
  """Shape function for image decoding ops."""
  unused_input_shape = op.inputs[0].get_shape().merge_with(
      tensor_shape.scalar())
  channels = op.get_attr('channels') or None
  return [tensor_shape.TensorShape([None, None, channels])]


@ops.RegisterShape('EncodeJpeg')
@ops.RegisterShape('EncodePng')
def _ImageEncodeShape(op):
  """Shape function for image encoding ops."""
  unused_input_shape = op.inputs[0].get_shape().with_rank(3)
  return [tensor_shape.scalar()]


def convert_image_dtype(image, dtype, saturate=False, name=None):
  """Convert `image` to `dtype`, scaling its values if needed.

  Images that are represented using floating point values are expected to have
  values in the range [0,1). Image data stored in integer data types are
  expected to have values in the range `[0,MAX]`, where `MAX` is the largest
  positive representable number for the data type.

  This op converts between data types, scaling the values appropriately before
  casting.

  Note that converting from floating point inputs to integer types may lead to
  over/underflow problems. Set saturate to `True` to avoid such problem in
  problematic conversions. If enabled, saturation will clip the output into the
  allowed range before performing a potentially dangerous cast (and only before
  performing such a cast, i.e., when casting from a floating point to an integer
  type, and when casting from a signed to an unsigned type; `saturate` has no
  effect on casts between floats, or on casts that increase the type's range).

  Args:
    image: An image.
    dtype: A `DType` to convert `image` to.
    saturate: If `True`, clip the input before casting (if necessary).
    name: A name for this operation (optional).

  Returns:
    `image`, converted to `dtype`.
  """
  image = ops.convert_to_tensor(image, name='image')
  if dtype == image.dtype:
    return array_ops.identity(image, name=name)

  with ops.op_scope([image], name, 'convert_image') as name:
    # Both integer: use integer multiplication in the larger range
    if image.dtype.is_integer and dtype.is_integer:
      scale_in = image.dtype.max
      scale_out = dtype.max
      if scale_in > scale_out:
        # Scaling down, scale first, then cast. The scaling factor will
        # cause in.max to be mapped to above out.max but below out.max+1,
        # so that the output is safely in the supported range.
        scale = (scale_in + 1) // (scale_out + 1)
        scaled = math_ops.div(image, scale)

        if saturate:
          return math_ops.saturate_cast(scaled, dtype, name=name)
        else:
          return math_ops.cast(scaled, dtype, name=name)
      else:
        # Scaling up, cast first, then scale. The scale will not map in.max to
        # out.max, but converting back and forth should result in no change.
        if saturate:
          cast = math_ops.saturate_cast(scaled, dtype)
        else:
          cast = math_ops.cast(image, dtype)
        scale = (scale_out + 1) // (scale_in + 1)
        return math_ops.mul(cast, scale, name=name)
    elif image.dtype.is_floating and dtype.is_floating:
      # Both float: Just cast, no possible overflows in the allowed ranges.
      # Note: We're ignoreing float overflows. If your image dynamic range
      # exceeds float range you're on your own.
      return math_ops.cast(image, dtype, name=name)
    else:
      if image.dtype.is_integer:
        # Converting to float: first cast, then scale. No saturation possible.
        cast = math_ops.cast(image, dtype)
        scale = 1. / image.dtype.max
        return math_ops.mul(cast, scale, name=name)
      else:
        # Converting from float: first scale, then cast
        scale = dtype.max + 0.5  # avoid rounding problems in the cast
        scaled = math_ops.mul(image, scale)
        if saturate:
          return math_ops.saturate_cast(scaled, dtype, name=name)
        else:
          return math_ops.cast(scaled, dtype, name=name)


def rgb_to_grayscale(images, name=None):
  """Converts one or more images from RGB to Grayscale.

  Outputs a tensor of the same `DType` and rank as `images`.  The size of the
  last dimension of the output is 1, containing the Grayscale value of the
  pixels.

  Args:
    images: The RGB tensor to convert. Last dimension must have size 3 and
      should contain RGB values.
    name: A name for the operation (optional).

  Returns:
    The converted grayscale image(s).
  """
  with ops.op_scope([images], name, 'rgb_to_grayscale') as name:
    images = ops.convert_to_tensor(images, name='images')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = images.dtype
    flt_image = convert_image_dtype(images, dtypes.float32)

    # Reference for converting between RGB and grayscale.
    # https://en.wikipedia.org/wiki/Luma_%28video%29
    rgb_weights = [0.2989, 0.5870, 0.1140]
    rank_1 = array_ops.expand_dims(array_ops.rank(images) - 1, 0)
    gray_float = math_ops.reduce_sum(flt_image * rgb_weights,
                                     rank_1,
                                     keep_dims=True)
    gray_float.set_shape(images.get_shape()[:-1].concatenate([1]))
    return convert_image_dtype(gray_float, orig_dtype, name=name)


def grayscale_to_rgb(images, name=None):
  """Converts one or more images from Grayscale to RGB.

  Outputs a tensor of the same `DType` and rank as `images`.  The size of the
  last dimension of the output is 3, containing the RGB value of the pixels.

  Args:
    images: The Grayscale tensor to convert. Last dimension must be size 1.
    name: A name for the operation (optional).

  Returns:
    The converted grayscale image(s).
  """
  with ops.op_scope([images], name, 'grayscale_to_rgb') as name:
    images = ops.convert_to_tensor(images, name='images')
    rank_1 = array_ops.expand_dims(array_ops.rank(images) - 1, 0)
    shape_list = (
        [array_ops.ones(rank_1,
                        dtype=dtypes.int32)] + [array_ops.expand_dims(3, 0)])
    multiples = array_ops.concat(0, shape_list)
    rgb = array_ops.tile(images, multiples, name=name)
    rgb.set_shape(images.get_shape()[:-1].concatenate([3]))
    return rgb


# pylint: disable=invalid-name
@ops.RegisterShape('HSVToRGB')
@ops.RegisterShape('RGBToHSV')
def _ColorspaceShape(op):
  """Shape function for colorspace ops."""
  input_shape = op.inputs[0].get_shape().with_rank_at_least(1)
  input_rank = input_shape.ndims
  if input_rank is not None:
    input_shape = input_shape.merge_with([None] * (input_rank - 1) + [3])
  return [input_shape]
# pylint: enable=invalid-name


def random_hue(image, max_delta, seed=None):
  """Adjust the hue of an RGB image by a random factor.

  Equivalent to `adjust_hue()` but uses a `delta` randomly
  picked in the interval `[-max_delta, max_delta]`.

  `max_delta` must be in the interval `[0, 0.5]`.

  Args:
    image: RGB image or images. Size of the last dimension must be 3.
    max_delta: float.  Maximum value for the random delta.
    seed: An operation-specific seed. It will be used in conjunction
      with the graph-level seed to determine the real seeds that will be
      used in this operation. Please see the documentation of
      set_random_seed for its interaction with the graph-level random seed.

  Returns:
    3-D float tensor of shape `[height, width, channels]`.

  Raises:
    ValueError: if `max_delta` is invalid.
  """
  if max_delta > 0.5:
    raise ValueError('max_delta must be <= 0.5.')

  if max_delta < 0:
    raise ValueError('max_delta must be non-negative.')

  delta = random_ops.random_uniform([], -max_delta, max_delta, seed=seed)
  return adjust_hue(image, delta)


def adjust_hue(image, delta, name=None):
  """Adjust hue of an RGB image.

  This is a convenience method that converts an RGB image to float
  representation, converts it to HSV, add an offset to the hue channel, converts
  back to RGB and then back to the original data type. If several adjustments
  are chained it is advisable to minimize the number of redundant conversions.

  `image` is an RGB image.  The image hue is adjusted by converting the
  image to HSV and rotating the hue channel (H) by
  `delta`.  The image is then converted back to RGB.

  `delta` must be in the interval `[-1, 1]`.

  Args:
    image: RGB image or images. Size of the last dimension must be 3.
    delta: float.  How much to add to the hue channel.
    name: A name for this operation (optional).

  Returns:
    Adjusted image(s), same shape and DType as `image`.
  """
  with ops.op_scope([image], name, 'adjust_hue') as name:
    image = ops.convert_to_tensor(image, name='image')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    flt_image = convert_image_dtype(image, dtypes.float32)

    hsv = gen_image_ops.rgb_to_hsv(flt_image)

    hue = array_ops.slice(hsv, [0, 0, 0], [-1, -1, 1])
    saturation = array_ops.slice(hsv, [0, 0, 1], [-1, -1, 1])
    value = array_ops.slice(hsv, [0, 0, 2], [-1, -1, 1])

    # Note that we add 2*pi to guarantee that the resulting hue is a positive
    # floating point number since delta is [-0.5, 0.5].
    hue = math_ops.mod(hue + (delta + 1.), 1.)

    hsv_altered = array_ops.concat(2, [hue, saturation, value])
    rgb_altered = gen_image_ops.hsv_to_rgb(hsv_altered)

    return convert_image_dtype(rgb_altered, orig_dtype)


def random_saturation(image, lower, upper, seed=None):
  """Adjust the saturation of an RGB image by a random factor.

  Equivalent to `adjust_saturation()` but uses a `saturation_factor` randomly
  picked in the interval `[lower, upper]`.

  Args:
    image: RGB image or images. Size of the last dimension must be 3.
    lower: float.  Lower bound for the random saturation factor.
    upper: float.  Upper bound for the random saturation factor.
    seed: An operation-specific seed. It will be used in conjunction
      with the graph-level seed to determine the real seeds that will be
      used in this operation. Please see the documentation of
      set_random_seed for its interaction with the graph-level random seed.

  Returns:
    Adjusted image(s), same shape and DType as `image`.

  Raises:
    ValueError: if `upper <= lower` or if `lower < 0`.
  """
  if upper <= lower:
    raise ValueError('upper must be > lower.')

  if lower < 0:
    raise ValueError('lower must be non-negative.')

  # Pick a float in [lower, upper]
  saturation_factor = random_ops.random_uniform([], lower, upper, seed=seed)
  return adjust_saturation(image, saturation_factor)


def adjust_saturation(image, saturation_factor, name=None):
  """Adjust saturation of an RGB image.

  This is a convenience method that converts an RGB image to float
  representation, converts it to HSV, add an offset to the saturation channel,
  converts back to RGB and then back to the original data type. If several
  adjustments are chained it is advisable to minimize the number of redundant
  conversions.

  `image` is an RGB image.  The image saturation is adjusted by converting the
  image to HSV and multiplying the saturation (S) channel by
  `saturation_factor` and clipping. The image is then converted back to RGB.

  Args:
    image: RGB image or images. Size of the last dimension must be 3.
    saturation_factor: float. Factor to multiply the saturation by.
    name: A name for this operation (optional).

  Returns:
    Adjusted image(s), same shape and DType as `image`.
  """
  with ops.op_scope([image], name, 'adjust_saturation') as name:
    image = ops.convert_to_tensor(image, name='image')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    flt_image = convert_image_dtype(image, dtypes.float32)

    hsv = gen_image_ops.rgb_to_hsv(flt_image)

    hue = array_ops.slice(hsv, [0, 0, 0], [-1, -1, 1])
    saturation = array_ops.slice(hsv, [0, 0, 1], [-1, -1, 1])
    value = array_ops.slice(hsv, [0, 0, 2], [-1, -1, 1])

    saturation *= saturation_factor
    saturation = clip_ops.clip_by_value(saturation, 0.0, 1.0)

    hsv_altered = array_ops.concat(2, [hue, saturation, value])
    rgb_altered = gen_image_ops.hsv_to_rgb(hsv_altered)

    return convert_image_dtype(rgb_altered, orig_dtype)


# TODO(irving): Remove once the C++ RandomCrop op is deprecated.
@ops.RegisterShape('RandomCrop')
def _random_crop_shape(op):
  """Shape function for RandomCrop op."""
  image_shape = op.inputs[0].get_shape().with_rank(3)
  if image_shape.ndims is not None:
    channels = image_shape[-1].value
  else:
    channels = None

  size = tensor_util.constant_value(op.inputs[1])
  if size is None:
    output_shape = [None, None, channels]
  elif size.shape == (2,):
    output_shape = [size[0], size[1], channels]
  else:
    raise ValueError('Input "size" must be a vector of two elements.')

  return [tensor_shape.TensorShape(output_shape)]


@ops.RegisterShape("ExtractGlimpse")
def _ExtractGlimpseShape(op):
  """Shape function for ExtractGlimpse op."""
  input_shape = op.inputs[0].get_shape().with_rank(4)
  unused_size_shape = op.inputs[1].get_shape().merge_with(
      tensor_shape.vector(2))
  offsets_shape = op.inputs[2].get_shape().merge_with(
      input_shape[:1].concatenate([2]))
  offsets_shape = offsets_shape
  size_value = tensor_util.constant_value(op.inputs[1])
  if size_value is not None:
    height = size_value[0]
    width = size_value[1]
  else:
    height = None
    width = None
  return [tensor_shape.TensorShape(
      [input_shape[0], height, width, input_shape[3]])]


__all__ = make_all(__name__)
# ResizeMethod is not documented, but is documented in functions that use it.
__all__.append('ResizeMethod')
