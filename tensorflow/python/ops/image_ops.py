# pylint: disable=g-short-docstring-punctuation
"""## Encoding and Decoding

TensorFlow provides Ops to decode and encode JPEG and PNG formats.  Encoded
images are represented by scalar string Tensors, decoded images by 3-D uint8
tensors of shape `[height, width, channels]`.

The encode and decode Ops apply to one image at a time.  Their input and output
are all of variable size.  If you need fixed size images, pass the output of
the decode Ops to one of the cropping and resizing Ops.

Note: The PNG encode and decode Ops support RGBA, but the conversions Ops
presently only support RGB, HSV, and GrayScale.

@@decode_jpeg
@@encode_jpeg

@@decode_png
@@encode_png

## Resizing

The resizing Ops accept input images as tensors of several types.  They always
output resized images as float32 tensors.

The convenience function [resize_images()](#resize_images) supports both 4-D
and 3-D tensors as input and output.  4-D tensors are for batches of images,
3-D tensors for individual images.

Other resizing Ops only support 3-D individual images as input:
[resize_area](#resize_area), [resize_bicubic](#resize_bicubic),
[resize_bilinear](#resize_bilinear),
[resize_nearest_neighbor](#resize_nearest_neighbor).

Example:

```python
# Decode a JPG image and resize it to 299 by 299.
image = tf.image.decode_jpeg(...)
resized_image = tf.image.resize_bilinear(image, [299, 299])
```

<i>Maybe refer to the Queue examples that show how to add images to a Queue
after resizing them to a fixed size, and how to dequeue batches of resized
images from the Queue.</i>

@@resize_images

@@resize_area
@@resize_bicubic
@@resize_bilinear
@@resize_nearest_neighbor


## Cropping

@@resize_image_with_crop_or_pad

@@pad_to_bounding_box
@@crop_to_bounding_box
@@random_crop
@@extract_glimpse

## Flipping and Transposing

@@flip_up_down
@@random_flip_up_down

@@flip_left_right
@@random_flip_left_right

@@transpose_image

## Image Adjustments

TensorFlow provides functions to adjust images in various ways: brightness,
contrast, hue, and saturation.  Each adjustment can be done with predefined
parameters or with random parameters picked from predefined intervals.  Random
adjustments are often useful to expand a training set and reduce overfitting.

@@adjust_brightness
@@random_brightness

@@adjust_contrast
@@random_contrast

@@per_image_whitening
"""
import math

import tensorflow.python.platform

from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import types
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_image_ops import *
from tensorflow.python.ops.gen_attention_ops import *
# pylint: enable=wildcard-import

ops.NoGradient('ResizeBilinear')
ops.NoGradient('RandomCrop')


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


def _Check3DImage(image):
  """Assert that we are working with properly shaped image.

  Args:
    image: 3-D Tensor of shape [height, width, channels]

  Raises:
    ValueError: if image.shape is not a [3] vector.
  """
  if not image.get_shape().is_fully_defined():
    raise ValueError('\'image\' must be fully defined.')
  if image.get_shape().ndims != 3:
    raise ValueError('\'image\' must be three-dimensional.')
  if not all(x > 0 for x in image.get_shape()):
    raise ValueError('all dims of \'image.shape\' must be > 0: %s' %
                     image.get_shape())


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
  _Check3DImage(image)
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
  _Check3DImage(image)
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
  _Check3DImage(image)
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
  _Check3DImage(image)
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
  _Check3DImage(image)
  return array_ops.transpose(image, [1, 0, 2], name='transpose_image')


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
  _Check3DImage(image)
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
  `offset_height + target_height, offset_width + target_width'.

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
  _Check3DImage(image)
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
  _Check3DImage(image)
  original_height, original_width, _ = _ImageDimensions(image)

  if target_width <= 0:
    raise ValueError('target_width must be > 0.')
  if target_height <= 0:
    raise ValueError('target_height must be > 0.')

  offset_crop_width = 0
  offset_pad_width = 0
  if target_width < original_width:
    offset_crop_width = int((original_width - target_width) / 2)
  elif target_width > original_width:
    offset_pad_width = int((target_width - original_width) / 2)

  offset_crop_height = 0
  offset_pad_height = 0
  if target_height < original_height:
    offset_crop_height = int((original_height - target_height) / 2)
  elif target_height > original_height:
    offset_pad_height = int((target_height - original_height) / 2)

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


def resize_images(images, new_height, new_width, method=ResizeMethod.BILINEAR):
  """Resize `images` to `new_width`, `new_height` using the specified `method`.

  Resized images will be distorted if their original aspect ratio is not
  the same as `new_width`, `new_height`.  To avoid distortions see
  [resize_image_with_crop_or_pad](#resize_image_with_crop_or_pad).

  `method` can be one of:

  *   <b>ResizeMethod.BILINEAR</b>: [Bilinear interpolation.]
      (https://en.wikipedia.org/wiki/Bilinear_interpolation)
  *   <b>ResizeMethod.NEAREST_NEIGHBOR</b>: [Nearest neighbor interpolation.]
      (https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
  *   <b>ResizeMethod.BICUBIC</b>: [Bicubic interpolation.]
      (https://en.wikipedia.org/wiki/Bicubic_interpolation)
  *   <b>ResizeMethod.AREA</b>: Area interpolation.

  Args:
    images: 4-D Tensor of shape `[batch, height, width, channels]` or
            3-D Tensor of shape `[height, width, channels]`.
    new_height: integer.
    new_width: integer.
    method: ResizeMethod.  Defaults to `ResizeMethod.BILINEAR`.

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
  if images.get_shape().ndims is None:
    raise ValueError('\'images\' contains no shape.')
  # TODO(shlens): Migrate this functionality to the underlying Op's.
  is_batch = True
  if len(images.get_shape()) == 3:
    is_batch = False
    images = array_ops.expand_dims(images, 0)

  _, height, width, depth = _ImageDimensions(images)

  if width == new_width and height == new_height:
    return images

  if method == ResizeMethod.BILINEAR:
    images = gen_image_ops.resize_bilinear(images, [new_height, new_width])
  elif method == ResizeMethod.NEAREST_NEIGHBOR:
    images = gen_image_ops.resize_nearest_neighbor(images, [new_height,
                                                            new_width])
  elif method == ResizeMethod.BICUBIC:
    images = gen_image_ops.resize_bicubic(images, [new_height, new_width])
  elif method == ResizeMethod.AREA:
    images = gen_image_ops.resize_area(images, [new_height, new_width])
  else:
    raise ValueError('Resize method is not implemented.')

  if not is_batch:
    images = array_ops.reshape(images, [new_height, new_width, depth])
  return images


def per_image_whitening(image):
  """Linearly scales `image` to have zero mean and unit norm.

  This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
  of all values in image, and
  `adjusted_stddev = max(stddev, 1.0/srqt(image.NumElements()))`.

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
  _Check3DImage(image)
  height, width, depth = _ImageDimensions(image)
  num_pixels = height * width * depth

  image = math_ops.cast(image, dtype=types.float32)
  image_mean = math_ops.reduce_mean(image)

  variance = (math_ops.reduce_mean(math_ops.square(image)) -
              math_ops.square(image_mean))
  stddev = math_ops.sqrt(variance)

  # Apply a minimum normalization that protects us against uniform images.
  min_stddev = constant_op.constant(1.0 / math.sqrt(num_pixels))
  pixel_value_scale = math_ops.maximum(stddev, min_stddev)
  pixel_value_offset = image_mean

  image = math_ops.sub(image, pixel_value_offset)
  image = math_ops.div(image, pixel_value_scale)
  return image


def random_brightness(image, max_delta, seed=None):
  """Adjust the brightness of images by a random factor.

  Equivalent to `adjust_brightness()` using a `delta` randomly picked in the
  interval `[-max_delta, max_delta)`.

  Note that `delta` is picked as a float. Because for integer type images,
  the brightness adjusted result is rounded before casting, integer images may
  have modifications in the range `[-max_delta,max_delta]`.

  Args:
    image: 3-D tensor of shape `[height, width, channels]`.
    max_delta: float, must be non-negative.
    seed: A Python integer. Used to create a random seed. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.

  Returns:
    3-D tensor of images of shape `[height, width, channels]`

  Raises:
    ValueError: if max_delta is negative.
  """
  _Check3DImage(image)

  if max_delta < 0:
    raise ValueError('max_delta must be non-negative.')

  delta = random_ops.random_uniform([], -max_delta, max_delta, seed=seed)
  return adjust_brightness(image, delta)


def random_contrast(image, lower, upper, seed=None):
  """Adjust the contrase of an image by a random factor.

  Equivalent to `adjust_constrast()` but uses a `contrast_factor` randomly
  picked in the interval `[lower, upper]`.

  Args:
    image: 3-D tensor of shape `[height, width, channels]`.
    lower: float.  Lower bound for the random contrast factor.
    upper: float.  Upper bound for the random contrast factor.
    seed: A Python integer. Used to create a random seed. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.

  Returns:
    3-D tensor of shape `[height, width, channels]`.

  Raises:
    ValueError: if `upper <= lower` or if `lower < 0`.
  """
  _Check3DImage(image)

  if upper <= lower:
    raise ValueError('upper must be > lower.')

  if lower < 0:
    raise ValueError('lower must be non-negative.')

  # Generate an a float in [lower, upper]
  contrast_factor = random_ops.random_uniform([], lower, upper, seed=seed)
  return adjust_contrast(image, contrast_factor)


def adjust_brightness(image, delta, min_value=None, max_value=None):
  """Adjust the brightness of RGB or Grayscale images.

  The value `delta` is added to all components of the tensor `image`. `image`
  and `delta` are cast to `float` before adding, and the resulting values are
  clamped to `[min_value, max_value]`. Finally, the result is cast back to
  `images.dtype`.

  If `min_value` or `max_value` are not given, they are set to the minimum and
  maximum allowed values for `image.dtype` respectively.

  Args:
    image: A tensor.
    delta: A scalar. Amount to add to the pixel values.
    min_value: Minimum value for output.
    max_value: Maximum value for output.

  Returns:
    A tensor of the same shape and type as `image`.
  """
  if min_value is None:
    min_value = image.dtype.min
  if max_value is None:
    max_value = image.dtype.max

  with ops.op_scope([image, delta, min_value, max_value], None,
                    'adjust_brightness') as name:
    adjusted = math_ops.add(
        math_ops.cast(image, types.float32),
        math_ops.cast(delta, types.float32),
        name=name)
    if image.dtype.is_integer:
      rounded = math_ops.round(adjusted)
    else:
      rounded = adjusted
    clipped = clip_ops.clip_by_value(rounded, float(min_value),
                                     float(max_value))
    output = math_ops.cast(clipped, image.dtype)
    return output


def adjust_contrast(images, contrast_factor, min_value=None, max_value=None):
  """Adjust contrast of RGB or grayscale images.

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

  Args:
    images: Images to adjust.  At least 3-D.
    contrast_factor: A float multiplier for adjusting contrast.
    min_value: Minimum value for clipping the adjusted pixels.
    max_value: Maximum value for clipping the adjusted pixels.

  Returns:
    The constrast-adjusted image or images.

  Raises:
    ValueError: if the arguments are invalid.
  """
  _CheckAtLeast3DImage(images)

  # If these are None, the min/max should be a nop, but still prevent overflows
  # from the cast back to images.dtype at the end of adjust_contrast.
  if min_value is None:
    min_value = images.dtype.min
  if max_value is None:
    max_value = images.dtype.max

  with ops.op_scope(
      [images, contrast_factor, min_value,
       max_value], None, 'adjust_contrast') as name:
    adjusted = gen_image_ops.adjust_contrast(images,
                                             contrast_factor=contrast_factor,
                                             min_value=min_value,
                                             max_value=max_value,
                                             name=name)
    if images.dtype.is_integer:
      return math_ops.cast(math_ops.round(adjusted), images.dtype)
    else:
      return math_ops.cast(adjusted, images.dtype)


ops.RegisterShape('AdjustContrast')(
    common_shapes.unchanged_shape_with_rank_at_least(3))


@ops.RegisterShape('ResizeBilinear')
@ops.RegisterShape('ResizeNearestNeighbor')
@ops.RegisterShape('ResizeBicubic')
@ops.RegisterShape('ResizeArea')
def _ResizeShape(op):
  """Shape function for the resize_bilinear and resize_nearest_neighbor ops."""
  input_shape = op.inputs[0].get_shape().with_rank(4)
  size = tensor_util.ConstantValue(op.inputs[1])
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


@ops.RegisterShape('RandomCrop')
def _random_cropShape(op):
  """Shape function for the random_crop op."""
  input_shape = op.inputs[0].get_shape().with_rank(3)
  unused_size_shape = op.inputs[1].get_shape().merge_with(
      tensor_shape.vector(2))
  size = tensor_util.ConstantValue(op.inputs[1])
  if size is not None:
    height = size[0]
    width = size[1]
  else:
    height = None
    width = None
  channels = input_shape[2]
  return [tensor_shape.TensorShape([height, width, channels])]


def random_crop(image, size, seed=None, name=None):
  """Randomly crops `image` to size `[target_height, target_width]`.

  The offset of the output within `image` is uniformly random. `image` always
  fully contains the result.

  Args:
    image: 3-D tensor of shape `[height, width, channels]`
    size: 1-D tensor with two elements, specifying target `[height, width]`
    seed: A Python integer. Used to create a random seed. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for this operation (optional).

  Returns:
    A cropped 3-D tensor of shape `[target_height, target_width, channels]`.
  """
  seed1, seed2 = random_seed.get_seed(seed)
  return gen_image_ops.random_crop(image, size, seed=seed1, seed2=seed2,
                                   name=name)
