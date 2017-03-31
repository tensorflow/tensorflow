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

"""Implementation of image ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables


ops.NotDifferentiable('RandomCrop')
# TODO(b/31222613): This op may be differentiable, and there may be
# latent bugs here.
ops.NotDifferentiable('RGBToHSV')
# TODO(b/31222613): This op may be differentiable, and there may be
# latent bugs here.
ops.NotDifferentiable('HSVToRGB')
ops.NotDifferentiable('DrawBoundingBoxes')
ops.NotDifferentiable('SampleDistortedBoundingBox')
# TODO(bsteiner): Implement the gradient function for extract_glimpse
# TODO(b/31222613): This op may be differentiable, and there may be
# latent bugs here.
ops.NotDifferentiable('ExtractGlimpse')
ops.NotDifferentiable('NonMaxSuppression')


def _assert(cond, ex_type, msg):
  """A polymorphic assert, works with tensors and boolean expressions.

  If `cond` is not a tensor, behave like an ordinary assert statement, except
  that a empty list is returned. If `cond` is a tensor, return a list
  containing a single TensorFlow assert op.

  Args:
    cond: Something evaluates to a boolean value. May be a tensor.
    ex_type: The exception class to use.
    msg: The error message.

  Returns:
    A list, containing at most one assert op.
  """
  if _is_tensor(cond):
    return [control_flow_ops.Assert(cond, [msg])]
  else:
    if not cond:
      raise ex_type(msg)
    else:
      return []


def _is_tensor(x):
  """Returns `True` if `x` is a symbolic tensor-like object.

  Args:
    x: A python object to check.

  Returns:
    `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.
  """
  return isinstance(x, (ops.Tensor, variables.Variable))


def _ImageDimensions(image, rank):
  """Returns the dimensions of an image tensor.

  Args:
    image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
    rank: The expected rank of the image

  Returns:
    A list of corresponding to the dimensions of the
    input image.  Dimensions that are statically known are python integers,
    otherwise they are integer scalar tensors.
  """
  if image.get_shape().is_fully_defined():
    return image.get_shape().as_list()
  else:
    static_shape = image.get_shape().with_rank(rank).as_list()
    dynamic_shape = array_ops.unstack(array_ops.shape(image), rank)
    return [s if s is not None else d
            for s, d in zip(static_shape, dynamic_shape)]


def _Check3DImage(image, require_static=True):
  """Assert that we are working with properly shaped image.

  Args:
    image: 3-D Tensor of shape [height, width, channels]
    require_static: If `True`, requires that all dimensions of `image` are
      known and non-zero.

  Raises:
    ValueError: if `image.shape` is not a 3-vector.

  Returns:
    An empty list, if `image` has fully defined dimensions. Otherwise, a list
    containing an assert op is returned.
  """
  try:
    image_shape = image.get_shape().with_rank(3)
  except ValueError:
    raise ValueError("'image' (shape %s) must be three-dimensional." %
                     image.shape)
  if require_static and not image_shape.is_fully_defined():
    raise ValueError("'image' (shape %s) must be fully defined." %
                     image_shape)
  if any(x == 0 for x in image_shape):
    raise ValueError("all dims of 'image.shape' must be > 0: %s" %
                     image_shape)
  if not image_shape.is_fully_defined():
    return [check_ops.assert_positive(array_ops.shape(image),
                                      ["all dims of 'image.shape' "
                                       "must be > 0."])]
  else:
    return []


def _CheckAtLeast3DImage(image, require_static=True):
  """Assert that we are working with properly shaped image.

  Args:
    image: >= 3-D Tensor of size [*, height, width, depth]
    require_static: If `True`, requires that all dimensions of `image` are
      known and non-zero.

  Raises:
    ValueError: if image.shape is not a [>= 3] vector.

  Returns:
    An empty list, if `image` has fully defined dimensions. Otherwise, a list
    containing an assert op is returned.
  """
  try:
    if image.get_shape().ndims is None:
      image_shape = image.get_shape().with_rank(3)
    else:
      image_shape = image.get_shape().with_rank_at_least(3)
  except ValueError:
    raise ValueError("'image' must be at least three-dimensional.")
  if require_static and not image_shape.is_fully_defined():
    raise ValueError('\'image\' must be fully defined.')
  if any(x == 0 for x in image_shape):
    raise ValueError('all dims of \'image.shape\' must be > 0: %s' %
                     image_shape)
  if not image_shape.is_fully_defined():
    return [check_ops.assert_positive(array_ops.shape(image),
                                      ["all dims of 'image.shape' "
                                       "must be > 0."])]
  else:
    return []


def fix_image_flip_shape(image, result):
  """Set the shape to 3 dimensional if we don't know anything else.

  Args:
    image: original image size
    result: flipped or transformed image

  Returns:
    An image whose shape is at least None,None,None.
  """

  image_shape = image.get_shape()
  if image_shape == tensor_shape.unknown_shape():
    result.set_shape([None, None, None])
  else:
    result.set_shape(image_shape)
  return result


def random_flip_up_down(image, seed=None):
  """Randomly flips an image vertically (upside down).

  With a 1 in 2 chance, outputs the contents of `image` flipped along the first
  dimension, which is `height`.  Otherwise output the image as-is.

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    seed: A Python integer. Used to create a random seed. See
      @{tf.set_random_seed}
      for behavior.

  Returns:
    A 3-D tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  image = ops.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=False)
  uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
  mirror_cond = math_ops.less(uniform_random, .5)
  result = control_flow_ops.cond(mirror_cond,
                                 lambda: array_ops.reverse(image, [0]),
                                 lambda: image)
  return fix_image_flip_shape(image, result)


def random_flip_left_right(image, seed=None):
  """Randomly flip an image horizontally (left to right).

  With a 1 in 2 chance, outputs the contents of `image` flipped along the
  second dimension, which is `width`.  Otherwise output the image as-is.

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    seed: A Python integer. Used to create a random seed. See
      @{tf.set_random_seed}
      for behavior.

  Returns:
    A 3-D tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  image = ops.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=False)
  uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
  mirror_cond = math_ops.less(uniform_random, .5)
  result = control_flow_ops.cond(mirror_cond,
                                 lambda: array_ops.reverse(image, [1]),
                                 lambda: image)
  return fix_image_flip_shape(image, result)


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
  return fix_image_flip_shape(image, array_ops.reverse(image, [1]))


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
  return fix_image_flip_shape(image, array_ops.reverse(image, [0]))


def rot90(image, k=1, name=None):
  """Rotate an image counter-clockwise by 90 degrees.

  Args:
    image: A 3-D tensor of shape `[height, width, channels]`.
    k: A scalar integer. The number of times the image is rotated by 90 degrees.
    name: A name for this operation (optional).

  Returns:
    A rotated 3-D tensor of the same type and shape as `image`.
  """
  with ops.name_scope(name, 'rot90', [image, k]) as scope:
    image = ops.convert_to_tensor(image, name='image')
    _Check3DImage(image, require_static=False)
    k = ops.convert_to_tensor(k, dtype=dtypes.int32, name='k')
    k.get_shape().assert_has_rank(0)
    k = math_ops.mod(k, 4)

    def _rot90():
      return array_ops.transpose(array_ops.reverse_v2(image, [1]),
                                 [1, 0, 2])
    def _rot180():
      return array_ops.reverse_v2(image, [0, 1])
    def _rot270():
      return array_ops.reverse_v2(array_ops.transpose(image, [1, 0, 2]),
                                  [1])
    cases = [(math_ops.equal(k, 1), _rot90),
             (math_ops.equal(k, 2), _rot180),
             (math_ops.equal(k, 3), _rot270)]

    ret = control_flow_ops.case(cases, default=lambda: image, exclusive=True,
                                name=scope)
    ret.set_shape([None, None, image.get_shape()[2]])
    return ret


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

  bbox_begin = array_ops.stack([bbox_h_start, bbox_w_start, 0])
  bbox_size = array_ops.stack([bbox_h_size, bbox_w_size, -1])
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
    image: 4-D Tensor of shape `[batch, height, width, channels]` or
           3-D Tensor of shape `[height, width, channels]`.
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.

  Returns:
    If `image` was 4-D, a 4-D float Tensor of shape
    `[batch, target_height, target_width, channels]`
    If `image` was 3-D, a 3-D float Tensor of shape
    `[target_height, target_width, channels]`

  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
      `target_*` arguments, or either `offset_height` or `offset_width` is
      negative.
  """
  image = ops.convert_to_tensor(image, name='image')

  assert_ops = []
  assert_ops += _CheckAtLeast3DImage(image, require_static=False)

  is_batch = True
  image_shape = image.get_shape()
  if image_shape.ndims == 3:
    is_batch = False
    image = array_ops.expand_dims(image, 0)
  elif image_shape.ndims is None:
    is_batch = False
    image = array_ops.expand_dims(image, 0)
    image.set_shape([None] * 4)
  elif image_shape.ndims != 4:
    raise ValueError('\'image\' must have either 3 or 4 dimensions.')

  batch, height, width, depth = _ImageDimensions(image, rank=4)

  after_padding_width = target_width - offset_width - width
  after_padding_height = target_height - offset_height - height

  assert_ops += _assert(offset_height >= 0, ValueError,
                        'offset_height must be >= 0')
  assert_ops += _assert(offset_width >= 0, ValueError,
                        'offset_width must be >= 0')
  assert_ops += _assert(after_padding_width >= 0, ValueError,
                        'width must be <= target - offset')
  assert_ops += _assert(after_padding_height >= 0, ValueError,
                        'height must be <= target - offset')
  image = control_flow_ops.with_dependencies(assert_ops, image)

  # Do not pad on the depth dimensions.
  paddings = array_ops.reshape(
      array_ops.stack([
          0, 0, offset_height, after_padding_height, offset_width,
          after_padding_width, 0, 0
      ]), [4, 2])
  padded = array_ops.pad(image, paddings)

  padded_shape = [None if _is_tensor(i) else i
                  for i in [batch, target_height, target_width, depth]]
  padded.set_shape(padded_shape)

  if not is_batch:
    padded = array_ops.squeeze(padded, squeeze_dims=[0])

  return padded


def crop_to_bounding_box(image, offset_height, offset_width, target_height,
                         target_width):
  """Crops an image to a specified bounding box.

  This op cuts a rectangular part out of `image`. The top-left corner of the
  returned image is at `offset_height, offset_width` in `image`, and its
  lower-right corner is at
  `offset_height + target_height, offset_width + target_width`.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or
           3-D Tensor of shape `[height, width, channels]`.
    offset_height: Vertical coordinate of the top-left corner of the result in
                   the input.
    offset_width: Horizontal coordinate of the top-left corner of the result in
                  the input.
    target_height: Height of the result.
    target_width: Width of the result.

  Returns:
    If `image` was 4-D, a 4-D float Tensor of shape
    `[batch, target_height, target_width, channels]`
    If `image` was 3-D, a 3-D float Tensor of shape
    `[target_height, target_width, channels]`

  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
      `target_*` arguments, or either `offset_height` or `offset_width` is
      negative, or either `target_height` or `target_width` is not positive.
  """
  image = ops.convert_to_tensor(image, name='image')

  assert_ops = []
  assert_ops += _CheckAtLeast3DImage(image, require_static=False)

  is_batch = True
  image_shape = image.get_shape()
  if image_shape.ndims == 3:
    is_batch = False
    image = array_ops.expand_dims(image, 0)
  elif image_shape.ndims is None:
    is_batch = False
    image = array_ops.expand_dims(image, 0)
    image.set_shape([None] * 4)
  elif image_shape.ndims != 4:
    raise ValueError('\'image\' must have either 3 or 4 dimensions.')

  batch, height, width, depth = _ImageDimensions(image, rank=4)

  assert_ops += _assert(offset_width >= 0, ValueError,
                        'offset_width must be >= 0.')
  assert_ops += _assert(offset_height >= 0, ValueError,
                        'offset_height must be >= 0.')
  assert_ops += _assert(target_width > 0, ValueError,
                        'target_width must be > 0.')
  assert_ops += _assert(target_height > 0, ValueError,
                        'target_height must be > 0.')
  assert_ops += _assert(width >= (target_width + offset_width), ValueError,
                        'width must be >= target + offset.')
  assert_ops += _assert(height >= (target_height + offset_height), ValueError,
                        'height must be >= target + offset.')
  image = control_flow_ops.with_dependencies(assert_ops, image)

  cropped = array_ops.slice(
      image,
      array_ops.stack([0, offset_height, offset_width, 0]),
      array_ops.stack([-1, target_height, target_width, -1]))

  cropped_shape = [None if _is_tensor(i) else i
                   for i in [batch, target_height, target_width, depth]]
  cropped.set_shape(cropped_shape)

  if not is_batch:
    cropped = array_ops.squeeze(cropped, squeeze_dims=[0])

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
    image: 4-D Tensor of shape `[batch, height, width, channels]` or
           3-D Tensor of shape `[height, width, channels]`.
    target_height: Target height.
    target_width: Target width.

  Raises:
    ValueError: if `target_height` or `target_width` are zero or negative.

  Returns:
    Cropped and/or padded image.
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """
  image = ops.convert_to_tensor(image, name='image')
  image_shape = image.get_shape()
  is_batch = True
  if image_shape.ndims == 3:
    is_batch = False
    image = array_ops.expand_dims(image, 0)
  elif image_shape.ndims is None:
    is_batch = False
    image = array_ops.expand_dims(image, 0)
    image.set_shape([None] * 4)
  elif image_shape.ndims != 4:
    raise ValueError('\'image\' must have either 3 or 4 dimensions.')

  assert_ops = []
  assert_ops += _CheckAtLeast3DImage(image, require_static=False)
  assert_ops += _assert(target_width > 0, ValueError,
                        'target_width must be > 0.')
  assert_ops += _assert(target_height > 0, ValueError,
                        'target_height must be > 0.')

  image = control_flow_ops.with_dependencies(assert_ops, image)
  # `crop_to_bounding_box` and `pad_to_bounding_box` have their own checks.
  # Make sure our checks come first, so that error messages are clearer.
  if _is_tensor(target_height):
    target_height = control_flow_ops.with_dependencies(
      assert_ops, target_height)
  if _is_tensor(target_width):
    target_width = control_flow_ops.with_dependencies(assert_ops, target_width)

  def max_(x, y):
    if _is_tensor(x) or _is_tensor(y):
      return math_ops.maximum(x, y)
    else:
      return max(x, y)

  def min_(x, y):
    if _is_tensor(x) or _is_tensor(y):
      return math_ops.minimum(x, y)
    else:
      return min(x, y)

  def equal_(x, y):
    if _is_tensor(x) or _is_tensor(y):
      return math_ops.equal(x, y)
    else:
      return x == y

  _, height, width, _ = _ImageDimensions(image, rank=4)
  width_diff = target_width - width
  offset_crop_width = max_(-width_diff // 2, 0)
  offset_pad_width = max_(width_diff // 2, 0)

  height_diff = target_height - height
  offset_crop_height = max_(-height_diff // 2, 0)
  offset_pad_height = max_(height_diff // 2, 0)

  # Maybe crop if needed.
  cropped = crop_to_bounding_box(image, offset_crop_height, offset_crop_width,
                                 min_(target_height, height),
                                 min_(target_width, width))

  # Maybe pad if needed.
  resized = pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                target_height, target_width)

  # In theory all the checks below are redundant.
  if resized.get_shape().ndims is None:
    raise ValueError('resized contains no shape.')

  _, resized_height, resized_width, _ = _ImageDimensions(resized, rank=4)

  assert_ops = []
  assert_ops += _assert(equal_(resized_height, target_height), ValueError,
                        'resized height is not correct.')
  assert_ops += _assert(equal_(resized_width, target_width), ValueError,
                        'resized width is not correct.')

  resized = control_flow_ops.with_dependencies(assert_ops, resized)

  if not is_batch:
    resized = array_ops.squeeze(resized, squeeze_dims=[0])

  return resized


class ResizeMethod(object):
  BILINEAR = 0
  NEAREST_NEIGHBOR = 1
  BICUBIC = 2
  AREA = 3


def resize_images(images,
                  size,
                  method=ResizeMethod.BILINEAR,
                  align_corners=False):
  """Resize `images` to `size` using the specified `method`.

  Resized images will be distorted if their original aspect ratio is not
  the same as `size`.  To avoid distortions see
  @{tf.image.resize_image_with_crop_or_pad}.

  `method` can be one of:

  *   <b>`ResizeMethod.BILINEAR`</b>: [Bilinear interpolation.](https://en.wikipedia.org/wiki/Bilinear_interpolation)
  *   <b>`ResizeMethod.NEAREST_NEIGHBOR`</b>: [Nearest neighbor interpolation.](https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
  *   <b>`ResizeMethod.BICUBIC`</b>: [Bicubic interpolation.](https://en.wikipedia.org/wiki/Bicubic_interpolation)
  *   <b>`ResizeMethod.AREA`</b>: Area interpolation.

  Args:
    images: 4-D Tensor of shape `[batch, height, width, channels]` or
            3-D Tensor of shape `[height, width, channels]`.
    size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
          new size for the images.
    method: ResizeMethod.  Defaults to `ResizeMethod.BILINEAR`.
    align_corners: bool. If true, exactly align all 4 corners of the input and
                   output. Defaults to `false`.

  Raises:
    ValueError: if the shape of `images` is incompatible with the
      shape arguments to this function
    ValueError: if `size` has invalid shape or type.
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
  if images.get_shape().ndims == 3:
    is_batch = False
    images = array_ops.expand_dims(images, 0)
  elif images.get_shape().ndims != 4:
    raise ValueError('\'images\' must have either 3 or 4 dimensions.')

  _, height, width, _ = images.get_shape().as_list()

  try:
    size = ops.convert_to_tensor(size, dtypes.int32, name='size')
  except (TypeError, ValueError):
    raise ValueError('\'size\' must be a 1-D int32 Tensor')
  if not size.get_shape().is_compatible_with([2]):
    raise ValueError('\'size\' must be a 1-D Tensor of 2 elements: '
                     'new_height, new_width')
  size_const_as_shape = tensor_util.constant_value_as_shape(size)
  new_height_const = size_const_as_shape[0].value
  new_width_const = size_const_as_shape[1].value

  # If we can determine that the height and width will be unmodified by this
  # transformation, we avoid performing the resize.
  if all(x is not None
         for x in [new_width_const, width, new_height_const, height]) and (
             width == new_width_const and height == new_height_const):
    if not is_batch:
      images = array_ops.squeeze(images, squeeze_dims=[0])
    return images

  if method == ResizeMethod.BILINEAR:
    images = gen_image_ops.resize_bilinear(images,
                                           size,
                                           align_corners=align_corners)
  elif method == ResizeMethod.NEAREST_NEIGHBOR:
    images = gen_image_ops.resize_nearest_neighbor(images,
                                                   size,
                                                   align_corners=align_corners)
  elif method == ResizeMethod.BICUBIC:
    images = gen_image_ops.resize_bicubic(images,
                                          size,
                                          align_corners=align_corners)
  elif method == ResizeMethod.AREA:
    images = gen_image_ops.resize_area(images,
                                       size,
                                       align_corners=align_corners)
  else:
    raise ValueError('Resize method is not implemented.')

  # NOTE(mrry): The shape functions for the resize ops cannot unpack
  # the packed values in `new_size`, so set the shape here.
  images.set_shape([None, new_height_const, new_width_const, None])

  if not is_batch:
    images = array_ops.squeeze(images, squeeze_dims=[0])
  return images


def per_image_standardization(image):
  """Linearly scales `image` to have zero mean and unit norm.

  This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
  of all values in image, and
  `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.

  `stddev` is the standard deviation of all values in `image`. It is capped
  away from zero to protect against division by 0 when handling uniform images.

  Args:
    image: 3-D tensor of shape `[height, width, channels]`.

  Returns:
    The standardized image with same shape as `image`.

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
  min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, dtypes.float32))
  pixel_value_scale = math_ops.maximum(stddev, min_stddev)
  pixel_value_offset = image_mean

  image = math_ops.subtract(image, pixel_value_offset)
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
      @{tf.set_random_seed}
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
      @{tf.set_random_seed}
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
  with ops.name_scope(None, 'adjust_brightness', [image, delta]) as name:
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
  with ops.name_scope(None, 'adjust_contrast',
                      [images, contrast_factor]) as name:
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


def adjust_gamma(image, gamma=1, gain=1):
  """Performs Gamma Correction on the input image.
    Also known as Power Law Transform. This function transforms the
    input image pixelwise according to the equation Out = In**gamma
    after scaling each pixel to the range 0 to 1.

  Args:
    image : A Tensor.
    gamma : A scalar. Non negative real number.
    gain  : A scalar. The constant multiplier.

  Returns:
    A Tensor. Gamma corrected output image.

  Notes:
    For gamma greater than 1, the histogram will shift towards left and
    the output image will be darker than the input image.
    For gamma less than 1, the histogram will shift towards right and
    the output image will be brighter than the input image.

  References:
    [1] http://en.wikipedia.org/wiki/Gamma_correction
  """

  with ops.op_scope([image, gamma, gain], None, 'adjust_gamma') as name:
    # Convert pixel value to DT_FLOAT for computing adjusted image
    img = ops.convert_to_tensor(image, name='img', dtype=dtypes.float32)
    # Keep image dtype for computing the scale of corresponding dtype
    image = ops.convert_to_tensor(image, name='image')

    if gamma < 0:
      raise ValueError("Gamma should be a non-negative real number")
    # scale = max(dtype) - min(dtype)
    scale = constant_op.constant(image.dtype.limits[1] - image.dtype.limits[0], dtype=dtypes.float32)
    # According to the definition of gamma correction
    adjusted_img = (img / scale) ** gamma * scale * gain

    return adjusted_img


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

  with ops.name_scope(name, 'convert_image', [image]) as name:
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
        return math_ops.multiply(cast, scale, name=name)
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
        return math_ops.multiply(cast, scale, name=name)
      else:
        # Converting from float: first scale, then cast
        scale = dtype.max + 0.5  # avoid rounding problems in the cast
        scaled = math_ops.multiply(image, scale)
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
  with ops.name_scope(name, 'rgb_to_grayscale', [images]) as name:
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
  with ops.name_scope(name, 'grayscale_to_rgb', [images]) as name:
    images = ops.convert_to_tensor(images, name='images')
    rank_1 = array_ops.expand_dims(array_ops.rank(images) - 1, 0)
    shape_list = (
        [array_ops.ones(rank_1,
                        dtype=dtypes.int32)] + [array_ops.expand_dims(3, 0)])
    multiples = array_ops.concat(shape_list, 0)
    rgb = array_ops.tile(images, multiples, name=name)
    rgb.set_shape(images.get_shape()[:-1].concatenate([3]))
    return rgb


# pylint: disable=invalid-name
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
  with ops.name_scope(name, 'adjust_hue', [image]) as name:
    image = ops.convert_to_tensor(image, name='image')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    flt_image = convert_image_dtype(image, dtypes.float32)

    # TODO(zhengxq): we will switch to the fused version after we add a GPU
    # kernel for that.
    fused = os.environ.get('TF_ADJUST_HUE_FUSED', '')
    fused = fused.lower() in ('true', 't', '1')

    if not fused:
      hsv = gen_image_ops.rgb_to_hsv(flt_image)

      hue = array_ops.slice(hsv, [0, 0, 0], [-1, -1, 1])
      saturation = array_ops.slice(hsv, [0, 0, 1], [-1, -1, 1])
      value = array_ops.slice(hsv, [0, 0, 2], [-1, -1, 1])

      # Note that we add 2*pi to guarantee that the resulting hue is a positive
      # floating point number since delta is [-0.5, 0.5].
      hue = math_ops.mod(hue + (delta + 1.), 1.)

      hsv_altered = array_ops.concat([hue, saturation, value], 2)
      rgb_altered = gen_image_ops.hsv_to_rgb(hsv_altered)
    else:
      rgb_altered = gen_image_ops.adjust_hue(flt_image, delta)

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
  with ops.name_scope(name, 'adjust_saturation', [image]) as name:
    image = ops.convert_to_tensor(image, name='image')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    flt_image = convert_image_dtype(image, dtypes.float32)

    # TODO(zhengxq): we will switch to the fused version after we add a GPU
    # kernel for that.
    fused = os.environ.get('TF_ADJUST_SATURATION_FUSED', '')
    fused = fused.lower() in ('true', 't', '1')

    if fused:
      return convert_image_dtype(
          gen_image_ops.adjust_saturation(flt_image, saturation_factor),
          orig_dtype)

    hsv = gen_image_ops.rgb_to_hsv(flt_image)

    hue = array_ops.slice(hsv, [0, 0, 0], [-1, -1, 1])
    saturation = array_ops.slice(hsv, [0, 0, 1], [-1, -1, 1])
    value = array_ops.slice(hsv, [0, 0, 2], [-1, -1, 1])

    saturation *= saturation_factor
    saturation = clip_ops.clip_by_value(saturation, 0.0, 1.0)

    hsv_altered = array_ops.concat([hue, saturation, value], 2)
    rgb_altered = gen_image_ops.hsv_to_rgb(hsv_altered)

    return convert_image_dtype(rgb_altered, orig_dtype)


def decode_image(contents, channels=None, name=None):
  """Convenience function for `decode_gif`, `decode_jpeg`, and `decode_png`.
  Detects whether an image is a GIF, JPEG, or PNG, and performs the appropriate
  operation to convert the input bytes `string` into a `Tensor` of type `uint8`.

  Note: `decode_gif` returns a 4-D array `[num_frames, height, width, 3]`, as
  opposed to `decode_jpeg` and `decode_png`, which return 3-D arrays
  `[height, width, num_channels]`. Make sure to take this into account when
  constructing your graph if you are intermixing GIF files with JPEG and/or PNG
  files.

  Args:
    contents: 0-D `string`. The encoded image bytes.
    channels: An optional `int`. Defaults to `0`. Number of color channels for
      the decoded image.
    name: A name for the operation (optional)

  Returns:
    `Tensor` with type `uint8` with shape `[height, width, num_channels]` for
      JPEG and PNG images and shape `[num_frames, height, width, 3]` for GIF
      images.
  """
  with ops.name_scope(name, 'decode_image') as scope:
    if channels not in (None, 0, 1, 3):
      raise ValueError('channels must be in (None, 0, 1, 3)')
    substr = string_ops.substr(contents, 0, 4)

    def _gif():
      # Create assert op to check that bytes are GIF decodable
      is_gif = math_ops.equal(substr, b'\x47\x49\x46\x38', name='is_gif')
      decode_msg = 'Unable to decode bytes as JPEG, PNG, or GIF'
      assert_decode = control_flow_ops.Assert(is_gif, [decode_msg])
      # Create assert to make sure that channels is not set to 1
      # Already checked above that channels is in (None, 0, 1, 3)
      gif_channels = 0 if channels is None else channels
      good_channels = math_ops.not_equal(gif_channels, 1, name='check_channels')
      channels_msg = 'Channels must be in (None, 0, 3) when decoding GIF images'
      assert_channels = control_flow_ops.Assert(good_channels, [channels_msg])
      with ops.control_dependencies([assert_decode, assert_channels]):
        return gen_image_ops.decode_gif(contents)

    def _png():
      return gen_image_ops.decode_png(contents, channels)

    def check_png():
      is_png = math_ops.equal(substr, b'\211PNG', name='is_png')
      return control_flow_ops.cond(is_png, _png, _gif, name='cond_png')

    def _jpeg():
      return gen_image_ops.decode_jpeg(contents, channels)

    is_jpeg = math_ops.equal(substr, b'\xff\xd8\xff\xe0', name='is_jpeg')
    return control_flow_ops.cond(is_jpeg, _jpeg, check_png, name='cond_jpeg')


def total_variation(images, name=None):
  """Calculate and return the total variation for one or more images.

  The total variation is the sum of the absolute differences for neighboring
  pixel-values in the input images. This measures how much noise is in the
  images.

  This can be used as a loss-function during optimization so as to suppress
  noise in images. If you have a batch of images, then you should calculate
  the scalar loss-value as the sum:
  `loss = tf.reduce_sum(tf.image.total_variation(images))`

  This implements the anisotropic 2-D version of the formula described here:

  https://en.wikipedia.org/wiki/Total_variation_denoising

  Args:
    images: 4-D Tensor of shape `[batch, height, width, channels]` or
            3-D Tensor of shape `[height, width, channels]`.

    name: A name for the operation (optional).

  Raises:
    ValueError: if images.shape is not a 3-D or 4-D vector.

  Returns:
    The total variation of `images`.

    If `images` was 4-D, return a 1-D float Tensor of shape `[batch]` with the
    total variation for each image in the batch.
    If `images` was 3-D, return a scalar float with the total variation for
    that image.
  """

  with ops.name_scope(name, 'total_variation'):
    ndims = images.get_shape().ndims

    if ndims == 3:
      # The input is a single image with shape [height, width, channels].

      # Calculate the difference of neighboring pixel-values.
      # The images are shifted one pixel along the height and width by slicing.
      pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
      pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]

      # Sum for all axis. (None is an alias for all axis.)
      sum_axis = None
    elif ndims == 4:
      # The input is a batch of images with shape:
      # [batch, height, width, channels].

      # Calculate the difference of neighboring pixel-values.
      # The images are shifted one pixel along the height and width by slicing.
      pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
      pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]

      # Only sum for the last 3 axis.
      # This results in a 1-D tensor with the total variation for each image.
      sum_axis = [1, 2, 3]
    else:
      raise ValueError('\'images\' must be either 3 or 4-dimensional.')

    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tot_var = math_ops.reduce_sum(math_ops.abs(pixel_dif1), axis=sum_axis) + \
              math_ops.reduce_sum(math_ops.abs(pixel_dif2), axis=sum_axis)

  return tot_var
