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

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

ops.NotDifferentiable('RandomCrop')
# TODO(b/31222613): This op may be differentiable, and there may be
# latent bugs here.
ops.NotDifferentiable('HSVToRGB')
ops.NotDifferentiable('DrawBoundingBoxes')
ops.NotDifferentiable('SampleDistortedBoundingBox')
ops.NotDifferentiable('SampleDistortedBoundingBoxV2')
# TODO(bsteiner): Implement the gradient function for extract_glimpse
# TODO(b/31222613): This op may be differentiable, and there may be
# latent bugs here.
ops.NotDifferentiable('ExtractGlimpse')
ops.NotDifferentiable('NonMaxSuppression')
ops.NotDifferentiable('NonMaxSuppressionV2')
ops.NotDifferentiable('NonMaxSuppressionWithOverlaps')
ops.NotDifferentiable('GenerateBoundingBoxProposals')

# pylint: disable=invalid-name
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
    otherwise, they are integer scalar tensors.
  """
  if image.get_shape().is_fully_defined():
    return image.get_shape().as_list()
  else:
    static_shape = image.get_shape().with_rank(rank).as_list()
    dynamic_shape = array_ops.unstack(array_ops.shape(image), rank)
    return [
        s if s is not None else d for s, d in zip(static_shape, dynamic_shape)
    ]


def _Check3DImage(image, require_static=True):
  """Assert that we are working with a properly shaped image.

  Args:
    image: 3-D Tensor of shape [height, width, channels]
    require_static: If `True`, requires that all dimensions of `image` are known
      and non-zero.

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
    raise ValueError("'image' (shape %s) must be fully defined." % image_shape)
  if any(x == 0 for x in image_shape):
    raise ValueError("all dims of 'image.shape' must be > 0: %s" % image_shape)
  if not image_shape.is_fully_defined():
    return [
        check_ops.assert_positive(
            array_ops.shape(image),
            ["all dims of 'image.shape' "
             'must be > 0.'])
    ]
  else:
    return []


def _Assert3DImage(image):
  """Assert that we are working with a properly shaped image.

  Performs the check statically if possible (i.e. if the shape
  is statically known). Otherwise adds a control dependency
  to an assert op that checks the dynamic shape.

  Args:
    image: 3-D Tensor of shape [height, width, channels]

  Raises:
    ValueError: if `image.shape` is not a 3-vector.

  Returns:
    If the shape of `image` could be verified statically, `image` is
    returned unchanged, otherwise there will be a control dependency
    added that asserts the correct dynamic shape.
  """
  return control_flow_ops.with_dependencies(
      _Check3DImage(image, require_static=False), image)


def _AssertAtLeast3DImage(image):
  """Assert that we are working with a properly shaped image.

  Performs the check statically if possible (i.e. if the shape
  is statically known). Otherwise adds a control dependency
  to an assert op that checks the dynamic shape.

  Args:
    image: >= 3-D Tensor of size [*, height, width, depth]

  Raises:
    ValueError: if image.shape is not a [>= 3] vector.

  Returns:
    If the shape of `image` could be verified statically, `image` is
    returned unchanged, otherwise there will be a control dependency
    added that asserts the correct dynamic shape.
  """
  return control_flow_ops.with_dependencies(
      _CheckAtLeast3DImage(image, require_static=False), image)


def _CheckAtLeast3DImage(image, require_static=True):
  """Assert that we are working with a properly shaped image.

  Args:
    image: >= 3-D Tensor of size [*, height, width, depth]
    require_static: If `True`, requires that all dimensions of `image` are known
      and non-zero.

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
  if any(x == 0 for x in image_shape[-3:]):
    raise ValueError('inner 3 dims of \'image.shape\' must be > 0: %s' %
                     image_shape)
  if not image_shape[-3:].is_fully_defined():
    return [
        check_ops.assert_positive(
            array_ops.shape(image)[-3:],
            ["inner 3 dims of 'image.shape' "
             'must be > 0.']),
        check_ops.assert_greater_equal(
            array_ops.rank(image),
            3,
            message="'image' must be at least three-dimensional.")
    ]
  else:
    return []


def _AssertGrayscaleImage(image):
  """Assert that we are working with a properly shaped grayscale image.

  Performs the check statically if possible (i.e. if the shape
  is statically known). Otherwise adds a control dependency
  to an assert op that checks the dynamic shape.

  Args:
    image: >= 2-D Tensor of size [*, 1]

  Raises:
    ValueError: if image.shape is not a [>= 2] vector or if
              last dimension is not size 1.

  Returns:
    If the shape of `image` could be verified statically, `image` is
    returned unchanged, otherwise there will be a control dependency
    added that asserts the correct dynamic shape.
  """
  return control_flow_ops.with_dependencies(
      _CheckGrayscaleImage(image, require_static=False), image)


def _CheckGrayscaleImage(image, require_static=True):
  """Assert that we are working with properly shaped grayscale image.

  Args:
    image: >= 2-D Tensor of size [*, 1]
    require_static: Boolean, whether static shape is required.

  Raises:
    ValueError: if image.shape is not a [>= 2] vector or if
              last dimension is not size 1.

  Returns:
    An empty list, if `image` has fully defined dimensions. Otherwise, a list
    containing an assert op is returned.
  """
  try:
    if image.get_shape().ndims is None:
      image_shape = image.get_shape().with_rank(2)
    else:
      image_shape = image.get_shape().with_rank_at_least(2)
  except ValueError:
    raise ValueError('A grayscale image must be at least two-dimensional.')
  if require_static and not image_shape.is_fully_defined():
    raise ValueError('\'image\' must be fully defined.')
  if image_shape.is_fully_defined():
    if image_shape[-1] != 1:
      raise ValueError('Last dimension of a grayscale image should be size 1.')
  if not image_shape.is_fully_defined():
    return [
        check_ops.assert_equal(
            array_ops.shape(image)[-1],
            1,
            message='Last dimension of a grayscale image should be size 1.'),
        check_ops.assert_greater_equal(
            array_ops.rank(image),
            3,
            message='A grayscale image must be at least two-dimensional.')
    ]
  else:
    return []


def fix_image_flip_shape(image, result):
  """Set the shape to 3 dimensional if we don't know anything else.

  Args:
    image: original image size
    result: flipped or transformed image

  Returns:
    An image whose shape is at least (None, None, None).
  """

  image_shape = image.get_shape()
  if image_shape == tensor_shape.unknown_shape():
    result.set_shape([None, None, None])
  else:
    result.set_shape(image_shape)
  return result


@tf_export('image.random_flip_up_down')
def random_flip_up_down(image, seed=None):
  """Randomly flips an image vertically (upside down).

  With a 1 in 2 chance, outputs the contents of `image` flipped along the first
  dimension, which is `height`.  Otherwise, output the image as-is.
  When passing a batch of images, each image will be randomly flipped
  independent of other images.

  Example usage:

    Randomly flip a single image.
    >>> import numpy as np

    >>> image = np.array([[[1], [2]], [[3], [4]]])
    >>> tf.image.random_flip_up_down(image, 3).numpy().tolist()
    [[[3], [4]], [[1], [2]]]

    Randomly flip multiple images.
    >>> images = np.array(
    ... [
    ...     [[[1], [2]], [[3], [4]]],
    ...     [[[5], [6]], [[7], [8]]]
    ... ])
    >>> tf.image.random_flip_up_down(images, 4).numpy().tolist()
    [[[[3], [4]], [[1], [2]]], [[[5], [6]], [[7], [8]]]]

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    seed: A Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed` for behavior.

  Returns:
    A tensor of the same type and shape as `image`.
  Raises:
    ValueError: if the shape of `image` not supported.
  """
  return _random_flip(image, 0, seed, 'random_flip_up_down')


@tf_export('image.random_flip_left_right')
def random_flip_left_right(image, seed=None):
  """Randomly flip an image horizontally (left to right).

  With a 1 in 2 chance, outputs the contents of `image` flipped along the
  second dimension, which is `width`.  Otherwise output the image as-is.
  When passing a batch of images, each image will be randomly flipped
  independent of other images.

  Example usage:
    Randomly flip a single image.
    >>> import numpy as np

    >>> image = np.array([[[1], [2]], [[3], [4]]])
    >>> tf.image.random_flip_left_right(image, 5).numpy().tolist()
    [[[2], [1]], [[4], [3]]]

    Randomly flip multiple images.
    >>> images = np.array(
    ... [
    ...     [[[1], [2]], [[3], [4]]],
    ...     [[[5], [6]], [[7], [8]]]
    ... ])
    >>> tf.image.random_flip_left_right(images, 6).numpy().tolist()
    [[[[2], [1]], [[4], [3]]], [[[5], [6]], [[7], [8]]]]

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    seed: A Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed` for behavior.

  Returns:
    A tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  return _random_flip(image, 1, seed, 'random_flip_left_right')


def _random_flip(image, flip_index, seed, scope_name):
  """Randomly (50% chance) flip an image along axis `flip_index`.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    flip_index: Dimension along which to flip the image.
      Vertical: 0, Horizontal: 1
    seed: A Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed` for behavior.
    scope_name: Name of the scope in which the ops are added.

  Returns:
    A tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  with ops.name_scope(None, scope_name, [image]) as scope:
    image = ops.convert_to_tensor(image, name='image')
    image = _AssertAtLeast3DImage(image)
    shape = image.get_shape()
    if shape.ndims == 3 or shape.ndims is None:
      uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
      mirror_cond = math_ops.less(uniform_random, .5)
      result = control_flow_ops.cond(
          mirror_cond,
          lambda: array_ops.reverse(image, [flip_index]),
          lambda: image,
          name=scope)
      return fix_image_flip_shape(image, result)
    elif shape.ndims == 4:
      batch_size = array_ops.shape(image)[0]
      uniform_random = random_ops.random_uniform([batch_size],
                                                 0,
                                                 1.0,
                                                 seed=seed)
      flips = math_ops.round(
          array_ops.reshape(uniform_random, [batch_size, 1, 1, 1]))
      flips = math_ops.cast(flips, image.dtype)
      flipped_input = array_ops.reverse(image, [flip_index + 1])
      return flips * flipped_input + (1 - flips) * image
    else:
      raise ValueError('\'image\' must have either 3 or 4 dimensions.')


@tf_export('image.flip_left_right')
def flip_left_right(image):
  """Flip an image horizontally (left to right).

  Outputs the contents of `image` flipped along the width dimension.

  See also `reverse()`.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.

  Returns:
    A tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  return _flip(image, 1, 'flip_left_right')


@tf_export('image.flip_up_down')
def flip_up_down(image):
  """Flip an image vertically (upside down).

  Outputs the contents of `image` flipped along the height dimension.

  See also `reverse()`.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.

  Returns:
    A `Tensor` of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  return _flip(image, 0, 'flip_up_down')


def _flip(image, flip_index, scope_name):
  """Flip an image either horizontally or vertically.

  Outputs the contents of `image` flipped along the dimension `flip_index`.

  See also `reverse()`.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    flip_index: 0 For vertical, 1 for horizontal.
    scope_name: string, scope name.

  Returns:
    A `Tensor` of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  with ops.name_scope(None, scope_name, [image]):
    image = ops.convert_to_tensor(image, name='image')
    image = _AssertAtLeast3DImage(image)
    shape = image.get_shape()
    if shape.ndims == 3 or shape.ndims is None:
      return fix_image_flip_shape(image, array_ops.reverse(image, [flip_index]))
    elif shape.ndims == 4:
      return array_ops.reverse(image, [flip_index + 1])
    else:
      raise ValueError('\'image\' must have either 3 or 4 dimensions.')


@tf_export('image.rot90')
def rot90(image, k=1, name=None):
  """Rotate image(s) counter-clockwise by 90 degrees.


  For example:
  ```python
  a=tf.constant([[[1],[2]],[[3],[4]]])
  # rotating `a` counter clockwise by 90 degrees
  a_rot=tf.image.rot90(a,k=1) #rotated `a`
  print(a_rot) # [[[2],[4]],[[1],[3]]]
  ```
  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    k: A scalar integer. The number of times the image is rotated by 90 degrees.
    name: A name for this operation (optional).

  Returns:
    A rotated tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  with ops.name_scope(name, 'rot90', [image, k]) as scope:
    image = ops.convert_to_tensor(image, name='image')
    image = _AssertAtLeast3DImage(image)
    k = ops.convert_to_tensor(k, dtype=dtypes.int32, name='k')
    k.get_shape().assert_has_rank(0)
    k = math_ops.mod(k, 4)

    shape = image.get_shape()
    if shape.ndims == 3 or shape.ndims is None:
      return _rot90_3D(image, k, scope)
    elif shape.ndims == 4:
      return _rot90_4D(image, k, scope)
    else:
      raise ValueError('\'image\' must have either 3 or 4 dimensions.')


def _rot90_3D(image, k, name_scope):
  """Rotate image counter-clockwise by 90 degrees `k` times.

  Args:
    image: 3-D Tensor of shape `[height, width, channels]`.
    k: A scalar integer. The number of times the image is rotated by 90 degrees.
    name_scope: A valid TensorFlow name scope.

  Returns:
    A 3-D tensor of the same type and shape as `image`.

  """

  def _rot90():
    return array_ops.transpose(array_ops.reverse_v2(image, [1]), [1, 0, 2])

  def _rot180():
    return array_ops.reverse_v2(image, [0, 1])

  def _rot270():
    return array_ops.reverse_v2(array_ops.transpose(image, [1, 0, 2]), [1])

  cases = [(math_ops.equal(k, 1), _rot90), (math_ops.equal(k, 2), _rot180),
           (math_ops.equal(k, 3), _rot270)]

  result = control_flow_ops.case(
      cases, default=lambda: image, exclusive=True, name=name_scope)
  result.set_shape([None, None, image.get_shape()[2]])
  return result


def _rot90_4D(images, k, name_scope):
  """Rotate batch of images counter-clockwise by 90 degrees `k` times.

  Args:
    images: 4-D Tensor of shape `[height, width, channels]`.
    k: A scalar integer. The number of times the images are rotated by 90
      degrees.
    name_scope: A valid TensorFlow name scope.

  Returns:
    A 4-D `Tensor` of the same type and shape as `images`.
  """

  def _rot90():
    return array_ops.transpose(array_ops.reverse_v2(images, [2]), [0, 2, 1, 3])

  def _rot180():
    return array_ops.reverse_v2(images, [1, 2])

  def _rot270():
    return array_ops.reverse_v2(array_ops.transpose(images, [0, 2, 1, 3]), [2])

  cases = [(math_ops.equal(k, 1), _rot90), (math_ops.equal(k, 2), _rot180),
           (math_ops.equal(k, 3), _rot270)]

  result = control_flow_ops.case(
      cases, default=lambda: images, exclusive=True, name=name_scope)
  shape = result.get_shape()
  result.set_shape([shape[0], None, None, shape[3]])
  return result


@tf_export('image.transpose', v1=['image.transpose', 'image.transpose_image'])
def transpose(image, name=None):
  """Transpose image(s) by swapping the height and width dimension.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    name: A name for this operation (optional).

  Returns:
    If `image` was 4-D, a 4-D float Tensor of shape
   `[batch, width, height, channels]`
    If `image` was 3-D, a 3-D float Tensor of shape
   `[width, height, channels]`

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  with ops.name_scope(name, 'transpose', [image]):
    image = ops.convert_to_tensor(image, name='image')
    image = _AssertAtLeast3DImage(image)
    shape = image.get_shape()
    if shape.ndims == 3 or shape.ndims is None:
      return array_ops.transpose(image, [1, 0, 2], name=name)
    elif shape.ndims == 4:
      return array_ops.transpose(image, [0, 2, 1, 3], name=name)
    else:
      raise ValueError('\'image\' must have either 3 or 4 dimensions.')


@tf_export('image.central_crop')
def central_crop(image, central_fraction):
  """Crop the central region of the image(s).

  Remove the outer parts of an image but retain the central region of the image
  along each dimension. If we specify central_fraction = 0.5, this function
  returns the region marked with "X" in the below diagram.

       --------
      |        |
      |  XXXX  |
      |  XXXX  |
      |        |   where "X" is the central 50% of the image.
       --------

  This function works on either a single image (`image` is a 3-D Tensor), or a
  batch of images (`image` is a 4-D Tensor).

  Args:
    image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D
      Tensor of shape [batch_size, height, width, depth].
    central_fraction: float (0, 1], fraction of size to crop
  Usage Example: ```python >> import tensorflow as tf >> x =
    tf.random.normal(shape=(256, 256, 3)) >> tf.image.central_crop(x, 0.5) ```

  Raises:
    ValueError: if central_crop_fraction is not within (0, 1].

  Returns:
    3-D / 4-D float Tensor, as per the input.
  """
  with ops.name_scope(None, 'central_crop', [image]):
    image = ops.convert_to_tensor(image, name='image')
    if central_fraction <= 0.0 or central_fraction > 1.0:
      raise ValueError('central_fraction must be within (0, 1]')
    if central_fraction == 1.0:
      return image

    _AssertAtLeast3DImage(image)
    rank = image.get_shape().ndims
    if rank != 3 and rank != 4:
      raise ValueError('`image` should either be a Tensor with rank = 3 or '
                       'rank = 4. Had rank = {}.'.format(rank))

    # Helper method to return the `idx`-th dimension of `tensor`, along with
    # a boolean signifying if the dimension is dynamic.
    def _get_dim(tensor, idx):
      static_shape = tensor.get_shape().dims[idx].value
      if static_shape is not None:
        return static_shape, False
      return array_ops.shape(tensor)[idx], True

    # Get the height, width, depth (and batch size, if the image is a 4-D
    # tensor).
    if rank == 3:
      img_h, dynamic_h = _get_dim(image, 0)
      img_w, dynamic_w = _get_dim(image, 1)
      img_d = image.get_shape()[2]
    else:
      img_bs = image.get_shape()[0]
      img_h, dynamic_h = _get_dim(image, 1)
      img_w, dynamic_w = _get_dim(image, 2)
      img_d = image.get_shape()[3]

    # Compute the bounding boxes for the crop. The type and value of the
    # bounding boxes depend on the `image` tensor's rank and whether / not the
    # dimensions are statically defined.
    if dynamic_h:
      img_hd = math_ops.cast(img_h, dtypes.float64)
      bbox_h_start = math_ops.cast((img_hd - img_hd * central_fraction) / 2,
                                   dtypes.int32)
    else:
      img_hd = float(img_h)
      bbox_h_start = int((img_hd - img_hd * central_fraction) / 2)

    if dynamic_w:
      img_wd = math_ops.cast(img_w, dtypes.float64)
      bbox_w_start = math_ops.cast((img_wd - img_wd * central_fraction) / 2,
                                   dtypes.int32)
    else:
      img_wd = float(img_w)
      bbox_w_start = int((img_wd - img_wd * central_fraction) / 2)

    bbox_h_size = img_h - bbox_h_start * 2
    bbox_w_size = img_w - bbox_w_start * 2

    if rank == 3:
      bbox_begin = array_ops.stack([bbox_h_start, bbox_w_start, 0])
      bbox_size = array_ops.stack([bbox_h_size, bbox_w_size, -1])
    else:
      bbox_begin = array_ops.stack([0, bbox_h_start, bbox_w_start, 0])
      bbox_size = array_ops.stack([-1, bbox_h_size, bbox_w_size, -1])

    image = array_ops.slice(image, bbox_begin, bbox_size)

    # Reshape the `image` tensor to the desired size.
    if rank == 3:
      image.set_shape([
          None if dynamic_h else bbox_h_size,
          None if dynamic_w else bbox_w_size, img_d
      ])
    else:
      image.set_shape([
          img_bs, None if dynamic_h else bbox_h_size,
          None if dynamic_w else bbox_w_size, img_d
      ])
    return image


@tf_export('image.pad_to_bounding_box')
def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width):
  """Pad `image` with zeros to the specified `height` and `width`.

  Adds `offset_height` rows of zeros on top, `offset_width` columns of
  zeros on the left, and then pads the image on the bottom and right
  with zeros until it has dimensions `target_height`, `target_width`.

  This op does nothing if `offset_*` is zero and the image already has size
  `target_height` by `target_width`.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
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
  with ops.name_scope(None, 'pad_to_bounding_box', [image]):
    image = ops.convert_to_tensor(image, name='image')

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

    assert_ops = _CheckAtLeast3DImage(image, require_static=False)
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

    padded_shape = [
        None if _is_tensor(i) else i
        for i in [batch, target_height, target_width, depth]
    ]
    padded.set_shape(padded_shape)

    if not is_batch:
      padded = array_ops.squeeze(padded, axis=[0])

    return padded


@tf_export('image.crop_to_bounding_box')
def crop_to_bounding_box(image, offset_height, offset_width, target_height,
                         target_width):
  """Crops an image to a specified bounding box.

  This op cuts a rectangular part out of `image`. The top-left corner of the
  returned image is at `offset_height, offset_width` in `image`, and its
  lower-right corner is at
  `offset_height + target_height, offset_width + target_width`.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
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
  with ops.name_scope(None, 'crop_to_bounding_box', [image]):
    image = ops.convert_to_tensor(image, name='image')

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

    assert_ops = _CheckAtLeast3DImage(image, require_static=False)

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
        image, array_ops.stack([0, offset_height, offset_width, 0]),
        array_ops.stack([-1, target_height, target_width, -1]))

    cropped_shape = [
        None if _is_tensor(i) else i
        for i in [batch, target_height, target_width, depth]
    ]
    cropped.set_shape(cropped_shape)

    if not is_batch:
      cropped = array_ops.squeeze(cropped, axis=[0])

    return cropped


@tf_export(
    'image.resize_with_crop_or_pad',
    v1=['image.resize_with_crop_or_pad', 'image.resize_image_with_crop_or_pad'])
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
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
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
  with ops.name_scope(None, 'resize_image_with_crop_or_pad', [image]):
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

    assert_ops = _CheckAtLeast3DImage(image, require_static=False)
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
      target_width = control_flow_ops.with_dependencies(assert_ops,
                                                        target_width)

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
    assert_ops += _assert(
        equal_(resized_height, target_height), ValueError,
        'resized height is not correct.')
    assert_ops += _assert(
        equal_(resized_width, target_width), ValueError,
        'resized width is not correct.')

    resized = control_flow_ops.with_dependencies(assert_ops, resized)

    if not is_batch:
      resized = array_ops.squeeze(resized, axis=[0])

    return resized


@tf_export(v1=['image.ResizeMethod'])
class ResizeMethodV1(object):
  BILINEAR = 0
  NEAREST_NEIGHBOR = 1
  BICUBIC = 2
  AREA = 3


@tf_export('image.ResizeMethod', v1=[])
class ResizeMethod(object):
  BILINEAR = 'bilinear'
  NEAREST_NEIGHBOR = 'nearest'
  BICUBIC = 'bicubic'
  AREA = 'area'
  LANCZOS3 = 'lanczos3'
  LANCZOS5 = 'lanczos5'
  GAUSSIAN = 'gaussian'
  MITCHELLCUBIC = 'mitchellcubic'


def _resize_images_common(images, resizer_fn, size, preserve_aspect_ratio, name,
                          skip_resize_if_same):
  """Core functionality for v1 and v2 resize functions."""
  with ops.name_scope(name, 'resize', [images, size]):
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
    new_height_const = size_const_as_shape.dims[0].value
    new_width_const = size_const_as_shape.dims[1].value

    if preserve_aspect_ratio:
      # Get the current shapes of the image, even if dynamic.
      _, current_height, current_width, _ = _ImageDimensions(images, rank=4)

      # do the computation to find the right scale and height/width.
      scale_factor_height = (
          math_ops.cast(new_height_const, dtypes.float32) /
          math_ops.cast(current_height, dtypes.float32))
      scale_factor_width = (
          math_ops.cast(new_width_const, dtypes.float32) /
          math_ops.cast(current_width, dtypes.float32))
      scale_factor = math_ops.minimum(scale_factor_height, scale_factor_width)
      scaled_height_const = math_ops.cast(
          math_ops.round(scale_factor *
                         math_ops.cast(current_height, dtypes.float32)),
          dtypes.int32)
      scaled_width_const = math_ops.cast(
          math_ops.round(scale_factor *
                         math_ops.cast(current_width, dtypes.float32)),
          dtypes.int32)

      # NOTE: Reset the size and other constants used later.
      size = ops.convert_to_tensor([scaled_height_const, scaled_width_const],
                                   dtypes.int32,
                                   name='size')
      size_const_as_shape = tensor_util.constant_value_as_shape(size)
      new_height_const = size_const_as_shape.dims[0].value
      new_width_const = size_const_as_shape.dims[1].value

    # If we can determine that the height and width will be unmodified by this
    # transformation, we avoid performing the resize.
    if skip_resize_if_same and all(
        x is not None
        for x in [new_width_const, width, new_height_const, height]) and (
            width == new_width_const and height == new_height_const):
      if not is_batch:
        images = array_ops.squeeze(images, axis=[0])
      return images

    images = resizer_fn(images, size)

    # NOTE(mrry): The shape functions for the resize ops cannot unpack
    # the packed values in `new_size`, so set the shape here.
    images.set_shape([None, new_height_const, new_width_const, None])

    if not is_batch:
      images = array_ops.squeeze(images, axis=[0])
    return images


@tf_export(v1=['image.resize_images', 'image.resize'])
def resize_images(images,
                  size,
                  method=ResizeMethodV1.BILINEAR,
                  align_corners=False,
                  preserve_aspect_ratio=False,
                  name=None):
  """Resize `images` to `size` using the specified `method`.

  Resized images will be distorted if their original aspect ratio is not
  the same as `size`.  To avoid distortions see
  `tf.compat.v1.image.resize_image_with_pad`.

  `method` can be one of:

  *   <b>`ResizeMethod.BILINEAR`</b>: [Bilinear interpolation.](
    https://en.wikipedia.org/wiki/Bilinear_interpolation)
  *   <b>`ResizeMethod.NEAREST_NEIGHBOR`</b>: [Nearest neighbor interpolation.](
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
  *   <b>`ResizeMethod.BICUBIC`</b>: [Bicubic interpolation.](
    https://en.wikipedia.org/wiki/Bicubic_interpolation)
  *   <b>`ResizeMethod.AREA`</b>: Area interpolation.

  The return value has the same type as `images` if `method` is
  `ResizeMethod.NEAREST_NEIGHBOR`. It will also have the same type as `images`
  if the size of `images` can be statically determined to be the same as `size`,
  because `images` is returned in this case. Otherwise, the return value has
  type `float32`.

  Args:
    images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new
      size for the images.
    method: ResizeMethod.  Defaults to `ResizeMethod.BILINEAR`.
    align_corners: bool.  If True, the centers of the 4 corner pixels of the
      input and output tensors are aligned, preserving the values at the corner
      pixels. Defaults to `False`.
    preserve_aspect_ratio: Whether to preserve the aspect ratio. If this is set,
      then `images` will be resized to a size that fits in `size` while
      preserving the aspect ratio of the original image. Scales up the image if
      `size` is bigger than the current size of the `image`. Defaults to False.
    name: A name for this operation (optional).

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

  def resize_fn(images_t, new_size):
    """Legacy resize core function, passed to _resize_images_common."""
    if method == ResizeMethodV1.BILINEAR or method == ResizeMethod.BILINEAR:
      return gen_image_ops.resize_bilinear(
          images_t, new_size, align_corners=align_corners)
    elif (method == ResizeMethodV1.NEAREST_NEIGHBOR or
          method == ResizeMethod.NEAREST_NEIGHBOR):
      return gen_image_ops.resize_nearest_neighbor(
          images_t, new_size, align_corners=align_corners)
    elif method == ResizeMethodV1.BICUBIC or method == ResizeMethod.BICUBIC:
      return gen_image_ops.resize_bicubic(
          images_t, new_size, align_corners=align_corners)
    elif method == ResizeMethodV1.AREA or method == ResizeMethod.AREA:
      return gen_image_ops.resize_area(
          images_t, new_size, align_corners=align_corners)
    else:
      raise ValueError('Resize method is not implemented.')

  return _resize_images_common(
      images,
      resize_fn,
      size,
      preserve_aspect_ratio=preserve_aspect_ratio,
      name=name,
      skip_resize_if_same=True)


@tf_export('image.resize', v1=[])
def resize_images_v2(images,
                     size,
                     method=ResizeMethod.BILINEAR,
                     preserve_aspect_ratio=False,
                     antialias=False,
                     name=None):
  """Resize `images` to `size` using the specified `method`.

  Resized images will be distorted if their original aspect ratio is not
  the same as `size`.  To avoid distortions see
  `tf.image.resize_with_pad`.

  When 'antialias' is true, the sampling filter will anti-alias the input image
  as well as interpolate.  When downsampling an image with [anti-aliasing](
  https://en.wikipedia.org/wiki/Spatial_anti-aliasing) the sampling filter
  kernel is scaled in order to properly anti-alias the input image signal.
  'antialias' has no effect when upsampling an image.

  *   <b>`bilinear`</b>: [Bilinear interpolation.](
    https://en.wikipedia.org/wiki/Bilinear_interpolation) If 'antialias' is
    true, becomes a hat/tent filter function with radius 1 when downsampling.
  *   <b>`lanczos3`</b>:  [Lanczos kernel](
    https://en.wikipedia.org/wiki/Lanczos_resampling) with radius 3.
    High-quality practical filter but may have some ringing, especially on
    synthetic images.
  *   <b>`lanczos5`</b>: [Lanczos kernel] (
    https://en.wikipedia.org/wiki/Lanczos_resampling) with radius 5.
    Very-high-quality filter but may have stronger ringing.
  *   <b>`bicubic`</b>: [Cubic interpolant](
    https://en.wikipedia.org/wiki/Bicubic_interpolation) of Keys. Equivalent to
    Catmull-Rom kernel. Reasonably good quality and faster than Lanczos3Kernel,
    particularly when upsampling.
  *   <b>`gaussian`</b>: [Gaussian kernel](
    https://en.wikipedia.org/wiki/Gaussian_filter) with radius 3,
    sigma = 1.5 / 3.0.
  *   <b>`nearest`</b>: [Nearest neighbor interpolation.](
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
    'antialias' has no effect when used with nearest neighbor interpolation.
  *   <b>`area`</b>: Anti-aliased resampling with area interpolation.
    'antialias' has no effect when used with area interpolation; it
    always anti-aliases.
  *   <b>`mitchellcubic`</b>: Mitchell-Netravali Cubic non-interpolating filter.
    For synthetic images (especially those lacking proper prefiltering), less
    ringing than Keys cubic kernel but less sharp.

  Note that near image edges the filtering kernel may be partially outside the
  image boundaries. For these pixels, only input pixels inside the image will be
  included in the filter sum, and the output value will be appropriately
  normalized.

  The return value has the same type as `images` if `method` is
  `ResizeMethod.NEAREST_NEIGHBOR`. Otherwise, the return value has type
  `float32`.

  Args:
    images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new
      size for the images.
    method: ResizeMethod.  Defaults to `bilinear`.
    preserve_aspect_ratio: Whether to preserve the aspect ratio. If this is set,
      then `images` will be resized to a size that fits in `size` while
      preserving the aspect ratio of the original image. Scales up the image if
      `size` is bigger than the current size of the `image`. Defaults to False.
    antialias: Whether to use an anti-aliasing filter when downsampling an
      image.
    name: A name for this operation (optional).

  Raises:
    ValueError: if the shape of `images` is incompatible with the
      shape arguments to this function
    ValueError: if `size` has an invalid shape or type.
    ValueError: if an unsupported resize method is specified.

  Returns:
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """

  def resize_fn(images_t, new_size):
    """Resize core function, passed to _resize_images_common."""
    scale_and_translate_methods = [
        ResizeMethod.LANCZOS3, ResizeMethod.LANCZOS5, ResizeMethod.GAUSSIAN,
        ResizeMethod.MITCHELLCUBIC
    ]

    def resize_with_scale_and_translate(method):
      scale = (
          math_ops.cast(new_size, dtype=dtypes.float32) /
          math_ops.cast(array_ops.shape(images_t)[1:3], dtype=dtypes.float32))
      return gen_image_ops.scale_and_translate(
          images_t,
          new_size,
          scale,
          array_ops.zeros([2]),
          kernel_type=method,
          antialias=antialias)

    if method == ResizeMethod.BILINEAR:
      if antialias:
        return resize_with_scale_and_translate('triangle')
      else:
        return gen_image_ops.resize_bilinear(
            images_t, new_size, half_pixel_centers=True)
    elif method == ResizeMethod.NEAREST_NEIGHBOR:
      return gen_image_ops.resize_nearest_neighbor(
          images_t, new_size, half_pixel_centers=True)
    elif method == ResizeMethod.BICUBIC:
      if antialias:
        return resize_with_scale_and_translate('keyscubic')
      else:
        return gen_image_ops.resize_bicubic(
            images_t, new_size, half_pixel_centers=True)
    elif method == ResizeMethod.AREA:
      return gen_image_ops.resize_area(images_t, new_size)
    elif method in scale_and_translate_methods:
      return resize_with_scale_and_translate(method)
    else:
      raise ValueError('Resize method is not implemented.')

  return _resize_images_common(
      images,
      resize_fn,
      size,
      preserve_aspect_ratio=preserve_aspect_ratio,
      name=name,
      skip_resize_if_same=False)


def _resize_image_with_pad_common(image, target_height, target_width,
                                  resize_fn):
  """Core functionality for v1 and v2 resize_image_with_pad functions."""
  with ops.name_scope(None, 'resize_image_with_pad', [image]):
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

    assert_ops = _CheckAtLeast3DImage(image, require_static=False)
    assert_ops += _assert(target_width > 0, ValueError,
                          'target_width must be > 0.')
    assert_ops += _assert(target_height > 0, ValueError,
                          'target_height must be > 0.')

    image = control_flow_ops.with_dependencies(assert_ops, image)

    def max_(x, y):
      if _is_tensor(x) or _is_tensor(y):
        return math_ops.maximum(x, y)
      else:
        return max(x, y)

    _, height, width, _ = _ImageDimensions(image, rank=4)

    # convert values to float, to ease divisions
    f_height = math_ops.cast(height, dtype=dtypes.float32)
    f_width = math_ops.cast(width, dtype=dtypes.float32)
    f_target_height = math_ops.cast(target_height, dtype=dtypes.float32)
    f_target_width = math_ops.cast(target_width, dtype=dtypes.float32)

    # Find the ratio by which the image must be adjusted
    # to fit within the target
    ratio = max_(f_width / f_target_width, f_height / f_target_height)
    resized_height_float = f_height / ratio
    resized_width_float = f_width / ratio
    resized_height = math_ops.cast(
        math_ops.floor(resized_height_float), dtype=dtypes.int32)
    resized_width = math_ops.cast(
        math_ops.floor(resized_width_float), dtype=dtypes.int32)

    padding_height = (f_target_height - resized_height_float) / 2
    padding_width = (f_target_width - resized_width_float) / 2
    f_padding_height = math_ops.floor(padding_height)
    f_padding_width = math_ops.floor(padding_width)
    p_height = max_(0, math_ops.cast(f_padding_height, dtype=dtypes.int32))
    p_width = max_(0, math_ops.cast(f_padding_width, dtype=dtypes.int32))

    # Resize first, then pad to meet requested dimensions
    resized = resize_fn(image, [resized_height, resized_width])

    padded = pad_to_bounding_box(resized, p_height, p_width, target_height,
                                 target_width)

    if padded.get_shape().ndims is None:
      raise ValueError('padded contains no shape.')

    _ImageDimensions(padded, rank=4)

    if not is_batch:
      padded = array_ops.squeeze(padded, axis=[0])

    return padded


@tf_export(v1=['image.resize_image_with_pad'])
def resize_image_with_pad_v1(image,
                             target_height,
                             target_width,
                             method=ResizeMethodV1.BILINEAR,
                             align_corners=False):
  """Resizes and pads an image to a target width and height.

  Resizes an image to a target width and height by keeping
  the aspect ratio the same without distortion. If the target
  dimensions don't match the image dimensions, the image
  is resized and then padded with zeroes to match requested
  dimensions.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    target_height: Target height.
    target_width: Target width.
    method: Method to use for resizing image. See `resize_images()`
    align_corners: bool.  If True, the centers of the 4 corner pixels of the
      input and output tensors are aligned, preserving the values at the corner
      pixels. Defaults to `False`.

  Raises:
    ValueError: if `target_height` or `target_width` are zero or negative.

  Returns:
    Resized and padded image.
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """

  def _resize_fn(im, new_size):
    return resize_images(im, new_size, method, align_corners=align_corners)

  return _resize_image_with_pad_common(image, target_height, target_width,
                                       _resize_fn)


@tf_export('image.resize_with_pad', v1=[])
def resize_image_with_pad_v2(image,
                             target_height,
                             target_width,
                             method=ResizeMethod.BILINEAR,
                             antialias=False):
  """Resizes and pads an image to a target width and height.

  Resizes an image to a target width and height by keeping
  the aspect ratio the same without distortion. If the target
  dimensions don't match the image dimensions, the image
  is resized and then padded with zeroes to match requested
  dimensions.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    target_height: Target height.
    target_width: Target width.
    method: Method to use for resizing image. See `image.resize()`
    antialias: Whether to use anti-aliasing when resizing. See 'image.resize()'.

  Raises:
    ValueError: if `target_height` or `target_width` are zero or negative.

  Returns:
    Resized and padded image.
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """

  def _resize_fn(im, new_size):
    return resize_images_v2(im, new_size, method, antialias=antialias)

  return _resize_image_with_pad_common(image, target_height, target_width,
                                       _resize_fn)


@tf_export('image.per_image_standardization')
def per_image_standardization(image):
  """Linearly scales each image in `image` to have mean 0 and variance 1.

  For each 3-D image `x` in `image`, computes `(x - mean) / adjusted_stddev`,
  where

  - `mean` is the average of all values in `x`
  - `adjusted_stddev = max(stddev, 1.0/sqrt(N))` is capped away from 0 to
    protect against division by 0 when handling uniform images
    - `N` is the number of elements in `x`
    - `stddev` is the standard deviation of all values in `x`

  Args:
    image: An n-D Tensor with at least 3 dimensions, the last 3 of which are the
      dimensions of each image.

  Returns:
    A `Tensor` with the same shape and dtype as `image`.

  Raises:
    ValueError: if the shape of 'image' is incompatible with this function.
  """
  with ops.name_scope(None, 'per_image_standardization', [image]) as scope:
    image = ops.convert_to_tensor(image, name='image')
    image = _AssertAtLeast3DImage(image)

    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    if orig_dtype not in [dtypes.float16, dtypes.float32]:
      image = convert_image_dtype(image, dtypes.float32)

    num_pixels = math_ops.reduce_prod(array_ops.shape(image)[-3:])
    image_mean = math_ops.reduce_mean(image, axis=[-1, -2, -3], keepdims=True)

    # Apply a minimum normalization that protects us against uniform images.
    stddev = math_ops.reduce_std(image, axis=[-1, -2, -3], keepdims=True)
    min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, image.dtype))
    adjusted_stddev = math_ops.maximum(stddev, min_stddev)

    image -= image_mean
    image = math_ops.div(image, adjusted_stddev, name=scope)
    return convert_image_dtype(image, orig_dtype, saturate=True)


@tf_export('image.random_brightness')
def random_brightness(image, max_delta, seed=None):
  """Adjust the brightness of images by a random factor.

  Equivalent to `adjust_brightness()` using a `delta` randomly picked in the
  interval `[-max_delta, max_delta)`.

  Args:
    image: An image or images to adjust.
    max_delta: float, must be non-negative.
    seed: A Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed` for behavior.

  Returns:
    The brightness-adjusted image(s).

  Raises:
    ValueError: if `max_delta` is negative.
  """
  if max_delta < 0:
    raise ValueError('max_delta must be non-negative.')

  delta = random_ops.random_uniform([], -max_delta, max_delta, seed=seed)
  return adjust_brightness(image, delta)


@tf_export('image.random_contrast')
def random_contrast(image, lower, upper, seed=None):
  """Adjust the contrast of an image or images by a random factor.

  Equivalent to `adjust_contrast()` but uses a `contrast_factor` randomly
  picked in the interval `[lower, upper]`.

  Args:
    image: An image tensor with 3 or more dimensions.
    lower: float.  Lower bound for the random contrast factor.
    upper: float.  Upper bound for the random contrast factor.
    seed: A Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed` for behavior.

  Returns:
    The contrast-adjusted image(s).

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


@tf_export('image.adjust_brightness')
def adjust_brightness(image, delta):
  """Adjust the brightness of RGB or Grayscale images.

  This is a convenience method that converts RGB images to float
  representation, adjusts their brightness, and then converts them back to the
  original data type. If several adjustments are chained, it is advisable to
  minimize the number of redundant conversions.

  The value `delta` is added to all components of the tensor `image`. `image` is
  converted to `float` and scaled appropriately if it is in fixed-point
  representation, and `delta` is converted to the same data type. For regular
  images, `delta` should be in the range `[0,1)`, as it is added to the image in
  floating point representation, where pixel values are in the `[0,1)` range.

  Args:
    image: RGB image or images to adjust.
    delta: A scalar. Amount to add to the pixel values.

  Returns:
    A brightness-adjusted tensor of the same shape and type as `image`.

  Usage Example:
    ```python
    import tensorflow as tf
    x = tf.random.normal(shape=(256, 256, 3))
    tf.image.adjust_brightness(x, delta=0.1)
    ```
  """
  with ops.name_scope(None, 'adjust_brightness', [image, delta]) as name:
    image = ops.convert_to_tensor(image, name='image')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype

    if orig_dtype in [dtypes.float16, dtypes.float32]:
      flt_image = image
    else:
      flt_image = convert_image_dtype(image, dtypes.float32)

    adjusted = math_ops.add(
        flt_image, math_ops.cast(delta, flt_image.dtype), name=name)

    return convert_image_dtype(adjusted, orig_dtype, saturate=True)


@tf_export('image.adjust_contrast')
def adjust_contrast(images, contrast_factor):
  """Adjust contrast of RGB or grayscale images.

  This is a convenience method that converts RGB images to float
  representation, adjusts their contrast, and then converts them back to the
  original data type. If several adjustments are chained, it is advisable to
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

  Usage Example:
    ```python
    import tensorflow as tf
    x = tf.random.normal(shape=(256, 256, 3))
    tf.image.adjust_contrast(x,2)
    ```
  """
  with ops.name_scope(None, 'adjust_contrast',
                      [images, contrast_factor]) as name:
    images = ops.convert_to_tensor(images, name='images')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = images.dtype

    if orig_dtype in (dtypes.float16, dtypes.float32):
      flt_images = images
    else:
      flt_images = convert_image_dtype(images, dtypes.float32)

    adjusted = gen_image_ops.adjust_contrastv2(
        flt_images, contrast_factor=contrast_factor, name=name)

    return convert_image_dtype(adjusted, orig_dtype, saturate=True)


@tf_export('image.adjust_gamma')
def adjust_gamma(image, gamma=1, gain=1):
  """Performs [Gamma Correction](http://en.wikipedia.org/wiki/Gamma_correction).

  on the input image.

  Also known as Power Law Transform. This function converts the
  input images at first to float representation, then transforms them
  pixelwise according to the equation `Out = gain * In**gamma`,
  and then converts the back to the original data type.

  Args:
    image : RGB image or images to adjust.
    gamma : A scalar or tensor. Non-negative real number.
    gain  : A scalar or tensor. The constant multiplier.

  Returns:
    A Tensor. A Gamma-adjusted tensor of the same shape and type as `image`.
  Usage Example:
    ```python
    >> import tensorflow as tf
    >> x = tf.random.normal(shape=(256, 256, 3))
    >> tf.image.adjust_gamma(x, 0.2)
    ```
  Raises:
    ValueError: If gamma is negative.
  Notes:
    For gamma greater than 1, the histogram will shift towards left and
    the output image will be darker than the input image.
    For gamma less than 1, the histogram will shift towards right and
    the output image will be brighter than the input image.
  References:
    [1] http://en.wikipedia.org/wiki/Gamma_correction
  """

  with ops.name_scope(None, 'adjust_gamma', [image, gamma, gain]) as name:
    image = ops.convert_to_tensor(image, name='image')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype

    if orig_dtype in [dtypes.float16, dtypes.float32]:
      flt_image = image
    else:
      flt_image = convert_image_dtype(image, dtypes.float32)

    assert_op = _assert(gamma >= 0, ValueError,
                        'Gamma should be a non-negative real number.')
    if assert_op:
      gamma = control_flow_ops.with_dependencies(assert_op, gamma)

    # According to the definition of gamma correction.
    adjusted_img = gain * flt_image**gamma

    return convert_image_dtype(adjusted_img, orig_dtype, saturate=True)


@tf_export('image.convert_image_dtype')
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

  Usage Example:
    ```python
    >> import tensorflow as tf
    >> x = tf.random.normal(shape=(256, 256, 3), dtype=tf.float32)
    >> tf.image.convert_image_dtype(x, dtype=tf.float16, saturate=False)
    ```

  Raises:
    AttributeError: Raises an attribute error when dtype is neither
    float nor integer
  """
  image = ops.convert_to_tensor(image, name='image')
  dtype = dtypes.as_dtype(dtype)
  if not dtype.is_floating and not dtype.is_integer:
    raise AttributeError('dtype must be either floating point or integer')
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
          cast = math_ops.saturate_cast(image, dtype)
        else:
          cast = math_ops.cast(image, dtype)
        scale = (scale_out + 1) // (scale_in + 1)
        return math_ops.multiply(cast, scale, name=name)
    elif image.dtype.is_floating and dtype.is_floating:
      # Both float: Just cast, no possible overflows in the allowed ranges.
      # Note: We're ignoring float overflows. If your image dynamic range
      # exceeds float range, you're on your own.
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


@tf_export('image.rgb_to_grayscale')
def rgb_to_grayscale(images, name=None):
  """Converts one or more images from RGB to Grayscale.

  Outputs a tensor of the same `DType` and rank as `images`.  The size of the
  last dimension of the output is 1, containing the Grayscale value of the
  pixels.
  
  ```python
  >>> original = tf.constant([[[1.0, 2.0, 3.0]]])
  >>> converted = tf.image.rgb_to_grayscale(original)
  >>> print(converted.numpy())
  [[[1.81...]]]

  ```
  
  Args:
    images: The RGB tensor to convert. The last dimension must have size 3 and
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
    gray_float = math_ops.tensordot(flt_image, rgb_weights, [-1, -1])
    gray_float = array_ops.expand_dims(gray_float, -1)
    return convert_image_dtype(gray_float, orig_dtype, name=name)


@tf_export('image.grayscale_to_rgb')
def grayscale_to_rgb(images, name=None):
  """Converts one or more images from Grayscale to RGB.

  Outputs a tensor of the same `DType` and rank as `images`.  The size of the
  last dimension of the output is 3, containing the RGB value of the pixels.
  The input images' last dimension must be size 1.
 
  ```python
  >>> original = tf.constant([[[1.0], [2.0], [3.0]]])
  >>> converted = tf.image.grayscale_to_rgb(original)
  >>> print(converted.numpy())
  [[[1. 1. 1.]
    [2. 2. 2.]
    [3. 3. 3.]]]

  ```
  
  Args:
    images: The Grayscale tensor to convert. The last dimension must be size 1.
    name: A name for the operation (optional).

  Returns:
    The converted grayscale image(s).
  """
  with ops.name_scope(name, 'grayscale_to_rgb', [images]) as name:
    images = _AssertGrayscaleImage(images)

    images = ops.convert_to_tensor(images, name='images')
    rank_1 = array_ops.expand_dims(array_ops.rank(images) - 1, 0)
    shape_list = ([array_ops.ones(rank_1, dtype=dtypes.int32)] +
                  [array_ops.expand_dims(3, 0)])
    multiples = array_ops.concat(shape_list, 0)
    rgb = array_ops.tile(images, multiples, name=name)
    rgb.set_shape(images.get_shape()[:-1].concatenate([3]))
    return rgb


# pylint: disable=invalid-name
@tf_export('image.random_hue')
def random_hue(image, max_delta, seed=None):
  """Adjust the hue of RGB images by a random factor.

  Equivalent to `adjust_hue()` but uses a `delta` randomly
  picked in the interval `[-max_delta, max_delta]`.

  `max_delta` must be in the interval `[0, 0.5]`.

  Args:
    image: RGB image or images. The size of the last dimension must be 3.
    max_delta: float. The maximum value for the random delta.
    seed: An operation-specific seed. It will be used in conjunction with the
      graph-level seed to determine the real seeds that will be used in this
      operation. Please see the documentation of set_random_seed for its
      interaction with the graph-level random seed.

  Returns:
    Adjusted image(s), same shape and DType as `image`.

  Raises:
    ValueError: if `max_delta` is invalid.
  """
  if max_delta > 0.5:
    raise ValueError('max_delta must be <= 0.5.')

  if max_delta < 0:
    raise ValueError('max_delta must be non-negative.')

  delta = random_ops.random_uniform([], -max_delta, max_delta, seed=seed)
  return adjust_hue(image, delta)


@tf_export('image.adjust_hue')
def adjust_hue(image, delta, name=None):
  """Adjust hue of RGB images.

  This is a convenience method that converts an RGB image to float
  representation, converts it to HSV, adds an offset to the
  hue channel, converts back to RGB and then back to the original
  data type. If several adjustments are chained it is advisable to minimize
  the number of redundant conversions.

  `image` is an RGB image.  The image hue is adjusted by converting the
  image(s) to HSV and rotating the hue channel (H) by
  `delta`.  The image is then converted back to RGB.

  `delta` must be in the interval `[-1, 1]`.

  Args:
    image: RGB image or images. The size of the last dimension must be 3.
    delta: float.  How much to add to the hue channel.
    name: A name for this operation (optional).

  Returns:
    Adjusted image(s), same shape and DType as `image`.

  Usage Example:
    ```python
    >> import tensorflow as tf
    >> x = tf.random.normal(shape=(256, 256, 3))
    >> tf.image.adjust_hue(x, 0.2)
    ```
  """
  with ops.name_scope(name, 'adjust_hue', [image]) as name:
    image = ops.convert_to_tensor(image, name='image')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    if orig_dtype in (dtypes.float16, dtypes.float32):
      flt_image = image
    else:
      flt_image = convert_image_dtype(image, dtypes.float32)

    rgb_altered = gen_image_ops.adjust_hue(flt_image, delta)

    return convert_image_dtype(rgb_altered, orig_dtype)


# pylint: disable=invalid-name
@tf_export('image.random_jpeg_quality')
def random_jpeg_quality(image, min_jpeg_quality, max_jpeg_quality, seed=None):
  """Randomly changes jpeg encoding quality for inducing jpeg noise.

  `min_jpeg_quality` must be in the interval `[0, 100]` and less than
  `max_jpeg_quality`.
  `max_jpeg_quality` must be in the interval `[0, 100]`.

  Args:
    image: 3D image. Size of the last dimension must be 1 or 3.
    min_jpeg_quality: Minimum jpeg encoding quality to use.
    max_jpeg_quality: Maximum jpeg encoding quality to use.
    seed: An operation-specific seed. It will be used in conjunction with the
      graph-level seed to determine the real seeds that will be used in this
      operation. Please see the documentation of set_random_seed for its
      interaction with the graph-level random seed.

  Returns:
    Adjusted image(s), same shape and DType as `image`.

  Raises:
    ValueError: if `min_jpeg_quality` or `max_jpeg_quality` is invalid.
  """
  if (min_jpeg_quality < 0 or max_jpeg_quality < 0 or min_jpeg_quality > 100 or
      max_jpeg_quality > 100):
    raise ValueError('jpeg encoding range must be between 0 and 100.')

  if min_jpeg_quality >= max_jpeg_quality:
    raise ValueError('`min_jpeg_quality` must be less than `max_jpeg_quality`.')

  jpeg_quality = random_ops.random_uniform([],
                                           min_jpeg_quality,
                                           max_jpeg_quality,
                                           seed=seed,
                                           dtype=dtypes.int32)
  return adjust_jpeg_quality(image, jpeg_quality)


@tf_export('image.adjust_jpeg_quality')
def adjust_jpeg_quality(image, jpeg_quality, name=None):
  """Adjust jpeg encoding quality of an image.

  This is a convenience method that converts an image to uint8 representation,
  encodes it to jpeg with `jpeg_quality`, decodes it, and then converts back
  to the original data type.

  `jpeg_quality` must be in the interval `[0, 100]`.

  Args:
    image: 3D image. The size of the last dimension must be None, 1 or 3.
    jpeg_quality: Python int or Tensor of type int32. jpeg encoding quality.
    name: A name for this operation (optional).

  Returns:
    Adjusted image, same shape and DType as `image`.

  Usage Example:
    ```python
    >> import tensorflow as tf
    >> x = tf.random.normal(shape=(256, 256, 3))
    >> tf.image.adjust_jpeg_quality(x, 75)
    ```
  Raises:
    InvalidArgumentError: quality must be in [0,100]
    InvalidArgumentError: image must have 1 or 3 channels
  """
  with ops.name_scope(name, 'adjust_jpeg_quality', [image]):
    image = ops.convert_to_tensor(image, name='image')
    channels = image.shape.as_list()[-1]
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    image = convert_image_dtype(image, dtypes.uint8)
    if not _is_tensor(jpeg_quality):
      # If jpeg_quality is a int (not tensor).
      jpeg_quality = ops.convert_to_tensor(jpeg_quality, dtype=dtypes.int32)
    image = gen_image_ops.encode_jpeg_variable_quality(image, jpeg_quality)

    image = gen_image_ops.decode_jpeg(image, channels=channels)
    return convert_image_dtype(image, orig_dtype)


@tf_export('image.random_saturation')
def random_saturation(image, lower, upper, seed=None):
  """Adjust the saturation of RGB images by a random factor.

  Equivalent to `adjust_saturation()` but uses a `saturation_factor` randomly
  picked in the interval `[lower, upper]`.

  Args:
    image: RGB image or images. The size of the last dimension must be 3.
    lower: float.  Lower bound for the random saturation factor.
    upper: float.  Upper bound for the random saturation factor.
    seed: An operation-specific seed. It will be used in conjunction with the
      graph-level seed to determine the real seeds that will be used in this
      operation. Please see the documentation of set_random_seed for its
      interaction with the graph-level random seed.

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


@tf_export('image.adjust_saturation')
def adjust_saturation(image, saturation_factor, name=None):
  """Adjust saturation of RGB images.

  This is a convenience method that converts RGB images to float
  representation, converts them to HSV, adds an offset to the
  saturation channel, converts back to RGB and then back to the original
  data type. If several adjustments are chained it is advisable to minimize
  the number of redundant conversions.

  `image` is an RGB image or images.  The image saturation is adjusted by
  converting the images to HSV and multiplying the saturation (S) channel by
  `saturation_factor` and clipping. The images are then converted back to RGB.

  Args:
    image: RGB image or images. The size of the last dimension must be 3.
    saturation_factor: float. Factor to multiply the saturation by.
    name: A name for this operation (optional).

  Returns:
    Adjusted image(s), same shape and DType as `image`.

  Usage Example:
    ```python
    >> import tensorflow as tf
    >> x = tf.random.normal(shape=(256, 256, 3))
    >> tf.image.adjust_saturation(x, 0.5)
    ```

  Raises:
    InvalidArgumentError: input must have 3 channels
  """
  with ops.name_scope(name, 'adjust_saturation', [image]) as name:
    image = ops.convert_to_tensor(image, name='image')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    if orig_dtype in (dtypes.float16, dtypes.float32):
      flt_image = image
    else:
      flt_image = convert_image_dtype(image, dtypes.float32)

    adjusted = gen_image_ops.adjust_saturation(flt_image, saturation_factor)

    return convert_image_dtype(adjusted, orig_dtype)


@tf_export('io.is_jpeg', 'image.is_jpeg', v1=['io.is_jpeg', 'image.is_jpeg'])
def is_jpeg(contents, name=None):
  r"""Convenience function to check if the 'contents' encodes a JPEG image.

  Args:
    contents: 0-D `string`. The encoded image bytes.
    name: A name for the operation (optional)

  Returns:
     A scalar boolean tensor indicating if 'contents' may be a JPEG image.
     is_jpeg is susceptible to false positives.
  """
  # Normal JPEGs start with \xff\xd8\xff\xe0
  # JPEG with EXIF starts with \xff\xd8\xff\xe1
  # Use \xff\xd8\xff to cover both.
  with ops.name_scope(name, 'is_jpeg'):
    substr = string_ops.substr(contents, 0, 3)
    return math_ops.equal(substr, b'\xff\xd8\xff', name=name)


def _is_png(contents, name=None):
  r"""Convenience function to check if the 'contents' encodes a PNG image.

  Args:
    contents: 0-D `string`. The encoded image bytes.
    name: A name for the operation (optional)

  Returns:
     A scalar boolean tensor indicating if 'contents' may be a PNG image.
     is_png is susceptible to false positives.
  """
  with ops.name_scope(name, 'is_png'):
    substr = string_ops.substr(contents, 0, 3)
    return math_ops.equal(substr, b'\211PN', name=name)


tf_export(
    'io.decode_and_crop_jpeg',
    'image.decode_and_crop_jpeg',
    v1=['io.decode_and_crop_jpeg', 'image.decode_and_crop_jpeg'])(
        gen_image_ops.decode_and_crop_jpeg)

tf_export(
    'io.decode_bmp',
    'image.decode_bmp',
    v1=['io.decode_bmp', 'image.decode_bmp'])(
        gen_image_ops.decode_bmp)
tf_export(
    'io.decode_gif',
    'image.decode_gif',
    v1=['io.decode_gif', 'image.decode_gif'])(
        gen_image_ops.decode_gif)
tf_export(
    'io.decode_jpeg',
    'image.decode_jpeg',
    v1=['io.decode_jpeg', 'image.decode_jpeg'])(
        gen_image_ops.decode_jpeg)
tf_export(
    'io.decode_png',
    'image.decode_png',
    v1=['io.decode_png', 'image.decode_png'])(
        gen_image_ops.decode_png)

tf_export(
    'io.encode_jpeg',
    'image.encode_jpeg',
    v1=['io.encode_jpeg', 'image.encode_jpeg'])(
        gen_image_ops.encode_jpeg)
tf_export(
    'io.extract_jpeg_shape',
    'image.extract_jpeg_shape',
    v1=['io.extract_jpeg_shape', 'image.extract_jpeg_shape'])(
        gen_image_ops.extract_jpeg_shape)


@tf_export('image.encode_png')
def encode_png(image, compression=-1, name=None):
  r"""PNG-encode an image.

  `image` is a 3-D uint8 or uint16 Tensor of shape `[height, width, channels]`
  where `channels` is:

  *   1: for grayscale.
  *   2: for grayscale + alpha.
  *   3: for RGB.
  *   4: for RGBA.

  The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
  default or a value from 0 to 9.  9 is the highest compression level,
  generating the smallest output, but is slower.

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `uint16`.
      3-D with shape `[height, width, channels]`.
    compression: An optional `int`. Defaults to `-1`. Compression level.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  return gen_image_ops.encode_png(
      ops.convert_to_tensor(image), compression, name)


@tf_export(
    'io.decode_image',
    'image.decode_image',
    v1=['io.decode_image', 'image.decode_image'])
def decode_image(contents,
                 channels=None,
                 dtype=dtypes.uint8,
                 name=None,
                 expand_animations=True):
  """Function for `decode_bmp`, `decode_gif`, `decode_jpeg`, and `decode_png`.

  Detects whether an image is a BMP, GIF, JPEG, or PNG, and performs the
  appropriate operation to convert the input bytes `string` into a `Tensor`
  of type `dtype`.

  Note: `decode_gif` returns a 4-D array `[num_frames, height, width, 3]`, as
  opposed to `decode_bmp`, `decode_jpeg` and `decode_png`, which return 3-D
  arrays `[height, width, num_channels]`. Make sure to take this into account
  when constructing your graph if you are intermixing GIF files with BMP, JPEG,
  and/or PNG files. Alternately, set the `expand_animations` argument of this
  function to `False`, in which case the op will return 3-dimensional tensors
  and will truncate animated GIF files to the first frame.

  Args:
    contents: 0-D `string`. The encoded image bytes.
    channels: An optional `int`. Defaults to `0`. Number of color channels for
      the decoded image.
    dtype: The desired DType of the returned `Tensor`.
    name: A name for the operation (optional)
    expand_animations: Controls the shape of the returned op's output. If
      `True`, the returned op will produce a 3-D tensor for PNG, JPEG, and BMP
      files; and a 4-D tensor for all GIFs, whether animated or not. If,
      `False`, the returned op will produce a 3-D tensor for all file types and
      will truncate animated GIFs to the first frame.

  Returns:
    `Tensor` with type `dtype` and a 3- or 4-dimensional shape, depending on
    the file type and the value of the `expand_animations` parameter.

  Raises:
    ValueError: On incorrect number of channels.
  """
  with ops.name_scope(name, 'decode_image'):
    if channels not in (None, 0, 1, 3, 4):
      raise ValueError('channels must be in (None, 0, 1, 3, 4)')
    substr = string_ops.substr(contents, 0, 3)

    def _bmp():
      """Decodes a BMP image."""
      signature = string_ops.substr(contents, 0, 2)
      # Create assert op to check that bytes are BMP decodable
      is_bmp = math_ops.equal(signature, 'BM', name='is_bmp')
      decode_msg = 'Unable to decode bytes as JPEG, PNG, GIF, or BMP'
      assert_decode = control_flow_ops.Assert(is_bmp, [decode_msg])
      bmp_channels = 0 if channels is None else channels
      good_channels = math_ops.not_equal(bmp_channels, 1, name='check_channels')
      channels_msg = 'Channels must be in (None, 0, 3) when decoding BMP images'
      assert_channels = control_flow_ops.Assert(good_channels, [channels_msg])
      with ops.control_dependencies([assert_decode, assert_channels]):
        return convert_image_dtype(gen_image_ops.decode_bmp(contents), dtype)

    def _gif():
      """Decodes a GIF image."""
      # Create assert to make sure that channels is not set to 1
      # Already checked above that channels is in (None, 0, 1, 3)
      gif_channels = 0 if channels is None else channels
      good_channels = math_ops.logical_and(
          math_ops.not_equal(gif_channels, 1, name='check_gif_channels'),
          math_ops.not_equal(gif_channels, 4, name='check_gif_channels'))
      channels_msg = 'Channels must be in (None, 0, 3) when decoding GIF images'
      assert_channels = control_flow_ops.Assert(good_channels, [channels_msg])
      with ops.control_dependencies([assert_channels]):
        result = convert_image_dtype(gen_image_ops.decode_gif(contents), dtype)
        if not expand_animations:
          # For now we decode animated GIFs fully and toss out all but the
          # first frame when expand_animations is False
          result = array_ops.gather(result, 0)
        return result

    def check_gif():
      # Create assert op to check that bytes are GIF decodable
      is_gif = math_ops.equal(substr, b'\x47\x49\x46', name='is_gif')
      return control_flow_ops.cond(is_gif, _gif, _bmp, name='cond_gif')

    def _png():
      """Decodes a PNG image."""
      return convert_image_dtype(
          gen_image_ops.decode_png(
              contents,
              channels,
              dtype=dtypes.uint8 if dtype == dtypes.uint8 else dtypes.uint16),
          dtype)

    def check_png():
      """Checks if an image is PNG."""
      return control_flow_ops.cond(
          _is_png(contents), _png, check_gif, name='cond_png')

    def _jpeg():
      """Decodes a jpeg image."""
      jpeg_channels = 0 if channels is None else channels
      good_channels = math_ops.not_equal(
          jpeg_channels, 4, name='check_jpeg_channels')
      channels_msg = ('Channels must be in (None, 0, 1, 3) when decoding JPEG '
                      'images')
      assert_channels = control_flow_ops.Assert(good_channels, [channels_msg])
      with ops.control_dependencies([assert_channels]):
        return convert_image_dtype(
            gen_image_ops.decode_jpeg(contents, channels), dtype)

    # Decode normal JPEG images (start with \xff\xd8\xff\xe0)
    # as well as JPEG images with EXIF data (start with \xff\xd8\xff\xe1).
    return control_flow_ops.cond(
        is_jpeg(contents), _jpeg, check_png, name='cond_jpeg')


@tf_export('image.total_variation')
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
    images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
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
    tot_var = (
        math_ops.reduce_sum(math_ops.abs(pixel_dif1), axis=sum_axis) +
        math_ops.reduce_sum(math_ops.abs(pixel_dif2), axis=sum_axis))

  return tot_var


@tf_export('image.sample_distorted_bounding_box', v1=[])
def sample_distorted_bounding_box_v2(image_size,
                                     bounding_boxes,
                                     seed=0,
                                     min_object_covered=0.1,
                                     aspect_ratio_range=None,
                                     area_range=None,
                                     max_attempts=None,
                                     use_image_if_no_bounding_boxes=None,
                                     name=None):
  """Generate a single randomly distorted bounding box for an image.

  Bounding box annotations are often supplied in addition to ground-truth labels
  in image recognition or object localization tasks. A common technique for
  training such a system is to randomly distort an image while preserving
  its content, i.e. *data augmentation*. This Op outputs a randomly distorted
  localization of an object, i.e. bounding box, given an `image_size`,
  `bounding_boxes` and a series of constraints.

  The output of this Op is a single bounding box that may be used to crop the
  original image. The output is returned as 3 tensors: `begin`, `size` and
  `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
  image. The latter may be supplied to `tf.image.draw_bounding_boxes` to
  visualize what the bounding box looks like.

  Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`.
  The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width
  and the height of the underlying image.

  For example,

  ```python
      # Generate a single distorted bounding box.
      begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          bounding_boxes=bounding_boxes,
          min_object_covered=0.1)

      # Draw the bounding box in an image summary.
      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                    bbox_for_draw)
      tf.compat.v1.summary.image('images_with_box', image_with_box)

      # Employ the bounding box to distort the image.
      distorted_image = tf.slice(image, begin, size)
  ```

  Note that if no bounding box information is available, setting
  `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
  bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
  false and no bounding boxes are supplied, an error is raised.

  Args:
    image_size: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`. 1-D, containing `[height, width, channels]`.
    bounding_boxes: A `Tensor` of type `float32`. 3-D with shape `[batch, N, 4]`
      describing the N bounding boxes associated with the image.
    seed: An optional `int`. Defaults to `0`. If `seed` is set to non-zero, the
      random number generator is seeded by the given `seed`.  Otherwise, it is
      seeded by a random seed.
    min_object_covered: A Tensor of type `float32`. Defaults to `0.1`. The
      cropped area of the image must contain at least this fraction of any
      bounding box supplied. The value of this parameter should be non-negative.
      In the case of 0, the cropped area does not need to overlap any of the
      bounding boxes supplied.
    aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75,
      1.33]`. The cropped area of the image must have an aspect `ratio = width /
      height` within this range.
    area_range: An optional list of `floats`. Defaults to `[0.05, 1]`. The
      cropped area of the image must contain a fraction of the supplied image
      within this range.
    max_attempts: An optional `int`. Defaults to `100`. Number of attempts at
      generating a cropped region of the image of the specified constraints.
      After `max_attempts` failures, return the entire image.
    use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.
      Controls behavior if no bounding boxes supplied. If true, assume an
      implicit bounding box covering the whole input. If false, raise an error.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (begin, size, bboxes).

    begin: A `Tensor`. Has the same type as `image_size`. 1-D, containing
    `[offset_height, offset_width, 0]`. Provide as input to
      `tf.slice`.
    size: A `Tensor`. Has the same type as `image_size`. 1-D, containing
    `[target_height, target_width, -1]`. Provide as input to
      `tf.slice`.
    bboxes: A `Tensor` of type `float32`. 3-D with shape `[1, 1, 4]` containing
    the distorted bounding box.
    Provide as input to `tf.image.draw_bounding_boxes`.
  """
  seed1, seed2 = random_seed.get_seed(seed) if seed else (0, 0)
  return sample_distorted_bounding_box(image_size, bounding_boxes, seed1, seed2,
                                       min_object_covered, aspect_ratio_range,
                                       area_range, max_attempts,
                                       use_image_if_no_bounding_boxes, name)


@tf_export(v1=['image.sample_distorted_bounding_box'])
@deprecation.deprecated(
    date=None,
    instructions='`seed2` arg is deprecated.'
    'Use sample_distorted_bounding_box_v2 instead.')
def sample_distorted_bounding_box(image_size,
                                  bounding_boxes,
                                  seed=None,
                                  seed2=None,
                                  min_object_covered=0.1,
                                  aspect_ratio_range=None,
                                  area_range=None,
                                  max_attempts=None,
                                  use_image_if_no_bounding_boxes=None,
                                  name=None):
  """Generate a single randomly distorted bounding box for an image.

  Bounding box annotations are often supplied in addition to ground-truth labels
  in image recognition or object localization tasks. A common technique for
  training such a system is to randomly distort an image while preserving
  its content, i.e. *data augmentation*. This Op outputs a randomly distorted
  localization of an object, i.e. bounding box, given an `image_size`,
  `bounding_boxes` and a series of constraints.

  The output of this Op is a single bounding box that may be used to crop the
  original image. The output is returned as 3 tensors: `begin`, `size` and
  `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
  image. The latter may be supplied to `tf.image.draw_bounding_boxes` to
  visualize what the bounding box looks like.

  Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`.
  The
  bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
  height of the underlying image.

  For example,

  ```python
      # Generate a single distorted bounding box.
      begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          bounding_boxes=bounding_boxes,
          min_object_covered=0.1)

      # Draw the bounding box in an image summary.
      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                    bbox_for_draw)
      tf.compat.v1.summary.image('images_with_box', image_with_box)

      # Employ the bounding box to distort the image.
      distorted_image = tf.slice(image, begin, size)
  ```

  Note that if no bounding box information is available, setting
  `use_image_if_no_bounding_boxes = True` will assume there is a single implicit
  bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
  false and no bounding boxes are supplied, an error is raised.

  Args:
    image_size: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`. 1-D, containing `[height, width, channels]`.
    bounding_boxes: A `Tensor` of type `float32`. 3-D with shape `[batch, N, 4]`
      describing the N bounding boxes associated with the image.
    seed: An optional `int`. Defaults to `0`. If either `seed` or `seed2` are
      set to non-zero, the random number generator is seeded by the given
      `seed`.  Otherwise, it is seeded by a random seed.
    seed2: An optional `int`. Defaults to `0`. A second seed to avoid seed
      collision.
    min_object_covered: A Tensor of type `float32`. Defaults to `0.1`. The
      cropped area of the image must contain at least this fraction of any
      bounding box supplied. The value of this parameter should be non-negative.
      In the case of 0, the cropped area does not need to overlap any of the
      bounding boxes supplied.
    aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75,
      1.33]`. The cropped area of the image must have an aspect ratio = width /
      height within this range.
    area_range: An optional list of `floats`. Defaults to `[0.05, 1]`. The
      cropped area of the image must contain a fraction of the supplied image
      within this range.
    max_attempts: An optional `int`. Defaults to `100`. Number of attempts at
      generating a cropped region of the image of the specified constraints.
      After `max_attempts` failures, return the entire image.
    use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.
      Controls behavior if no bounding boxes supplied. If true, assume an
      implicit bounding box covering the whole input. If false, raise an error.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (begin, size, bboxes).

    begin: A `Tensor`. Has the same type as `image_size`. 1-D, containing
    `[offset_height, offset_width, 0]`. Provide as input to
      `tf.slice`.
    size: A `Tensor`. Has the same type as `image_size`. 1-D, containing
    `[target_height, target_width, -1]`. Provide as input to
      `tf.slice`.
    bboxes: A `Tensor` of type `float32`. 3-D with shape `[1, 1, 4]` containing
    the distorted bounding box.
      Provide as input to `tf.image.draw_bounding_boxes`.
  """
  with ops.name_scope(name, 'sample_distorted_bounding_box'):
    return gen_image_ops.sample_distorted_bounding_box_v2(
        image_size,
        bounding_boxes,
        seed=seed,
        seed2=seed2,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes,
        name=name)


@tf_export('image.non_max_suppression')
def non_max_suppression(boxes,
                        scores,
                        max_output_size,
                        iou_threshold=0.5,
                        score_threshold=float('-inf'),
                        name=None):
  """Greedily selects a subset of bounding boxes in descending order of score.

  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval `[0, 1]`) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system.  Note that this
  algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather` operation.  For example:
    ```python
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size, iou_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```

  Args:
    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non-max suppression.
    iou_threshold: A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    name: A name for the operation (optional).

  Returns:
    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the
      selected indices from the boxes tensor, where `M <= max_output_size`.
  """
  with ops.name_scope(name, 'non_max_suppression'):
    iou_threshold = ops.convert_to_tensor(iou_threshold, name='iou_threshold')
    score_threshold = ops.convert_to_tensor(
        score_threshold, name='score_threshold')
    return gen_image_ops.non_max_suppression_v3(boxes, scores, max_output_size,
                                                iou_threshold, score_threshold)


@tf_export('image.non_max_suppression_with_scores')
def non_max_suppression_with_scores(boxes,
                                    scores,
                                    max_output_size,
                                    iou_threshold=0.5,
                                    score_threshold=float('-inf'),
                                    soft_nms_sigma=0.0,
                                    name=None):
  """Greedily selects a subset of bounding boxes in descending order of score.

  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval `[0, 1]`) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system.  Note that this
  algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather` operation.  For example:
    ```python
    selected_indices, selected_scores = tf.image.non_max_suppression_v2(
        boxes, scores, max_output_size, iou_threshold=1.0, score_threshold=0.1,
        soft_nms_sigma=0.5)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```

  This function generalizes the `tf.image.non_max_suppression` op by also
  supporting a Soft-NMS (with Gaussian weighting) mode (c.f.
  Bodla et al, https://arxiv.org/abs/1704.04503) where boxes reduce the score
  of other overlapping boxes instead of directly causing them to be pruned.
  Consequently, in contrast to `tf.image.non_max_suppression`,
  `tf.image.non_max_suppression_v2` returns the new scores of each input box in
  the second output, `selected_scores`.

  To enable this Soft-NMS mode, set the `soft_nms_sigma` parameter to be
  larger than 0.  When `soft_nms_sigma` equals 0, the behavior of
  `tf.image.non_max_suppression_v2` is identical to that of
  `tf.image.non_max_suppression` (except for the extra output) both in function
  and in running time.

  Args:
    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non-max suppression.
    iou_threshold: A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    soft_nms_sigma: A scalar float representing the Soft NMS sigma parameter;
      See Bodla et al, https://arxiv.org/abs/1704.04503).  When
        `soft_nms_sigma=0.0` (which is default), we fall back to standard (hard)
        NMS.
    name: A name for the operation (optional).

  Returns:
    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the
      selected indices from the boxes tensor, where `M <= max_output_size`.
    selected_scores: A 1-D float tensor of shape `[M]` representing the
      corresponding scores for each selected box, where `M <= max_output_size`.
      Scores only differ from corresponding input scores when using Soft NMS
      (i.e. when `soft_nms_sigma>0`)
  """
  with ops.name_scope(name, 'non_max_suppression_with_scores'):
    iou_threshold = ops.convert_to_tensor(iou_threshold, name='iou_threshold')
    score_threshold = ops.convert_to_tensor(
        score_threshold, name='score_threshold')
    soft_nms_sigma = ops.convert_to_tensor(
        soft_nms_sigma, name='soft_nms_sigma')
    (selected_indices, selected_scores,
     _) = gen_image_ops.non_max_suppression_v5(
         boxes,
         scores,
         max_output_size,
         iou_threshold,
         score_threshold,
         soft_nms_sigma,
         pad_to_max_output_size=False)
    return selected_indices, selected_scores


@tf_export('image.non_max_suppression_padded')
def non_max_suppression_padded(boxes,
                               scores,
                               max_output_size,
                               iou_threshold=0.5,
                               score_threshold=float('-inf'),
                               pad_to_max_output_size=False,
                               name=None):
  """Greedily selects a subset of bounding boxes in descending order of score.

  Performs algorithmically equivalent operation to tf.image.non_max_suppression,
  with the addition of an optional parameter which zero-pads the output to
  be of size `max_output_size`.
  The output of this operation is a tuple containing the set of integers
  indexing into the input collection of bounding boxes representing the selected
  boxes and the number of valid indices in the index set.  The bounding box
  coordinates corresponding to the selected indices can then be obtained using
  the `tf.slice` and `tf.gather` operations.  For example:
    ```python
    selected_indices_padded, num_valid = tf.image.non_max_suppression_padded(
        boxes, scores, max_output_size, iou_threshold,
        score_threshold, pad_to_max_output_size=True)
    selected_indices = tf.slice(
        selected_indices_padded, tf.constant([0]), num_valid)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```

  Args:
    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non-max suppression.
    iou_threshold: A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    pad_to_max_output_size: bool.  If True, size of `selected_indices` output is
      padded to `max_output_size`.
    name: A name for the operation (optional).

  Returns:
    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the
      selected indices from the boxes tensor, where `M <= max_output_size`.
    valid_outputs: A scalar integer `Tensor` denoting how many elements in
    `selected_indices` are valid.  Valid elements occur first, then padding.
  """
  with ops.name_scope(name, 'non_max_suppression_padded'):
    iou_threshold = ops.convert_to_tensor(iou_threshold, name='iou_threshold')
    score_threshold = ops.convert_to_tensor(
        score_threshold, name='score_threshold')
    return gen_image_ops.non_max_suppression_v4(boxes, scores, max_output_size,
                                                iou_threshold, score_threshold,
                                                pad_to_max_output_size)


@tf_export('image.non_max_suppression_overlaps')
def non_max_suppression_with_overlaps(overlaps,
                                      scores,
                                      max_output_size,
                                      overlap_threshold=0.5,
                                      score_threshold=float('-inf'),
                                      name=None):
  """Greedily selects a subset of bounding boxes in descending order of score.

  Prunes away boxes that have high overlap with previously selected boxes.
  N-by-n overlap values are supplied as square matrix.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather` operation.  For example:
    ```python
    selected_indices = tf.image.non_max_suppression_overlaps(
        overlaps, scores, max_output_size, iou_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```

  Args:
    overlaps: A 2-D float `Tensor` of shape `[num_boxes, num_boxes]`.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non-max suppression.
    overlap_threshold: A float representing the threshold for deciding whether
      boxes overlap too much with respect to the provided overlap values.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    name: A name for the operation (optional).

  Returns:
    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the
      selected indices from the overlaps tensor, where `M <= max_output_size`.
  """
  with ops.name_scope(name, 'non_max_suppression_overlaps'):
    overlap_threshold = ops.convert_to_tensor(
        overlap_threshold, name='overlap_threshold')
    # pylint: disable=protected-access
    return gen_image_ops.non_max_suppression_with_overlaps(
        overlaps, scores, max_output_size, overlap_threshold, score_threshold)
    # pylint: enable=protected-access


_rgb_to_yiq_kernel = [[0.299, 0.59590059, 0.2115],
                      [0.587, -0.27455667, -0.52273617],
                      [0.114, -0.32134392, 0.31119955]]


@tf_export('image.rgb_to_yiq')
def rgb_to_yiq(images):
  """Converts one or more images from RGB to YIQ.

  Outputs a tensor of the same shape as the `images` tensor, containing the YIQ
  value of the pixels.
  The output is only well defined if the value in images are in [0,1].

  Args:
    images: 2-D or higher rank. Image data to convert. Last dimension must be
      size 3.

  Returns:
    images: tensor with the same shape as `images`.
  """
  images = ops.convert_to_tensor(images, name='images')
  kernel = ops.convert_to_tensor(
      _rgb_to_yiq_kernel, dtype=images.dtype, name='kernel')
  ndims = images.get_shape().ndims
  return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])


_yiq_to_rgb_kernel = [[1, 1, 1], [0.95598634, -0.27201283, -1.10674021],
                      [0.6208248, -0.64720424, 1.70423049]]


@tf_export('image.yiq_to_rgb')
def yiq_to_rgb(images):
  """Converts one or more images from YIQ to RGB.

  Outputs a tensor of the same shape as the `images` tensor, containing the RGB
  value of the pixels.
  The output is only well defined if the Y value in images are in [0,1],
  I value are in [-0.5957,0.5957] and Q value are in [-0.5226,0.5226].

  Args:
    images: 2-D or higher rank. Image data to convert. Last dimension must be
      size 3.

  Returns:
    images: tensor with the same shape as `images`.
  """
  images = ops.convert_to_tensor(images, name='images')
  kernel = ops.convert_to_tensor(
      _yiq_to_rgb_kernel, dtype=images.dtype, name='kernel')
  ndims = images.get_shape().ndims
  return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])


_rgb_to_yuv_kernel = [[0.299, -0.14714119, 0.61497538],
                      [0.587, -0.28886916, -0.51496512],
                      [0.114, 0.43601035, -0.10001026]]


@tf_export('image.rgb_to_yuv')
def rgb_to_yuv(images):
  """Converts one or more images from RGB to YUV.

  Outputs a tensor of the same shape as the `images` tensor, containing the YUV
  value of the pixels.
  The output is only well defined if the value in images are in [0,1].

  Args:
    images: 2-D or higher rank. Image data to convert. Last dimension must be
      size 3.

  Returns:
    images: tensor with the same shape as `images`.
    
  Usage Example:
  ```python
  >> import tensorflow as tf
  >> x = tf.random.normal(shape=(256, 256, 3))
  >> tf.image.rgb_to_yuv(x)
  ```
    
  """
  images = ops.convert_to_tensor(images, name='images')
  kernel = ops.convert_to_tensor(
      _rgb_to_yuv_kernel, dtype=images.dtype, name='kernel')
  ndims = images.get_shape().ndims
  return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])


_yuv_to_rgb_kernel = [[1, 1, 1], [0, -0.394642334, 2.03206185],
                      [1.13988303, -0.58062185, 0]]


@tf_export('image.yuv_to_rgb')
def yuv_to_rgb(images):
  """Converts one or more images from YUV to RGB.

  Outputs a tensor of the same shape as the `images` tensor, containing the RGB
  value of the pixels.
  The output is only well defined if the Y value in images are in [0,1],
  U and V value are in [-0.5,0.5].

  Args:
    images: 2-D or higher rank. Image data to convert. Last dimension must be
      size 3.

  Returns:
    images: tensor with the same shape as `images`.
  """
  images = ops.convert_to_tensor(images, name='images')
  kernel = ops.convert_to_tensor(
      _yuv_to_rgb_kernel, dtype=images.dtype, name='kernel')
  ndims = images.get_shape().ndims
  return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])


def _verify_compatible_image_shapes(img1, img2):
  """Checks if two image tensors are compatible for applying SSIM or PSNR.

  This function checks if two sets of images have ranks at least 3, and if the
  last three dimensions match.

  Args:
    img1: Tensor containing the first image batch.
    img2: Tensor containing the second image batch.

  Returns:
    A tuple containing: the first tensor shape, the second tensor shape, and a
    list of control_flow_ops.Assert() ops implementing the checks.

  Raises:
    ValueError: When static shape check fails.
  """
  shape1 = img1.get_shape().with_rank_at_least(3)
  shape2 = img2.get_shape().with_rank_at_least(3)
  shape1[-3:].assert_is_compatible_with(shape2[-3:])

  if shape1.ndims is not None and shape2.ndims is not None:
    for dim1, dim2 in zip(
        reversed(shape1.dims[:-3]), reversed(shape2.dims[:-3])):
      if not (dim1 == 1 or dim2 == 1 or dim1.is_compatible_with(dim2)):
        raise ValueError('Two images are not compatible: %s and %s' %
                         (shape1, shape2))

  # Now assign shape tensors.
  shape1, shape2 = array_ops.shape_n([img1, img2])

  # TODO(sjhwang): Check if shape1[:-3] and shape2[:-3] are broadcastable.
  checks = []
  checks.append(
      control_flow_ops.Assert(
          math_ops.greater_equal(array_ops.size(shape1), 3), [shape1, shape2],
          summarize=10))
  checks.append(
      control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(shape1[-3:], shape2[-3:])),
          [shape1, shape2],
          summarize=10))
  return shape1, shape2, checks


@tf_export('image.psnr')
def psnr(a, b, max_val, name=None):
  """Returns the Peak Signal-to-Noise Ratio between a and b.

  This is intended to be used on signals (or images). Produces a PSNR value for
  each image in batch.

  The last three dimensions of input are expected to be [height, width, depth].

  Example:

  ```python
      # Read images from file.
      im1 = tf.decode_png('path/to/im1.png')
      im2 = tf.decode_png('path/to/im2.png')
      # Compute PSNR over tf.uint8 Tensors.
      psnr1 = tf.image.psnr(im1, im2, max_val=255)

      # Compute PSNR over tf.float32 Tensors.
      im1 = tf.image.convert_image_dtype(im1, tf.float32)
      im2 = tf.image.convert_image_dtype(im2, tf.float32)
      psnr2 = tf.image.psnr(im1, im2, max_val=1.0)
      # psnr1 and psnr2 both have type tf.float32 and are almost equal.
  ```

  Arguments:
    a: First set of images.
    b: Second set of images.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    name: Namespace to embed the computation in.

  Returns:
    The scalar PSNR between a and b. The returned tensor has type `tf.float32`
    and shape [batch_size, 1].
  """
  with ops.name_scope(name, 'PSNR', [a, b]):
    # Need to convert the images to float32.  Scale max_val accordingly so that
    # PSNR is computed correctly.
    max_val = math_ops.cast(max_val, a.dtype)
    max_val = convert_image_dtype(max_val, dtypes.float32)
    a = convert_image_dtype(a, dtypes.float32)
    b = convert_image_dtype(b, dtypes.float32)
    mse = math_ops.reduce_mean(math_ops.squared_difference(a, b), [-3, -2, -1])
    psnr_val = math_ops.subtract(
        20 * math_ops.log(max_val) / math_ops.log(10.0),
        np.float32(10 / np.log(10)) * math_ops.log(mse),
        name='psnr')

    _, _, checks = _verify_compatible_image_shapes(a, b)
    with ops.control_dependencies(checks):
      return array_ops.identity(psnr_val)


def _ssim_helper(x, y, reducer, max_val, compensation=1.0, k1=0.01, k2=0.03):
  r"""Helper function for computing SSIM.

  SSIM estimates covariances with weighted sums.  The default parameters
  use a biased estimate of the covariance:
  Suppose `reducer` is a weighted sum, then the mean estimators are
    \mu_x = \sum_i w_i x_i,
    \mu_y = \sum_i w_i y_i,
  where w_i's are the weighted-sum weights, and covariance estimator is
    cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
  with assumption \sum_i w_i = 1. This covariance estimator is biased, since
    E[cov_{xy}] = (1 - \sum_i w_i ^ 2) Cov(X, Y).
  For SSIM measure with unbiased covariance estimators, pass as `compensation`
  argument (1 - \sum_i w_i ^ 2).

  Arguments:
    x: First set of images.
    y: Second set of images.
    reducer: Function that computes 'local' averages from the set of images. For
      non-convolutional version, this is usually tf.reduce_mean(x, [1, 2]), and
      for convolutional version, this is usually tf.nn.avg_pool2d or
      tf.nn.conv2d with weighted-sum kernel.
    max_val: The dynamic range (i.e., the difference between the maximum
      possible allowed value and the minimum allowed value).
    compensation: Compensation factor. See above.
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we took the values in the range of 0 < K2 < 0.4).

  Returns:
    A pair containing the luminance measure, and the contrast-structure measure.
  """

  c1 = (k1 * max_val)**2
  c2 = (k2 * max_val)**2

  # SSIM luminance measure is
  # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
  mean0 = reducer(x)
  mean1 = reducer(y)
  num0 = mean0 * mean1 * 2.0
  den0 = math_ops.square(mean0) + math_ops.square(mean1)
  luminance = (num0 + c1) / (den0 + c1)

  # SSIM contrast-structure measure is
  #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
  # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
  #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
  #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
  num1 = reducer(x * y) * 2.0
  den1 = reducer(math_ops.square(x) + math_ops.square(y))
  c2 *= compensation
  cs = (num1 - num0 + c2) / (den1 - den0 + c2)

  # SSIM score is the product of the luminance and contrast-structure measures.
  return luminance, cs


def _fspecial_gauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  size = ops.convert_to_tensor(size, dtypes.int32)
  sigma = ops.convert_to_tensor(sigma)

  coords = math_ops.cast(math_ops.range(size), sigma.dtype)
  coords -= math_ops.cast(size - 1, sigma.dtype) / 2.0

  g = math_ops.square(coords)
  g *= -0.5 / math_ops.square(sigma)

  g = array_ops.reshape(g, shape=[1, -1]) + array_ops.reshape(g, shape=[-1, 1])
  g = array_ops.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
  g = nn_ops.softmax(g)
  return array_ops.reshape(g, shape=[size, size, 1, 1])


def _ssim_per_channel(img1,
                      img2,
                      max_val=1.0,
                      filter_size=11,
                      filter_sigma=1.5,
                      k1=0.01,
                      k2=0.03):
  """Computes SSIM index between img1 and img2 per color channel.

  This function matches the standard SSIM implementation from:
  Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
  quality assessment: from error visibility to structural similarity. IEEE
  transactions on image processing.

  Details:
    - 11x11 Gaussian filter of width 1.5 is used.
    - k1 = 0.01, k2 = 0.03 as in the original paper.

  Args:
    img1: First image batch.
    img2: Second image batch.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Default value 11 (size of gaussian filter).
    filter_sigma: Default value 1.5 (width of gaussian filter).
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we took the values in the range of 0 < K2 < 0.4).

  Returns:
    A pair of tensors containing and channel-wise SSIM and contrast-structure
    values. The shape is [..., channels].
  """
  filter_size = constant_op.constant(filter_size, dtype=dtypes.int32)
  filter_sigma = constant_op.constant(filter_sigma, dtype=img1.dtype)

  shape1, shape2 = array_ops.shape_n([img1, img2])
  checks = [
      control_flow_ops.Assert(
          math_ops.reduce_all(
              math_ops.greater_equal(shape1[-3:-1], filter_size)),
          [shape1, filter_size],
          summarize=8),
      control_flow_ops.Assert(
          math_ops.reduce_all(
              math_ops.greater_equal(shape2[-3:-1], filter_size)),
          [shape2, filter_size],
          summarize=8)
  ]

  # Enforce the check to run before computation.
  with ops.control_dependencies(checks):
    img1 = array_ops.identity(img1)

  # TODO(sjhwang): Try to cache kernels and compensation factor.
  kernel = _fspecial_gauss(filter_size, filter_sigma)
  kernel = array_ops.tile(kernel, multiples=[1, 1, shape1[-1], 1])

  # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
  # but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
  compensation = 1.0

  # TODO(sjhwang): Try FFT.
  # TODO(sjhwang): Gaussian kernel is separable in space. Consider applying
  #   1-by-n and n-by-1 Gaussain filters instead of an n-by-n filter.
  def reducer(x):
    shape = array_ops.shape(x)
    x = array_ops.reshape(x, shape=array_ops.concat([[-1], shape[-3:]], 0))
    y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
    return array_ops.reshape(
        y, array_ops.concat([shape[:-3], array_ops.shape(y)[1:]], 0))

  luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation, k1,
                               k2)

  # Average over the second and the third from the last: height, width.
  axes = constant_op.constant([-3, -2], dtype=dtypes.int32)
  ssim_val = math_ops.reduce_mean(luminance * cs, axes)
  cs = math_ops.reduce_mean(cs, axes)
  return ssim_val, cs


@tf_export('image.ssim')
def ssim(img1,
         img2,
         max_val,
         filter_size=11,
         filter_sigma=1.5,
         k1=0.01,
         k2=0.03):
  """Computes SSIM index between img1 and img2.

  This function is based on the standard SSIM implementation from:
  Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
  quality assessment: from error visibility to structural similarity. IEEE
  transactions on image processing.

  Note: The true SSIM is only defined on grayscale.  This function does not
  perform any colorspace transform.  (If the input is already YUV, then it will
  compute YUV SSIM average.)

  Details:
    - 11x11 Gaussian filter of width 1.5 is used.
    - k1 = 0.01, k2 = 0.03 as in the original paper.

  The image sizes must be at least 11x11 because of the filter size.

  Example:

  ```python
      # Read images from file.
      im1 = tf.decode_png('path/to/im1.png')
      im2 = tf.decode_png('path/to/im2.png')
      # Compute SSIM over tf.uint8 Tensors.
      ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)

      # Compute SSIM over tf.float32 Tensors.
      im1 = tf.image.convert_image_dtype(im1, tf.float32)
      im2 = tf.image.convert_image_dtype(im2, tf.float32)
      ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)
      # ssim1 and ssim2 both have type tf.float32 and are almost equal.
  ```

  Args:
    img1: First image batch.
    img2: Second image batch.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Default value 11 (size of gaussian filter).
    filter_sigma: Default value 1.5 (width of gaussian filter).
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we took the values in the range of 0 < K2 < 0.4).

  Returns:
    A tensor containing an SSIM value for each image in batch.  Returned SSIM
    values are in range (-1, 1], when pixel values are non-negative. Returns
    a tensor with shape: broadcast(img1.shape[:-3], img2.shape[:-3]).
  """
  _, _, checks = _verify_compatible_image_shapes(img1, img2)
  with ops.control_dependencies(checks):
    img1 = array_ops.identity(img1)

  # Need to convert the images to float32.  Scale max_val accordingly so that
  # SSIM is computed correctly.
  max_val = math_ops.cast(max_val, img1.dtype)
  max_val = convert_image_dtype(max_val, dtypes.float32)
  img1 = convert_image_dtype(img1, dtypes.float32)
  img2 = convert_image_dtype(img2, dtypes.float32)
  ssim_per_channel, _ = _ssim_per_channel(img1, img2, max_val, filter_size,
                                          filter_sigma, k1, k2)
  # Compute average over color channels.
  return math_ops.reduce_mean(ssim_per_channel, [-1])


# Default values obtained by Wang et al.
_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


@tf_export('image.ssim_multiscale')
def ssim_multiscale(img1,
                    img2,
                    max_val,
                    power_factors=_MSSSIM_WEIGHTS,
                    filter_size=11,
                    filter_sigma=1.5,
                    k1=0.01,
                    k2=0.03):
  """Computes the MS-SSIM between img1 and img2.

  This function assumes that `img1` and `img2` are image batches, i.e. the last
  three dimensions are [height, width, channels].

  Note: The true SSIM is only defined on grayscale.  This function does not
  perform any colorspace transform.  (If the input is already YUV, then it will
  compute YUV SSIM average.)

  Original paper: Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale
  structural similarity for image quality assessment." Signals, Systems and
  Computers, 2004.

  Arguments:
    img1: First image batch.
    img2: Second image batch. Must have the same rank as img1.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    power_factors: Iterable of weights for each of the scales. The number of
      scales used is the length of the list. Index 0 is the unscaled
      resolution's weight and each increasing scale corresponds to the image
      being downsampled by 2.  Defaults to (0.0448, 0.2856, 0.3001, 0.2363,
      0.1333), which are the values obtained in the original paper.
    filter_size: Default value 11 (size of gaussian filter).
    filter_sigma: Default value 1.5 (width of gaussian filter).
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we took the values in the range of 0 < K2 < 0.4).

  Returns:
    A tensor containing an MS-SSIM value for each image in batch.  The values
    are in range [0, 1].  Returns a tensor with shape:
    broadcast(img1.shape[:-3], img2.shape[:-3]).
  """
  with ops.name_scope(None, 'MS-SSIM', [img1, img2]):
    # Convert to tensor if needed.
    img1 = ops.convert_to_tensor(img1, name='img1')
    img2 = ops.convert_to_tensor(img2, name='img2')
    # Shape checking.
    shape1, shape2, checks = _verify_compatible_image_shapes(img1, img2)
    with ops.control_dependencies(checks):
      img1 = array_ops.identity(img1)

    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    max_val = math_ops.cast(max_val, img1.dtype)
    max_val = convert_image_dtype(max_val, dtypes.float32)
    img1 = convert_image_dtype(img1, dtypes.float32)
    img2 = convert_image_dtype(img2, dtypes.float32)

    imgs = [img1, img2]
    shapes = [shape1, shape2]

    # img1 and img2 are assumed to be a (multi-dimensional) batch of
    # 3-dimensional images (height, width, channels). `heads` contain the batch
    # dimensions, and `tails` contain the image dimensions.
    heads = [s[:-3] for s in shapes]
    tails = [s[-3:] for s in shapes]

    divisor = [1, 2, 2, 1]
    divisor_tensor = constant_op.constant(divisor[1:], dtype=dtypes.int32)

    def do_pad(images, remainder):
      padding = array_ops.expand_dims(remainder, -1)
      padding = array_ops.pad(padding, [[1, 0], [1, 0]])
      return [array_ops.pad(x, padding, mode='SYMMETRIC') for x in images]

    mcs = []
    for k in range(len(power_factors)):
      with ops.name_scope(None, 'Scale%d' % k, imgs):
        if k > 0:
          # Avg pool takes rank 4 tensors. Flatten leading dimensions.
          flat_imgs = [
              array_ops.reshape(x, array_ops.concat([[-1], t], 0))
              for x, t in zip(imgs, tails)
          ]

          remainder = tails[0] % divisor_tensor
          need_padding = math_ops.reduce_any(math_ops.not_equal(remainder, 0))
          # pylint: disable=cell-var-from-loop
          padded = control_flow_ops.cond(need_padding,
                                         lambda: do_pad(flat_imgs, remainder),
                                         lambda: flat_imgs)
          # pylint: enable=cell-var-from-loop

          downscaled = [
              nn_ops.avg_pool(
                  x, ksize=divisor, strides=divisor, padding='VALID')
              for x in padded
          ]
          tails = [x[1:] for x in array_ops.shape_n(downscaled)]
          imgs = [
              array_ops.reshape(x, array_ops.concat([h, t], 0))
              for x, h, t in zip(downscaled, heads, tails)
          ]

        # Overwrite previous ssim value since we only need the last one.
        ssim_per_channel, cs = _ssim_per_channel(
            *imgs,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        mcs.append(nn_ops.relu(cs))

    # Remove the cs score for the last scale. In the MS-SSIM calculation,
    # we use the l(p) at the highest scale. l(p) * cs(p) is ssim(p).
    mcs.pop()  # Remove the cs score for the last scale.
    mcs_and_ssim = array_ops.stack(
        mcs + [nn_ops.relu(ssim_per_channel)], axis=-1)
    # Take weighted geometric mean across the scale axis.
    ms_ssim = math_ops.reduce_prod(
        math_ops.pow(mcs_and_ssim, power_factors), [-1])

    return math_ops.reduce_mean(ms_ssim, [-1])  # Avg over color channels.


@tf_export('image.image_gradients')
def image_gradients(image):
  """Returns image gradients (dy, dx) for each color channel.

  Both output tensors have the same shape as the input: [batch_size, h, w,
  d]. The gradient values are organized so that [I(x+1, y) - I(x, y)] is in
  location (x, y). That means that dy will always have zeros in the last row,
  and dx will always have zeros in the last column.

  Arguments:
    image: Tensor with shape [batch_size, h, w, d].

  Returns:
    Pair of tensors (dy, dx) holding the vertical and horizontal image
    gradients (1-step finite difference).

  Usage Example:
    ```python
    BATCH_SIZE = 1
    IMAGE_HEIGHT = 5
    IMAGE_WIDTH = 5
    CHANNELS = 1
    image = tf.reshape(tf.range(IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS,
      delta=1, dtype=tf.float32),
      shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    dx, dy = tf.image.image_gradients(image)
    print(image[0, :,:,0])
    tf.Tensor(
      [[ 0.  1.  2.  3.  4.]
      [ 5.  6.  7.  8.  9.]
      [10. 11. 12. 13. 14.]
      [15. 16. 17. 18. 19.]
      [20. 21. 22. 23. 24.]], shape=(5, 5), dtype=float32)
    print(dx[0, :,:,0])
    tf.Tensor(
      [[5. 5. 5. 5. 5.]
      [5. 5. 5. 5. 5.]
      [5. 5. 5. 5. 5.]
      [5. 5. 5. 5. 5.]
      [0. 0. 0. 0. 0.]], shape=(5, 5), dtype=float32)
    print(dy[0, :,:,0])
    tf.Tensor(
      [[1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]], shape=(5, 5), dtype=float32)
    ```

  Raises:
    ValueError: If `image` is not a 4D tensor.
  """
  if image.get_shape().ndims != 4:
    raise ValueError(
        'image_gradients expects a 4D tensor '
        '[batch_size, h, w, d], not %s.', image.get_shape())
  image_shape = array_ops.shape(image)
  batch_size, height, width, depth = array_ops.unstack(image_shape)
  dy = image[:, 1:, :, :] - image[:, :-1, :, :]
  dx = image[:, :, 1:, :] - image[:, :, :-1, :]

  # Return tensors with same size as original image by concatenating
  # zeros. Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
  shape = array_ops.stack([batch_size, 1, width, depth])
  dy = array_ops.concat([dy, array_ops.zeros(shape, image.dtype)], 1)
  dy = array_ops.reshape(dy, image_shape)

  shape = array_ops.stack([batch_size, height, 1, depth])
  dx = array_ops.concat([dx, array_ops.zeros(shape, image.dtype)], 2)
  dx = array_ops.reshape(dx, image_shape)

  return dy, dx


@tf_export('image.sobel_edges')
def sobel_edges(image):
  """Returns a tensor holding Sobel edge maps.

  Arguments:
    image: Image tensor with shape [batch_size, h, w, d] and type float32 or
      float64.  The image(s) must be 2x2 or larger.

  Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
  """
  # Define vertical and horizontal Sobel filters.
  static_image_shape = image.get_shape()
  image_shape = array_ops.shape(image)
  kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
  num_kernels = len(kernels)
  kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
  kernels = np.expand_dims(kernels, -2)
  kernels_tf = constant_op.constant(kernels, dtype=image.dtype)

  kernels_tf = array_ops.tile(
      kernels_tf, [1, 1, image_shape[-1], 1], name='sobel_filters')

  # Use depth-wise convolution to calculate edge maps per channel.
  pad_sizes = [[0, 0], [1, 1], [1, 1], [0, 0]]
  padded = array_ops.pad(image, pad_sizes, mode='REFLECT')

  # Output tensor has shape [batch_size, h, w, d * num_kernels].
  strides = [1, 1, 1, 1]
  output = nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')

  # Reshape to [batch_size, h, w, d, num_kernels].
  shape = array_ops.concat([image_shape, [num_kernels]], 0)
  output = array_ops.reshape(output, shape=shape)
  output.set_shape(static_image_shape.concatenate([num_kernels]))
  return output


def resize_bicubic(images,
                   size,
                   align_corners=False,
                   name=None,
                   half_pixel_centers=False):
  return gen_image_ops.resize_bicubic(
      images=images,
      size=size,
      align_corners=align_corners,
      half_pixel_centers=half_pixel_centers,
      name=name)


def resize_bilinear(images,
                    size,
                    align_corners=False,
                    name=None,
                    half_pixel_centers=False):
  return gen_image_ops.resize_bilinear(
      images=images,
      size=size,
      align_corners=align_corners,
      half_pixel_centers=half_pixel_centers,
      name=name)


def resize_nearest_neighbor(images,
                            size,
                            align_corners=False,
                            name=None,
                            half_pixel_centers=False):
  return gen_image_ops.resize_nearest_neighbor(
      images=images,
      size=size,
      align_corners=align_corners,
      half_pixel_centers=half_pixel_centers,
      name=name)


resize_area_deprecation = deprecation.deprecated(
    date=None,
    instructions=(
        'Use `tf.image.resize(...method=ResizeMethod.AREA...)` instead.'))
tf_export(v1=['image.resize_area'])(
    resize_area_deprecation(gen_image_ops.resize_area))

resize_bicubic_deprecation = deprecation.deprecated(
    date=None,
    instructions=(
        'Use `tf.image.resize(...method=ResizeMethod.BICUBIC...)` instead.'))
tf_export(v1=['image.resize_bicubic'])(
    resize_bicubic_deprecation(resize_bicubic))

resize_bilinear_deprecation = deprecation.deprecated(
    date=None,
    instructions=(
        'Use `tf.image.resize(...method=ResizeMethod.BILINEAR...)` instead.'))
tf_export(v1=['image.resize_bilinear'])(
    resize_bilinear_deprecation(resize_bilinear))

resize_nearest_neighbor_deprecation = deprecation.deprecated(
    date=None,
    instructions=(
        'Use `tf.image.resize(...method=ResizeMethod.NEAREST_NEIGHBOR...)` '
        'instead.'))
tf_export(v1=['image.resize_nearest_neighbor'])(
    resize_nearest_neighbor_deprecation(resize_nearest_neighbor))


@tf_export('image.crop_and_resize', v1=[])
def crop_and_resize_v2(image,
                       boxes,
                       box_indices,
                       crop_size,
                       method='bilinear',
                       extrapolation_value=0,
                       name=None):
  """Extracts crops from the input image tensor and resizes them.

  Extracts crops from the input image tensor and resizes them using bilinear
  sampling or nearest neighbor sampling (possibly with aspect ratio change) to a
  common output size specified by `crop_size`. This is more general than the
  `crop_to_bounding_box` op which extracts a fixed size slice from the input
  image and does not allow resizing or aspect ratio change.

  Returns a tensor with `crops` from the input `image` at positions defined at
  the bounding box locations in `boxes`. The cropped boxes are all resized (with
  bilinear or nearest neighbor interpolation) to a fixed
  `size = [crop_height, crop_width]`. The result is a 4-D tensor
  `[num_boxes, crop_height, crop_width, depth]`. The resizing is corner aligned.
  In particular, if `boxes = [[0, 0, 1, 1]]`, the method will give identical
  results to using `tf.compat.v1.image.resize_bilinear()` or
  `tf.compat.v1.image.resize_nearest_neighbor()`(depends on the `method`
  argument) with
  `align_corners=True`.

  Args:
    image: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
      specifies the coordinates of a box in the `box_ind[i]` image and is
      specified in normalized coordinates `[y1, x1, y2, x2]`. A normalized
      coordinate value of `y` is mapped to the image coordinate at `y *
      (image_height - 1)`, so as the `[0, 1]` interval of normalized image
      height is mapped to `[0, image_height - 1]` in image height coordinates.
      We do allow `y1` > `y2`, in which case the sampled crop is an up-down
      flipped version of the original image. The width dimension is treated
      similarly. Normalized coordinates outside the `[0, 1]` range are allowed,
      in which case we use `extrapolation_value` to extrapolate the input image
      values.
    box_indices: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0,
      batch)`. The value of `box_ind[i]` specifies the image that the `i`-th box
      refers to.
    crop_size: A 1-D tensor of 2 elements, `size = [crop_height, crop_width]`.
      All cropped image patches are resized to this size. The aspect ratio of
      the image content is not preserved. Both `crop_height` and `crop_width`
      need to be positive.
    method: An optional string specifying the sampling method for resizing. It
      can be either `"bilinear"` or `"nearest"` and default to `"bilinear"`.
      Currently two sampling methods are supported: Bilinear and Nearest
        Neighbor.
    extrapolation_value: An optional `float`. Defaults to `0`. Value used for
      extrapolation, when applicable.
    name: A name for the operation (optional).

  Returns:
    A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.

  Example:

  ```python
  import tensorflow as tf
  BATCH_SIZE = 1
  NUM_BOXES = 5
  IMAGE_HEIGHT = 256
  IMAGE_WIDTH = 256
  CHANNELS = 3
  CROP_SIZE = (24, 24)

  image = tf.random.normal(shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH,
  CHANNELS) )
  boxes = tf.random.uniform(shape=(NUM_BOXES, 4))
  box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0,
  maxval=BATCH_SIZE, dtype=tf.int32)
  output = tf.image.crop_and_resize(image, boxes, box_indices, CROP_SIZE)
  output.shape  #=> (5, 24, 24, 3)
  ```
  """
  return gen_image_ops.crop_and_resize(image, boxes, box_indices, crop_size,
                                       method, extrapolation_value, name)


@tf_export(v1=['image.crop_and_resize'])
@deprecation.deprecated_args(None,
                             'box_ind is deprecated, use box_indices instead',
                             'box_ind')
def crop_and_resize_v1(  # pylint: disable=missing-docstring
    image,
    boxes,
    box_ind=None,
    crop_size=None,
    method='bilinear',
    extrapolation_value=0,
    name=None,
    box_indices=None):
  box_ind = deprecation.deprecated_argument_lookup('box_indices', box_indices,
                                                   'box_ind', box_ind)
  return gen_image_ops.crop_and_resize(image, boxes, box_ind, crop_size, method,
                                       extrapolation_value, name)


crop_and_resize_v1.__doc__ = gen_image_ops.crop_and_resize.__doc__


@tf_export(v1=['image.extract_glimpse'])
def extract_glimpse(
    input,  # pylint: disable=redefined-builtin
    size,
    offsets,
    centered=True,
    normalized=True,
    uniform_noise=True,
    name=None):
  """Extracts a glimpse from the input tensor.

  Returns a set of windows called glimpses extracted at location
  `offsets` from the input tensor. If the windows only partially
  overlaps the inputs, the non-overlapping areas will be filled with
  random noise.

  The result is a 4-D tensor of shape `[batch_size, glimpse_height,
  glimpse_width, channels]`. The channels and batch dimensions are the
  same as that of the input tensor. The height and width of the output
  windows are specified in the `size` parameter.

  The argument `normalized` and `centered` controls how the windows are built:

  * If the coordinates are normalized but not centered, 0.0 and 1.0
    correspond to the minimum and maximum of each height and width
    dimension.
  * If the coordinates are both normalized and centered, they range from
    -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
    left corner, the lower right corner is located at (1.0, 1.0) and the
    center is at (0, 0).
  * If the coordinates are not normalized they are interpreted as
    numbers of pixels.

  Args:
    input: A `Tensor` of type `float32`. A 4-D float tensor of shape
      `[batch_size, height, width, channels]`.
    size: A `Tensor` of type `int32`. A 1-D tensor of 2 elements containing the
      size of the glimpses to extract.  The glimpse height must be specified
      first, following by the glimpse width.
    offsets: A `Tensor` of type `float32`. A 2-D integer tensor of shape
      `[batch_size, 2]` containing the y, x locations of the center of each
      window.
    centered: An optional `bool`. Defaults to `True`. indicates if the offset
      coordinates are centered relative to the image, in which case the (0, 0)
      offset is relative to the center of the input images. If false, the (0,0)
      offset corresponds to the upper left corner of the input images.
    normalized: An optional `bool`. Defaults to `True`. indicates if the offset
      coordinates are normalized.
    uniform_noise: An optional `bool`. Defaults to `True`. indicates if the
      noise should be generated using a uniform distribution or a Gaussian
      distribution.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.

  Usage Example:
    ```python
    BATCH_SIZE = 1
    IMAGE_HEIGHT = 3
    IMAGE_WIDTH = 3
    CHANNELS = 1
    GLIMPSE_SIZE = (2, 2)
    image = tf.reshape(tf.range(9, delta=1, dtype=tf.float32),
      shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    output = tf.image.extract_glimpse(image, size=GLIMPSE_SIZE,
      offsets=[[1, 1]], centered=False, normalized=False)
     ```
  """
  return gen_image_ops.extract_glimpse(
      input=input,
      size=size,
      offsets=offsets,
      centered=centered,
      normalized=normalized,
      uniform_noise=uniform_noise,
      name=name)


@tf_export('image.extract_glimpse', v1=[])
def extract_glimpse_v2(
    input,  # pylint: disable=redefined-builtin
    size,
    offsets,
    centered=True,
    normalized=True,
    noise='uniform',
    name=None):
  """Extracts a glimpse from the input tensor.

  Returns a set of windows called glimpses extracted at location
  `offsets` from the input tensor. If the windows only partially
  overlaps the inputs, the non-overlapping areas will be filled with
  random noise.

  The result is a 4-D tensor of shape `[batch_size, glimpse_height,
  glimpse_width, channels]`. The channels and batch dimensions are the
  same as that of the input tensor. The height and width of the output
  windows are specified in the `size` parameter.

  The argument `normalized` and `centered` controls how the windows are built:

  * If the coordinates are normalized but not centered, 0.0 and 1.0
    correspond to the minimum and maximum of each height and width
    dimension.
  * If the coordinates are both normalized and centered, they range from
    -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
    left corner, the lower right corner is located at (1.0, 1.0) and the
    center is at (0, 0).
  * If the coordinates are not normalized they are interpreted as
    numbers of pixels.

  Args:
    input: A `Tensor` of type `float32`. A 4-D float tensor of shape
      `[batch_size, height, width, channels]`.
    size: A `Tensor` of type `int32`. A 1-D tensor of 2 elements containing the
      size of the glimpses to extract.  The glimpse height must be specified
      first, following by the glimpse width.
    offsets: A `Tensor` of type `float32`. A 2-D integer tensor of shape
      `[batch_size, 2]` containing the y, x locations of the center of each
      window.
    centered: An optional `bool`. Defaults to `True`. indicates if the offset
      coordinates are centered relative to the image, in which case the (0, 0)
      offset is relative to the center of the input images. If false, the (0,0)
      offset corresponds to the upper left corner of the input images.
    normalized: An optional `bool`. Defaults to `True`. indicates if the offset
      coordinates are normalized.
    noise: An optional `string`. Defaults to `uniform`. indicates if the noise
      should be `uniform` (uniform distribution), `gaussian` (gaussian
      distribution), or `zero` (zero padding).
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.

  Usage Example:
    ```python
    BATCH_SIZE = 1
    IMAGE_HEIGHT = 3
    IMAGE_WIDTH = 3
    CHANNELS = 1
    GLIMPSE_SIZE = (2, 2)
    image = tf.reshape(tf.range(9, delta=1, dtype=tf.float32),
      shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    output = tf.image.extract_glimpse(image, size=GLIMPSE_SIZE,
      offsets=[[1, 1]], centered=False, normalized=False)
     ```
  """
  return gen_image_ops.extract_glimpse(
      input=input,
      size=size,
      offsets=offsets,
      centered=centered,
      normalized=normalized,
      noise=noise,
      uniform_noise=False,
      name=name)


@tf_export('image.combined_non_max_suppression')
def combined_non_max_suppression(boxes,
                                 scores,
                                 max_output_size_per_class,
                                 max_total_size,
                                 iou_threshold=0.5,
                                 score_threshold=float('-inf'),
                                 pad_per_class=False,
                                 clip_boxes=True,
                                 name=None):
  """Greedily selects a subset of bounding boxes in descending order of score.

  This operation performs non_max_suppression on the inputs per batch, across
  all classes.
  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system. Also note that
  this algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is the final boxes, scores and classes tensor
  returned after performing non_max_suppression.

  Args:
    boxes: A 4-D float `Tensor` of shape `[batch_size, num_boxes, q, 4]`. If `q`
      is 1 then same boxes are used for all classes otherwise, if `q` is equal
      to number of classes, class-specific boxes are used.
    scores: A 3-D float `Tensor` of shape `[batch_size, num_boxes, num_classes]`
      representing a single score corresponding to each box (each row of boxes).
    max_output_size_per_class: A scalar integer `Tensor` representing the
      maximum number of boxes to be selected by non-max suppression per class
    max_total_size: A scalar representing the maximum number of boxes retained
    over all classes.
    iou_threshold: A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    pad_per_class: If false, the output nmsed boxes, scores and classes are
      padded/clipped to `max_total_size`. If true, the output nmsed boxes,
      scores and classes are padded to be of length
      `max_size_per_class`*`num_classes`, unless it exceeds `max_total_size` in
      which case it is clipped to `max_total_size`. Defaults to false.
    clip_boxes: If true, the coordinates of output nmsed boxes will be clipped
      to [0, 1]. If false, output the box coordinates as it is. Defaults to
      true.
    name: A name for the operation (optional).

  Returns:
    'nmsed_boxes': A [batch_size, max_detections, 4] float32 tensor
      containing the non-max suppressed boxes.
    'nmsed_scores': A [batch_size, max_detections] float32 tensor containing
      the scores for the boxes.
    'nmsed_classes': A [batch_size, max_detections] float32 tensor
      containing the class for boxes.
    'valid_detections': A [batch_size] int32 tensor indicating the number of
      valid detections per batch item. Only the top valid_detections[i] entries
      in nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
      entries are zero paddings.
  """
  with ops.name_scope(name, 'combined_non_max_suppression'):
    iou_threshold = ops.convert_to_tensor(
        iou_threshold, dtype=dtypes.float32, name='iou_threshold')
    score_threshold = ops.convert_to_tensor(
        score_threshold, dtype=dtypes.float32, name='score_threshold')
    return gen_image_ops.combined_non_max_suppression(
        boxes, scores, max_output_size_per_class, max_total_size, iou_threshold,
        score_threshold, pad_per_class, clip_boxes)


@tf_export('image.draw_bounding_boxes', v1=[])
def draw_bounding_boxes_v2(images, boxes, colors, name=None):
  """Draw bounding boxes on a batch of images.

  Outputs a copy of `images` but draws on top of the pixels zero or more
  bounding boxes specified by the locations in `boxes`. The coordinates of the
  each bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`.
  The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width
  and the height of the underlying image.

  For example, if an image is 100 x 200 pixels (height x width) and the bounding
  box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of
  the bounding box will be `(40, 10)` to `(180, 50)` (in (x,y) coordinates).

  Parts of the bounding box may fall outside the image.

  Args:
    images: A `Tensor`. Must be one of the following types: `float32`, `half`.
      4-D with shape `[batch, height, width, depth]`. A batch of images.
    boxes: A `Tensor` of type `float32`. 3-D with shape `[batch,
      num_bounding_boxes, 4]` containing bounding boxes.
    colors: A `Tensor` of type `float32`. 2-D. A list of RGBA colors to cycle
      through for the boxes.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
  if colors is None:
    return gen_image_ops.draw_bounding_boxes(images, boxes, name)
  return gen_image_ops.draw_bounding_boxes_v2(images, boxes, colors, name)


@tf_export(v1=['image.draw_bounding_boxes'])
def draw_bounding_boxes(images, boxes, name=None, colors=None):
  """Draw bounding boxes on a batch of images.

  Outputs a copy of `images` but draws on top of the pixels zero or more
  bounding boxes specified by the locations in `boxes`. The coordinates of the
  each bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`.
  The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width
  and the height of the underlying image.

  For example, if an image is 100 x 200 pixels (height x width) and the bounding
  box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of
  the bounding box will be `(40, 10)` to `(180, 50)` (in (x,y) coordinates).

  Parts of the bounding box may fall outside the image.

  Args:
    images: A `Tensor`. Must be one of the following types: `float32`, `half`.
      4-D with shape `[batch, height, width, depth]`. A batch of images.
    boxes: A `Tensor` of type `float32`. 3-D with shape `[batch,
      num_bounding_boxes, 4]` containing bounding boxes.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
  return draw_bounding_boxes_v2(images, boxes, colors, name)


@tf_export('image.generate_bounding_box_proposals')
def generate_bounding_box_proposals(scores,
                                    bbox_deltas,
                                    image_info,
                                    anchors,
                                    nms_threshold=0.7,
                                    pre_nms_topn=6000,
                                    min_size=16,
                                    post_nms_topn=300,
                                    name=None):
  """ Generate bounding box proposals from encoded bounding boxes.

  Returns:
    rois: Region of interest boxes sorted by their scores.
    roi_probabilities: scores of the ROI boxes in the ROIs' tensor.
  """
  return gen_image_ops.generate_bounding_box_proposals(
      scores=scores,
      bbox_deltas=bbox_deltas,
      image_info=image_info,
      anchors=anchors,
      nms_threshold=nms_threshold,
      pre_nms_topn=pre_nms_topn,
      min_size=min_size,
      post_nms_topn=post_nms_topn,
      name=name)
