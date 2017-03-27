# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Python layer for image_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.image.ops import gen_image_ops
from tensorflow.contrib.util import loader
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import resource_loader

_image_ops_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("_image_ops.so"))

_IMAGE_DTYPES = set(
    [dtypes.uint8, dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64])

ops.RegisterShape("ImageProjectiveTransform")(common_shapes.call_cpp_shape_fn)


def rotate(images, angles):
  """Rotate image(s) by the passed angle(s) in radians.

  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
       (NHWC), (num_rows, num_columns, num_channels) (HWC), or
       (num_rows, num_columns) (HW).
    angles: A scalar angle to rotate all images by, or (if images has rank 4)
       a vector of length num_images, with an angle for each image in the batch.

  Returns:
    Image(s) with the same type and shape as `images`, rotated by the given
    angle(s). Empty space due to the rotation will be filled with zeros.

  Raises:
    TypeError: If `image` is an invalid type.
  """
  image_or_images = ops.convert_to_tensor(images, name="images")
  if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
    raise TypeError("Invalid dtype %s." % image_or_images.dtype)
  if len(image_or_images.get_shape()) == 2:
    images = image_or_images[None, :, :, None]
  elif len(image_or_images.get_shape()) == 3:
    images = image_or_images[None, :, :, :]
  elif len(image_or_images.get_shape()) == 4:
    images = image_or_images
  else:
    raise TypeError("Images should have rank between 2 and 4.")

  image_height = math_ops.cast(array_ops.shape(images)[1], dtypes.float32)[None]
  image_width = math_ops.cast(array_ops.shape(images)[2], dtypes.float32)[None]
  output = transform(
      images,
      angles_to_projective_transforms(angles, image_width, image_height))
  if len(image_or_images.get_shape()) == 2:
    return output[0, :, :, 0]
  elif len(image_or_images.get_shape()) == 3:
    return output[0, :, :, :]
  else:
    return output


def angles_to_projective_transforms(angles, image_height, image_width):
  """Returns projective transform(s) for the given angle(s).

  Args:
    angles: A scalar angle to rotate all images by, or (for batches of images)
      a vector with an angle to rotate each image in the batch.
    image_height: Height of the image(s) to be transformed.
    image_width: Width of the image(s) to be transformed.

  Returns:
    A tensor of shape (num_images, 8). Projective transforms which can be given
      to `tf.contrib.image.transform`.
  """
  angle_or_angles = ops.convert_to_tensor(
      angles, name="angles", dtype=dtypes.float32)
  if len(angle_or_angles.get_shape()) == 0:  # pylint: disable=g-explicit-length-test
    angles = angle_or_angles[None]
  elif len(angle_or_angles.get_shape()) == 1:
    angles = angle_or_angles
  else:
    raise TypeError("Angles should have rank 0 or 1.")
  x_offset = ((image_width - 1) - (math_ops.cos(angles) *
                                   (image_width - 1) - math_ops.sin(angles) *
                                   (image_height - 1))) / 2.0
  y_offset = ((image_height - 1) - (math_ops.sin(angles) *
                                    (image_width - 1) + math_ops.cos(angles) *
                                    (image_height - 1))) / 2.0
  num_angles = array_ops.shape(angles)[0]
  return array_ops.concat(
      values=[
          math_ops.cos(angles)[:, None],
          -math_ops.sin(angles)[:, None],
          x_offset[:, None],
          math_ops.sin(angles)[:, None],
          math_ops.cos(angles)[:, None],
          y_offset[:, None],
          array_ops.zeros((num_angles, 2), dtypes.float32),
      ],
      axis=1)


def transform(images, transforms):
  """Applies the given transform(s) to the image(s).

  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
       (NHWC), (num_rows, num_columns, num_channels) (HWC), or
       (num_rows, num_columns) (HW).
    transforms: Projective transform matrix/matrices. A vector of length 8 or
       tensor of size N x 8. If one row of transforms is
       [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
       `(x, y)` to a transformed *input* point
       `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
       where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to
       the transform mapping input points to output points.

  Returns:
    Image(s) with the same type and shape as `images`, with the given
    transform(s) applied. Transformed coordinates outside of the input image
    will be filled with zeros.

  Raises:
    TypeError: If `image` is an invalid type.
  """
  image_or_images = ops.convert_to_tensor(images, name="images")
  transform_or_transforms = ops.convert_to_tensor(
      transforms, name="transforms", dtype=dtypes.float32)
  if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
    raise TypeError("Invalid dtype %s." % image_or_images.dtype)
  if len(image_or_images.get_shape()) == 2:
    images = image_or_images[None, :, :, None]
  elif len(image_or_images.get_shape()) == 3:
    images = image_or_images[None, :, :, :]
  elif len(image_or_images.get_shape()) == 4:
    images = image_or_images
  else:
    raise TypeError("Images should have rank between 2 and 4.")

  if len(transform_or_transforms.get_shape()) == 1:
    transforms = transform_or_transforms[None]
  elif len(transform_or_transforms.get_shape()) == 2:
    transforms = transform_or_transforms
  else:
    raise TypeError("Transforms should have rank 1 or 2.")
  # pylint: disable=protected-access
  output = gen_image_ops.image_projective_transform(images, transforms)
  if len(image_or_images.get_shape()) == 2:
    return output[0, :, :, 0]
  elif len(image_or_images.get_shape()) == 3:
    return output[0, :, :, :]
  else:
    return output


def compose_transforms(*transforms):
  """Composes the transforms tensors.

  Args:
    *transforms: List of image projective transforms to be composed. Each
        transform is length 8 (single transform) or shape (N, 8) (batched
        transforms). The shapes of all inputs must be equal, and at least one
        input must be given.

  Returns:
    A composed transform tensor. When passed to `tf.contrib.image.transform`,
        equivalent to applying each of the given transforms to the image in
        order.
  """
  assert transforms, "transforms cannot be empty"
  composed = _flat_transforms_to_matrices(transforms[0])
  for tr in transforms[1:]:
    # Multiply batches of matrices.
    composed = math_ops.matmul(composed, _flat_transforms_to_matrices(tr))
  return _transform_matrices_to_flat(composed)


def _flat_transforms_to_matrices(transforms):
  # Make the transform(s) 2D in case the input is a single transform.
  transforms = array_ops.reshape(transforms, constant_op.constant([-1, 8]))
  num_transforms = array_ops.shape(transforms)[0]
  # Add a column of ones for the implicit last entry in the matrix.
  return array_ops.reshape(
      array_ops.concat(
          [transforms, array_ops.ones([num_transforms, 1])], axis=1),
      constant_op.constant([-1, 3, 3]))


def _transform_matrices_to_flat(transform_matrices):
  # Flatten each matrix.
  transforms = array_ops.reshape(
      transform_matrices, constant_op.constant([-1, 9]))
  # Divide each matrix by the last entry (normally 1).
  transforms /= transforms[:, 8:9]
  return transforms[:, :8]


ops.NotDifferentiable("ImageProjectiveTransform")
