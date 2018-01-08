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
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import resource_loader

_image_ops_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("_image_ops.so"))

_IMAGE_DTYPES = set(
    [dtypes.uint8, dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64])

ops.RegisterShape("ImageProjectiveTransform")(common_shapes.call_cpp_shape_fn)


def rotate(images, angles, interpolation="NEAREST", name=None):
  """Rotate image(s) by the passed angle(s) in radians.

  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
       (NHWC), (num_rows, num_columns, num_channels) (HWC), or
       (num_rows, num_columns) (HW). The rank must be statically known (the
       shape is not `TensorShape(None)`.
    angles: A scalar angle to rotate all images by, or (if images has rank 4)
       a vector of length num_images, with an angle for each image in the batch.
    interpolation: Interpolation mode. Supported values: "NEAREST", "BILINEAR".
    name: The name of the op.

  Returns:
    Image(s) with the same type and shape as `images`, rotated by the given
    angle(s). Empty space due to the rotation will be filled with zeros.

  Raises:
    TypeError: If `image` is an invalid type.
  """
  with ops.name_scope(name, "rotate"):
    image_or_images = ops.convert_to_tensor(images)
    if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
      raise TypeError("Invalid dtype %s." % image_or_images.dtype)
    elif image_or_images.get_shape().ndims is None:
      raise TypeError("image_or_images rank must be statically known")
    elif len(image_or_images.get_shape()) == 2:
      images = image_or_images[None, :, :, None]
    elif len(image_or_images.get_shape()) == 3:
      images = image_or_images[None, :, :, :]
    elif len(image_or_images.get_shape()) == 4:
      images = image_or_images
    else:
      raise TypeError("Images should have rank between 2 and 4.")

    image_height = math_ops.cast(array_ops.shape(images)[1],
                                 dtypes.float32)[None]
    image_width = math_ops.cast(array_ops.shape(images)[2],
                                dtypes.float32)[None]
    output = transform(
        images,
        angles_to_projective_transforms(angles, image_height, image_width),
        interpolation=interpolation)
    if image_or_images.get_shape().ndims is None:
      raise TypeError("image_or_images rank must be statically known")
    elif len(image_or_images.get_shape()) == 2:
      return output[0, :, :, 0]
    elif len(image_or_images.get_shape()) == 3:
      return output[0, :, :, :]
    else:
      return output


def translate(images, translations, interpolation="NEAREST", name=None):
  """Translate image(s) by the passed vectors(s).

  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
        (NHWC), (num_rows, num_columns, num_channels) (HWC), or
        (num_rows, num_columns) (HW). The rank must be statically known (the
        shape is not `TensorShape(None)`.
    translations: A vector representing [dx, dy] or (if images has rank 4)
        a matrix of length num_images, with a [dx, dy] vector for each image in
        the batch.
    interpolation: Interpolation mode. Supported values: "NEAREST", "BILINEAR".
    name: The name of the op.

  Returns:
    Image(s) with the same type and shape as `images`, translated by the given
        vector(s). Empty space due to the translation will be filled with zeros.

  Raises:
    TypeError: If `image` is an invalid type.
  """
  with ops.name_scope(name, "translate"):
    return transform(
        images,
        translations_to_projective_transforms(translations),
        interpolation=interpolation)


def angles_to_projective_transforms(angles,
                                    image_height,
                                    image_width,
                                    name=None):
  """Returns projective transform(s) for the given angle(s).

  Args:
    angles: A scalar angle to rotate all images by, or (for batches of images)
        a vector with an angle to rotate each image in the batch. The rank must
        be statically known (the shape is not `TensorShape(None)`.
    image_height: Height of the image(s) to be transformed.
    image_width: Width of the image(s) to be transformed.

  Returns:
    A tensor of shape (num_images, 8). Projective transforms which can be given
      to `tf.contrib.image.transform`.
  """
  with ops.name_scope(name, "angles_to_projective_transforms"):
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


def translations_to_projective_transforms(translations, name=None):
  """Returns projective transform(s) for the given translation(s).

  Args:
      translations: A 2-element list representing [dx, dy] or a matrix of
          2-element lists representing [dx, dy] to translate for each image
          (for a batch of images). The rank must be statically known (the shape
          is not `TensorShape(None)`.
      name: The name of the op.

  Returns:
      A tensor of shape (num_images, 8) projective transforms which can be given
          to `tf.contrib.image.transform`.
  """
  with ops.name_scope(name, "translations_to_projective_transforms"):
    translation_or_translations = ops.convert_to_tensor(
        translations, name="translations", dtype=dtypes.float32)
    if translation_or_translations.get_shape().ndims is None:
      raise TypeError(
          "translation_or_translations rank must be statically known")
    elif len(translation_or_translations.get_shape()) == 1:
      translations = translation_or_translations[None]
    elif len(translation_or_translations.get_shape()) == 2:
      translations = translation_or_translations
    else:
      raise TypeError("Translations should have rank 1 or 2.")
    num_translations = array_ops.shape(translations)[0]
    # The translation matrix looks like:
    #     [[1 0 -dx]
    #      [0 1 -dy]
    #      [0 0 1]]
    # where the last entry is implicit.
    # Translation matrices are always float32.
    return array_ops.concat(
        values=[
            array_ops.ones((num_translations, 1), dtypes.float32),
            array_ops.zeros((num_translations, 1), dtypes.float32),
            -translations[:, 0, None],
            array_ops.zeros((num_translations, 1), dtypes.float32),
            array_ops.ones((num_translations, 1), dtypes.float32),
            -translations[:, 1, None],
            array_ops.zeros((num_translations, 2), dtypes.float32),
        ],
        axis=1)


def transform(images, transforms, interpolation="NEAREST", name=None):
  """Applies the given transform(s) to the image(s).

  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
       (NHWC), (num_rows, num_columns, num_channels) (HWC), or
       (num_rows, num_columns) (HW). The rank must be statically known (the
       shape is not `TensorShape(None)`.
    transforms: Projective transform matrix/matrices. A vector of length 8 or
       tensor of size N x 8. If one row of transforms is
       [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
       `(x, y)` to a transformed *input* point
       `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
       where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to
       the transform mapping input points to output points. Note that gradients
       are not backpropagated into transformation parameters.
    interpolation: Interpolation mode. Supported values: "NEAREST", "BILINEAR".

  Returns:
    Image(s) with the same type and shape as `images`, with the given
    transform(s) applied. Transformed coordinates outside of the input image
    will be filled with zeros.

  Raises:
    TypeError: If `image` is an invalid type.
  """
  with ops.name_scope(name, "transform"):
    image_or_images = ops.convert_to_tensor(images, name="images")
    transform_or_transforms = ops.convert_to_tensor(
        transforms, name="transforms", dtype=dtypes.float32)
    if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
      raise TypeError("Invalid dtype %s." % image_or_images.dtype)
    elif image_or_images.get_shape().ndims is None:
      raise TypeError("image_or_images rank must be statically known")
    elif len(image_or_images.get_shape()) == 2:
      images = image_or_images[None, :, :, None]
    elif len(image_or_images.get_shape()) == 3:
      images = image_or_images[None, :, :, :]
    elif len(image_or_images.get_shape()) == 4:
      images = image_or_images
    else:
      raise TypeError("Images should have rank between 2 and 4.")

    if len(transform_or_transforms.get_shape()) == 1:
      transforms = transform_or_transforms[None]
    elif transform_or_transforms.get_shape().ndims is None:
      raise TypeError(
          "transform_or_transforms rank must be statically known")
    elif len(transform_or_transforms.get_shape()) == 2:
      transforms = transform_or_transforms
    else:
      raise TypeError("Transforms should have rank 1 or 2.")
    output = gen_image_ops.image_projective_transform(
        images, transforms, interpolation=interpolation.upper())
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
  with ops.name_scope("compose_transforms"):
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
  transforms = array_ops.reshape(transform_matrices,
                                 constant_op.constant([-1, 9]))
  # Divide each matrix by the last entry (normally 1).
  transforms /= transforms[:, 8:9]
  return transforms[:, :8]


@ops.RegisterGradient("ImageProjectiveTransform")
def _image_projective_transform_grad(op, grad):
  """Computes the gradient for ImageProjectiveTransform."""
  images = op.inputs[0]
  transforms = op.inputs[1]
  interpolation = op.get_attr("interpolation")

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
    raise TypeError("Images should have rank between 2 and 4")
  if len(transform_or_transforms.get_shape()) == 1:
    transforms = transform_or_transforms[None]
  elif len(transform_or_transforms.get_shape()) == 2:
    transforms = transform_or_transforms
  else:
    raise TypeError("Transforms should have rank 1 or 2.")

  # Invert transformations
  transforms = _flat_transforms_to_matrices(transforms=transforms)
  inverse = linalg_ops.matrix_inverse(transforms)
  transforms = _transform_matrices_to_flat(inverse)
  output = gen_image_ops.image_projective_transform(
      grad, transforms, interpolation=interpolation)
  if len(image_or_images.get_shape()) == 2:
    return [output[0, :, :, 0], None]
  elif len(image_or_images.get_shape()) == 3:
    return [output[0, :, :, :], None]
  else:
    return [output, None]


def bipartite_match(distance_mat,
                    num_valid_rows,
                    top_k=-1,
                    name="bipartite_match"):
  """Find bipartite matching based on a given distance matrix.

  A greedy bi-partite matching algorithm is used to obtain the matching with
  the (greedy) minimum distance.

  Args:
    distance_mat: A 2-D float tensor of shape `[num_rows, num_columns]`. It is a
      pair-wise distance matrix between the entities represented by each row and
      each column. It is an asymmetric matrix. The smaller the distance is, the
      more similar the pairs are. The bipartite matching is to minimize the
      distances.
    num_valid_rows: A scalar or a 1-D tensor with one element describing the
      number of valid rows of distance_mat to consider for the bipartite
      matching. If set to be negative, then all rows from `distance_mat` are
      used.
    top_k: A scalar that specifies the number of top-k matches to retrieve.
      If set to be negative, then is set according to the maximum number of
      matches from `distance_mat`.
    name: The name of the op.

  Returns:
    row_to_col_match_indices: A vector of length num_rows, which is the number
      of rows of the input `distance_matrix`. If `row_to_col_match_indices[i]`
      is not -1, row i is matched to column `row_to_col_match_indices[i]`.
    col_to_row_match_indices: A vector of length num_columns, which is the
      number of columns of the input ditance matrix.
      If `col_to_row_match_indices[j]` is not -1, column j is matched to row
      `col_to_row_match_indices[j]`.
  """
  result = gen_image_ops.bipartite_match(
      distance_mat, num_valid_rows, top_k, name=name)
  return result


ops.NotDifferentiable("BipartiteMatch")
