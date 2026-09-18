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

"""Image ops.

The `tf.image` module contains various functions for image
processing and decoding-encoding Ops.

Many of the encoding/decoding functions are also available in the
core `tf.io` module.

## Image processing

### Resizing

The resizing Ops accept input images as tensors of several types. They always
output resized images as float32 tensors.

The convenience function `tf.image.resize` supports both 4-D
and 3-D tensors as input and output.  4-D tensors are for batches of images,
3-D tensors for individual images.

Resized images will be distorted if their original aspect ratio is not the
same as size. To avoid distortions see tf.image.resize_with_pad.

*   `tf.image.resize`
*   `tf.image.resize_with_pad`
*   `tf.image.resize_with_crop_or_pad`

The Class `tf.image.ResizeMethod` provides various resize methods like
`bilinear`, `nearest_neighbor`.

### Converting Between Colorspaces

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

TensorFlow can convert between images in RGB or HSV or YIQ.

*   `tf.image.rgb_to_grayscale`, `tf.image.grayscale_to_rgb`
*   `tf.image.rgb_to_hsv`, `tf.image.hsv_to_rgb`
*   `tf.image.rgb_to_yiq`, `tf.image.yiq_to_rgb`
*   `tf.image.rgb_to_yuv`, `tf.image.yuv_to_rgb`
*   `tf.image.image_gradients`
*   `tf.image.convert_image_dtype`

### Image Adjustments

TensorFlow provides functions to adjust images in various ways: brightness,
contrast, hue, and saturation.  Each adjustment can be done with predefined
parameters or with random parameters picked from predefined intervals. Random
adjustments are often useful to expand a training set and reduce overfitting.

If several adjustments are chained it is advisable to minimize the number of
redundant conversions by first converting the images to the most natural data
type and representation.

*   `tf.image.adjust_brightness`
*   `tf.image.adjust_contrast`
*   `tf.image.adjust_gamma`
*   `tf.image.adjust_hue`
*   `tf.image.adjust_jpeg_quality`
*   `tf.image.adjust_saturation`
*   `tf.image.random_brightness`
*   `tf.image.random_contrast`
*   `tf.image.random_hue`
*   `tf.image.random_saturation`
*   `tf.image.per_image_standardization`

### Working with Bounding Boxes

*   `tf.image.draw_bounding_boxes`
*   `tf.image.combined_non_max_suppression`
*   `tf.image.generate_bounding_box_proposals`
*   `tf.image.non_max_suppression`
*   `tf.image.non_max_suppression_overlaps`
*   `tf.image.non_max_suppression_padded`
*   `tf.image.non_max_suppression_with_scores`
*   `tf.image.pad_to_bounding_box`
*   `tf.image.sample_distorted_bounding_box`

### Cropping

*   `tf.image.central_crop`
*   `tf.image.crop_and_resize`
*   `tf.image.crop_to_bounding_box`
*   `tf.io.decode_and_crop_jpeg`
*   `tf.image.extract_glimpse`
*   `tf.image.random_crop`
*   `tf.image.resize_with_crop_or_pad`

### Flipping, Rotating and Transposing

*   `tf.image.flip_left_right`
*   `tf.image.flip_up_down`
*   `tf.image.random_flip_left_right`
*   `tf.image.random_flip_up_down`
*   `tf.image.rot90`
*   `tf.image.transpose`

## Image decoding and encoding

TensorFlow provides Ops to decode and encode JPEG and PNG formats.  Encoded
images are represented by scalar string Tensors, decoded images by 3-D uint8
tensors of shape `[height, width, channels]`. (PNG also supports uint16.)

Note: `decode_gif` returns a 4-D array `[num_frames, height, width, 3]`

The encode and decode Ops apply to one image at a time.  Their input and output
are all of variable size.  If you need fixed size images, pass the output of
the decode Ops to one of the cropping and resizing Ops.

*   `tf.io.decode_bmp`
*   `tf.io.decode_gif`
*   `tf.io.decode_image`
*   `tf.io.decode_jpeg`
*   `tf.io.decode_and_crop_jpeg`
*   `tf.io.decode_png`
*   `tf.io.encode_jpeg`
*   `tf.io.encode_png`

API docstring: tensorflow.image
"""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_image_ops import *
from tensorflow.python.ops.image_ops_impl import *
# pylint: enable=wildcard-import

# TODO(drpng): remove these once internal use has discontinued.
# pylint: disable=unused-import
from tensorflow.python.ops.image_ops_impl import _Check3DImage
from tensorflow.python.ops.image_ops_impl import _ImageDimensions
# pylint: enable=unused-import

_IMAGE_DTYPES = frozenset([
    dtypes.uint8, dtypes.int32, dtypes.int64, dtypes.float16, dtypes.float32,
    dtypes.float64
])


def flat_transforms_to_matrices(transforms):
  """Converts `tf.contrib.image` projective transforms to affine matrices.

  Note that the output matrices map output coordinates to input coordinates. For
  the forward transformation matrix, call `tf.linalg.inv` on the result.

  Args:
    transforms: Vector of length 8, or batches of transforms with shape `(N,
      8)`.

  Returns:
    3D tensor of matrices with shape `(N, 3, 3)`. The output matrices map the
      *output coordinates* (in homogeneous coordinates) of each transform to the
      corresponding *input coordinates*.

  Raises:
    ValueError: If `transforms` have an invalid shape.
  """
  with ops.name_scope("flat_transforms_to_matrices"):
    transforms = ops.convert_to_tensor(transforms, name="transforms")
    if transforms.shape.ndims not in (1, 2):
      raise ValueError("Transforms should be 1D or 2D, got: %s" % transforms)
    # Make the transform(s) 2D in case the input is a single transform.
    transforms = array_ops.reshape(transforms, constant_op.constant([-1, 8]))
    num_transforms = array_ops.shape(transforms)[0]
    # Add a column of ones for the implicit last entry in the matrix.
    return array_ops.reshape(
        array_ops.concat(
            [transforms, array_ops.ones([num_transforms, 1])], axis=1),
        constant_op.constant([-1, 3, 3]))


def matrices_to_flat_transforms(transform_matrices):
  """Converts affine matrices to `tf.contrib.image` projective transforms.

  Note that we expect matrices that map output coordinates to input coordinates.
  To convert forward transformation matrices, call `tf.linalg.inv` on the
  matrices and use the result here.

  Args:
    transform_matrices: One or more affine transformation matrices, for the
      reverse transformation in homogeneous coordinates. Shape `(3, 3)` or `(N,
      3, 3)`.

  Returns:
    2D tensor of flat transforms with shape `(N, 8)`, which may be passed into
      `tf.contrib.image.transform`.

  Raises:
    ValueError: If `transform_matrices` have an invalid shape.
  """
  with ops.name_scope("matrices_to_flat_transforms"):
    transform_matrices = ops.convert_to_tensor(
        transform_matrices, name="transform_matrices")
    if transform_matrices.shape.ndims not in (2, 3):
      raise ValueError("Matrices should be 2D or 3D, got: %s" %
                       transform_matrices)
    # Flatten each matrix.
    transforms = array_ops.reshape(transform_matrices,
                                   constant_op.constant([-1, 9]))
    # Divide each matrix by the last entry (normally 1).
    transforms /= transforms[:, 8:9]
    return transforms[:, :8]


def _trunc_div(a, b):
  """Integer division that truncates toward zero, like C++'s `/` on ints.

  `a / b` followed by a cast matches `static_cast<DenseIndex>(a / b)` in
  `MapCoordinate` (tensorflow/core/kernels/image/image_ops.h); Python and
  TensorFlow's own integer division instead floor, which disagrees for
  negative inputs.
  """
  return math_ops.cast(math_ops.cast(a / b, dtypes.int32), dtypes.float32)


def _map_coordinate(coord, length, fill_mode):
  """Maps an output-space coordinate to an input-space one, per `fill_mode`.

  Mirrors `MapCoordinate<Mode>` in
  tensorflow/core/kernels/image/image_ops.h exactly, so that the adjoint
  below reads from (in the forward op, sense) or writes to (here) the same
  pixel the forward kernel does.

  Args:
    coord: float32 tensor, arbitrary shape.
    length: scalar int32 tensor, the input extent along this axis.
    fill_mode: Python str, one of "CONSTANT", "NEAREST", "REFLECT", "WRAP".

  Returns:
    float32 tensor, same shape as `coord`.
  """
  if fill_mode == "CONSTANT":
    return coord

  length_f = math_ops.cast(length, dtypes.float32)
  last = length_f - 1.0

  if fill_mode == "NEAREST":
    return clip_ops.clip_by_value(coord, 0.0, last)

  # `length <= 1` makes `last`/`sz2` 0, which would divide by zero below;
  # the result is discarded by the `length <= 1` select further down, but
  # an intermediate inf/nan cast to int32 is platform-dependent undefined
  # behavior, so avoid it rather than merely discard it.
  safe_last = array_ops.where_v2(length <= 1, 1.0, last)
  if fill_mode == "WRAP":
    below = coord + length_f * (_trunc_div(-coord, safe_last) + 1.0)
    above = coord - length_f * _trunc_div(coord, safe_last)
  elif fill_mode == "REFLECT":
    sz2 = 2.0 * length_f
    safe_sz2 = array_ops.where_v2(length <= 1, 2.0, sz2)
    folded_low = sz2 * _trunc_div(-coord, safe_sz2) + coord
    below = array_ops.where_v2(folded_low < -length_f, folded_low + sz2,
                               -folded_low - 1.0)
    folded_high = coord - sz2 * _trunc_div(coord, safe_sz2)
    above = array_ops.where_v2(folded_high >= length_f,
                               sz2 - folded_high - 1.0, folded_high)
  else:
    raise ValueError("Unknown fill_mode %r" % (fill_mode,))

  mapped = array_ops.where_v2(coord < 0.0, below,
                              array_ops.where_v2(coord > last, above, coord))
  # `length <= 1` collapses every coordinate to 0, matching the kernel.
  mapped = array_ops.where_v2(length <= 1, array_ops.zeros_like(coord), mapped)
  return clip_ops.clip_by_value(mapped, 0.0, last)


def _round_half_away_from_zero(x):
  """Matches C++ `std::round`, used by the forward kernel's NEAREST tap.

  TensorFlow's own `round` breaks ties to even instead, which disagrees with
  the kernel at exact `.5` boundaries.
  """
  return math_ops.sign(x) * math_ops.floor(math_ops.abs(x) + 0.5)


def _image_projective_transform_grad_impl(images, transforms, grad,
                                          interpolation, fill_mode):
  """Shared adjoint for ImageProjectiveTransformV2/V3.

  The forward op resamples `images` at a coordinate that `transforms` maps
  each output pixel to, optionally blending up to 4 input pixels
  (BILINEAR) or reading exactly 1 (NEAREST). Its true gradient scatter-adds
  each output pixel's incoming gradient onto the same input pixel(s), with
  the same weights. (The previous implementation instead re-ran the forward
  *resampling* op on `grad` through the inverted transform, which is only
  correct when the forward coordinate map is a bijection on the pixel grid
  -- never true once `fill_mode` clamps or folds an out-of-range coordinate,
  or the transform downscales.)

  Args:
    images: the forward op's `images` input; only its shape/dtype are used.
    transforms: the forward op's `transforms` input, rank 1 `[8]` or rank 2
      `[N, 8]` with `N` equal to 1 (broadcast) or to the batch size.
    grad: gradient w.r.t. the forward op's output, `[B, out_H, out_W, C]`.
    interpolation: Python str, "NEAREST" or "BILINEAR".
    fill_mode: Python str, "CONSTANT", "NEAREST", "REFLECT", or "WRAP".

  Returns:
    Gradient w.r.t. `images`, shape `[B, in_H, in_W, C]`, dtype `grad.dtype`.
  """
  if images.dtype.base_dtype not in _IMAGE_DTYPES:
    raise TypeError("Invalid dtype %s." % images.dtype)

  transforms = ops.convert_to_tensor(
      transforms, name="transforms", dtype=dtypes.float32)
  if transforms.shape.ndims == 1:
    transforms = transforms[None]
  elif transforms.shape.ndims != 2:
    raise TypeError("Transforms should have rank 1 or 2.")

  image_shape = array_ops.shape(images)
  batch, in_h, in_w, channels = (image_shape[0], image_shape[1],
                                 image_shape[2], image_shape[3])
  grad_shape = array_ops.shape(grad)
  out_h, out_w = grad_shape[1], grad_shape[2]

  # `transforms` may carry one row (broadcast to the whole batch) or one row
  # per batch element, decided by its actual leading dimension at runtime --
  # gather a `[batch, 8]` view either way rather than branching on shape.
  num_transforms = array_ops.shape(transforms)[0]
  transform_row = array_ops.where_v2(
      math_ops.equal(num_transforms, 1),
      array_ops.zeros([batch], dtype=dtypes.int32), math_ops.range(batch))
  t = array_ops.gather(transforms, transform_row, axis=0)  # [batch, 8]

  oy, ox = array_ops.meshgrid(
      math_ops.range(out_h), math_ops.range(out_w), indexing="ij")
  oy = math_ops.cast(oy, dtypes.float32)  # [out_H, out_W]
  ox = math_ops.cast(ox, dtypes.float32)

  def _coef(i):
    return t[:, i, None, None]  # [batch, 1, 1], broadcasts against ox/oy.

  proj = _coef(6) * ox + _coef(7) * oy + 1.0  # [batch, out_H, out_W]
  valid_proj = math_ops.not_equal(proj, 0.0)
  safe_proj = array_ops.where_v2(valid_proj, proj, array_ops.ones_like(proj))
  input_x = (_coef(0) * ox + _coef(1) * oy + _coef(2)) / safe_proj
  input_y = (_coef(3) * ox + _coef(4) * oy + _coef(5)) / safe_proj

  x = _map_coordinate(input_x, in_w, fill_mode)
  y = _map_coordinate(input_y, in_h, fill_mode)

  batch_idx = array_ops.broadcast_to(
      array_ops.reshape(math_ops.range(batch), [-1, 1, 1]),
      array_ops.shape(x))  # int32, [batch, out_H, out_W]

  grad = ops.convert_to_tensor(grad)
  compute_dtype = grad.dtype.base_dtype  # accumulate at the tape's precision
  grad_flat = array_ops.reshape(grad, [-1, channels])

  def _scatter_tap(iy, ix, weight):
    """One interpolation tap's indices/updates, out-of-range taps zeroed."""
    in_bounds = math_ops.logical_and(
        math_ops.logical_and(iy >= 0.0, iy < math_ops.cast(in_h, iy.dtype)),
        math_ops.logical_and(ix >= 0.0, ix < math_ops.cast(in_w, ix.dtype)))
    keep = math_ops.logical_and(in_bounds, valid_proj)
    zero_i = array_ops.zeros_like(iy)
    safe_iy = math_ops.cast(array_ops.where_v2(keep, iy, zero_i), dtypes.int32)
    safe_ix = math_ops.cast(array_ops.where_v2(keep, ix, zero_i), dtypes.int32)
    indices = array_ops_stack.stack([batch_idx, safe_iy, safe_ix], axis=-1)
    weight = array_ops.where_v2(keep, weight, array_ops.zeros_like(weight))
    weight = math_ops.cast(array_ops.reshape(weight, [-1, 1]), compute_dtype)
    return array_ops.reshape(indices, [-1, 3]), weight * grad_flat

  if interpolation == "NEAREST":
    iy = _round_half_away_from_zero(y)
    ix = _round_half_away_from_zero(x)
    taps = [(iy, ix, array_ops.ones_like(x))]
  elif interpolation == "BILINEAR":
    y_floor, x_floor = math_ops.floor(y), math_ops.floor(x)
    y_ceil, x_ceil = y_floor + 1.0, x_floor + 1.0
    wy_floor, wy_ceil = y_ceil - y, y - y_floor
    wx_floor, wx_ceil = x_ceil - x, x - x_floor
    taps = [
        (y_floor, x_floor, wy_floor * wx_floor),
        (y_floor, x_ceil, wy_floor * wx_ceil),
        (y_ceil, x_floor, wy_ceil * wx_floor),
        (y_ceil, x_ceil, wy_ceil * wx_ceil),
    ]
  else:
    raise ValueError("Unknown interpolation %r" % (interpolation,))

  all_indices, all_updates = [], []
  for iy, ix, weight in taps:
    indices, updates = _scatter_tap(iy, ix, weight)
    all_indices.append(indices)
    all_updates.append(updates)

  return array_ops.scatter_nd(
      array_ops.concat(all_indices, axis=0),
      array_ops.concat(all_updates, axis=0),
      array_ops_stack.stack([batch, in_h, in_w, channels]))


@ops.RegisterGradient("ImageProjectiveTransformV2")
def _image_projective_transform_grad(op, grad):
  """Computes the gradient for ImageProjectiveTransform."""
  output = _image_projective_transform_grad_impl(
      images=op.inputs[0],
      transforms=op.inputs[1],
      grad=grad,
      interpolation=op.get_attr("interpolation").decode(),
      fill_mode=op.get_attr("fill_mode").decode())
  return [output, None, None]


@ops.RegisterGradient("ImageProjectiveTransformV3")
def _image_projective_transform_v3_grad(op, grad):
  """Computes the gradient for ImageProjectiveTransform."""
  output = _image_projective_transform_grad_impl(
      images=op.inputs[0],
      transforms=op.inputs[1],
      grad=grad,
      interpolation=op.get_attr("interpolation").decode(),
      fill_mode=op.get_attr("fill_mode").decode())
  return [output, None, None, None]
