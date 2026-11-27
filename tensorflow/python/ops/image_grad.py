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
"""Contains Gradient functions for image ops."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops


@ops.RegisterGradient("ResizeNearestNeighbor")
def _ResizeNearestNeighborGrad(op: ops.Operation, grad):
  """The derivatives for nearest neighbor resizing.

  Args:
    op: The ResizeNearestNeighbor op.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the input and the output.
  """
  image = op.inputs[0]
  if image.get_shape()[1:3].is_fully_defined():
    image_shape = image.get_shape()[1:3]
  else:
    image_shape = array_ops.shape(image)[1:3]

  grads = gen_image_ops.resize_nearest_neighbor_grad(
      grad,
      image_shape,
      align_corners=op.get_attr("align_corners"),
      half_pixel_centers=op.get_attr("half_pixel_centers"))
  return [grads, None]


@ops.RegisterGradient("ResizeNearestNeighborGrad")
def _ResizeNearestNeighborGradGrad(op: ops.Operation, grad):
  """The transpose of the nearest neighbor resize gradient is a resize."""
  image = op.inputs[0]
  if image.get_shape()[1:3].is_fully_defined():
    image_shape = image.get_shape()[1:3]
  else:
    image_shape = array_ops.shape(image)[1:3]
  return [
      gen_image_ops.resize_nearest_neighbor(
          grad,
          image_shape,
          align_corners=op.get_attr("align_corners"),
          half_pixel_centers=op.get_attr("half_pixel_centers"),
      ),
      None,
  ]


@ops.RegisterGradient("AdjustContrastv2")
def _AdjustContrastGrad(op: ops.Operation, grad):
  """The derivatives for `tf.image.adjust_contrast`.

  The kernel computes `(images - mean) * contrast_factor + mean`, where
  `mean` is taken per batch and channel over the last three dimensions,
  which are interpreted as [height, width, channels].

  Args:
    op: The `AdjustContrastv2` `Operation`.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the images and the contrast factor.
  """
  images = op.inputs[0]
  factor = op.inputs[1]
  factor_t = math_ops.cast(factor, images.dtype)
  static_rank = images.shape.rank
  if static_rank is not None:
    spatial_axes = list(range(static_rank - 3, static_rank - 1))
  else:
    dynamic_rank = array_ops.rank(images)
    spatial_axes = math_ops.range(dynamic_rank - 3, dynamic_rank - 1)
  mean = math_ops.reduce_mean(images, axis=spatial_axes, keepdims=True)
  grad_images = grad * factor_t + (1.0 - factor_t) * math_ops.reduce_mean(
      grad,
      axis=spatial_axes,
      keepdims=True,
  )
  grad_factor = math_ops.reduce_sum(
      math_ops.cast(grad, dtypes.float32)
      * (
          math_ops.cast(images, dtypes.float32)
          - math_ops.cast(mean, dtypes.float32)
      )
  )
  return grad_images, grad_factor


@ops.RegisterGradient("ResizeBilinear")
def _ResizeBilinearGrad(op: ops.Operation, grad):
  """The derivatives for bilinear resizing.

  Args:
    op: The ResizeBilinear op.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the input.
  """
  image = op.inputs[0]
  grad0 = None
  if image.dtype.is_floating:
    grad0 = gen_image_ops.resize_bilinear_grad(
        grad,
        image,
        align_corners=op.get_attr("align_corners"),
        half_pixel_centers=op.get_attr("half_pixel_centers"),
    )
  return [grad0, None]


ops.NotDifferentiable("ResizeBilinearGrad")


@ops.RegisterGradient("ScaleAndTranslate")
def _ScaleAndTranslateGrad(op, grad):
  """The derivatives for ScaleAndTranslate transformation op.

  Args:
    op: The ScaleAndTranslate op.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the input.
  """

  image = op.inputs[0]
  grad0 = None
  if image.dtype.is_floating:
    # ScaleAndTranslateGrad only has a float32 kernel and only reads the shape
    # of the original image, while the forward op also accepts half, bfloat16
    # and double images. Run it in float32 and return the input's dtype.
    grad0 = gen_image_ops.scale_and_translate_grad(
        grad,
        math_ops.cast(image, dtypes.float32),
        op.inputs[2],
        op.inputs[3],
        kernel_type=op.get_attr("kernel_type"),
        antialias=op.get_attr("antialias"),
    )
    grad0 = math_ops.cast(grad0, image.dtype)
  return [grad0, None, None, None]


@ops.RegisterGradient("ResizeBicubic")
def _ResizeBicubicGrad(op: ops.Operation, grad):
  """The derivatives for bicubic resizing.

  Args:
    op: The ResizeBicubic op.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the input.
  """
  allowed_types = [dtypes.float32, dtypes.float64]
  image = op.inputs[0]
  grad0 = None
  if image.dtype.is_floating:
    # ResizeBicubicGrad only takes float32 and float64 images, so compute the
    # gradient for half and bfloat16 images in float32.
    if image.dtype not in allowed_types:
      image = math_ops.cast(image, dtypes.float32)
    grad0 = gen_image_ops.resize_bicubic_grad(
        grad,
        image,
        align_corners=op.get_attr("align_corners"),
        half_pixel_centers=op.get_attr("half_pixel_centers"),
    )
    grad0 = math_ops.cast(grad0, op.inputs[0].dtype)
  return [grad0, None]


@ops.RegisterGradient("CropAndResize")
def _CropAndResizeGrad(op: ops.Operation, grad):
  """The derivatives for crop_and_resize.

  We back-propagate to the image only when the input image tensor has floating
  point dtype but we always back-propagate to the input boxes tensor.

  Args:
    op: The CropAndResize op.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the input image, boxes, as well as the always-None
    gradients w.r.t. box_ind and crop_size.
  """
  image = op.inputs[0]
  if image.get_shape().is_fully_defined():
    image_shape = image.get_shape().as_list()
  else:
    image_shape = array_ops.shape(image)

  allowed_types = [dtypes.float16, dtypes.float32, dtypes.float64]
  if op.inputs[0].dtype in allowed_types:
    # pylint: disable=protected-access
    grad0 = gen_image_ops.crop_and_resize_grad_image(
        grad, op.inputs[1], op.inputs[2], image_shape, T=op.get_attr("T"),
        method=op.get_attr("method"))
    # pylint: enable=protected-access
  else:
    grad0 = None

  # `grad0` is the gradient to the input image pixels and it
  # has been implemented for nearest neighbor and bilinear sampling
  # respectively. `grad1` is the gradient to the input crop boxes' coordinates.
  # When using nearest neighbor sampling, the gradient to crop boxes'
  # coordinates are not well defined. In practice, we still approximate
  # grad1 using the gradient derived from bilinear sampling.
  grad1 = gen_image_ops.crop_and_resize_grad_boxes(
      grad, op.inputs[0], op.inputs[1], op.inputs[2])

  return [grad0, grad1, None, None]


def _CustomReciprocal(x):
  """Wrapper function around `math_ops.div_no_nan()` to perform a "safe" reciprocal incase the input is zero. Avoids divide by zero and NaNs.

  Input:
    x -> input tensor to be reciprocat-ed.
  Returns:
    x_reciprocal -> reciprocal of x without NaNs.
  """
  return math_ops.div_no_nan(math_ops.cast(1.0, x.dtype), x)


def _rgb_to_hsv_grad_math(rgb, grad, hsv=None):
  """The gradients for the `rgb_to_hsv` operation, as a plain function.

  This function is a piecewise continuous function as defined here:
  https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB
  We perform the multivariate derivative and compute all partial derivatives
  separately before adding them in the end. Formulas are given before each
  partial derivative calculation.

  Args:
    rgb: The RGB input of the `rgb_to_hsv` operation, with the channels in the
      last dimension.
    grad: Gradient with respect to the output of the `rgb_to_hsv` op.
    hsv: The output of the `rgb_to_hsv` operation, or `None`. When `None`, the
      saturation and value are recomputed from `rgb` with the same tie-breaking
      masks used for the derivatives.

  Returns:
    Gradients with respect to the input of `rgb_to_hsv`.
  """
  # Input Channels
  reds = rgb[..., 0]
  greens = rgb[..., 1]
  blues = rgb[..., 2]

  dtype = rgb.dtype

  # Mask/Indicator for max and min values of each pixel.
  # Arbitrary assignment in case of tie breakers with R>G>B.
  # Max values
  red_biggest = math_ops.cast((reds >= blues) & \
                 (reds >= greens), dtype)
  green_biggest = math_ops.cast((greens > reds) & \
                   (greens >= blues), dtype)
  blue_biggest = math_ops.cast((blues > reds) & \
                  (blues > greens), dtype)
  # Min values
  red_smallest = math_ops.cast((reds < blues) & \
                  (reds < greens), dtype)
  green_smallest = math_ops.cast((greens <= reds) & \
                    (greens < blues), dtype)
  blue_smallest = math_ops.cast((blues <= reds) & \
                   (blues <= greens), dtype)

  # Output Channels
  if hsv is None:
    value = (red_biggest * reds + green_biggest * greens +
             blue_biggest * blues)
    minimum = (red_smallest * reds + green_smallest * greens +
               blue_smallest * blues)
    saturation = math_ops.div_no_nan(value - minimum, value)
  else:
    saturation = hsv[..., 1]
    value = hsv[..., 2]

  # Derivatives of R, G, B wrt Value slice
  dv_dr = red_biggest
  dv_dg = green_biggest
  dv_db = blue_biggest

  # Derivatives of R, G, B wrt Saturation slice

  # The first term in the addition is the case when the corresponding color
  # from (r,g,b) was "MAX"
  # -> derivative = MIN/square(MAX), MIN could be one of the other two colors
  # The second term is the case when the corresponding color from
  # (r,g,b) was "MIN"
  # -> derivative = -1/MAX, MAX could be one of the other two colours.
  ds_dr = math_ops.cast(reds > 0, dtype) * math_ops.add(
      red_biggest * math_ops.add(green_smallest * greens, blue_smallest * blues)
      * _CustomReciprocal(math_ops.square(reds)), red_smallest * -1 *
      _CustomReciprocal((green_biggest * greens) + (blue_biggest * blues)))
  ds_dg = math_ops.cast(greens > 0, dtype) * math_ops.add(
      green_biggest * math_ops.add(red_smallest * reds, blue_smallest * blues) *
      _CustomReciprocal(math_ops.square(greens)), green_smallest * -1 *
      _CustomReciprocal((red_biggest * reds) + (blue_biggest * blues)))
  ds_db = math_ops.cast(blues > 0, dtype) * math_ops.add(
      blue_biggest * math_ops.add(green_smallest * greens, red_smallest * reds)
      * _CustomReciprocal(math_ops.square(blues)), blue_smallest * -1 *
      _CustomReciprocal((green_biggest * greens) + (red_biggest * reds)))

  # Derivatives of R, G, B wrt Hue slice

  # Need to go case by case for each color.
  # for red, dh_dr -> dh_dr_1 + dh_dr_2 + dh_dr_3 + dh_dr_4 + dh_dr_5
  # dh_dr_1 ->
  # if red was MAX, then derivative = 60 * -1 * (G-B)/square(MAX-MIN) == 60 *\
  #  -1 * (greens-blues) * reciprocal(square(saturation)) * \
  #  reciprocal(square(value))
  # elif green was MAX, there are two subcases
  # ie when red was MIN and when red was NOT MIN
  #   dh_dr_2 ->
  #   if red was MIN (use UV rule) ->  60 * ((1 * -1/(MAX-MIN)) +\
  #     (B-R)*(-1/square(MAX-MIN) * -1)) == 60 * (blues - greens) *\
  #     reciprocal(square(reds - greens))
  #   dh_dr_3 ->
  #   if red was NOT MIN -> 60 * -1/MAX-MIN == -60 * reciprocal(greens-blues)
  # elif blue was MAX, there are two subcases
  #   dh_dr_4 ->
  #   if red was MIN (similarly use the UV rule) -> 60 * (blues - greens) *\
  #     reciprocal(square(blues - reds))
  #   dh_dr_5 ->
  #   if red was NOT MIN -> 60 * 1/MAX-MIN == 60 * reciprocal(blues-greens)
  dh_dr_1 = 60 * (
      math_ops.cast(reds > 0, dtype) * red_biggest * -1 *
      (greens - blues) * _CustomReciprocal(math_ops.square(saturation)) *
      _CustomReciprocal(math_ops.square(value)))
  dh_dr_2 = 60 * (
      math_ops.cast(greens > 0, dtype) * green_biggest * red_smallest *
      (blues - greens) * _CustomReciprocal(math_ops.square(reds - greens)))
  dh_dr_3 = 60 * (
      math_ops.cast(greens > 0, dtype) * green_biggest * blue_smallest * -1 *
      _CustomReciprocal(greens - blues))
  dh_dr_4 = 60 * (
      math_ops.cast(blues > 0, dtype) * blue_biggest * red_smallest *
      (blues - greens) * _CustomReciprocal(math_ops.square(blues - reds)))
  dh_dr_5 = 60 * (
      math_ops.cast(blues > 0, dtype) * blue_biggest * green_smallest *
      _CustomReciprocal(blues - greens))

  dh_dr = dh_dr_1 + dh_dr_2 + dh_dr_3 + dh_dr_4 + dh_dr_5
  # Converting from degrees to [0,1] scale as specified in
  # https://www.tensorflow.org/api_docs/python/tf/image/rgb_to_hsv
  dh_dr = dh_dr / 360

  # for green, dh_dg -> dh_dg_1 + dh_dg_2 + dh_dg_3 + dh_dg_4 + dh_dg_5
  # dh_dg_1 ->
  # if green was MAX, then derivative = 60 * -1 * (B-R)/square(MAX-MIN) == 60 *\
  #   -1 * (blues - reds) * reciprocal(square(saturation)) * \
  #   reciprocal(square(value))
  # elif red was MAX, there are two subcases ie
  # when green was MIN and when green was NOT MIN
  #   dh_dg_2 ->
  #   if green was MIN (use UV rule) ->  60 * ((1 * 1/(MAX-MIN)) + \
  #     (greens-blues) * (-1/square(MAX-MIN) * -1)) == 60 * \
  #     ((reciprocal(reds-greens) + (greens-blues) * \
  #     reciprocal(square(reds-greens))))
  #   dh_dg_3 ->
  #   if green was NOT MIN -> 60 * 1/MAX-MIN == 60 * reciprocal(reds - blues)
  # elif blue was MAX, there are two subcases
  #   dh_dg_4 ->
  #   if green was MIN (similarly use the UV rule) -> 60 * -1 * \
  #     (reciprocal(blues - greens) + (reds-greens)* -1 * \
  #     reciprocal(square(blues-greens)))
  #   dh_dr_5 ->
  #   if green was NOT MIN -> 60 * -1/MAX-MIN == -60 * reciprocal(blues - reds)
  dh_dg_1 = 60 * (
      math_ops.cast(greens > 0, dtype) * green_biggest * -1 *
      (blues - reds) * _CustomReciprocal(math_ops.square(saturation)) *
      _CustomReciprocal(math_ops.square(value)))
  dh_dg_2 = 60 * (
      math_ops.cast(reds > 0, dtype) * red_biggest * green_smallest *
      (reds - blues) * _CustomReciprocal(math_ops.square(reds - greens)))
  dh_dg_3 = 60 * (
      math_ops.cast(reds > 0, dtype) * red_biggest * blue_smallest *
      _CustomReciprocal(reds - blues))
  dh_dg_4 = 60 * (
      math_ops.cast(blues > 0, dtype) * blue_biggest * green_smallest *
      (reds - blues) * _CustomReciprocal(math_ops.square(blues - greens)))
  dh_dg_5 = 60 * (
      math_ops.cast(blues > 0, dtype) * blue_biggest * red_smallest * -1 *
      _CustomReciprocal(blues - reds))

  dh_dg = dh_dg_1 + dh_dg_2 + dh_dg_3 + dh_dg_4 + dh_dg_5
  # Converting from degrees to [0,1] scale as specified in
  # https://www.tensorflow.org/api_docs/python/tf/image/rgb_to_hsv
  dh_dg = dh_dg / 360

  # for blue, dh_db -> dh_db_1 + dh_db_2 + dh_db_3 + dh_db_4 + dh_db_5
  # dh_db_1 ->
  # if blue was MAX, then derivative = 60 * -1 * (R-G)/square(MAX-MIN) == 60 *\
  #   -1 * reciprocal(square(saturation)) * reciprocal(square(value))
  # elif red was MAX, there are two subcases
  # ie when blue was MIN and when blue was NOT MIN
  #   dh_dg_2 ->
  #   if blue was MIN (use UV rule) ->  60 * ((1 * -1/(MAX-MIN)) + \
  #     (greens-blues) * (-1/square(MAX-MIN) * -1)) == 60 * (greens - reds) *\
  #     reciprocal(square(reds - blues))
  #   dh_dg_3 ->
  #   if blue was NOT MIN -> 60 * -1/MAX-MIN == 60 * -1 * \
  #     reciprocal(reds - greens)
  # elif green was MAX, there are two subcases
  #   dh_dg_4 ->
  #   if blue was MIN (similarly use the UV rule) -> 60 * -1 * \
  #     (reciprocal(greens - blues) + (blues - reds) * -1 * \
  #     reciprocal(square(greens - blues)))
  #   dh_dr_5 ->
  #   if blue was NOT MIN -> 60 * 1/MAX-MIN == 60 * reciprocal(greens - reds)
  dh_db_1 = 60 * (
      math_ops.cast(blues > 0, dtype) * blue_biggest * -1 *
      (reds - greens) * _CustomReciprocal(math_ops.square(saturation)) *
      _CustomReciprocal(math_ops.square(value)))
  dh_db_2 = 60 * (
      math_ops.cast(reds > 0, dtype) * red_biggest * blue_smallest *
      (greens - reds) * _CustomReciprocal(math_ops.square(reds - blues)))
  dh_db_3 = 60 * (
      math_ops.cast(reds > 0, dtype) * red_biggest * green_smallest * -1 *
      _CustomReciprocal(reds - greens))
  dh_db_4 = 60 * (
      math_ops.cast(greens > 0, dtype) * green_biggest * blue_smallest *
      (greens - reds) * _CustomReciprocal(math_ops.square(greens - blues)))
  dh_db_5 = 60 * (
      math_ops.cast(greens > 0, dtype) * green_biggest * red_smallest *
      _CustomReciprocal(greens - reds))

  dh_db = dh_db_1 + dh_db_2 + dh_db_3 + dh_db_4 + dh_db_5
  # Converting from degrees to [0,1] scale as specified in
  # https://www.tensorflow.org/api_docs/python/tf/image/rgb_to_hsv
  dh_db = dh_db / 360

  # Gradients wrt to inputs
  dv_drgb = array_ops_stack.stack(
      [grad[..., 2] * dv_dr, grad[..., 2] * dv_dg, grad[..., 2] * dv_db],
      axis=-1)
  ds_drgb = array_ops_stack.stack(
      [grad[..., 1] * ds_dr, grad[..., 1] * ds_dg, grad[..., 1] * ds_db],
      axis=-1)
  dh_drgb = array_ops_stack.stack(
      [grad[..., 0] * dh_dr, grad[..., 0] * dh_dg, grad[..., 0] * dh_db],
      axis=-1)

  gradient_input = math_ops.add(math_ops.add(dv_drgb, ds_drgb), dh_drgb)
  return gradient_input


@ops.RegisterGradient("RGBToHSV")
def _RGBToHSVGrad(op: ops.Operation, grad):
  """The gradients for `rgb_to_hsv` operation.

  The formulas are in `_rgb_to_hsv_grad_math`; the saturation and value slices
  are taken from the forward op's output.

  Args:
    op: The `rgb_to_hsv` `Operation` that we are differentiating.
    grad: Gradient with respect to the output of the `rgb_to_hsv` op.

  Returns:
    Gradients with respect to the input of `rgb_to_hsv`.
  """
  return _rgb_to_hsv_grad_math(op.inputs[0], grad, hsv=op.outputs[0])


def _hsv_to_rgb_grad_math(hsv, grad):
  """The gradients for the `hsv_to_rgb` operation, as a plain function.

  The kernel computes, with `dh = 6 * h`,

    r = v * (1 - s * (1 - clip(|dh - 3| - 1, 0, 1)))
    g = v * (1 - s * (1 - clip(2 - |dh - 2|, 0, 1)))
    b = v * (1 - s * (1 - clip(2 - |dh - 4|, 0, 1)))

  which is linear in `v`, linear in `s` and piecewise linear in `h`. The hue
  derivative of each clipped term is +-6 on the two ramps of its hue sector and
  zero where the term is clipped. At the breakpoints between those pieces the
  function is not differentiable and the zero subgradient is used.

  Args:
    hsv: The HSV input of the `hsv_to_rgb` operation, with the channels in the
      last dimension.
    grad: Gradient with respect to the output of the `hsv_to_rgb` op.

  Returns:
    Gradients with respect to the input of `hsv_to_rgb`.
  """
  dtype = hsv.dtype
  hue = hsv[..., 0]
  saturation = hsv[..., 1]
  value = hsv[..., 2]
  grad_r = grad[..., 0]
  grad_g = grad[..., 1]
  grad_b = grad[..., 2]

  dh = hue * 6

  def _ramp(distance, offset, sign):
    """Returns `clip(sign * (|distance| - offset), 0, 1)` and its hue slope."""
    ramp_arg = sign * (math_ops.abs(distance) - offset)
    ramp = math_ops.minimum(math_ops.maximum(ramp_arg, 0), 1)
    on_ramp = math_ops.cast((ramp_arg > 0) & (ramp_arg < 1), dtype)
    ramp_slope = 6 * sign * math_ops.sign(distance) * on_ramp
    return ramp, ramp_slope

  dr, dr_dh = _ramp(dh - 3, 1, 1)
  dg, dg_dh = _ramp(dh - 2, 2, -1)
  db, db_dh = _ramp(dh - 4, 2, -1)

  grad_h = value * saturation * (grad_r * dr_dh + grad_g * dg_dh +
                                 grad_b * db_dh)
  grad_s = value * ((dr - 1) * grad_r + (dg - 1) * grad_g + (db - 1) * grad_b)
  grad_v = ((1 - saturation * (1 - dr)) * grad_r +
            (1 - saturation * (1 - dg)) * grad_g +
            (1 - saturation * (1 - db)) * grad_b)

  return array_ops_stack.stack([grad_h, grad_s, grad_v], axis=-1)


@ops.RegisterGradient("HSVToRGB")
def _HSVToRGBGrad(op: ops.Operation, grad):
  """The gradients for the `hsv_to_rgb` operation.

  The formulas are in `_hsv_to_rgb_grad_math`.

  Args:
    op: The `hsv_to_rgb` `Operation` that we are differentiating.
    grad: Gradient with respect to the output of the `hsv_to_rgb` op.

  Returns:
    Gradients with respect to the input of `hsv_to_rgb`.
  """
  return _hsv_to_rgb_grad_math(op.inputs[0], grad)


def _image_adjust_hsv_forward(rgb):
  """Recomputes the per-pixel HSV values used by the adjust op kernels.

  Matches the helpers in `adjust_hue_op.cc` and `adjust_saturation_op.cc`:
  `value` is the largest channel, the saturation is the channel range divided
  by `value` (zero when `value` is zero) and the hue lives in `[0, 1)`. Ties
  between equal channels are broken the same way the kernels compare them, and
  the hue is zero where the range is zero because it has no influence there.

  Args:
    rgb: RGB tensor with the channels in the last dimension.

  Returns:
    Tuple of `hue`, `saturation` and `value` tensors with the shape of each
    pixel (all dimensions except the last one of `rgb`).
  """
  reds = rgb[..., 0]
  greens = rgb[..., 1]
  blues = rgb[..., 2]
  value = math_ops.reduce_max(rgb, axis=-1)
  minimum = math_ops.reduce_min(rgb, axis=-1)
  channel_range = value - minimum
  saturation = math_ops.div_no_nan(channel_range, value)
  hue_step = math_ops.div_no_nan(1.0 / 6.0, channel_range)
  hue_red = (greens - blues) * hue_step
  hue_green = (blues - reds) * hue_step + 1.0 / 3.0
  hue_blue = (reds - greens) * hue_step + 2.0 / 3.0
  hue = array_ops.where_v2(
      reds >= math_ops.maximum(greens, blues), hue_red,
      array_ops.where_v2(greens >= blues, hue_green, hue_blue))
  # The kernels wrap the hue back into [0, 1); the second wrap only guards
  # against rounding at the seam.
  hue = array_ops.where_v2(hue < 0.0, hue + 1.0, hue)
  hue = array_ops.where_v2(hue >= 1.0, hue - 1.0, hue)
  return hue, saturation, value


@ops.RegisterGradient("AdjustHue")
def _AdjustHueGrad(op: ops.Operation, grad):
  """The derivatives for `tf.image.adjust_hue`.

  The kernel converts each pixel to (hue in `[0, 6]`, `v_min`, `v_max`), adds
  `6 * delta` to the hue, wraps it back into `[0, 6)` and reconstructs the RGB
  channels sector-wise linearly from the shifted hue and the unchanged
  `v_min`/`v_max`. That reconstruction is the `hsv_to_rgb` formula with the
  same saturation and value, so the output gradient is chained through the
  `hsv_to_rgb` gradient evaluated at the shifted hue and through the
  `rgb_to_hsv` gradient back to the input channels. `delta` only shifts the
  hue with slope 1, and the output is continuous across the wrap points, so
  the hue gradient doubles as the delta gradient.

  Args:
    op: The `AdjustHue` `Operation` that we are differentiating.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the images and the delta.
  """
  images = op.inputs[0]
  delta = op.inputs[1]
  grad = math_ops.cast(grad, dtypes.float32)
  rgb = math_ops.cast(images, dtypes.float32)
  hue, saturation, value = _image_adjust_hsv_forward(rgb)
  hue_out = math_ops.mod(hue + delta, 1.0)
  grad_hue_sat_val = _hsv_to_rgb_grad_math(
      array_ops_stack.stack([hue_out, saturation, value], axis=-1), grad)
  grad_delta = math_ops.reduce_sum(grad_hue_sat_val[..., 0])
  grad_images = _rgb_to_hsv_grad_math(rgb, grad_hue_sat_val)
  return [math_ops.cast(grad_images, images.dtype), grad_delta]


@ops.RegisterGradient("AdjustSaturation")
def _AdjustSaturationGrad(op: ops.Operation, grad):
  """The derivatives for `tf.image.adjust_saturation`.

  The kernel converts each pixel RGB -> HSV, multiplies the saturation by
  `scale`, clamps it into `[0, 1]` and converts back. The conversion back is
  the `hsv_to_rgb` formula with the clamped saturation, so the output gradient
  is chained through the `hsv_to_rgb` gradient evaluated at the clamped
  saturation and through the `rgb_to_hsv` gradient back to the input channels.
  Inside the clamp the saturation is `s * scale`, so it differentiates to
  `scale` with respect to the input saturation and to `s` with respect to
  `scale`; the clamp ends use the zero subgradient, so `scale` only receives
  gradient while `0 < s * scale < 1`.

  Args:
    op: The `AdjustSaturation` `Operation` that we are differentiating.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the images and the scale.
  """
  images = op.inputs[0]
  scale = op.inputs[1]
  grad = math_ops.cast(grad, dtypes.float32)
  rgb = math_ops.cast(images, dtypes.float32)
  hue, saturation, value = _image_adjust_hsv_forward(rgb)
  scaled = saturation * scale
  saturation_out = math_ops.minimum(math_ops.maximum(scaled, 0.0), 1.0)
  inside = math_ops.cast((scaled > 0.0) & (scaled < 1.0), dtypes.float32)
  grad_hue_sat_val = _hsv_to_rgb_grad_math(
      array_ops_stack.stack([hue, saturation_out, value], axis=-1), grad)
  grad_saturation = grad_hue_sat_val[..., 1] * scale * inside
  grad_scale = math_ops.reduce_sum(
      grad_hue_sat_val[..., 1] * saturation * inside)
  grad_hue_sat_val = array_ops_stack.stack(
      [grad_hue_sat_val[..., 0], grad_saturation, grad_hue_sat_val[..., 2]],
      axis=-1)
  grad_images = _rgb_to_hsv_grad_math(rgb, grad_hue_sat_val)
  return [math_ops.cast(grad_images, images.dtype), grad_scale]
