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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_image_ops


@ops.RegisterGradient("ResizeNearestNeighbor")
def _ResizeNearestNeighborGrad(op, grad):
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


@ops.RegisterGradient("ResizeBilinear")
def _ResizeBilinearGrad(op, grad):
  """The derivatives for bilinear resizing.

  Args:
    op: The ResizeBilinear op.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the input.
  """
  grad0 = gen_image_ops.resize_bilinear_grad(
      grad,
      op.inputs[0],
      align_corners=op.get_attr("align_corners"),
      half_pixel_centers=op.get_attr("half_pixel_centers"))
  return [grad0, None]


@ops.RegisterGradient("ScaleAndTranslate")
def _ScaleAndTranslateGrad(op, grad):
  """The derivatives for ScaleAndTranslate transformation op.

  Args:
    op: The ScaleAndTranslate op.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the input.
  """

  grad0 = gen_image_ops.scale_and_translate_grad(
      grad,
      op.inputs[0],
      op.inputs[2],
      op.inputs[3],
      kernel_type=op.get_attr("kernel_type"),
      antialias=op.get_attr("antialias"))
  return [grad0, None, None, None]


@ops.RegisterGradient("ResizeBicubic")
def _ResizeBicubicGrad(op, grad):
  """The derivatives for bicubic resizing.

  Args:
    op: The ResizeBicubic op.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the input.
  """
  allowed_types = [dtypes.float32, dtypes.float64]
  grad0 = None
  if op.inputs[0].dtype in allowed_types:
    grad0 = gen_image_ops.resize_bicubic_grad(
        grad,
        op.inputs[0],
        align_corners=op.get_attr("align_corners"),
        half_pixel_centers=op.get_attr("half_pixel_centers"))
  return [grad0, None]


@ops.RegisterGradient("CropAndResize")
def _CropAndResizeGrad(op, grad):
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

@ops.RegisterGradient("RGBToHSV")
def _rgb_to_hsv_grad(op, grad):
  """The gradients for `zero_out`.

  Args:
    op: The `rgb_to_hsv` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `rgb_to_hsv` op.

  Returns:
    Gradients with respect to the input of `rgb_to_hsv`.
  """
  print("******** This is a test implementation ********** \n")
    
  # Input Channels
  reds = op.inputs[0][..., 0]
  greens = op.inputs[0][..., 1]
  blues = op.inputs[0][..., 2]
  # Output Channels
  hue = op.outputs[0][..., 0]
  saturation = op.outputs[0][..., 1]
  value = op.outputs[0][..., 2]

  # Mask/Indicator for max and min values of each pixel. Arbitrary assignment in case 
  # of tie breakers with R>G>B.
  # Max values
  red_biggest = cast((reds >= blues) & (reds >= greens), dtypes.float32)
  green_biggest = cast((greens > reds) & (greens >= blues), dtypes.float32)
  blue_biggest = cast((blues > reds) & (blues > greens), dtypes.float32)
  # Min values
  red_smallest = cast((reds < blues) & (reds < greens), dtypes.float32)
  green_smallest = cast((greens <= reds) & (greens < blues), dtypes.float32)
  blue_smallest = cast((blues <= reds) & (blues <= greens), dtypes.float32)

  ##############################################################
  #  Derivatives of R, G, B wrt Value slice
  ##############################################################
  dv_dr = red_biggest
  dv_dg = green_biggest
  dv_db = blue_biggest
  ##############################################################
  # Derivatives of R, G, B wrt Saturation slice
  ##############################################################
  # The first term in the addition is the case when the corresponding color from (r,g,b) was "MAX" 
  # -> derivative = MIN/square(MAX), MIN could be one of the other two colors
  # The second term is the case when the corresponding color from (r,g,b) was "MIN" 
  # -> derivative = -1/MAX, MAX could be one of the other two colours.
  # Defining a custom replacement for machine epsilon (eps) to avoid NaNs or divide by zeros
  my_eps = 0.000000001
  ds_dr = cast(reds > 0, dtypes.float32) * add(red_biggest * divide(add(green_smallest * greens, blue_smallest * blues), square(reds + my_eps)), 
              red_smallest * -reciprocal(add(green_biggest * greens, blue_biggest * blues + my_eps))) 
  ds_dg = cast(greens > 0, dtypes.float32) * add(green_biggest * divide(add(red_smallest * reds, blue_smallest * blues), square(greens + my_eps)), 
              green_smallest * -reciprocal(add(red_biggest * reds, blue_biggest * blues + my_eps)))
  ds_db = cast(blues > 0, dtypes.float32) * add(blue_biggest * divide(add(green_smallest * greens, red_smallest * reds), square(blues + my_eps)), 
              blue_smallest * -reciprocal(add(green_biggest * greens, red_biggest * reds + my_eps)))
  ##############################################################
  # Derivatives of R, G, B wrt Hue slice
  ##############################################################
  # Need to go case by case for each color.

  # for red, dh_dr -> dh_dr_1 + dh_dr_2 + dh_dr_3 + dh_dr_4 + dh_dr_5
  # dh_dr_1 ->
  # if red was MAX, then derivative = 60 * -1 * (G-B)/square(MAX-MIN) == 60 * -1 * (greens-blues) * reciprocal(square(saturation)) * reciprical(square(value))
  # elif green was MAX, there are two subcases ie when red was MIN and when red was NOT MIN
  #   dh_dr_2 ->
  #   if red was MIN (use UV rule) ->  60 * ((1 * -1/(MAX-MIN))  + (B-R)*(-1/square(MAX-MIN) * -1)) == 60 * (blues - greens) * reciprocal(square(reds - greens))
  #   dh_dr_3 ->
  #   if red was NOT MIN -> 60 * -1/MAX-MIN == -60 * reciprocal(greens-blues)
  # elif blue was MAX, there are two subcases
  #   dh_dr_4 ->
  #   if red was MIN (similarly use the UV rule) -> 60 * (blues - greens) * reciprocal(square(blues - reds))
  #   dh_dr_5 -> 
  #   if red was NOT MIN -> 60 * 1/MAX-MIN == 60 * reciprocal(blues-greens)
  dh_dr_1 = 60 * (cast(reds > 0, dtypes.float32) * red_biggest * -1  * (greens - blues) * _custom_reciprocal(square(saturation)) * _custom_reciprocal(square(value)))
  dh_dr_2 = 60 * (cast(greens > 0, dtypes.float32) * green_biggest * red_smallest  * (blues - greens) * _custom_reciprocal(square(reds - greens)))
  dh_dr_3 = 60 * (cast(greens > 0, dtypes.float32) * green_biggest * blue_smallest * -1 * _custom_reciprocal(greens - blues))
  dh_dr_4 = 60 * (cast(blues > 0, dtypes.float32) * blue_biggest * red_smallest * (blues - greens) * _custom_reciprocal(square(blues - reds)))
  dh_dr_5 = 60 * (cast(blues > 0, dtypes.float32) * blue_biggest * green_smallest * _custom_reciprocal(blues - greens))

  dh_dr = dh_dr_1 + dh_dr_2 + dh_dr_3 + dh_dr_4 + dh_dr_5

  # for green, dh_dg -> dh_dg_1 + dh_dg_2 + dh_dg_3 + dh_dg_4 + dh_dg_5
  # dh_dg_1 ->
  # if green was MAX, then derivative = 60 * -1 * (B-R)/square(MAX-MIN) == 60 * -1 * (blues - reds) * reciprocal(square(saturation)) * reciprocal(square(value))
  # elif red was MAX, there are two subcases ie when green was MIN and when green was NOT MIN
  #   dh_dg_2 ->
  #   if green was MIN (use UV rule) ->  60 * ((1 * 1/(MAX-MIN))  + (greens-blues)*(-1/square(MAX-MIN) * -1)) == 60 * ((reciprocal(reds-greens) + (greens-blues)*reciprocal(square(reds-greens))))
  #   dh_dg_3 ->
  #   if green was NOT MIN -> 60 * 1/MAX-MIN == 60 * reciprocal(reds - blues)
  # elif blue was MAX, there are two subcases
  #   dh_dg_4 ->
  #   if green was MIN (similarly use the UV rule) -> 60 * -1 * (reciprocal(blues - greens) + (reds-greens)* -1 * reciprocal(square(blues-greens)))
  #   dh_dr_5 -> 
  #   if green was NOT MIN -> 60 * -1/MAX-MIN == 60 * -1 * reciprocal(blues - reds)
  dh_dg_1 = 60 * (cast(greens > 0, dtypes.float32) * green_biggest * -1 * (blues - reds) * _custom_reciprocal(square(saturation)) * _custom_reciprocal(square(value)))
  dh_dg_2 = 60 * (cast(reds > 0, dtypes.float32) * red_biggest * green_smallest * (reds - blues) * _custom_reciprocal(square(reds - greens)))
  dh_dg_3 = 60 * (cast(reds > 0, dtypes.float32) * red_biggest * blue_smallest * _custom_reciprocal(reds - blues))
  dh_dg_4 = 60 * (cast(blues > 0, dtypes.float32) * blue_biggest * green_smallest * (reds - blues) * _custom_reciprocal(square(blues - greens)))
  dh_dg_5 = 60 * (cast(blues > 0, dtypes.float32) * blue_biggest * red_smallest * -1 * _custom_reciprocal(blues - reds))

  dh_dg = dh_dg_1 + dh_dg_2 + dh_dg_3 + dh_dg_4 + dh_dg_5


  return None
