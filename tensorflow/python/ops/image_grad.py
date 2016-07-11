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
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
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
  # pylint: disable=protected-access
  grads = gen_image_ops._resize_nearest_neighbor_grad(
      grad,
      op.inputs[0].get_shape()[1:3],
      align_corners=op.get_attr("align_corners"))
  # pylint: enable=protected-access
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
  allowed_types = [dtypes.float32, dtypes.float64]
  grad0 = None
  if op.inputs[0].dtype in allowed_types:
    # pylint: disable=protected-access
    grad0 = gen_image_ops._resize_bilinear_grad(
        grad,
        op.inputs[0],
        align_corners=op.get_attr("align_corners"))
    # pylint: enable=protected-access
  return [grad0, None]


@ops.RegisterShape("ResizeNearestNeighborGrad")
def _ResizeShape(op):
  """Shape function for the resize grad ops."""
  input_shape = op.inputs[0].get_shape().with_rank(4)
  size = tensor_util.constant_value(op.inputs[1])
  if size is not None:
    height = size[0]
    width = size[1]
  else:
    height = None
    width = None
  return [
      tensor_shape.TensorShape([input_shape[0], height, width, input_shape[3]])
  ]


@ops.RegisterShape("ResizeBilinearGrad")
def _ResizeBilinearGradShape(op):
  """Shape function for ResizeBilinearGrad."""
  return [op.inputs[1].get_shape()]


@ops.RegisterShape("CropAndResizeGradImage")
def _CropAndResizeGradImageShape(op):
  """Shape function for CropAndResizeGradImage."""
  image_size = tensor_util.constant_value(op.inputs[3])
  if image_size is not None:
    batch = image_size[0]
    height = image_size[1]
    width = image_size[2]
    depth = image_size[3]
  else:
    batch = None
    height = None
    width = None
    depth = None
  return [tensor_shape.TensorShape([batch, height, width, depth])]


@ops.RegisterShape("CropAndResizeGradBoxes")
def _CropAndResizeGradBoxesShape(op):
  """Shape function for CropAndResizeGradBoxes."""
  return [op.inputs[2].get_shape()]


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
    grad0 = gen_image_ops.crop_and_resize_grad_image(grad,
                                                     op.inputs[1],
                                                     op.inputs[2],
                                                     image_shape,
                                                     T=op.get_attr("T"))
    # pylint: enable=protected-access
  else:
    grad0 = None

  grad1 = gen_image_ops.crop_and_resize_grad_boxes(grad, op.inputs[0],
                                                   op.inputs[1], op.inputs[2])

  return [grad0, grad1, None, None]
