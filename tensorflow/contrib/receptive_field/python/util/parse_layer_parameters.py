# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Functions to parse RF-related parameters from TF layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from tensorflow.contrib.util import make_ndarray
from tensorflow.python.platform import tf_logging as logging

# White-listed layer operations, which do not affect the receptive field
# computation.
_UNCHANGED_RF_LAYER_OPS = [
    "Add", "BiasAdd", "Cast", "Ceil", "ConcatV2", "Const", "Floor",
    "FusedBatchNorm", "Identity", "Log", "Mul", "Pow", "RealDiv", "Relu",
    "Relu6", "Round", "Rsqrt", "Softplus", "Sub", "VariableV2"
]

# Different ways in which padding modes may be spelled.
_VALID_PADDING = ["VALID", b"VALID"]
_SAME_PADDING = ["SAME", b"SAME"]


def _stride_size(node, name_to_node):
  """Computes stride size given a TF node.

  Args:
    node: Tensorflow node (NodeDef proto).

  Returns:
    stride_x: Stride size for horizontal direction (integer).
    stride_y: Stride size for vertical direction (integer).
  """
  if node.op == "MaxPoolV2":
    strides_input_name = node.input[2]
    if not strides_input_name.endswith("/strides"):
      raise ValueError("Strides name does not end with '/strides'")
    strides_node = name_to_node[strides_input_name]
    value = strides_node.attr["value"]
    t = make_ndarray(value.tensor)
    stride_y = t[1]
    stride_x = t[2]
  else:
    strides_attr = node.attr["strides"]
    logging.vlog(4, "strides_attr = %s", strides_attr)
    stride_y = strides_attr.list.i[1]
    stride_x = strides_attr.list.i[2]
  return stride_x, stride_y


def _conv_kernel_size(node, name_to_node):
  """Computes kernel size given a TF convolution or pooling node.

  Args:
    node: Tensorflow node (NodeDef proto).
    name_to_node: Dict keyed by node name, each entry containing the node's
      NodeDef.

  Returns:
    kernel_size_x: Kernel size for horizontal direction (integer).
    kernel_size_y: Kernel size for vertical direction (integer).

  Raises:
    ValueError: If the weight layer node is invalid.
  """
  weights_layer_read_name = node.input[1]
  if not weights_layer_read_name.endswith("/read"):
    raise ValueError(
        "Weight layer's name input to conv layer does not end with '/read'")
  weights_layer_param_name = weights_layer_read_name[:-5]
  weights_node = name_to_node[weights_layer_param_name]
  if weights_node.op != "VariableV2":
    raise ValueError("Weight layer is not of type VariableV2")
  shape = weights_node.attr["shape"]
  logging.vlog(4, "weight shape = %s", shape)
  kernel_size_y = shape.shape.dim[0].size
  kernel_size_x = shape.shape.dim[1].size
  return kernel_size_x, kernel_size_y


def _padding_size_conv_pool(node, kernel_size, stride, input_resolution=None):
  """Computes padding size given a TF convolution or pooling node.

  Args:
    node: Tensorflow node (NodeDef proto).
    kernel_size: Kernel size of node (integer).
    stride: Stride size of node (integer).
    input_resolution: Input resolution to assume, if not None (integer).

  Returns:
    total_padding: Total padding size (integer).
    padding: Padding size, applied to the left or top (integer).

  Raises:
    ValueError: If padding is invalid.
  """
  # In this case, we need to carefully consider the different TF padding modes.
  # The padding depends on kernel size, and may depend on input size. If it
  # depends on input size and input_resolution is None, we raise an exception.
  padding_attr = node.attr["padding"]
  logging.vlog(4, "padding_attr = %s", padding_attr)
  if padding_attr.s in _VALID_PADDING:
    total_padding = 0
    padding = 0
  elif padding_attr.s in _SAME_PADDING:
    if input_resolution is None:
      # In this case, we do not know the input resolution, so we can only know
      # the padding in some special cases.
      if kernel_size == 1:
        total_padding = 0
        padding = 0
      elif stride == 1:
        total_padding = kernel_size - 1
        padding = int(math.floor(float(total_padding) / 2))
      elif stride == 2 and kernel_size % 2 == 0:
        # In this case, we can be sure of the left/top padding, but not of the
        # total padding.
        total_padding = None
        padding = int(math.floor((float(kernel_size) - 1) / 2))
      else:
        total_padding = None
        padding = None
        logging.warning(
            "Padding depends on input size, which means that the effective "
            "padding may be different depending on the input image "
            "dimensionality. In this case, alignment check will be skipped. If"
            " you know the input resolution, please set it.")
    else:
      # First, compute total_padding based on documentation.
      if input_resolution % stride == 0:
        total_padding = int(max(float(kernel_size - stride), 0.0))
      else:
        total_padding = int(
            max(float(kernel_size - (input_resolution % stride)), 0.0))
      # Then, compute left/top padding.
      padding = int(math.floor(float(total_padding) / 2))

  else:
    raise ValueError("Invalid padding operation %s" % padding_attr.s)
  return total_padding, padding


def _pool_kernel_size(node, name_to_node):
  """Computes kernel size given a TF pooling node.

  Args:
    node: Tensorflow node (NodeDef proto).

  Returns:
    kernel_size_x: Kernel size for horizontal direction (integer).
    kernel_size_y: Kernel size for vertical direction (integer).

  Raises:
    ValueError: If pooling is invalid.
  """
  if node.op == "MaxPoolV2":
    ksize_input_name = node.input[1]
    if not ksize_input_name.endswith("/ksize"):
      raise ValueError("Kernel size name does not end with '/ksize'")
    ksize_node = name_to_node[ksize_input_name]
    value = ksize_node.attr["value"]
    t = make_ndarray(value.tensor)
    kernel_size_y = t[1]
    kernel_size_x = t[2]
    if t[0] != 1:
      raise ValueError("pool ksize for first dim is not 1")
    if t[3] != 1:
      raise ValueError("pool ksize for last dim is not 1")
  else:
    ksize = node.attr["ksize"]
    kernel_size_y = ksize.list.i[1]
    kernel_size_x = ksize.list.i[2]
    if ksize.list.i[0] != 1:
      raise ValueError("pool ksize for first dim is not 1")
    if ksize.list.i[3] != 1:
      raise ValueError("pool ksize for last dim is not 1")
  return kernel_size_x, kernel_size_y


def _padding_size_pad_layer(node, name_to_node):
  """Computes padding size given a TF padding node.

  Args:
    node: Tensorflow node (NodeDef proto).
    name_to_node: Dict keyed by node name, each entry containing the node's
      NodeDef.

  Returns:
    total_padding_x: Total padding size for horizontal direction (integer).
    padding_x: Padding size for horizontal direction, left side (integer).
    total_padding_y: Total padding size for vertical direction (integer).
    padding_y: Padding size for vertical direction, top side (integer).

  Raises:
    ValueError: If padding layer is invalid.
  """
  paddings_layer_name = node.input[1]
  if not paddings_layer_name.endswith("/paddings"):
    raise ValueError("Padding layer name does not end with '/paddings'")
  paddings_node = name_to_node[paddings_layer_name]
  if paddings_node.op != "Const":
    raise ValueError("Padding op is not Const")
  value = paddings_node.attr["value"]
  t = make_ndarray(value.tensor)
  padding_y = t[1][0]
  padding_x = t[2][0]
  total_padding_y = padding_y + t[1][1]
  total_padding_x = padding_x + t[2][1]
  if (t[0][0] != 0) or (t[0][1] != 0):
    raise ValueError("padding is not zero for first tensor dim")
  if (t[3][0] != 0) or (t[3][1] != 0):
    raise ValueError("padding is not zero for last tensor dim")
  return total_padding_x, padding_x, total_padding_y, padding_y


def get_layer_params(node, name_to_node, input_resolution=None, force=False):
  """Gets layer parameters relevant for RF computation.

  Currently, only these nodes are supported:
  - Conv2D
  - DepthwiseConv2dNative
  - Pad
  - MaxPool
  - AvgPool
  - all nodes listed in _UNCHANGED_RF_LAYER_OPS

  Args:
    node: Tensorflow node (NodeDef proto).
    name_to_node: Dict keyed by node name, each entry containing the node's
      NodeDef.
    input_resolution: List with 2 dimensions, denoting the height/width of the
      input feature map to this layer. If set to None, then the padding may be
      undefined (in tensorflow, SAME padding depends on input spatial
      resolution).
    force: If True, the function does not raise a ValueError if the layer op is
      unknown. Instead, in this case it sets each of the returned parameters to
      None.

  Returns:
    kernel_size_x: Kernel size for horizontal direction (integer).
    kernel_size_y: Kernel size for vertical direction (integer).
    stride_x: Stride size for horizontal direction (integer).
    stride_y: Stride size for vertical direction (integer).
    padding_x: Padding size for horizontal direction, left side (integer).
    padding_y: Padding size for vertical direction, top side (integer).
    total_padding_x: Total padding size for horizontal direction (integer).
    total_padding_y: Total padding size for vertical direction (integer).

  Raises:
    ValueError: If layer op is unknown and force is False.
  """
  logging.vlog(3, "node.name = %s", node.name)
  logging.vlog(3, "node.op = %s", node.op)
  logging.vlog(4, "node = %s", node)
  if node.op == "Conv2D" or node.op == "DepthwiseConv2dNative":
    stride_x, stride_y = _stride_size(node, name_to_node)
    kernel_size_x, kernel_size_y = _conv_kernel_size(node, name_to_node)
    # Compute the padding for this node separately for each direction.
    total_padding_x, padding_x = _padding_size_conv_pool(
        node, kernel_size_x, stride_x, input_resolution[1]
        if input_resolution is not None else None)
    total_padding_y, padding_y = _padding_size_conv_pool(
        node, kernel_size_y, stride_y, input_resolution[0]
        if input_resolution is not None else None)
  elif node.op == "Pad":
    # Kernel and stride are simply 1 in this case.
    kernel_size_x = 1
    kernel_size_y = 1
    stride_x = 1
    stride_y = 1
    total_padding_x, padding_x, total_padding_y, padding_y = (
        _padding_size_pad_layer(node, name_to_node))
  elif node.op == "MaxPool" or node.op == "MaxPoolV2" or node.op == "AvgPool":
    stride_x, stride_y = _stride_size(node, name_to_node)
    kernel_size_x, kernel_size_y = _pool_kernel_size(node, name_to_node)
    # Compute the padding for this node separately for each direction.
    total_padding_x, padding_x = _padding_size_conv_pool(
        node, kernel_size_x, stride_x, input_resolution[1]
        if input_resolution is not None else None)
    total_padding_y, padding_y = _padding_size_conv_pool(
        node, kernel_size_y, stride_y, input_resolution[0]
        if input_resolution is not None else None)
  elif node.op in _UNCHANGED_RF_LAYER_OPS:
    # These nodes do not modify the RF parameters.
    kernel_size_x = 1
    kernel_size_y = 1
    stride_x = 1
    stride_y = 1
    total_padding_x = 0
    padding_x = 0
    total_padding_y = 0
    padding_y = 0
  else:
    if force:
      kernel_size_x = None
      kernel_size_y = None
      stride_x = None
      stride_y = None
      total_padding_x = None
      padding_x = None
      total_padding_y = None
      padding_y = None
    else:
      raise ValueError("Unknown layer for operation '%s': %s" % (node.name,
                                                                 node.op))
  return (kernel_size_x, kernel_size_y, stride_x, stride_y, padding_x,
          padding_y, total_padding_x, total_padding_y)
