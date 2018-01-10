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
"""Functions to compute receptive field of a fully-convolutional network.

Please refer to the following g3doc for detailed explanation on how this
computation is performed, and why it is important:
g3doc/photos/vision/features/delf/g3doc/rf_computation.md
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from tensorflow.contrib.receptive_field.python.util import graph_compute_order
from tensorflow.contrib.util import make_ndarray
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import ops as framework_ops
import numpy as np

# White-listed layer operations, which do not affect the receptive field
# computation.
_UNCHANGED_RF_LAYER_OPS = [
    "Add", "BiasAdd", "Cast", "Ceil", "ConcatV2", "Const", "Floor", "Identity",
    "Log", "Mul", "Pow", "RealDiv", "Relu", "Relu6", "Round", "Rsqrt",
    "Softplus", "Sub", "VariableV2"
]

# Different ways in which padding modes may be spelled.
_VALID_PADDING = ["VALID", b"VALID"]
_SAME_PADDING = ["SAME", b"SAME"]


def _stride_size(node):
  """Computes stride size given a TF node.

  Args:
    node: Tensorflow node (NodeDef proto).

  Returns:
    stride_x: Stride size for horizontal direction (integer).
    stride_y: Stride size for vertical direction (integer).
  """
  strides_attr = node.attr["strides"]
  logging.vlog(4, "strides_attr = %s", strides_attr)
  stride_y = strides_attr.list.i[1]
  stride_x = strides_attr.list.i[2]
  return stride_x, stride_y


def _conv_kernel_size(node, name_to_order_node):
  """Computes kernel size given a TF convolution or pooling node.

  Args:
    node: Tensorflow node (NodeDef proto).
    name_to_order_node: Map from name to {order, node}. Output of
      graph_compute_order.get_compute_order().

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
  weights_node = name_to_order_node[weights_layer_param_name].node
  if weights_node.op != "VariableV2":
    raise ValueError("Weight layer is not of type VariableV2")
  shape = weights_node.attr["shape"]
  logging.vlog(4, "weight shape = %s", shape)
  kernel_size_y = shape.shape.dim[0].size
  kernel_size_x = shape.shape.dim[1].size
  return kernel_size_x, kernel_size_y


def _padding_size_conv_pool(node, kernel_size, stride):
  """Computes padding size given a TF convolution or pooling node.

  Args:
    node: Tensorflow node (NodeDef proto).
    kernel_size: Kernel size of node (integer).
    stride: Stride size of node (integer).

  Returns:
    padding: Padding size (integer).

  Raises:
    ValueError: If padding is invalid.
  """
  # In this case, we need to carefully consider the different TF padding modes.
  # The padding depends on kernel size, and may depend on input size. If it
  # depends on input size, we raise an exception.
  padding_attr = node.attr["padding"]
  logging.vlog(4, "padding_attr = %s", padding_attr)
  if padding_attr.s in _VALID_PADDING:
    padding = 0
  elif padding_attr.s in _SAME_PADDING:
    if kernel_size == 1:
      padding = 0
    elif stride == 1:
      padding = int(math.floor((float(kernel_size) - 1) / 2))
    elif stride == 2 and kernel_size % 2 == 0:
      padding = int(math.floor((float(kernel_size) - 1) / 2))
    else:
      padding = None
      logging.warning(
          "Padding depends on input size, which means that the effective "
          "padding may be different depending on the input image "
          "dimensionality. In this case, alignment check will be skipped.")
  else:
    raise ValueError("Invalid padding operation %s" % padding_attr.s)
  return padding


def _pool_kernel_size(node):
  """Computes kernel size given a TF pooling node.

  Args:
    node: Tensorflow node (NodeDef proto).

  Returns:
    kernel_size_x: Kernel size for horizontal direction (integer).
    kernel_size_y: Kernel size for vertical direction (integer).

  Raises:
    ValueError: If pooling is invalid.
  """
  ksize = node.attr["ksize"]
  kernel_size_y = ksize.list.i[1]
  kernel_size_x = ksize.list.i[2]
  if ksize.list.i[0] != 1:
    raise ValueError("pool ksize for first dim is not 1")
  if ksize.list.i[3] != 1:
    raise ValueError("pool ksize for last dim is not 1")
  return kernel_size_x, kernel_size_y


def _padding_size_pad_layer(node, name_to_order_node):
  """Computes padding size given a TF padding node.

  Args:
    node: Tensorflow node (NodeDef proto).
    name_to_order_node: Map from name to {order, node}. Output of
      graph_compute_order.get_compute_order().

  Returns:
    padding_x: Padding size for horizontal direction (integer).
    padding_y: Padding size for vertical direction (integer).

  Raises:
    ValueError: If padding layer is invalid.
  """
  paddings_layer_name = node.input[1]
  if not paddings_layer_name.endswith("/paddings"):
    raise ValueError("Padding layer name does not end with '/paddings'")
  paddings_node = name_to_order_node[paddings_layer_name].node
  if paddings_node.op != "Const":
    raise ValueError("Padding op is not Const")
  value = paddings_node.attr["value"]
  t = make_ndarray(value.tensor)
  padding_y = t[1][0]
  padding_x = t[2][0]
  if t[0][0] != 0:
    raise ValueError("padding is not zero for first tensor dim")
  if t[3][0] != 0:
    raise ValueError("padding is not zero for last tensor dim")
  return padding_x, padding_y


def _get_layer_params(node, name_to_order_node):
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
    name_to_order_node: Map from name to {order, node}. Output of
      graph_compute_order.get_compute_order().

  Returns:
    kernel_size_x: Kernel size for horizontal direction (integer).
    kernel_size_y: Kernel size for vertical direction (integer).
    stride_x: Stride size for horizontal direction (integer).
    stride_y: Stride size for vertical direction (integer).
    padding_x: Padding size for horizontal direction (integer).
    padding_y: Padding size for vertical direction (integer).

  Raises:
    ValueError: If layer op is unknown.
  """
  logging.vlog(3, "node.op = %s", node.op)
  logging.vlog(4, "node = %s", node)
  if node.op == "Conv2D" or node.op == "DepthwiseConv2dNative":
    stride_x, stride_y = _stride_size(node)
    kernel_size_x, kernel_size_y = _conv_kernel_size(node, name_to_order_node)
    # Compute the padding for this node separately for each direction.
    padding_x = _padding_size_conv_pool(node, kernel_size_x, stride_x)
    padding_y = _padding_size_conv_pool(node, kernel_size_y, stride_y)
  elif node.op == "Pad":
    # Kernel and stride are simply 1 in this case.
    kernel_size_x = 1
    kernel_size_y = 1
    stride_x = 1
    stride_y = 1
    padding_x, padding_y = _padding_size_pad_layer(node, name_to_order_node)
  elif node.op == "MaxPool" or node.op == "AvgPool":
    stride_x, stride_y = _stride_size(node)
    kernel_size_x, kernel_size_y = _pool_kernel_size(node)
    # Compute the padding for this node separately for each direction.
    padding_x = _padding_size_conv_pool(node, kernel_size_x, stride_x)
    padding_y = _padding_size_conv_pool(node, kernel_size_y, stride_y)
  elif node.op in _UNCHANGED_RF_LAYER_OPS:
    # These nodes do not modify the RF parameters.
    kernel_size_x = 1
    kernel_size_y = 1
    stride_x = 1
    stride_y = 1
    padding_x = 0
    padding_y = 0
  else:
    raise ValueError("Unknown layer for operation '%s': %s" % (node.name,
                                                               node.op))
  return kernel_size_x, kernel_size_y, stride_x, stride_y, padding_x, padding_y


def _reverse_sort_by_order(name_to_order_node):
  """Sorts map of name_to_order_node nodes in reverse order.

  The output is such that the nodes in name_to_order_node are sorted in
  descending order of the "order" field.

  Args:
    name_to_order_node: Map from name to {order, node}. Output of
      graph_compute_order.get_compute_order().

  Returns:
    sorted_name_to_order_node: Sorted version of the input, in descending order.
  """
  return sorted(name_to_order_node.items(), key=lambda x: -x[1].order)


def _get_rf_size_node_input(stride, kernel_size, rf_size_output):
  """Computes RF size at the input of a given layer.

  Args:
    stride: Stride of given layer (integer).
    kernel_size: Kernel size of given layer (integer).
    rf_size_output: RF size at output of given layer (integer).

  Returns:
    rf_size_input: RF size at input of given layer (integer).
  """
  return stride * rf_size_output + kernel_size - stride


def _get_effective_stride_node_input(stride, effective_stride_output):
  """Computes effective stride at the input of a given layer.

  Args:
    stride: Stride of given layer (integer).
    effective_stride_output: Effective stride at output of given layer
      (integer).

  Returns:
    effective_stride_input: Effective stride at input of given layer
      (integer).
  """
  return stride * effective_stride_output


def _get_effective_padding_node_input(stride, padding,
                                      effective_padding_output):
  """Computes effective padding at the input of a given layer.

  Args:
    stride: Stride of given layer (integer).
    padding: Padding of given layer (integer).
    effective_padding_output: Effective padding at output of given layer
      (integer).

  Returns:
    effective_padding_input: Effective padding at input of given layer
      (integer).
  """
  return stride * effective_padding_output + padding


class ReceptiveField:
  """Receptive field of a convolutional neural network.

  Args:
    size: Receptive field size.
    stride: Effective stride.
    padding: Effective padding.
  """

  def __init__(self, size, stride, padding):
    self.size = np.asarray(size)
    self.stride = np.asarray(stride)
    self.padding = np.asarray(padding)

  def compute_input_center_coordinates(self, y, axis=None):
    """Computes the center of the receptive field that generated a feature.

    Args:
      y: An array of feature coordinates with shape `(..., d)`, where `d` is the
        number of dimensions of the coordinates.
      axis: The dimensions for which to compute the input center coordinates.
        If `None` (the default), compute the input center coordinates for all
        dimensions.

    Returns:
      x: Center of the receptive field that generated the features, at the input
        of the network.

    Raises:
      ValueError: If the number of dimensions of the feature coordinates does
        not match the number of elements in `axis`.
    """
    # Use all dimensions.
    if axis is None:
      axis = range(self.size.size)
    # Ensure axis is a list because tuples have different indexing behavior.
    axis = list(axis)
    y = np.asarray(y)
    if y.shape[-1] != len(axis):
      raise ValueError("Dimensionality of the feature coordinates `y` (%d) "
                       "does not match dimensionality of `axis` (%d)" %
                       (y.shape[-1], len(axis)))
    return - self.padding[axis] + y * self.stride[axis] + \
      (self.size[axis] - 1) / 2

  def compute_feature_coordinates(self, x, axis=None):
    """Computes the position of a feature given the center of a receptive field.

    Args:
      x: An array of input center coordinates with shape `(..., d)`, where `d`
        is the number of dimensions of the coordinates.
      axis: The dimensions for which to compute the feature coordinates.
        If `None` (the default), compute the feature coordinates for all
        dimensions.

    Returns:
      y: Coordinates of the features.

    Raises:
      ValueError: If the number of dimensions of the input center coordinates
        does not match the number of elements in `axis`.
    """
    # Use all dimensions.
    if axis is None:
      axis = range(self.size.size)
    # Ensure axis is a list because tuples have different indexing behavior.
    axis = list(axis)
    x = np.asarray(x)
    if x.shape[-1] != len(axis):
      raise ValueError("Dimensionality of the input center coordinates `x` "
                       "(%d) does not match dimensionality of `axis` (%d)" %
                       (x.shape[-1], len(axis)))
    return (x + self.padding[axis] + (1 - self.size[axis]) / 2) / \
      self.stride[axis]

  def __iter__(self):
    return iter(np.concatenate([self.size, self.stride, self.padding]))


def compute_receptive_field_from_graph_def(graph_def,
                                           input_node,
                                           output_node,
                                           stop_propagation=None):
  """Computes receptive field (RF) parameters from a Graph or GraphDef object.

  The algorithm stops the calculation of the receptive field whenever it
  encounters an operation in the list `stop_propagation`. Stopping the
  calculation early can be useful to calculate the receptive field of a
  subgraph such as a single branch of the
  [inception network](https://arxiv.org/abs/1512.00567).

  Args:
    graph_def: Graph or GraphDef object.
    input_node: Name of the input node or Tensor object from graph.
    output_node: Name of the output node or Tensor object from graph.
    stop_propagation: List of operation or scope names for which to stop the
      propagation of the receptive field.

  Returns:
    rf_size_x: Receptive field size of network in the horizontal direction, with
      respect to specified input and output.
    rf_size_y: Receptive field size of network in the vertical direction, with
      respect to specified input and output.
    effective_stride_x: Effective stride of network in the horizontal direction,
      with respect to specified input and output.
    effective_stride_y: Effective stride of network in the vertical direction,
      with respect to specified input and output.
    effective_padding_x: Effective padding of network in the horizontal
      direction, with respect to specified input and output.
    effective_padding_y: Effective padding of network in the vertical
      direction, with respect to specified input and output.

  Raises:
    ValueError: If network is not aligned or if either input or output nodes
      cannot be found. For network criterion alignment, see
      photos/vision/features/delf/g3doc/rf_computation.md
  """
  # Convert a graph to graph_def if necessary.
  if isinstance(graph_def, framework_ops.Graph):
    graph_def = graph_def.as_graph_def()

  # Convert tensors to names.
  if isinstance(input_node, framework_ops.Tensor):
    input_node = input_node.op.name
  if isinstance(output_node, framework_ops.Tensor):
    output_node = output_node.op.name

  stop_propagation = stop_propagation or []

  # Computes order of computation for a given graph.
  name_to_order_node = graph_compute_order.get_compute_order(
      graph_def=graph_def)

  # Sort in reverse topological order.
  order = _reverse_sort_by_order(name_to_order_node)

  # Dictionaries to keep track of receptive field, effective stride and
  # effective padding of different nodes.
  rf_sizes_x = {}
  rf_sizes_y = {}
  effective_strides_x = {}
  effective_strides_y = {}
  effective_paddings_x = {}
  effective_paddings_y = {}

  # Initialize dicts for output_node.
  rf_sizes_x[output_node] = 1
  rf_sizes_y[output_node] = 1
  effective_strides_x[output_node] = 1
  effective_strides_y[output_node] = 1
  effective_paddings_x[output_node] = 0
  effective_paddings_y[output_node] = 0

  # Flag to denote if we found output node yet. If we have not, we skip nodes
  # until the output node is found.
  found_output_node = False

  # Flag to denote if padding is undefined. This happens when SAME padding mode
  # is used in conjunction with stride and kernel sizes which make it such that
  # the padding to be applied would depend on the input size. In this case,
  # alignment checks are skipped, and the effective padding is None.
  undefined_padding = False

  for _, (o, node) in order:
    if node:
      logging.vlog(3, "%10d %-100s %-20s" % (o, node.name[:90], node.op))
    else:
      continue

    # When we find input node, we can stop.
    if node.name == input_node:
      break

    # Loop until we find the output node. All nodes before finding the output
    # one are irrelevant, so they can be skipped.
    if not found_output_node:
      if node.name == output_node:
        found_output_node = True

    if found_output_node:
      if node.name not in rf_sizes_x:
        assert node.name not in rf_sizes_y, ("Node %s is in rf_sizes_y, but "
                                             "not in rf_sizes_x" % node.name)
        # In this case, node is not relevant since it's not part of the
        # computation we're interested in.
        logging.vlog(3, "Irrelevant node %s, skipping it...", node.name)
        continue

      # Get params for this layer.
      kernel_size_x, kernel_size_y, stride_x, stride_y, padding_x, padding_y = (
          _get_layer_params(node, name_to_order_node))
      logging.vlog(3, "kernel_size_x = %s, kernel_size_y = %s, "
                   "stride_x = %s, stride_y = %s, "
                   "padding_x = %s, padding_y = %s" %
                   (kernel_size_x, kernel_size_y, stride_x, stride_y, padding_x,
                    padding_y))
      if padding_x is None or padding_y is None:
        undefined_padding = True

      # Get parameters at input of this layer which may or may not be propagated
      # to the input layers.
      rf_size_input_x = _get_rf_size_node_input(stride_x, kernel_size_x,
                                                rf_sizes_x[node.name])
      rf_size_input_y = _get_rf_size_node_input(stride_y, kernel_size_y,
                                                rf_sizes_y[node.name])
      effective_stride_input_x = _get_effective_stride_node_input(
          stride_x, effective_strides_x[node.name])
      effective_stride_input_y = _get_effective_stride_node_input(
          stride_y, effective_strides_y[node.name])
      if not undefined_padding:
        effective_padding_input_x = _get_effective_padding_node_input(
            stride_x, padding_x, effective_paddings_x[node.name])
        effective_padding_input_y = _get_effective_padding_node_input(
            stride_y, padding_y, effective_paddings_y[node.name])
      else:
        effective_padding_input_x = None
        effective_padding_input_y = None

      # Loop over this node's inputs and potentially propagate information down.
      for inp_name in node.input:
        # Stop the propagation of the receptive field.
        if any(inp_name.startswith(stop) for stop in stop_propagation):
          logging.vlog(3, "Skipping explicitly ignored node %s.", node.name)
          continue

        logging.vlog(4, "inp_name = %s", inp_name)
        if inp_name.startswith("^"):
          # The character "^" denotes a control dependency, so this input node
          # can be safely ignored.
          continue

        inp_node = name_to_order_node[inp_name].node
        logging.vlog(4, "inp_node = \n%s", inp_node)
        if inp_node.name in rf_sizes_x:
          assert inp_node.name in rf_sizes_y, (
              "Node %s is in rf_sizes_x, but "
              "not in rf_sizes_y" % inp_node.name)
          # This node was already discovered through a previous path, so we need
          # to make sure that graph is aligned. This alignment check is skipped
          # if the padding is not defined, since in this case alignment cannot
          # be checked.
          if not undefined_padding:
            if effective_strides_x[inp_node.name] != effective_stride_input_x:
              raise ValueError(
                  "Graph is not aligned since effective stride from different "
                  "paths is different in horizontal direction")
            if effective_strides_y[inp_node.name] != effective_stride_input_y:
              raise ValueError(
                  "Graph is not aligned since effective stride from different "
                  "paths is different in vertical direction")
            if (rf_sizes_x[inp_node.name] - 1
               ) / 2 - effective_paddings_x[inp_node.name] != (
                   rf_size_input_x - 1) / 2 - effective_padding_input_x:
              raise ValueError(
                  "Graph is not aligned since center shift from different "
                  "paths is different in horizontal direction")
            if (rf_sizes_y[inp_node.name] - 1
               ) / 2 - effective_paddings_y[inp_node.name] != (
                   rf_size_input_y - 1) / 2 - effective_padding_input_y:
              raise ValueError(
                  "Graph is not aligned since center shift from different "
                  "paths is different in vertical direction")
          # Keep track of path with largest RF, for both directions.
          if rf_sizes_x[inp_node.name] < rf_size_input_x:
            rf_sizes_x[inp_node.name] = rf_size_input_x
            effective_strides_x[inp_node.name] = effective_stride_input_x
            effective_paddings_x[inp_node.name] = effective_padding_input_x
          if rf_sizes_y[inp_node.name] < rf_size_input_y:
            rf_sizes_y[inp_node.name] = rf_size_input_y
            effective_strides_y[inp_node.name] = effective_stride_input_y
            effective_paddings_y[inp_node.name] = effective_padding_input_y
        else:
          assert inp_node.name not in rf_sizes_y, (
              "Node %s is in rf_sizes_y, but "
              "not in rf_sizes_x" % inp_node.name)
          # In this case, it is the first time we encounter this node. So we
          # propagate the RF parameters.
          rf_sizes_x[inp_node.name] = rf_size_input_x
          rf_sizes_y[inp_node.name] = rf_size_input_y
          effective_strides_x[inp_node.name] = effective_stride_input_x
          effective_strides_y[inp_node.name] = effective_stride_input_y
          effective_paddings_x[inp_node.name] = effective_padding_input_x
          effective_paddings_y[inp_node.name] = effective_padding_input_y

  if not found_output_node:
    raise ValueError("Output node was not found")
  if input_node not in rf_sizes_x:
    raise ValueError("Input node was not found")
  return ReceptiveField(
      (rf_sizes_x[input_node], rf_sizes_y[input_node]),
      (effective_strides_x[input_node], effective_strides_y[input_node]),
      (effective_paddings_x[input_node], effective_paddings_y[input_node]))
