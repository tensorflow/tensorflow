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
"""Functions to compute receptive field of a fully-convolutional network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib.receptive_field.python.util import graph_compute_order
from tensorflow.contrib.receptive_field.python.util import parse_layer_parameters
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.platform import tf_logging as logging


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


class ReceptiveField(object):
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
      axis: The dimensions for which to compute the input center coordinates. If
        `None` (the default), compute the input center coordinates for all
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
    return -self.padding[axis] + y * self.stride[axis] + (
        self.size[axis] - 1) / 2

  def compute_feature_coordinates(self, x, axis=None):
    """Computes the position of a feature given the center of a receptive field.

    Args:
      x: An array of input center coordinates with shape `(..., d)`, where `d`
        is the number of dimensions of the coordinates.
      axis: The dimensions for which to compute the feature coordinates. If
        `None` (the default), compute the feature coordinates for all
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
    return (x + self.padding[axis] +
            (1 - self.size[axis]) / 2) / self.stride[axis]

  def __iter__(self):
    return iter(np.concatenate([self.size, self.stride, self.padding]))


def compute_receptive_field_from_graph_def(graph_def,
                                           input_node,
                                           output_node,
                                           stop_propagation=None,
                                           input_resolution=None):
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
    stop_propagation: List of operations or scope names for which to stop the
      propagation of the receptive field.
    input_resolution: 2D list. If the input resolution to the model is fixed and
      known, this may be set. This is helpful for cases where the RF parameters
      vary depending on the input resolution (this happens since SAME padding in
      tensorflow depends on input resolution in general). If this is None, it is
      assumed that the input resolution is unknown, so some RF parameters may be
      unknown (depending on the model architecture).

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
  node_info, name_to_node = graph_compute_order.get_compute_order(
      graph_def=graph_def,
      input_node_name=input_node,
      input_node_size=input_resolution)

  # Sort in reverse topological order.
  ordered_node_info = sorted(node_info.items(), key=lambda x: -x[1].order)

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

  for _, (o, node, _, _) in ordered_node_info:
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
      (kernel_size_x, kernel_size_y, stride_x, stride_y, padding_x, padding_y,
       _, _) = parse_layer_parameters.get_layer_params(
           node, name_to_node, node_info[node.name].input_size)
      logging.vlog(
          3, "kernel_size_x = %s, kernel_size_y = %s, "
          "stride_x = %s, stride_y = %s, "
          "padding_x = %s, padding_y = %s, input size = %s" %
          (kernel_size_x, kernel_size_y, stride_x, stride_y, padding_x,
           padding_y, node_info[node.name].input_size))
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
      logging.vlog(
          4, "rf_size_input_x = %s, rf_size_input_y = %s, "
          "effective_stride_input_x = %s, effective_stride_input_y = %s, "
          "effective_padding_input_x = %s, effective_padding_input_y = %s" %
          (rf_size_input_x, rf_size_input_y, effective_stride_input_x,
           effective_stride_input_y, effective_padding_input_x,
           effective_padding_input_y))

      # Loop over this node's inputs and potentially propagate information down.
      for inp_name in node.input:
        # Stop the propagation of the receptive field.
        if any(inp_name.startswith(stop) for stop in stop_propagation):
          logging.vlog(3, "Skipping explicitly ignored node %s.", inp_name)
          continue

        logging.vlog(4, "inp_name = %s", inp_name)
        if inp_name.startswith("^"):
          # The character "^" denotes a control dependency, so this input node
          # can be safely ignored.
          continue

        inp_node = name_to_node[inp_name]
        logging.vlog(4, "inp_node = \n%s", inp_node)
        if inp_name in rf_sizes_x:
          assert inp_name in rf_sizes_y, ("Node %s is in rf_sizes_x, but "
                                          "not in rf_sizes_y" % inp_name)
          logging.vlog(
              4, "rf_sizes_x[inp_name] = %s,"
              " rf_sizes_y[inp_name] = %s, "
              "effective_strides_x[inp_name] = %s,"
              " effective_strides_y[inp_name] = %s, "
              "effective_paddings_x[inp_name] = %s,"
              " effective_paddings_y[inp_name] = %s" %
              (rf_sizes_x[inp_name], rf_sizes_y[inp_name],
               effective_strides_x[inp_name], effective_strides_y[inp_name],
               effective_paddings_x[inp_name], effective_paddings_y[inp_name]))
          # This node was already discovered through a previous path, so we need
          # to make sure that graph is aligned. This alignment check is skipped
          # if the padding is not defined, since in this case alignment cannot
          # be checked.
          if not undefined_padding:
            if effective_strides_x[inp_name] != effective_stride_input_x:
              raise ValueError(
                  "Graph is not aligned since effective stride from different "
                  "paths is different in horizontal direction")
            if effective_strides_y[inp_name] != effective_stride_input_y:
              raise ValueError(
                  "Graph is not aligned since effective stride from different "
                  "paths is different in vertical direction")
            if (rf_sizes_x[inp_name] -
                1) / 2 - effective_paddings_x[inp_name] != (
                    rf_size_input_x - 1) / 2 - effective_padding_input_x:
              raise ValueError(
                  "Graph is not aligned since center shift from different "
                  "paths is different in horizontal direction")
            if (rf_sizes_y[inp_name] -
                1) / 2 - effective_paddings_y[inp_name] != (
                    rf_size_input_y - 1) / 2 - effective_padding_input_y:
              raise ValueError(
                  "Graph is not aligned since center shift from different "
                  "paths is different in vertical direction")
          # Keep track of path with largest RF, for both directions.
          if rf_sizes_x[inp_name] < rf_size_input_x:
            rf_sizes_x[inp_name] = rf_size_input_x
            effective_strides_x[inp_name] = effective_stride_input_x
            effective_paddings_x[inp_name] = effective_padding_input_x
          if rf_sizes_y[inp_name] < rf_size_input_y:
            rf_sizes_y[inp_name] = rf_size_input_y
            effective_strides_y[inp_name] = effective_stride_input_y
            effective_paddings_y[inp_name] = effective_padding_input_y
        else:
          assert inp_name not in rf_sizes_y, ("Node %s is in rf_sizes_y, but "
                                              "not in rf_sizes_x" % inp_name)
          # In this case, it is the first time we encounter this node. So we
          # propagate the RF parameters.
          rf_sizes_x[inp_name] = rf_size_input_x
          rf_sizes_y[inp_name] = rf_size_input_y
          effective_strides_x[inp_name] = effective_stride_input_x
          effective_strides_y[inp_name] = effective_stride_input_y
          effective_paddings_x[inp_name] = effective_padding_input_x
          effective_paddings_y[inp_name] = effective_padding_input_y

  if not found_output_node:
    raise ValueError("Output node was not found")
  if input_node not in rf_sizes_x:
    raise ValueError("Input node was not found")
  return ReceptiveField(
      (rf_sizes_x[input_node], rf_sizes_y[input_node]),
      (effective_strides_x[input_node], effective_strides_y[input_node]),
      (effective_paddings_x[input_node], effective_paddings_y[input_node]))
