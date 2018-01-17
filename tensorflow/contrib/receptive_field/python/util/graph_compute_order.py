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
"""Library to compute order of computations in a graph.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
from tensorflow.contrib.receptive_field.python.util import parse_layer_parameters
from tensorflow.python.platform import tf_logging as logging


def parse_graph_nodes(graph_def):
  """Helper function to parse GraphDef's nodes.

  It returns a dict mapping from node name to NodeDef.

  Args:
    graph_def: A GraphDef object.

  Returns:
    name_to_node: Dict keyed by node name, each entry containing the node's
      NodeDef.
  """
  name_to_node = {}
  for node_def in graph_def.node:
    name_to_node[node_def.name] = node_def
  return name_to_node


# Named tuple used to collect information from each node in a computation graph.
_node_info = collections.namedtuple(
    'NodeInfo', field_names=['order', 'node', 'input_size', 'output_size'])


def _compute_output_resolution(input_spatial_resolution, kernel_size, stride,
                               total_padding):
  """Computes output resolution, given input resolution and layer parameters.

  Note that this computation is done only over one dimension (eg, x or y).
  If any of the inputs is None, returns None.

  Args:
    input_spatial_resolution: Input spatial resolution (int).
    kernel_size: Kernel size (int).
    stride: Stride (int).
    total_padding: Total padding to be applied (int).
  Returns:
    output_resolution: Ouput dimension (int) or None.
  """
  if (input_spatial_resolution is None) or (kernel_size is None) or (
      stride is None) or (total_padding is None):
    return None
  return int(
      math.ceil((
          input_spatial_resolution + total_padding - kernel_size + 1) / stride))


def _get_computed_nodes(name_to_node,
                        current,
                        node_info,
                        input_node_name='',
                        input_node_size=None):
  """Traverses the graph recursively to compute its topological order.

  Optionally, the function may also compute the input and output feature map
  resolutions at each node. In this case, input_node_name and input_node_size
  must be set. Note that if a node's op type is unknown, the input and output
  resolutions are ignored and set to None.

  Args:
    name_to_node: Dict keyed by node name, each entry containing the node's
      NodeDef.
    current: Current node name.
    node_info: Map of nodes we've already traversed, containing their _node_info
      information.
    input_node_name: Name of node with fixed input resolution (optional).
    input_node_size: Fixed input resolution to use (optional).
  Returns:
    order: Order in topological sort for 'current'.
    input_size: Tensor spatial resolution at input of current node.
    output_size: Tensor spatial resolution at output of current node.
  """
  if current in node_info:
    return (node_info[current].order, node_info[current].input_size,
            node_info[current].output_size)

  node_def = name_to_node[current]

  if current == input_node_name:
    order = 0
    input_size = None
    output_size = input_node_size
    node_info[current] = _node_info(order, node_def, input_size, output_size)
    return (order, input_size, output_size)

  input_size = None
  output_size = None

  order = 0
  number_inputs = 0
  for each in node_def.input:
    # Parses name of input node.
    if each.startswith('^'):
      # The character '^' denotes a control dependency, so this input node can
      # be safely ignored.
      continue
    each = each.split(':')[0]
    # Recursively computes ordering.
    (parent_order, _, parent_output_size) = _get_computed_nodes(
        name_to_node, each, node_info, input_node_name, input_node_size)
    order = max(order, parent_order + 1)
    if number_inputs == 0:
      # For all the types of nodes we consider, the first input corresponds to
      # the feature map.
      input_size = parent_output_size
    number_inputs += 1

  # Figure out output size for this layer.
  logging.vlog(3, 'input_size = %s', input_size)
  if input_size is None:
    output_size = None
  else:
    (kernel_size_x, kernel_size_y, stride_x, stride_y, _, _, total_padding_x,
     total_padding_y) = (
         parse_layer_parameters.get_layer_params(
             node_def, name_to_node, input_size, force=True))
    logging.vlog(3, 'kernel_size_x = %s, kernel_size_y = %s, '
                 'stride_x = %s, stride_y = %s, '
                 'total_padding_x = %s, total_padding_y = %s' %
                 (kernel_size_x, kernel_size_y, stride_x, stride_y,
                  total_padding_x, total_padding_y))
    output_size = [None] * 2
    output_size[0] = _compute_output_resolution(input_size[0], kernel_size_x,
                                                stride_x, total_padding_x)
    output_size[1] = _compute_output_resolution(input_size[1], kernel_size_y,
                                                stride_y, total_padding_y)

  logging.vlog(3, 'output_size = %s', output_size)
  node_info[current] = _node_info(order, node_def, input_size, output_size)

  return order, input_size, output_size


def get_compute_order(graph_def, input_node_name='', input_node_size=None):
  """Computes order of computation for a given CNN graph.

  Optionally, the function may also compute the input and output feature map
  resolutions at each node. In this case, input_node_name and input_node_size
  must be set. Note that if a node's op type is unknown, the input and output
  resolutions are ignored and set to None.

  Args:
    graph_def: GraphDef object.
    input_node_name: Name of node with fixed input resolution (optional). This
      is usually the node name for the input image in a CNN.
    input_node_size: 2D list of integers, fixed input resolution to use
      (optional). This is usually the input resolution used for the input image
      in a CNN (common examples are: [224, 224], [299, 299], [321, 321]).
  Returns:
    node_info: Default dict keyed by node name, mapping to a named tuple with
      the following fields:
      - order: Integer denoting topological order;
      - node: NodeDef for the given node;
      - input_size: 2D list of integers, denoting the input spatial resolution
        to the node;
      - output_size: 2D list of integers, denoting the output spatial resolution
        of the node.
    name_to_node: Dict keyed by node name, each entry containing the node's
      NodeDef.
  """
  name_to_node = parse_graph_nodes(graph_def)
  node_info = collections.defaultdict(_node_info)
  for each in graph_def.node:
    _get_computed_nodes(name_to_node, each.name, node_info, input_node_name,
                        input_node_size)
  return node_info, name_to_node
