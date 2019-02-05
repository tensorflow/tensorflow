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
"""Tests for parse_layer_parameters module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import slim
from tensorflow.contrib.receptive_field.python.util import graph_compute_order
from tensorflow.contrib.receptive_field.python.util import parse_layer_parameters
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


def create_test_network():
  """Convolutional neural network for test.

  Returns:
    name_to_node: Dict keyed by node name, each entry containing the node's
      NodeDef.
  """
  g = ops.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = array_ops.placeholder(
        dtypes.float32, (None, None, None, 1), name='input_image')
    # Left branch before first addition.
    l1 = slim.conv2d(x, 1, [1, 1], stride=4, scope='L1', padding='VALID')
    # Right branch before first addition.
    l2_pad = array_ops.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]], name='L2_pad')
    l2 = slim.conv2d(l2_pad, 1, [3, 3], stride=2, scope='L2', padding='VALID')
    l3 = slim.max_pool2d(l2, [3, 3], stride=2, scope='L3', padding='SAME')
    # First addition.
    l4 = nn.relu(l1 + l3, name='L4_relu')
    # Left branch after first addition.
    l5 = slim.conv2d(l4, 1, [1, 1], stride=2, scope='L5', padding='SAME')
    # Right branch after first addition.
    l6 = slim.conv2d(l4, 1, [3, 3], stride=2, scope='L6', padding='SAME')
    # Final addition.
    gen_math_ops.add(l5, l6, name='L7_add')

  name_to_node = graph_compute_order.parse_graph_nodes(g.as_graph_def())
  return name_to_node


class ParseLayerParametersTest(test.TestCase):

  def testParametersAreParsedCorrectly(self):
    """Checks parameters from create_test_network() are parsed correctly."""
    name_to_node = create_test_network()

    # L1.
    l1_node_name = 'L1/Conv2D'
    l1_params = parse_layer_parameters.get_layer_params(
        name_to_node[l1_node_name], name_to_node)
    expected_l1_params = (1, 1, 4, 4, 0, 0, 0, 0)
    self.assertEqual(l1_params, expected_l1_params)

    # L2 padding.
    l2_pad_name = 'L2_pad'
    l2_pad_params = parse_layer_parameters.get_layer_params(
        name_to_node[l2_pad_name], name_to_node)
    expected_l2_pad_params = (1, 1, 1, 1, 1, 1, 1, 1)
    self.assertEqual(l2_pad_params, expected_l2_pad_params)

    # L2.
    l2_node_name = 'L2/Conv2D'
    l2_params = parse_layer_parameters.get_layer_params(
        name_to_node[l2_node_name], name_to_node)
    expected_l2_params = (3, 3, 2, 2, 0, 0, 0, 0)
    self.assertEqual(l2_params, expected_l2_params)

    # L3.
    l3_node_name = 'L3/MaxPool'
    # - Without knowing input size.
    l3_params = parse_layer_parameters.get_layer_params(
        name_to_node[l3_node_name], name_to_node)
    expected_l3_params = (3, 3, 2, 2, None, None, None, None)
    self.assertEqual(l3_params, expected_l3_params)
    # - Input size is even.
    l3_even_params = parse_layer_parameters.get_layer_params(
        name_to_node[l3_node_name], name_to_node, input_resolution=[4, 4])
    expected_l3_even_params = (3, 3, 2, 2, 0, 0, 1, 1)
    self.assertEqual(l3_even_params, expected_l3_even_params)
    # - Input size is odd.
    l3_odd_params = parse_layer_parameters.get_layer_params(
        name_to_node[l3_node_name], name_to_node, input_resolution=[5, 5])
    expected_l3_odd_params = (3, 3, 2, 2, 1, 1, 2, 2)
    self.assertEqual(l3_odd_params, expected_l3_odd_params)

    # L4.
    l4_node_name = 'L4_relu'
    l4_params = parse_layer_parameters.get_layer_params(
        name_to_node[l4_node_name], name_to_node)
    expected_l4_params = (1, 1, 1, 1, 0, 0, 0, 0)
    self.assertEqual(l4_params, expected_l4_params)

    # L5.
    l5_node_name = 'L5/Conv2D'
    l5_params = parse_layer_parameters.get_layer_params(
        name_to_node[l5_node_name], name_to_node)
    expected_l5_params = (1, 1, 2, 2, 0, 0, 0, 0)
    self.assertEqual(l5_params, expected_l5_params)

    # L6.
    l6_node_name = 'L6/Conv2D'
    # - Without knowing input size.
    l6_params = parse_layer_parameters.get_layer_params(
        name_to_node[l6_node_name], name_to_node)
    expected_l6_params = (3, 3, 2, 2, None, None, None, None)
    self.assertEqual(l6_params, expected_l6_params)
    # - Input size is even.
    l6_even_params = parse_layer_parameters.get_layer_params(
        name_to_node[l6_node_name], name_to_node, input_resolution=[4, 4])
    expected_l6_even_params = (3, 3, 2, 2, 0, 0, 1, 1)
    self.assertEqual(l6_even_params, expected_l6_even_params)
    # - Input size is odd.
    l6_odd_params = parse_layer_parameters.get_layer_params(
        name_to_node[l6_node_name], name_to_node, input_resolution=[5, 5])
    expected_l6_odd_params = (3, 3, 2, 2, 1, 1, 2, 2)
    self.assertEqual(l6_odd_params, expected_l6_odd_params)

    # L7.
    l7_node_name = 'L7_add'
    l7_params = parse_layer_parameters.get_layer_params(
        name_to_node[l7_node_name], name_to_node)
    expected_l7_params = (1, 1, 1, 1, 0, 0, 0, 0)
    self.assertEqual(l7_params, expected_l7_params)


if __name__ == '__main__':
  test.main()
