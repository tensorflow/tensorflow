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
"""Tests for graph_compute_order module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import slim
from tensorflow.contrib.receptive_field import receptive_field_api as receptive_field
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


def create_test_network():
  """Convolutional neural network for test.

  Returns:
    g: Tensorflow graph object (Graph proto).
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

  return g


class GraphComputeOrderTest(test.TestCase):

  def check_topological_sort_and_sizes(self,
                                       node_info,
                                       expected_input_sizes=None,
                                       expected_output_sizes=None):
    """Helper function to check topological sorting and sizes are correct.

    The arguments expected_input_sizes and expected_output_sizes are used to
    check that the sizes are correct, if they are given.

    Args:
      node_info: Default dict keyed by node name, mapping to a named tuple with
        the following keys: {order, node, input_size, output_size}.
      expected_input_sizes: Dict mapping node names to expected input sizes
        (optional).
      expected_output_sizes: Dict mapping node names to expected output sizes
        (optional).
    """
    # Loop over nodes in sorted order, collecting those that were already seen.
    # These will be used to make sure that the graph is topologically sorted.
    # At the same time, we construct dicts from node name to input/output size,
    # which will be used to check those.
    already_seen_nodes = []
    input_sizes = {}
    output_sizes = {}
    for _, (_, node, input_size, output_size) in sorted(
        node_info.items(), key=lambda x: x[1].order):
      for inp_name in node.input:
        # Since the graph is topologically sorted, the inputs to the current
        # node must have been seen beforehand.
        self.assertIn(inp_name, already_seen_nodes)
      input_sizes[node.name] = input_size
      output_sizes[node.name] = output_size
      already_seen_nodes.append(node.name)

    # Check input sizes, if desired.
    if expected_input_sizes is not None:
      for k, v in expected_input_sizes.items():
        self.assertIn(k, input_sizes)
        self.assertEqual(input_sizes[k], v)

    # Check output sizes, if desired.
    if expected_output_sizes is not None:
      for k, v in expected_output_sizes.items():
        self.assertIn(k, output_sizes)
        self.assertEqual(output_sizes[k], v)

  def testGraphOrderIsCorrect(self):
    """Tests that the order and sizes of create_test_network() are correct."""

    graph_def = create_test_network().as_graph_def()

    # Case 1: Input node name/size are not given.
    node_info, _ = receptive_field.get_compute_order(graph_def)
    self.check_topological_sort_and_sizes(node_info)

    # Case 2: Input node name is given, but not size.
    node_info, _ = receptive_field.get_compute_order(
        graph_def, input_node_name='input_image')
    self.check_topological_sort_and_sizes(node_info)

    # Case 3: Input node name and size (224) are given.
    node_info, _ = receptive_field.get_compute_order(
        graph_def, input_node_name='input_image', input_node_size=[224, 224])
    expected_input_sizes = {
        'input_image': None,
        'L1/Conv2D': [224, 224],
        'L2_pad': [224, 224],
        'L2/Conv2D': [225, 225],
        'L3/MaxPool': [112, 112],
        'L4_relu': [56, 56],
        'L5/Conv2D': [56, 56],
        'L6/Conv2D': [56, 56],
        'L7_add': [28, 28],
    }
    expected_output_sizes = {
        'input_image': [224, 224],
        'L1/Conv2D': [56, 56],
        'L2_pad': [225, 225],
        'L2/Conv2D': [112, 112],
        'L3/MaxPool': [56, 56],
        'L4_relu': [56, 56],
        'L5/Conv2D': [28, 28],
        'L6/Conv2D': [28, 28],
        'L7_add': [28, 28],
    }
    self.check_topological_sort_and_sizes(node_info, expected_input_sizes,
                                          expected_output_sizes)


if __name__ == '__main__':
  test.main()
