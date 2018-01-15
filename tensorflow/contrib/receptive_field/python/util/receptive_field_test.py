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
"""Tests for receptive_fields module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import slim
from tensorflow.contrib.receptive_field import receptive_field_api as receptive_field
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


# TODO(andrearaujo): Rename the create_test_network_* functions in order to have
# more descriptive names.
def create_test_network_1():
  """Aligned network for test.

  The graph corresponds to the example from the second figure in
  go/cnn-rf-computation#arbitrary-computation-graphs

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = ops.Graph()
  with g.as_default():
    # An 8x8 test image.
    x = array_ops.placeholder(dtypes.float32, (1, 8, 8, 1), name='input_image')
    # Left branch.
    l1 = slim.conv2d(x, 1, [1, 1], stride=4, scope='L1', padding='VALID')
    # Right branch.
    l2_pad = array_ops.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l2 = slim.conv2d(l2_pad, 1, [3, 3], stride=2, scope='L2', padding='VALID')
    l3 = slim.conv2d(l2, 1, [1, 1], stride=2, scope='L3', padding='VALID')
    # Addition.
    nn.relu(l1 + l3, name='output')
  return g


def create_test_network_2():
  """Aligned network for test.

  The graph corresponds to a variation to the example from the second figure in
  go/cnn-rf-computation#arbitrary-computation-graphs. Layers 2 and 3 are changed
  to max-pooling operations. Since the functionality is the same as convolution,
  the network is aligned and the receptive field size is the same as from the
  network created using create_test_network_1().

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = ops.Graph()
  with g.as_default():
    # An 8x8 test image.
    x = array_ops.placeholder(dtypes.float32, (1, 8, 8, 1), name='input_image')
    # Left branch.
    l1 = slim.conv2d(x, 1, [1, 1], stride=4, scope='L1', padding='VALID')
    # Right branch.
    l2_pad = array_ops.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l2 = slim.max_pool2d(l2_pad, [3, 3], stride=2, scope='L2', padding='VALID')
    l3 = slim.max_pool2d(l2, [1, 1], stride=2, scope='L3', padding='VALID')
    # Addition.
    nn.relu(l1 + l3, name='output')
  return g


def create_test_network_3():
  """Misaligned network for test.

  The graph corresponds to the example from the first figure in
  go/cnn-rf-computation#arbitrary-computation-graphs

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = ops.Graph()
  with g.as_default():
    # An 8x8 test image.
    x = array_ops.placeholder(dtypes.float32, (1, 8, 8, 1), name='input_image')
    # Left branch.
    l1_pad = array_ops.pad(x, [[0, 0], [2, 1], [2, 1], [0, 0]])
    l1 = slim.conv2d(l1_pad, 1, [5, 5], stride=2, scope='L1', padding='VALID')
    # Right branch.
    l2 = slim.conv2d(x, 1, [3, 3], stride=1, scope='L2', padding='VALID')
    l3 = slim.conv2d(l2, 1, [3, 3], stride=1, scope='L3', padding='VALID')
    # Addition.
    nn.relu(l1 + l3, name='output')
  return g


def create_test_network_4():
  """Misaligned network for test.

  The graph corresponds to a variation from the example from the second figure
  in go/cnn-rf-computation#arbitrary-computation-graphs. Layer 2 uses 'SAME'
  padding, which makes its padding dependent on the input image dimensionality.
  In this case, the effective padding will be undetermined, and the utility is
  not able to check the network alignment.

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = ops.Graph()
  with g.as_default():
    # An 8x8 test image.
    x = array_ops.placeholder(dtypes.float32, (1, 8, 8, 1), name='input_image')
    # Left branch.
    l1 = slim.conv2d(x, 1, [1, 1], stride=4, scope='L1', padding='VALID')
    # Right branch.
    l2 = slim.conv2d(x, 1, [3, 3], stride=2, scope='L2', padding='SAME')
    l3 = slim.conv2d(l2, 1, [1, 1], stride=2, scope='L3', padding='VALID')
    # Addition.
    nn.relu(l1 + l3, name='output')
  return g


def create_test_network_5():
  """Single-path network for testing non-square kernels.

  The graph is similar to the right branch of the graph from
  create_test_network_1(), except that the kernel sizes are changed to be
  non-square.

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = ops.Graph()
  with g.as_default():
    # An 8x8 test image.
    x = array_ops.placeholder(dtypes.float32, (1, 8, 8, 1), name='input_image')
    # Two convolutional layers, where the first one has non-square kernel.
    l1 = slim.conv2d(x, 1, [3, 5], stride=2, scope='L1', padding='VALID')
    l2 = slim.conv2d(l1, 1, [3, 1], stride=2, scope='L2', padding='VALID')
    # ReLU.
    nn.relu(l2, name='output')
  return g


def create_test_network_6():
  """Aligned network with dropout for test.

  The graph is similar to create_test_network_1(), except that the right branch
  has dropout normalization.

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = ops.Graph()
  with g.as_default():
    # An 8x8 test image.
    x = array_ops.placeholder(dtypes.float32, (1, 8, 8, 1), name='input_image')
    # Left branch.
    l1 = slim.conv2d(x, 1, [1, 1], stride=4, scope='L1', padding='VALID')
    # Right branch.
    l2_pad = array_ops.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l2 = slim.conv2d(l2_pad, 1, [3, 3], stride=2, scope='L2', padding='VALID')
    l3 = slim.conv2d(l2, 1, [1, 1], stride=2, scope='L3', padding='VALID')
    dropout = slim.dropout(l3)
    # Addition.
    nn.relu(l1 + dropout, name='output')
  return g


def create_test_network_7():
  """Aligned network for test, with a control dependency.

  The graph is similar to create_test_network_1(), except that it includes an
  assert operation on the left branch.

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = ops.Graph()
  with g.as_default():
    # An 8x8 test image.
    x = array_ops.placeholder(dtypes.float32, (1, 8, 8, 1), name='input_image')
    # Left branch.
    l1 = slim.conv2d(x, 1, [1, 1], stride=4, scope='L1', padding='VALID')
    l1_shape = array_ops.shape(l1)
    assert_op = control_flow_ops.Assert(
        gen_math_ops.equal(l1_shape[1], 2), [l1_shape], summarize=4)
    # Right branch.
    l2_pad = array_ops.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l2 = slim.conv2d(l2_pad, 1, [3, 3], stride=2, scope='L2', padding='VALID')
    l3 = slim.conv2d(l2, 1, [1, 1], stride=2, scope='L3', padding='VALID')
    # Addition.
    with ops.control_dependencies([assert_op]):
      nn.relu(l1 + l3, name='output')
  return g


def create_test_network_8():
  """Aligned network for test, including an intermediate addition.

  The graph is similar to create_test_network_1(), except that it includes a few
  more layers on top. The added layers compose two different branches whose
  receptive fields are different. This makes this test case more challenging; in
  particular, this test fails if a naive DFS-like algorithm is used for RF
  computation.

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = ops.Graph()
  with g.as_default():
    # A 16x16 test image.
    x = array_ops.placeholder(
        dtypes.float32, (1, 16, 16, 1), name='input_image')
    # Left branch before first addition.
    l1 = slim.conv2d(x, 1, [1, 1], stride=4, scope='L1', padding='VALID')
    # Right branch before first addition.
    l2_pad = array_ops.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l2 = slim.conv2d(l2_pad, 1, [3, 3], stride=2, scope='L2', padding='VALID')
    l3 = slim.conv2d(l2, 1, [1, 1], stride=2, scope='L3', padding='VALID')
    # First addition.
    l4 = nn.relu(l1 + l3)
    # Left branch after first addition.
    l5 = slim.conv2d(l4, 1, [1, 1], stride=2, scope='L5', padding='VALID')
    # Right branch after first addition.
    l6_pad = array_ops.pad(l4, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l6 = slim.conv2d(l6_pad, 1, [3, 3], stride=2, scope='L6', padding='VALID')
    # Final addition.
    nn.relu(l5 + l6, name='output')

  return g


class RfUtilsTest(test.TestCase):

  def testComputeRFFromGraphDefAligned(self):
    graph_def = create_test_network_1().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         receptive_field.compute_receptive_field_from_graph_def(
             graph_def, input_node, output_node))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 1)
    self.assertEqual(effective_padding_y, 1)

  def testComputeRFFromGraphDefAligned2(self):
    graph_def = create_test_network_2().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         receptive_field.compute_receptive_field_from_graph_def(
             graph_def, input_node, output_node))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 1)
    self.assertEqual(effective_padding_y, 1)

  def testComputeRFFromGraphDefUnaligned(self):
    graph_def = create_test_network_3().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    with self.assertRaises(ValueError):
      receptive_field.compute_receptive_field_from_graph_def(
          graph_def, input_node, output_node)

  def testComputeRFFromGraphDefUnaligned2(self):
    graph_def = create_test_network_4().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         receptive_field.compute_receptive_field_from_graph_def(
             graph_def, input_node, output_node))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, None)
    self.assertEqual(effective_padding_y, None)

  def testComputeRFFromGraphDefNonSquareRF(self):
    graph_def = create_test_network_5().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         receptive_field.compute_receptive_field_from_graph_def(
             graph_def, input_node, output_node))
    self.assertEqual(receptive_field_x, 5)
    self.assertEqual(receptive_field_y, 7)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 0)
    self.assertEqual(effective_padding_y, 0)

  def testComputeRFFromGraphDefStopPropagation(self):
    graph_def = create_test_network_6().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    # Compute the receptive field but stop the propagation for the random
    # uniform variable of the dropout.
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         receptive_field.compute_receptive_field_from_graph_def(
             graph_def, input_node, output_node,
             ['Dropout/dropout/random_uniform']))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 1)
    self.assertEqual(effective_padding_y, 1)

  def testComputeCoordinatesRoundtrip(self):
    graph_def = create_test_network_1()
    input_node = 'input_image'
    output_node = 'output'
    rf = receptive_field.compute_receptive_field_from_graph_def(
        graph_def, input_node, output_node)

    x = np.random.randint(0, 100, (50, 2))
    y = rf.compute_feature_coordinates(x)
    x2 = rf.compute_input_center_coordinates(y)

    self.assertAllEqual(x, x2)

  def testComputeRFFromGraphDefAlignedWithControlDependencies(self):
    graph_def = create_test_network_7().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         receptive_field.compute_receptive_field_from_graph_def(
             graph_def, input_node, output_node))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 1)
    self.assertEqual(effective_padding_y, 1)

  def testComputeRFFromGraphDefWithIntermediateAddNode(self):
    graph_def = create_test_network_8().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         receptive_field.compute_receptive_field_from_graph_def(
             graph_def, input_node, output_node))
    self.assertEqual(receptive_field_x, 11)
    self.assertEqual(receptive_field_y, 11)
    self.assertEqual(effective_stride_x, 8)
    self.assertEqual(effective_stride_y, 8)
    self.assertEqual(effective_padding_x, 5)
    self.assertEqual(effective_padding_y, 5)


if __name__ == '__main__':
  test.main()
