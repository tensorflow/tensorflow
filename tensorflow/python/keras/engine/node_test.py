# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#,============================================================================
"""Tests for layer graphs construction & handling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.platform import test


class DummyTensor(object):

  def __init__(self, shape=None):
    self.shape = shape


class DummyLayer(base_layer.Layer):
  pass


class NetworkConstructionTest(keras_parameterized.TestCase):

  def test_chained_node_construction(self):
    # test basics
    a = DummyTensor(shape=(None, 32))
    b = DummyTensor(shape=(None, 32))

    a_layer = DummyLayer()
    node = node_module.Node(a_layer, outputs=a)
    self.assertEqual(node.outbound_layer, a_layer)

    self.assertTrue(node.is_input)
    self.assertListEqual(node.inbound_layers, [])
    self.assertListEqual(node.input_tensors, [a])
    self.assertListEqual(node.input_shapes, [(None, 32)])
    self.assertListEqual(node.output_tensors, [a])
    self.assertListEqual(node.output_shapes, [(None, 32)])

    b_layer = DummyLayer()
    node_module.Node(b_layer, outputs=b)

    dense = DummyLayer()
    a_2 = DummyTensor()
    node_a = node_module.Node(layer=dense, call_args=(a,), outputs=a_2)
    b_2 = DummyTensor()
    node_b = node_module.Node(layer=dense, call_args=(b,), outputs=b_2)

    # test the node attributes
    self.assertFalse(node_a.is_input)
    self.assertFalse(node_b.is_input)
    self.assertEqual(node_a.call_args, (a,))
    self.assertEqual(node_a.call_kwargs, {})
    self.assertEqual(node_a.outputs, a_2)

    # Test the layer wiring
    self.assertLen(dense._inbound_nodes, 2)
    self.assertLen(dense._outbound_nodes, 0)
    self.assertEqual(dense._inbound_nodes, [node_a, node_b])
    self.assertEqual(dense._inbound_nodes[0].inbound_layers, a_layer)
    self.assertEqual(dense._inbound_nodes[0].outbound_layer, dense)
    self.assertEqual(dense._inbound_nodes[1].inbound_layers, b_layer)
    self.assertEqual(dense._inbound_nodes[1].outbound_layer, dense)
    self.assertIs(dense._inbound_nodes[0].input_tensors, a)
    self.assertIs(dense._inbound_nodes[1].input_tensors, b)

  def test_multi_input_node(self):
    # test multi-input layer
    a = DummyTensor()
    b = DummyTensor()

    dense = DummyLayer()
    a_2 = DummyTensor()
    node_module.Node(layer=dense, call_args=(a,), outputs=a_2)
    b_2 = DummyTensor()
    node_module.Node(layer=dense, call_args=(b,), outputs=b_2)

    concat_layer = DummyLayer()
    merged = DummyTensor()
    node_module.Node(layer=concat_layer, call_args=([a_2, b_2],),
                     outputs=merged)

    merge_layer, merge_node_index, merge_tensor_index = merged._keras_history

    self.assertEqual(merge_node_index, 0)
    self.assertEqual(merge_tensor_index, 0)

    self.assertLen(merge_layer._inbound_nodes, 1)
    self.assertLen(merge_layer._outbound_nodes, 0)

    self.assertLen(merge_layer._inbound_nodes[0].input_tensors, 2)
    self.assertEqual(merge_layer._inbound_nodes[0].input_tensors, [a_2, b_2])
    self.assertLen(merge_layer._inbound_nodes[0].inbound_layers, 2)

  def test_arg_and_kwarg_mix(self):
    input_layer = DummyLayer()
    input_layer_2 = DummyLayer()
    a = DummyTensor()
    node_a = node_module.Node(layer=input_layer, outputs=a)
    b = DummyTensor()
    node_b = node_module.Node(layer=input_layer_2, outputs=b)

    arg_2 = DummyTensor()
    arg_3 = DummyTensor()
    node_c = node_module.Node(layer=input_layer, outputs=arg_3)

    kwarg_x = DummyTensor()
    kwarg_y = DummyTensor()
    node_d = node_module.Node(layer=input_layer, outputs=kwarg_y)

    merge_layer = DummyLayer()
    merged = DummyTensor()
    node = node_module.Node(layer=merge_layer,
                            call_args=([a, b], arg_2, arg_3),
                            call_kwargs={'x': kwarg_x, 'y': kwarg_y},
                            outputs=merged)

    merge_layer, merge_node_index, merge_tensor_index = merged._keras_history

    # Check the saved call args/kwargs
    self.assertEqual(([a, b], arg_2, arg_3), node.call_args)
    self.assertEqual({'x': kwarg_x, 'y': kwarg_y}, node.call_kwargs)

    # Only the inputs that were produced by input nodes should appear in
    # keras_tensors
    self.assertEqual({a, b, arg_3, kwarg_y}, set(node.keras_inputs))
    self.assertEqual(set(node.parent_nodes), {node_a, node_b, node_c, node_d})

    # Check the layer wirings
    self.assertEqual(merge_node_index, 0)
    self.assertEqual(merge_tensor_index, 0)
    self.assertLen(merge_layer._inbound_nodes, 1)
    self.assertLen(merge_layer._outbound_nodes, 0)
    self.assertLen(input_layer._outbound_nodes, 3)
    self.assertLen(input_layer_2._outbound_nodes, 1)

    # The 'backwards compatibility' attributes should only check the
    # first call argument
    self.assertLen(merge_layer._inbound_nodes[0].input_tensors, 2)
    self.assertEqual(merge_layer._inbound_nodes[0].input_tensors, [a, b])
    self.assertLen(merge_layer._inbound_nodes[0].inbound_layers, 2)


if __name__ == '__main__':
  test.main()
