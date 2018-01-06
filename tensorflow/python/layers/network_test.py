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
"""Tests for tf.layers.network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.layers import base as base_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import network as network_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class BaseLayerCompatibilityTest(test.TestCase):

  def test_get_updates_for(self):
    a = network_layers.Input(shape=(2,))
    dense_layer = core_layers.Dense(1)
    dense_layer.add_update(0, inputs=a)
    dense_layer.add_update(1, inputs=None)

    self.assertEqual(dense_layer.get_updates_for(a), [0])
    self.assertEqual(dense_layer.get_updates_for(None), [1])

  def test_get_losses_for(self):
    a = network_layers.Input(shape=(2,))
    dense_layer = core_layers.Dense(1)
    dense_layer.add_loss(0, inputs=a)
    dense_layer.add_loss(1, inputs=None)

    self.assertEqual(dense_layer.get_losses_for(a), [0])
    self.assertEqual(dense_layer.get_losses_for(None), [1])

  def testTopologicalAttributes(self):
    # test layer attributes / methods related to cross-layer connectivity.
    a = network_layers.Input(shape=(32,), name='input_a')
    b = network_layers.Input(shape=(32,), name='input_b')

    # test input, output, input_shape, output_shape
    test_layer = core_layers.Dense(16, name='test_layer')
    a_test = test_layer(a)
    self.assertEqual(test_layer.input, a)
    self.assertEqual(test_layer.output, a_test)
    self.assertEqual(test_layer.input_shape, (None, 32))
    self.assertEqual(test_layer.output_shape, (None, 16))

    # test `get_*_at` methods
    dense = core_layers.Dense(16, name='dense_1')
    a_2 = dense(a)
    b_2 = dense(b)

    self.assertEqual(dense.get_input_at(0), a)
    self.assertEqual(dense.get_input_at(1), b)
    self.assertEqual(dense.get_output_at(0), a_2)
    self.assertEqual(dense.get_output_at(1), b_2)
    self.assertEqual(dense.get_input_shape_at(0), (None, 32))
    self.assertEqual(dense.get_input_shape_at(1), (None, 32))
    self.assertEqual(dense.get_output_shape_at(0), (None, 16))
    self.assertEqual(dense.get_output_shape_at(1), (None, 16))

    # Test invalid value for attribute retrieval.
    with self.assertRaises(ValueError):
      dense.get_input_at(2)
    with self.assertRaises(AttributeError):
      new_dense = core_layers.Dense(16)
      _ = new_dense.input
    with self.assertRaises(AttributeError):
      new_dense = core_layers.Dense(16)
      _ = new_dense.output
    with self.assertRaises(AttributeError):
      new_dense = core_layers.Dense(16)
      _ = new_dense.output_shape
    with self.assertRaises(AttributeError):
      new_dense = core_layers.Dense(16)
      _ = new_dense.input_shape
    with self.assertRaises(AttributeError):
      new_dense = core_layers.Dense(16)
      a = network_layers.Input(shape=(3, 32))
      a = network_layers.Input(shape=(5, 32))
      a_2 = dense(a)
      b_2 = dense(b)
      _ = new_dense.input_shape
    with self.assertRaises(AttributeError):
      new_dense = core_layers.Dense(16)
      a = network_layers.Input(shape=(3, 32))
      a = network_layers.Input(shape=(5, 32))
      a_2 = dense(a)
      b_2 = dense(b)
      _ = new_dense.output_shape

  def testTopologicalAttributesMultiOutputLayer(self):

    class PowersLayer(base_layers.Layer):

      def call(self, inputs):
        return [inputs**2, inputs**3]

    x = network_layers.Input(shape=(32,))
    test_layer = PowersLayer()
    p1, p2 = test_layer(x)  # pylint: disable=not-callable

    self.assertEqual(test_layer.input, x)
    self.assertEqual(test_layer.output, [p1, p2])
    self.assertEqual(test_layer.input_shape, (None, 32))
    self.assertEqual(test_layer.output_shape, [(None, 32), (None, 32)])

  def testTopologicalAttributesMultiInputLayer(self):

    class AddLayer(base_layers.Layer):

      def call(self, inputs):
        assert len(inputs) == 2
        return inputs[0] + inputs[1]

    a = network_layers.Input(shape=(32,))
    b = network_layers.Input(shape=(32,))
    test_layer = AddLayer()
    y = test_layer([a, b])  # pylint: disable=not-callable

    self.assertEqual(test_layer.input, [a, b])
    self.assertEqual(test_layer.output, y)
    self.assertEqual(test_layer.input_shape, [(None, 32), (None, 32)])
    self.assertEqual(test_layer.output_shape, (None, 32))


class NetworkTest(test.TestCase):

  def testBasicNetwork(self):
    # minimum viable network
    x = network_layers.Input(shape=(32,))
    dense = core_layers.Dense(2)
    y = dense(x)
    network = network_layers.GraphNetwork(x, y, name='dense_network')

    # test basic attributes
    self.assertEqual(network.name, 'dense_network')
    self.assertEqual(len(network.layers), 2)  # InputLayer + Dense
    self.assertEqual(network.layers[1], dense)
    self.assertEqual(network.weights, dense.weights)
    self.assertEqual(network.trainable_weights, dense.trainable_weights)
    self.assertEqual(network.non_trainable_weights, dense.non_trainable_weights)

    # test callability on Input
    x_2 = network_layers.Input(shape=(32,))
    y_2 = network(x_2)
    self.assertEqual(y_2.get_shape().as_list(), [None, 2])

    # test callability on regular tensor
    x_2 = array_ops.placeholder(dtype='float32', shape=(None, 32))
    y_2 = network(x_2)
    self.assertEqual(y_2.get_shape().as_list(), [None, 2])

    # test network `trainable` attribute
    network.trainable = False
    self.assertEqual(network.weights, dense.weights)
    self.assertEqual(network.trainable_weights, [])
    self.assertEqual(network.non_trainable_weights,
                     dense.trainable_weights + dense.non_trainable_weights)

  def test_node_construction(self):
    # test graph topology construction basics
    a = network_layers.Input(shape=(32,), name='input_a')
    b = network_layers.Input(shape=(32,), name='input_b')

    self.assertEqual(a.get_shape().as_list(), [None, 32])
    a_layer, a_node_index, a_tensor_index = a._keras_history
    b_layer, _, _ = b._keras_history
    self.assertEqual(len(a_layer._inbound_nodes), 1)
    self.assertEqual(a_tensor_index, 0)
    node = a_layer._inbound_nodes[a_node_index]
    self.assertEqual(node.outbound_layer, a_layer)

    self.assertEqual(node.inbound_layers, [])
    self.assertEqual(node.input_tensors, [a])
    self.assertEqual(node.input_shapes, [(None, 32)])
    self.assertEqual(node.output_tensors, [a])
    self.assertEqual(node.output_shapes, [(None, 32)])

    dense = core_layers.Dense(16, name='dense_1')
    dense(a)
    dense(b)

    self.assertEqual(len(dense._inbound_nodes), 2)
    self.assertEqual(len(dense._outbound_nodes), 0)
    self.assertEqual(dense._inbound_nodes[0].inbound_layers, [a_layer])
    self.assertEqual(dense._inbound_nodes[0].outbound_layer, dense)
    self.assertEqual(dense._inbound_nodes[1].inbound_layers, [b_layer])
    self.assertEqual(dense._inbound_nodes[1].outbound_layer, dense)
    self.assertEqual(dense._inbound_nodes[0].input_tensors, [a])
    self.assertEqual(dense._inbound_nodes[1].input_tensors, [b])

    # Test config
    config_0 = dense._inbound_nodes[0].get_config()
    self.assertEqual(config_0['outbound_layer'], dense.name)

  def testMultiInputNetwork(self):
    a = network_layers.Input(shape=(32,), name='input_a')
    b = network_layers.Input(shape=(32,), name='input_b')

    class AddLayer(base_layers.Layer):

      def call(self, inputs):
        assert len(inputs) == 2
        return inputs[0] + inputs[1]

    c = AddLayer()([a, b])  # pylint: disable=not-callable
    network = network_layers.GraphNetwork([a, b], c)
    self.assertEqual(len(network.layers), 3)  # 2 * InputLayer + AddLayer

    # Test callability.
    a2 = network_layers.Input(shape=(32,))
    b2 = network_layers.Input(shape=(32,))
    c2 = network([a2, b2])
    self.assertEqual(c2.get_shape().as_list(), [None, 32])

  def testMultiOutputNetwork(self):
    x = network_layers.Input(shape=(32,))
    y1 = core_layers.Dense(2)(x)
    y2 = core_layers.Dense(3)(x)
    network = network_layers.GraphNetwork(x, [y1, y2])

    self.assertEqual(len(network.layers), 3)  # InputLayer + 2 * Dense

    # Test callability.
    x2 = network_layers.Input(shape=(32,))
    outputs = network(x2)

    self.assertEqual(type(outputs), list)
    self.assertEqual(len(outputs), 2)
    self.assertEqual(outputs[0].get_shape().as_list(), [None, 2])
    self.assertEqual(outputs[1].get_shape().as_list(), [None, 3])

  def testMultiInputMultiOutputNetworkSharedLayer(self):
    a = network_layers.Input(shape=(32,), name='input_a')
    b = network_layers.Input(shape=(32,), name='input_b')

    dense = core_layers.Dense(2)

    y1 = dense(a)
    y2 = dense(b)
    network = network_layers.GraphNetwork([a, b], [y1, y2])
    self.assertEqual(len(network.layers), 3)  # 2 * InputLayer + Dense

    # Test callability.
    a2 = network_layers.Input(shape=(32,))
    b2 = network_layers.Input(shape=(32,))
    outputs = network([a2, b2])

    self.assertEqual(type(outputs), list)
    self.assertEqual(len(outputs), 2)
    self.assertEqual(outputs[0].get_shape().as_list(), [None, 2])
    self.assertEqual(outputs[1].get_shape().as_list(), [None, 2])

  def testCrossDataFlows(self):
    # Test the ability to have multi-output layers with outputs that get routed
    # to separate layers

    class PowersLayer(base_layers.Layer):

      def call(self, inputs):
        return [inputs**2, inputs**3]

    x = network_layers.Input(shape=(32,))
    p1, p2 = PowersLayer()(x)  # pylint: disable=not-callable
    y1 = core_layers.Dense(2)(p1)
    y2 = core_layers.Dense(3)(p2)
    network = network_layers.GraphNetwork(x, [y1, y2])

    self.assertEqual(len(network.layers), 4)  # InputLayer + 2 * Dense + PLayer

    # Test callability.
    x2 = network_layers.Input(shape=(32,))
    outputs = network(x2)

    self.assertEqual(type(outputs), list)
    self.assertEqual(len(outputs), 2)
    self.assertEqual(outputs[0].get_shape().as_list(), [None, 2])
    self.assertEqual(outputs[1].get_shape().as_list(), [None, 3])

  def testNetworkAttributes(self):
    x = network_layers.Input(shape=(32,))
    z = core_layers.Dense(2, kernel_regularizer=lambda x: 0.01 * (x**2))(x)
    dense = core_layers.Dense(2, name='dense')
    dense.add_update(1)
    y = dense(z)
    net = network_layers.GraphNetwork(x, y)

    # losses
    self.assertEqual(len(net.losses), 1)

    # updates
    self.assertEqual(len(net.updates), 1)

    # get_layer
    self.assertEqual(net.get_layer('dense'), dense)
    self.assertEqual(net.get_layer(index=2), dense)
    with self.assertRaises(ValueError):
      net.get_layer('dense_unknown')
    with self.assertRaises(ValueError):
      net.get_layer()
    with self.assertRaises(ValueError):
      net.get_layer(index=4)

    # input, output
    self.assertEqual(net.input, x)
    self.assertEqual(net.output, y)

    # input_shape, output_shape
    self.assertEqual(net.input_shape, (None, 32))
    self.assertEqual(net.output_shape, (None, 2))

    # get_*_at
    self.assertEqual(net.get_input_at(0), x)
    self.assertEqual(net.get_output_at(0), y)

    # compute_output_shape
    self.assertEqual(net.compute_output_shape((3, 32)).as_list(), [3, 2])

  def testInvalidNetworks(self):
    # redundant inputs
    x = network_layers.Input(shape=(32,))
    y = core_layers.Dense(2)(x)
    with self.assertRaises(ValueError):
      network_layers.GraphNetwork([x, x], y)

    # inputs that don't come from Input
    x = array_ops.placeholder(dtype='float32', shape=(None, 32))
    y = core_layers.Dense(2)(x)
    with self.assertRaises(ValueError):
      network_layers.GraphNetwork(x, y)

    # inputs that don't come from Input but have a layer history
    x = network_layers.Input(shape=(32,))
    x = core_layers.Dense(32)(x)
    y = core_layers.Dense(2)(x)
    with self.assertRaises(ValueError):
      network_layers.GraphNetwork(x, y)

    # outputs that don't come from layers
    x = network_layers.Input(shape=(32,))
    y = core_layers.Dense(2)(x)
    y = 2 * y
    with self.assertRaises(ValueError):
      network_layers.GraphNetwork(x, y)

    # disconnected graphs
    x1 = network_layers.Input(shape=(32,))
    x2 = network_layers.Input(shape=(32,))
    y = core_layers.Dense(2)(x1)
    with self.assertRaises(ValueError):
      network_layers.GraphNetwork(x2, y)

    # redundant layer names
    x = network_layers.Input(shape=(32,))
    z = core_layers.Dense(2, name='dense')(x)
    y = core_layers.Dense(2, name='dense')(z)
    with self.assertRaises(ValueError):
      network_layers.GraphNetwork(x, y)

  def testInputTensorWrapping(self):
    x = array_ops.placeholder(dtype='float32', shape=(None, 32))
    x = network_layers.Input(tensor=x)
    y = core_layers.Dense(2)(x)
    network_layers.GraphNetwork(x, y)

  def testExplicitBatchSize(self):
    x = network_layers.Input(shape=(32,), batch_size=3)
    y = core_layers.Dense(2)(x)
    self.assertEqual(y.get_shape().as_list(), [3, 2])

  def testNetworkRecursion(self):
    # test the ability of networks to be used as layers inside networks.
    a = network_layers.Input(shape=(32,))
    b = core_layers.Dense(2)(a)
    net = network_layers.GraphNetwork(a, b)

    c = network_layers.Input(shape=(32,))
    d = net(c)

    recursive_net = network_layers.GraphNetwork(c, d)
    self.assertEqual(len(recursive_net.layers), 2)
    self.assertEqual(recursive_net.layers[1], net)
    self.assertEqual(len(recursive_net.weights), 2)

    # test callability
    x = array_ops.placeholder(dtype='float32', shape=(None, 32))
    y = recursive_net(x)
    self.assertEqual(y.get_shape().as_list(), [None, 2])

  def testSparseInput(self):

    class SparseSoftmax(base_layers.Layer):

      def call(self, inputs):
        return sparse_ops.sparse_softmax(inputs)

    x = network_layers.Input(shape=(32,), sparse=True)
    y = SparseSoftmax()(x)  # pylint: disable=not-callable
    network = network_layers.GraphNetwork(x, y)

    self.assertEqual(len(network.layers), 2)
    self.assertEqual(network.layers[0].sparse, True)

  @test_util.run_in_graph_and_eager_modes()
  def testMaskingSingleInput(self):

    class MaskedLayer(base_layers.Layer):

      def call(self, inputs, mask=None):
        if mask is not None:
          return inputs * mask
        return inputs

      def compute_mask(self, inputs, mask=None):
        return array_ops.ones_like(inputs)

    if context.in_graph_mode():
      x = network_layers.Input(shape=(32,))
      y = MaskedLayer()(x)  # pylint: disable=not-callable
      network = network_layers.GraphNetwork(x, y)

      # test callability on Input
      x_2 = network_layers.Input(shape=(32,))
      y_2 = network(x_2)
      self.assertEqual(y_2.get_shape().as_list(), [None, 32])

      # test callability on regular tensor
      x_2 = array_ops.placeholder(dtype='float32', shape=(None, 32))
      y_2 = network(x_2)
      self.assertEqual(y_2.get_shape().as_list(), [None, 32])
    else:
      a = constant_op.constant([2] * 32)
      mask = constant_op.constant([0, 1] * 16)
      a._keras_mask = mask
      b = MaskedLayer().apply(a)
      self.assertTrue(hasattr(b, '_keras_mask'))
      self.assertAllEqual(self.evaluate(array_ops.ones_like(mask)),
                          self.evaluate(getattr(b, '_keras_mask')))
      self.assertAllEqual(self.evaluate(a * mask), self.evaluate(b))


class DeferredModeTest(test.TestCase):

  def testDeferredTensorAttributes(self):
    x = base_layers._DeferredTensor(shape=(None, 2), dtype='float32', name='x')
    self.assertEqual(str(x),
                     'DeferredTensor(\'x\', shape=(?, 2), dtype=float32)')
    self.assertEqual(repr(x),
                     '<_DeferredTensor \'x\' shape=(?, 2) dtype=float32>')

  @test_util.run_in_graph_and_eager_modes()
  def testSimpleNetworkBuilding(self):
    inputs = network_layers.Input(shape=(32,))
    if context.in_eager_mode():
      self.assertIsInstance(inputs, base_layers._DeferredTensor)
      self.assertEqual(inputs.dtype.name, 'float32')
      self.assertEqual(inputs.shape.as_list(), [None, 32])

    x = core_layers.Dense(2)(inputs)
    if context.in_eager_mode():
      self.assertIsInstance(x, base_layers._DeferredTensor)
      self.assertEqual(x.dtype.name, 'float32')
      self.assertEqual(x.shape.as_list(), [None, 2])

    outputs = core_layers.Dense(4)(x)
    network = network_layers.GraphNetwork(inputs, outputs)
    self.assertIsInstance(network, network_layers.GraphNetwork)

    if context.in_eager_mode():
      # It should be possible to call such a network on EagerTensors.
      inputs = constant_op.constant(
          np.random.random((10, 32)).astype('float32'))
      outputs = network(inputs)
      self.assertEqual(outputs.shape.as_list(), [10, 4])

  @test_util.run_in_graph_and_eager_modes()
  def testMultiIONetworkbuilding(self):
    input_a = network_layers.Input(shape=(32,))
    input_b = network_layers.Input(shape=(16,))
    a = core_layers.Dense(16)(input_a)

    class AddLayer(base_layers.Layer):

      def call(self, inputs):
        return inputs[0] + inputs[1]

      def compute_output_shape(self, input_shape):
        return input_shape[0]

    c = AddLayer()([a, input_b])  # pylint: disable=not-callable
    c = core_layers.Dense(2)(c)

    network = network_layers.GraphNetwork([input_a, input_b], [a, c])
    if context.in_eager_mode():
      a_val = constant_op.constant(
          np.random.random((10, 32)).astype('float32'))
      b_val = constant_op.constant(
          np.random.random((10, 16)).astype('float32'))
      outputs = network([a_val, b_val])
      self.assertEqual(len(outputs), 2)
      self.assertEqual(outputs[0].shape.as_list(), [10, 16])
      self.assertEqual(outputs[1].shape.as_list(), [10, 2])

if __name__ == '__main__':
  test.main()
