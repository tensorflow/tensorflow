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
"""Tests for tf.layers.base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from tensorflow.python.framework import ops
from tensorflow.python.layers import base as base_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


class BaseLayerTest(test.TestCase):

  def testLayerProperties(self):
    layer = base_layers.Layer(name='my_layer')
    self.assertListEqual(layer.variables, [])
    self.assertListEqual(layer.trainable_variables, [])
    self.assertListEqual(layer.non_trainable_variables, [])
    self.assertListEqual(layer.updates, [])
    self.assertListEqual(layer.losses, [])
    self.assertEqual(layer.built, False)
    layer = base_layers.Layer(name='my_layer', trainable=False)
    self.assertEqual(layer.trainable, False)

  def testAddWeight(self):
    with self.test_session():
      layer = base_layers.Layer(name='my_layer')

      # Test basic variable creation.
      variable = layer.add_variable(
          'my_var', [2, 2], initializer=init_ops.zeros_initializer())
      self.assertEqual(variable.name, 'my_layer/my_var:0')
      self.assertListEqual(layer.variables, [variable])
      self.assertListEqual(layer.trainable_variables, [variable])
      self.assertListEqual(layer.non_trainable_variables, [])
      self.assertListEqual(
          layer.variables,
          ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))

      # Test non-trainable variable creation.
      # layer.add_variable should work even outside `build` and `call`.
      variable_2 = layer.add_variable(
          'non_trainable_var', [2, 2],
          initializer=init_ops.zeros_initializer(),
          trainable=False)
      self.assertListEqual(layer.variables, [variable, variable_2])
      self.assertListEqual(layer.trainable_variables, [variable])
      self.assertListEqual(layer.non_trainable_variables, [variable_2])
      self.assertEqual(
          len(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)), 1)

      # Test with regularizer.
      regularizer = lambda x: math_ops.reduce_sum(x) * 1e-3
      variable = layer.add_variable(
          'reg_var', [2, 2],
          initializer=init_ops.zeros_initializer(),
          regularizer=regularizer)
      self.assertEqual(len(layer.losses), 1)

  def testGetVariable(self):
    with self.test_session():

      class MyLayer(base_layers.Layer):

        def build(self, input_shape):
          self.my_var = self.add_variable(
              'my_var', [2, 2], initializer=init_ops.zeros_initializer())

        def call(self, inputs):
          return inputs * 2

      layer = MyLayer(name='my_layer')
      inputs = random_ops.random_uniform((5,), seed=1)
      layer.apply(inputs)
      layer.apply(inputs)
      self.assertListEqual([v.name for v in layer.variables],
                           ['my_layer/my_var:0'])

      # Creating a layer with no scope leads to lazy construction of
      # the scope at apply() time.  It uses scope "<current scope>/base_name"
      lazy_layer = MyLayer(_reuse=True)
      with variable_scope.variable_scope('new_scope'):
        # This should attempt to reuse 'my_var' in 'new_scope'
        with self.assertRaisesRegexp(
            ValueError, r'new_scope/my_layer/my_var does not exist'):
          lazy_layer.apply(inputs)
        with variable_scope.variable_scope('my_layer'):
          variable_scope.get_variable('my_var', [2, 2])

        # Smoke test: it runs.
        lazy_layer.apply(inputs)
        # The variables were created outside of the Layer, and
        # reuse=True, so the Layer does not own them and they are not
        # stored in its collection.
        self.assertListEqual(lazy_layer.variables, [])
        self.assertEqual(lazy_layer._scope.name, 'new_scope/my_layer')

      # Creating a layer with no scope leads to lazy construction of
      # the scope at apply() time.  If 'scope' argument is passed to
      # apply(), it uses that scope when accessing variables.
      lazy_layer = MyLayer(_reuse=True)
      with variable_scope.variable_scope('new_scope') as new_scope:
        # This should attempt to reuse 'my_var' in 'new_scope'
        with self.assertRaisesRegexp(
            ValueError, r'new_scope/my_var does not exist'):
          lazy_layer.apply(inputs, scope=new_scope)
        variable_scope.get_variable('my_var', [2, 2])

        # Smoke test: it runs.
        lazy_layer.apply(inputs, scope=new_scope)
        # The variables were created outside of the Layer, and
        # reuse=True, so the Layer does not own them and they are not
        # stored in its collection.
        self.assertListEqual(lazy_layer.variables, [])
        self.assertEqual(lazy_layer._scope.name, 'new_scope')

      with ops.Graph().as_default():
        inputs_ng = random_ops.random_uniform((5,), seed=1)
        with self.assertRaisesRegexp(ValueError,
                                     r'graph are not the same'):
          layer.apply(inputs_ng)

  def testCall(self):

    class MyLayer(base_layers.Layer):

      def call(self, inputs):
        return math_ops.square(inputs)

    layer = MyLayer(name='my_layer')
    inputs = random_ops.random_uniform((5,), seed=1)
    outputs = layer.apply(inputs)
    self.assertEqual(layer.built, True)
    self.assertEqual(outputs.op.name, 'my_layer/Square')

  def testFirstCallCanCreateVariablesButSecondCanNotWhenBuildEmpty(self):

    class MyLayer(base_layers.Layer):

      def build(self, _):
        # Do not mark the layer as built.
        pass

      def call(self, inputs):
        self.my_var = self.add_variable('my_var', [2, 2])
        if self.built:
          # Skip creating on the first call; try to create after it's
          # built.  This is expected to fail.
          self.add_variable('this_will_break_on_second_call', [2, 2])
        return inputs + math_ops.square(self.my_var)

    layer = MyLayer(name='my_layer')
    inputs = random_ops.random_uniform((2,), seed=1)
    outputs = layer.apply(inputs)
    self.assertEqual(layer.built, True)
    self.assertEqual(outputs.op.name, 'my_layer/add')
    self.assertListEqual(
        [v.name for v in layer.variables], ['my_layer/my_var:0'])
    with self.assertRaisesRegexp(ValueError,
                                 'my_layer/this_will_break_on_second_call'):
      layer.apply(inputs)
    # The list of variables hasn't changed.
    self.assertListEqual(
        [v.name for v in layer.variables], ['my_layer/my_var:0'])

  def testDeepCopy(self):

    class MyLayer(base_layers.Layer):

      def call(self, inputs):
        return math_ops.square(inputs)

    layer = MyLayer(name='my_layer')
    layer._private_tensor = random_ops.random_uniform(())
    inputs = random_ops.random_uniform((5,), seed=1)
    outputs = layer.apply(inputs)
    self.assertEqual(layer.built, True)
    self.assertEqual(outputs.op.name, 'my_layer/Square')

    layer_copy = copy.deepcopy(layer)
    self.assertEqual(layer_copy.name, layer.name)
    self.assertEqual(layer_copy._scope.name, layer._scope.name)
    self.assertEqual(layer_copy._graph, layer._graph)
    self.assertEqual(layer_copy._private_tensor, layer._private_tensor)

  def testScopeNaming(self):

    class PrivateLayer(base_layers.Layer):

      def call(self, inputs):
        return inputs

    inputs = random_ops.random_uniform((5,))
    default_layer = PrivateLayer()
    _ = default_layer.apply(inputs)
    self.assertEqual(default_layer._scope.name, 'private_layer')
    default_layer1 = PrivateLayer()
    default_layer1.apply(inputs)
    self.assertEqual(default_layer1._scope.name, 'private_layer_1')
    my_layer = PrivateLayer(name='my_layer')
    my_layer.apply(inputs)
    self.assertEqual(my_layer._scope.name, 'my_layer')
    my_layer1 = PrivateLayer(name='my_layer')
    my_layer1.apply(inputs)
    self.assertEqual(my_layer1._scope.name, 'my_layer_1')
    my_layer2 = PrivateLayer(name='my_layer')
    my_layer2.apply(inputs)
    self.assertEqual(my_layer2._scope.name, 'my_layer_2')
    # Name scope shouldn't affect names.
    with ops.name_scope('some_name_scope'):
      default_layer2 = PrivateLayer()
      default_layer2.apply(inputs)
      self.assertEqual(default_layer2._scope.name, 'private_layer_2')
      my_layer3 = PrivateLayer(name='my_layer')
      my_layer3.apply(inputs)
      self.assertEqual(my_layer3._scope.name, 'my_layer_3')
      other_layer = PrivateLayer(name='other_layer')
      other_layer.apply(inputs)
      self.assertEqual(other_layer._scope.name, 'other_layer')
    # Variable scope gets added to scope names.
    with variable_scope.variable_scope('var_scope'):
      default_layer_scoped = PrivateLayer()
      default_layer_scoped.apply(inputs)
      self.assertEqual(default_layer_scoped._scope.name,
                       'var_scope/private_layer')
      my_layer_scoped = PrivateLayer(name='my_layer')
      my_layer_scoped.apply(inputs)
      self.assertEqual(my_layer_scoped._scope.name, 'var_scope/my_layer')
      my_layer_scoped1 = PrivateLayer(name='my_layer')
      my_layer_scoped1.apply(inputs)
      self.assertEqual(my_layer_scoped1._scope.name, 'var_scope/my_layer_1')

  def testInputSpecNdimCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = base_layers.InputSpec(ndim=2)

      def call(self, inputs):
        return inputs

    layer = CustomerLayer()
    with self.assertRaisesRegexp(ValueError,
                                 r'requires a defined rank'):
      layer.apply(array_ops.placeholder('int32'))

    with self.assertRaisesRegexp(ValueError,
                                 r'expected ndim=2'):
      layer.apply(array_ops.placeholder('int32', shape=(None,)))

    # Works
    layer.apply(array_ops.placeholder('int32', shape=(None, None)))

  def testInputSpecMinNdimCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = base_layers.InputSpec(min_ndim=2)

      def call(self, inputs):
        return inputs

    layer = CustomerLayer()
    with self.assertRaisesRegexp(ValueError,
                                 r'requires a defined rank'):
      layer.apply(array_ops.placeholder('int32'))

    with self.assertRaisesRegexp(ValueError,
                                 r'expected min_ndim=2'):
      layer.apply(array_ops.placeholder('int32', shape=(None,)))

    # Works
    layer.apply(array_ops.placeholder('int32', shape=(None, None)))
    layer.apply(array_ops.placeholder('int32', shape=(None, None, None)))

  def testInputSpecMaxNdimCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = base_layers.InputSpec(max_ndim=2)

      def call(self, inputs):
        return inputs

    layer = CustomerLayer()
    with self.assertRaisesRegexp(ValueError,
                                 r'requires a defined rank'):
      layer.apply(array_ops.placeholder('int32'))

    with self.assertRaisesRegexp(ValueError,
                                 r'expected max_ndim=2'):
      layer.apply(array_ops.placeholder('int32', shape=(None, None, None)))

    # Works
    layer.apply(array_ops.placeholder('int32', shape=(None, None)))
    layer.apply(array_ops.placeholder('int32', shape=(None,)))

  def testInputSpecDtypeCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = base_layers.InputSpec(dtype='float32')

      def call(self, inputs):
        return inputs

    layer = CustomerLayer()
    with self.assertRaisesRegexp(ValueError,
                                 r'expected dtype=float32'):
      layer.apply(array_ops.placeholder('int32'))

    # Works
    layer.apply(array_ops.placeholder('float32', shape=(None, None)))

  def testInputSpecAxesCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = base_layers.InputSpec(axes={-1: 2})

      def call(self, inputs):
        return inputs

    layer = CustomerLayer()
    with self.assertRaisesRegexp(ValueError,
                                 r'expected axis'):
      layer.apply(array_ops.placeholder('int32', shape=(None, 3)))

    # Works
    layer.apply(array_ops.placeholder('int32', shape=(None, None, 2)))
    layer.apply(array_ops.placeholder('int32', shape=(None, 2)))

  def testInputSpecShapeCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = base_layers.InputSpec(shape=(None, 3))

      def call(self, inputs):
        return inputs

    layer = CustomerLayer()
    with self.assertRaisesRegexp(ValueError,
                                 r'expected shape'):
      layer.apply(array_ops.placeholder('int32', shape=(None, 2)))

    # Works
    layer.apply(array_ops.placeholder('int32', shape=(None, 3)))
    layer.apply(array_ops.placeholder('int32', shape=(2, 3)))

  def testNoInputSpec(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = None

      def call(self, inputs):
        return inputs

    layer = CustomerLayer()

    # Works
    layer.apply(array_ops.placeholder('int32'))
    layer.apply(array_ops.placeholder('int32', shape=(2, 3)))

  def test_get_updates_for(self):
    a = base_layers.Input(shape=(2,))
    dense_layer = core_layers.Dense(1)
    dense_layer.add_update(0, inputs=a)
    dense_layer.add_update(1, inputs=None)

    self.assertListEqual(dense_layer.get_updates_for(a), [0])
    self.assertListEqual(dense_layer.get_updates_for(None), [1])

  def test_get_losses_for(self):
    a = base_layers.Input(shape=(2,))
    dense_layer = core_layers.Dense(1)
    dense_layer.add_loss(0, inputs=a)
    dense_layer.add_loss(1, inputs=None)

    self.assertListEqual(dense_layer.get_losses_for(a), [0])
    self.assertListEqual(dense_layer.get_losses_for(None), [1])

  def testTopologicalAttributes(self):
    # test layer attributes / methods related to cross-layer connectivity.
    a = base_layers.Input(shape=(32,), name='input_a')
    b = base_layers.Input(shape=(32,), name='input_b')

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
      a = base_layers.Input(shape=(3, 32))
      a = base_layers.Input(shape=(5, 32))
      a_2 = dense(a)
      b_2 = dense(b)
      _ = new_dense.input_shape
    with self.assertRaises(AttributeError):
      new_dense = core_layers.Dense(16)
      a = base_layers.Input(shape=(3, 32))
      a = base_layers.Input(shape=(5, 32))
      a_2 = dense(a)
      b_2 = dense(b)
      _ = new_dense.output_shape

  def testTopologicalAttributesMultiOutputLayer(self):
    class PowersLayer(base_layers.Layer):

      def call(self, inputs):
        return [inputs ** 2, inputs ** 3]

    x = base_layers.Input(shape=(32,))
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

    a = base_layers.Input(shape=(32,))
    b = base_layers.Input(shape=(32,))
    test_layer = AddLayer()
    y = test_layer([a, b])  # pylint: disable=not-callable

    self.assertEqual(test_layer.input, [a, b])
    self.assertEqual(test_layer.output, y)
    self.assertEqual(test_layer.input_shape, [(None, 32), (None, 32)])
    self.assertEqual(test_layer.output_shape, (None, 32))

  def test_count_params(self):
    dense = core_layers.Dense(16)
    dense.build((None, 4))
    self.assertEqual(dense.count_params(), 16 * 4 + 16)

    dense = core_layers.Dense(16)
    with self.assertRaises(ValueError):
      dense.count_params()


class NetworkTest(test.TestCase):

  def testBasicNetwork(self):
    # minimum viable network
    x = base_layers.Input(shape=(32,))
    dense = core_layers.Dense(2)
    y = dense(x)
    network = base_layers.Network(x, y, name='dense_network')

    # test basic attributes
    self.assertEqual(network.name, 'dense_network')
    self.assertEqual(len(network.layers), 2)  # InputLayer + Dense
    self.assertEqual(network.layers[1], dense)
    self.assertEqual(network.weights, dense.weights)
    self.assertEqual(network.trainable_weights, dense.trainable_weights)
    self.assertEqual(network.non_trainable_weights, dense.non_trainable_weights)

    # test callability on Input
    x_2 = base_layers.Input(shape=(32,))
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
    a = base_layers.Input(shape=(32,), name='input_a')
    b = base_layers.Input(shape=(32,), name='input_b')

    self.assertListEqual(a.get_shape().as_list(), [None, 32])
    a_layer, a_node_index, a_tensor_index = a._keras_history
    b_layer, _, _ = b._keras_history
    self.assertEqual(len(a_layer.inbound_nodes), 1)
    self.assertEqual(a_tensor_index, 0)
    node = a_layer.inbound_nodes[a_node_index]
    self.assertEqual(node.outbound_layer, a_layer)

    self.assertListEqual(node.inbound_layers, [])
    self.assertListEqual(node.input_tensors, [a])
    self.assertListEqual(node.input_shapes, [(None, 32)])
    self.assertListEqual(node.output_tensors, [a])
    self.assertListEqual(node.output_shapes, [(None, 32)])

    dense = core_layers.Dense(16, name='dense_1')
    dense(a)
    dense(b)

    self.assertEqual(len(dense.inbound_nodes), 2)
    self.assertEqual(len(dense.outbound_nodes), 0)
    self.assertListEqual(dense.inbound_nodes[0].inbound_layers, [a_layer])
    self.assertEqual(dense.inbound_nodes[0].outbound_layer, dense)
    self.assertListEqual(dense.inbound_nodes[1].inbound_layers, [b_layer])
    self.assertEqual(dense.inbound_nodes[1].outbound_layer, dense)
    self.assertListEqual(dense.inbound_nodes[0].input_tensors, [a])
    self.assertListEqual(dense.inbound_nodes[1].input_tensors, [b])

    # Test config
    config_0 = dense.inbound_nodes[0].get_config()
    self.assertEqual(config_0['outbound_layer'], dense.name)

  def testMultiInputNetwork(self):
    a = base_layers.Input(shape=(32,), name='input_a')
    b = base_layers.Input(shape=(32,), name='input_b')

    class AddLayer(base_layers.Layer):

      def call(self, inputs):
        assert len(inputs) == 2
        return inputs[0] + inputs[1]

    c = AddLayer()([a, b])  # pylint: disable=not-callable
    network = base_layers.Network([a, b], c)
    self.assertEqual(len(network.layers), 3)  # 2 * InputLayer + AddLayer

    # Test callability.
    a2 = base_layers.Input(shape=(32,))
    b2 = base_layers.Input(shape=(32,))
    c2 = network([a2, b2])
    self.assertEqual(c2.get_shape().as_list(), [None, 32])

  def testMultiOutputNetwork(self):
    x = base_layers.Input(shape=(32,))
    y1 = core_layers.Dense(2)(x)
    y2 = core_layers.Dense(3)(x)
    network = base_layers.Network(x, [y1, y2])

    self.assertEqual(len(network.layers), 3)  # InputLayer + 2 * Dense

    # Test callability.
    x2 = base_layers.Input(shape=(32,))
    outputs = network(x2)

    self.assertEqual(type(outputs), list)
    self.assertEqual(len(outputs), 2)
    self.assertEqual(outputs[0].get_shape().as_list(), [None, 2])
    self.assertEqual(outputs[1].get_shape().as_list(), [None, 3])

  def testMultiInputMultiOutputNetworkSharedLayer(self):
    a = base_layers.Input(shape=(32,), name='input_a')
    b = base_layers.Input(shape=(32,), name='input_b')

    dense = core_layers.Dense(2)

    y1 = dense(a)
    y2 = dense(b)
    network = base_layers.Network([a, b], [y1, y2])
    self.assertEqual(len(network.layers), 3)  # 2 * InputLayer + Dense

    # Test callability.
    a2 = base_layers.Input(shape=(32,))
    b2 = base_layers.Input(shape=(32,))
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
        return [inputs ** 2, inputs ** 3]

    x = base_layers.Input(shape=(32,))
    p1, p2 = PowersLayer()(x)  # pylint: disable=not-callable
    y1 = core_layers.Dense(2)(p1)
    y2 = core_layers.Dense(3)(p2)
    network = base_layers.Network(x, [y1, y2])

    self.assertEqual(len(network.layers), 4)  # InputLayer + 2 * Dense + PLayer

    # Test callability.
    x2 = base_layers.Input(shape=(32,))
    outputs = network(x2)

    self.assertEqual(type(outputs), list)
    self.assertEqual(len(outputs), 2)
    self.assertEqual(outputs[0].get_shape().as_list(), [None, 2])
    self.assertEqual(outputs[1].get_shape().as_list(), [None, 3])

  def testNetworkAttributes(self):
    x = base_layers.Input(shape=(32,))
    z = core_layers.Dense(2, kernel_regularizer=lambda x: 0.01 * (x ** 2))(x)
    dense = core_layers.Dense(2, name='dense')
    dense.add_update(1)
    y = dense(z)
    net = base_layers.Network(x, y)

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

    # _compute_output_shape
    self.assertEqual(net._compute_output_shape((3, 32)).as_list(), [3, 2])

  def testInvalidNetworks(self):
    # redundant inputs
    x = base_layers.Input(shape=(32,))
    y = core_layers.Dense(2)(x)
    with self.assertRaises(ValueError):
      base_layers.Network([x, x], y)

    # inputs that don't come from Input
    x = array_ops.placeholder(dtype='float32', shape=(None, 32))
    y = core_layers.Dense(2)(x)
    with self.assertRaises(ValueError):
      base_layers.Network(x, y)

    # inputs that don't come from Input but have a layer history
    x = base_layers.Input(shape=(32,))
    x = core_layers.Dense(32)(x)
    y = core_layers.Dense(2)(x)
    with self.assertRaises(ValueError):
      base_layers.Network(x, y)

    # outputs that don't come from layers
    x = base_layers.Input(shape=(32,))
    y = core_layers.Dense(2)(x)
    y = 2 * y
    with self.assertRaises(ValueError):
      base_layers.Network(x, y)

    # disconnected graphs
    x1 = base_layers.Input(shape=(32,))
    x2 = base_layers.Input(shape=(32,))
    y = core_layers.Dense(2)(x1)
    with self.assertRaises(ValueError):
      base_layers.Network(x2, y)

    # redundant layer names
    x = base_layers.Input(shape=(32,))
    z = core_layers.Dense(2, name='dense')(x)
    y = core_layers.Dense(2, name='dense')(z)
    with self.assertRaises(ValueError):
      base_layers.Network(x, y)

  def testInputTensorWrapping(self):
    x = array_ops.placeholder(dtype='float32', shape=(None, 32))
    x = base_layers.Input(tensor=x)
    y = core_layers.Dense(2)(x)
    base_layers.Network(x, y)

  def testExplicitBatchSize(self):
    x = base_layers.Input(shape=(32,), batch_size=3)
    y = core_layers.Dense(2)(x)
    self.assertEqual(y.get_shape().as_list(), [3, 2])

  def testNetworkRecursion(self):
    # test the ability of networks to be used as layers inside networks.
    a = base_layers.Input(shape=(32,))
    b = core_layers.Dense(2)(a)
    net = base_layers.Network(a, b)

    c = base_layers.Input(shape=(32,))
    d = net(c)

    recursive_net = base_layers.Network(c, d)
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

    x = base_layers.Input(shape=(32,), sparse=True)
    y = SparseSoftmax()(x)  # pylint: disable=not-callable
    network = base_layers.Network(x, y)

    self.assertEqual(len(network.layers), 2)
    self.assertEqual(network.layers[0].sparse, True)

  def testMaskingSingleInput(self):

    class MaskedLayer(base_layers.Layer):

      def call(self, inputs, mask=None):
        if mask is not None:
          return inputs * mask
        return inputs

      def compute_mask(self, inputs, mask=None):
        return array_ops.ones_like(inputs)

    x = base_layers.Input(shape=(32,))
    y = MaskedLayer()(x)  # pylint: disable=not-callable
    network = base_layers.Network(x, y)

    # test callability on Input
    x_2 = base_layers.Input(shape=(32,))
    y_2 = network(x_2)
    self.assertEqual(y_2.get_shape().as_list(), [None, 32])

    # test callability on regular tensor
    x_2 = array_ops.placeholder(dtype='float32', shape=(None, 32))
    y_2 = network(x_2)
    self.assertEqual(y_2.get_shape().as_list(), [None, 32])


if __name__ == '__main__':
  test.main()
