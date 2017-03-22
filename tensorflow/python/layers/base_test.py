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

from tensorflow.python.framework import ops
from tensorflow.python.layers import base as base_layers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


class BaseLayerTest(test.TestCase):

  def testLayerProperties(self):
    layer = base_layers._Layer(name='my_layer')
    self.assertListEqual(layer.variables, [])
    self.assertListEqual(layer.trainable_variables, [])
    self.assertListEqual(layer.non_trainable_variables, [])
    self.assertListEqual(layer.updates, [])
    self.assertListEqual(layer.losses, [])
    self.assertEqual(layer.built, False)
    with self.assertRaisesRegexp(ValueError, 'not been used yet'):
      _ = layer.name
    layer = base_layers._Layer(name='my_layer', trainable=False)
    self.assertEqual(layer.trainable, False)

  def testAddWeight(self):
    with self.test_session():
      layer = base_layers._Layer(name='my_layer')

      # Test basic variable creation.
      variable = layer._add_variable(
          'my_var', [2, 2], initializer=init_ops.zeros_initializer())
      self.assertEqual(variable.name, 'my_var:0')
      self.assertListEqual(layer.variables, [variable])
      self.assertListEqual(layer.trainable_variables, [variable])
      self.assertListEqual(layer.non_trainable_variables, [])
      self.assertListEqual(
          layer.variables,
          ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))

      # Test non-trainable variable creation.
      # layer._add_variable should work even outside `build` and `call`.
      variable_2 = layer._add_variable(
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
      variable = layer._add_variable(
          'reg_var', [2, 2],
          initializer=init_ops.zeros_initializer(),
          regularizer=regularizer)
      self.assertEqual(len(layer.losses), 1)

  def testGetVariable(self):
    with self.test_session():
      # From inside `build` and `call` it should be possible to use
      # either tf.get_variable

      class MyLayer(base_layers._Layer):

        def build(self, input_shape):
          self.my_var = variable_scope.get_variable(
              'my_var', [2, 2], initializer=init_ops.zeros_initializer())

        def call(self, inputs):
          variable_scope.get_variable(
              'my_call_var', [2, 2], initializer=init_ops.zeros_initializer())
          return inputs

      layer = MyLayer(name='my_layer')
      inputs = random_ops.random_uniform((5,), seed=1)
      layer.apply(inputs)
      layer.apply(inputs)
      self.assertListEqual([v.name for v in layer.variables],
                           ['my_layer/my_var:0', 'my_layer/my_call_var:0'])

      # Creating a layer with _reuse=True and _scope=None delays
      # selecting the variable scope until call.
      lazy_layer = MyLayer(name='lazy_layer', _reuse=True)
      with variable_scope.variable_scope('new_scope'):
        # This should attempt to reuse 'my_var' and 'my_call_var' in 'new_scope'
        with self.assertRaisesRegexp(
            ValueError, r'new_scope/my_var does not exist'):
          lazy_layer.apply(inputs)
        variable_scope.get_variable('my_var', [2, 2])
        with self.assertRaisesRegexp(
            ValueError, r'new_scope/my_call_var does not exist'):
          lazy_layer.apply(inputs)
        variable_scope.get_variable('my_call_var', [2, 2])
        # Smoke test: it runs.
        lazy_layer.apply(inputs)
        # The variables were created outside of the Layer, and
        # reuse=True, so the Layer does not own them and they are not
        # stored in its collection.
        self.assertListEqual(lazy_layer.variables, [])
        self.assertEqual(lazy_layer.name, 'new_scope')

  def testCall(self):

    class MyLayer(base_layers._Layer):

      def call(self, inputs):
        return math_ops.square(inputs)

    layer = MyLayer(name='my_layer')
    inputs = random_ops.random_uniform((5,), seed=1)
    outputs = layer.apply(inputs)
    self.assertEqual(layer.built, True)
    self.assertEqual(outputs.op.name, 'my_layer/Square')

  def testNaming(self):

    class PrivateLayer(base_layers._Layer):

      def call(self, inputs):
        return None

    inputs = random_ops.random_uniform((5,))
    default_layer = PrivateLayer()
    _ = default_layer.apply(inputs)
    self.assertEqual(default_layer.name, 'private_layer')
    default_layer1 = PrivateLayer()
    default_layer1.apply(inputs)
    self.assertEqual(default_layer1.name, 'private_layer_1')
    my_layer = PrivateLayer(name='my_layer')
    my_layer.apply(inputs)
    self.assertEqual(my_layer.name, 'my_layer')
    my_layer1 = PrivateLayer(name='my_layer')
    my_layer1.apply(inputs)
    self.assertEqual(my_layer1.name, 'my_layer_1')
    # New graph has fully orthogonal names.
    with ops.Graph().as_default():
      my_layer_other_graph = PrivateLayer(name='my_layer')
      my_layer_other_graph.apply(inputs)
      self.assertEqual(my_layer_other_graph.name, 'my_layer')
    my_layer2 = PrivateLayer(name='my_layer')
    my_layer2.apply(inputs)
    self.assertEqual(my_layer2.name, 'my_layer_2')
    # Name scope shouldn't affect names.
    with ops.name_scope('some_name_scope'):
      default_layer2 = PrivateLayer()
      default_layer2.apply(inputs)
      self.assertEqual(default_layer2.name, 'private_layer_2')
      my_layer3 = PrivateLayer(name='my_layer')
      my_layer3.apply(inputs)
      self.assertEqual(my_layer3.name, 'my_layer_3')
      other_layer = PrivateLayer(name='other_layer')
      other_layer.apply(inputs)
      self.assertEqual(other_layer.name, 'other_layer')
    # Variable scope gets added to names.
    with variable_scope.variable_scope('var_scope'):
      default_layer_scoped = PrivateLayer()
      default_layer_scoped.apply(inputs)
      self.assertEqual(default_layer_scoped.name, 'var_scope/private_layer')
      my_layer_scoped = PrivateLayer(name='my_layer')
      my_layer_scoped.apply(inputs)
      self.assertEqual(my_layer_scoped.name, 'var_scope/my_layer')
      my_layer_scoped1 = PrivateLayer(name='my_layer')
      my_layer_scoped1.apply(inputs)
      self.assertEqual(my_layer_scoped1.name, 'var_scope/my_layer_1')


if __name__ == '__main__':
  test.main()
