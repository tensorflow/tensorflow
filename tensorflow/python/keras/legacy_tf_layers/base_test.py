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

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import combinations
from tensorflow.python.keras.engine import base_layer as keras_base_layer
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.legacy_tf_layers import base as base_layers
from tensorflow.python.keras.legacy_tf_layers import core as core_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


class BaseLayerTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testLayerProperties(self):
    layer = base_layers.Layer(name='my_layer')
    self.assertEqual(layer.variables, [])
    self.assertEqual(layer.trainable_variables, [])
    self.assertEqual(layer.non_trainable_variables, [])
    if not context.executing_eagerly():
      # updates, losses only supported in GRAPH mode
      self.assertEqual(layer.updates, [])
      self.assertEqual(layer.losses, [])
    self.assertEqual(layer.built, False)
    layer = base_layers.Layer(name='my_layer', trainable=False)
    self.assertEqual(layer.trainable, False)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInt64Layer(self):
    layer = base_layers.Layer(name='my_layer', dtype='int64')
    layer.add_variable('my_var', [2, 2])
    self.assertEqual(layer.name, 'my_layer')

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testKerasStyleAddWeight(self):
    keras_layer = keras_base_layer.Layer(name='keras_layer')
    with ops.name_scope('foo', skip_on_eager=False):
      keras_variable = keras_layer.add_variable(
          'my_var', [2, 2], initializer=init_ops.zeros_initializer())
    self.assertEqual(keras_variable.name, 'foo/my_var:0')

    with ops.name_scope('baz', skip_on_eager=False):
      old_style_layer = base_layers.Layer(name='my_layer')
      # Test basic variable creation.
      variable = old_style_layer.add_variable(
          'my_var', [2, 2], initializer=init_ops.zeros_initializer())
    self.assertEqual(variable.name, 'my_layer/my_var:0')

    with base_layers.keras_style_scope():
      layer = base_layers.Layer(name='my_layer')
    # Test basic variable creation.
    with ops.name_scope('bar', skip_on_eager=False):
      variable = layer.add_variable(
          'my_var', [2, 2], initializer=init_ops.zeros_initializer())
    self.assertEqual(variable.name, 'bar/my_var:0')

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testAddWeight(self):
    layer = base_layers.Layer(name='my_layer')

    # Test basic variable creation.
    variable = layer.add_variable(
        'my_var', [2, 2], initializer=init_ops.zeros_initializer())
    self.assertEqual(variable.name, 'my_layer/my_var:0')
    self.assertEqual(layer.variables, [variable])
    self.assertEqual(layer.trainable_variables, [variable])
    self.assertEqual(layer.non_trainable_variables, [])
    if not context.executing_eagerly():
      self.assertEqual(
          layer.variables,
          ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))

    # Test non-trainable variable creation.
    # layer.add_variable should work even outside `build` and `call`.
    variable_2 = layer.add_variable(
        'non_trainable_var', [2, 2],
        initializer=init_ops.zeros_initializer(),
        trainable=False)
    self.assertEqual(layer.variables, [variable, variable_2])
    self.assertEqual(layer.trainable_variables, [variable])
    self.assertEqual(layer.non_trainable_variables, [variable_2])

    if not context.executing_eagerly():
      self.assertEqual(
          len(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)), 1)

    regularizer = lambda x: math_ops.reduce_sum(x) * 1e-3
    _ = layer.add_variable(
        'reg_var', [2, 2],
        initializer=init_ops.zeros_initializer(),
        regularizer=regularizer)
    self.assertEqual(len(layer.losses), 1)

    added_variable = [False]

    # Test that sync `ON_READ` variables are defaulted to be non-trainable.
    variable_3 = layer.add_variable(
        'sync_on_read_var', [2, 2],
        initializer=init_ops.zeros_initializer(),
        synchronization=variable_scope.VariableSynchronization.ON_READ,
        aggregation=variable_scope.VariableAggregation.SUM)
    self.assertEqual(layer.non_trainable_variables, [variable_2, variable_3])

    @def_function.function
    def function_adds_weight():
      if not added_variable[0]:
        layer.add_variable(
            'reg_var_from_function', [2, 2],
            initializer=init_ops.zeros_initializer(),
            regularizer=regularizer)
        added_variable[0] = True

    function_adds_weight()
    self.assertEqual(len(layer.losses), 2)

  def testInvalidTrainableSynchronizationCombination(self):
    layer = base_layers.Layer(name='my_layer')

    with self.assertRaisesRegexp(
        ValueError, 'Synchronization value can be set to '
        'VariableSynchronization.ON_READ only for non-trainable variables. '
        'You have specified trainable=True and '
        'synchronization=VariableSynchronization.ON_READ.'):
      _ = layer.add_variable(
          'v', [2, 2],
          initializer=init_ops.zeros_initializer(),
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          trainable=True)

  @test_util.run_deprecated_v1
  def testReusePartitionedVariablesAndRegularizers(self):
    regularizer = lambda x: math_ops.reduce_sum(x) * 1e-3
    partitioner = partitioned_variables.fixed_size_partitioner(3)
    for reuse in [False, True]:
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         partitioner=partitioner,
                                         reuse=reuse):
        layer = base_layers.Layer(name='my_layer')
        _ = layer.add_variable(
            'reg_part_var', [4, 4],
            initializer=init_ops.zeros_initializer(),
            regularizer=regularizer)
    self.assertEqual(
        len(ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)), 3)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testCall(self):

    class MyLayer(base_layers.Layer):

      def call(self, inputs):
        return math_ops.square(inputs)

    layer = MyLayer(name='my_layer')
    inputs = random_ops.random_uniform((5,), seed=1)
    outputs = layer.apply(inputs)
    self.assertEqual(layer.built, True)
    if not context.executing_eagerly():
      # op is only supported in GRAPH mode
      self.assertEqual(outputs.op.name, 'my_layer/Square')

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testDeepCopy(self):

    class MyLayer(base_layers.Layer):

      def call(self, inputs):
        return math_ops.square(inputs)

    layer = MyLayer(name='my_layer')
    layer._private_tensor = random_ops.random_uniform(())
    inputs = random_ops.random_uniform((5,), seed=1)
    outputs = layer.apply(inputs)
    self.assertEqual(layer.built, True)
    if not context.executing_eagerly():
      # op only supported in GRAPH mode.
      self.assertEqual(outputs.op.name, 'my_layer/Square')

    layer_copy = copy.deepcopy(layer)
    self.assertEqual(layer_copy.name, layer.name)
    self.assertEqual(layer_copy._scope.name, layer._scope.name)
    self.assertEqual(layer_copy._private_tensor, layer._private_tensor)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
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

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInputSpecNdimCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = input_spec.InputSpec(ndim=2)

      def call(self, inputs):
        return inputs

    if not context.executing_eagerly():
      layer = CustomerLayer()
      with self.assertRaisesRegexp(ValueError, r'requires a defined rank'):
        layer.apply(array_ops.placeholder('int32'))

    layer = CustomerLayer()
    with self.assertRaisesRegexp(ValueError, r'expected ndim=2'):
      layer.apply(constant_op.constant([1]))

    # Note that we re-create the layer since in Eager mode, input spec checks
    # only happen on first call.
    # Works
    layer = CustomerLayer()
    layer.apply(constant_op.constant([[1], [2]]))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInputSpecMinNdimCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = input_spec.InputSpec(min_ndim=2)

      def call(self, inputs):
        return inputs

    if not context.executing_eagerly():
      layer = CustomerLayer()
      with self.assertRaisesRegexp(ValueError, r'requires a defined rank'):
        layer.apply(array_ops.placeholder('int32'))

    layer = CustomerLayer()
    with self.assertRaisesRegexp(ValueError, r'expected min_ndim=2'):
      layer.apply(constant_op.constant([1]))

    # Works
    layer = CustomerLayer()
    layer.apply(constant_op.constant([[1], [2]]))

    layer = CustomerLayer()
    layer.apply(constant_op.constant([[[1], [2]]]))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInputSpecMaxNdimCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = input_spec.InputSpec(max_ndim=2)

      def call(self, inputs):
        return inputs

    if not context.executing_eagerly():
      layer = CustomerLayer()
      with self.assertRaisesRegexp(ValueError, r'requires a defined rank'):
        layer.apply(array_ops.placeholder('int32'))

    layer = CustomerLayer()
    with self.assertRaisesRegexp(ValueError, r'expected max_ndim=2'):
      layer.apply(constant_op.constant([[[1], [2]]]))

    # Works
    layer = CustomerLayer()
    layer.apply(constant_op.constant([1]))

    layer = CustomerLayer()
    layer.apply(constant_op.constant([[1], [2]]))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInputSpecDtypeCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = input_spec.InputSpec(dtype='float32')

      def call(self, inputs):
        return inputs

    layer = CustomerLayer()
    with self.assertRaisesRegexp(ValueError, r'expected dtype=float32'):
      layer.apply(constant_op.constant(1, dtype=dtypes.int32))

    # Works
    layer = CustomerLayer()
    layer.apply(constant_op.constant(1.0, dtype=dtypes.float32))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInputSpecAxesCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = input_spec.InputSpec(axes={-1: 2})

      def call(self, inputs):
        return inputs

    layer = CustomerLayer()
    with self.assertRaisesRegexp(ValueError, r'expected axis'):
      layer.apply(constant_op.constant([1, 2, 3]))

    # Works
    layer = CustomerLayer()
    layer.apply(constant_op.constant([1, 2]))
    layer = CustomerLayer()
    layer.apply(constant_op.constant([[1, 2], [3, 4], [5, 6]]))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInputSpecShapeCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = input_spec.InputSpec(shape=(None, 3))

      def call(self, inputs):
        return inputs

    layer = CustomerLayer()
    with self.assertRaisesRegexp(ValueError, r'expected shape'):
      layer.apply(constant_op.constant([[1, 2]]))

    # Works
    layer = CustomerLayer()
    layer.apply(constant_op.constant([[1, 2, 3], [4, 5, 6]]))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testNoInputSpec(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = None

      def call(self, inputs):
        return inputs

    layer = CustomerLayer()

    layer.apply(constant_op.constant(1))

    # Works
    if not context.executing_eagerly():
      layer.apply(array_ops.placeholder('int32'))
      layer.apply(array_ops.placeholder('int32', shape=(2, 3)))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_count_params(self):
    dense = core_layers.Dense(16)
    dense.build((None, 4))
    self.assertEqual(dense.count_params(), 16 * 4 + 16)

    dense = core_layers.Dense(16)
    with self.assertRaises(ValueError):
      dense.count_params()

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testDictInputOutput(self):

    class DictLayer(base_layers.Layer):

      def call(self, inputs):
        return {'l' + key: inputs[key] for key in inputs}

    layer = DictLayer()
    if context.executing_eagerly():
      i1 = constant_op.constant(3)
      i2 = constant_op.constant(4.0)
      result = layer.apply({'abel': i1, 'ogits': i2})
      self.assertTrue(isinstance(result, dict))
      self.assertEqual(set(['label', 'logits']), set(result.keys()))
      self.assertEqual(3, result['label'].numpy())
      self.assertEqual(4.0, result['logits'].numpy())
    else:
      i1 = array_ops.placeholder('int32')
      i2 = array_ops.placeholder('float32')
      result = layer.apply({'abel': i1, 'ogits': i2})
      self.assertTrue(isinstance(result, dict))
      self.assertEqual(set(['label', 'logits']), set(result.keys()))

  @test_util.run_deprecated_v1
  def testActivityRegularizer(self):
    regularizer = math_ops.reduce_sum
    layer = base_layers.Layer(activity_regularizer=regularizer)
    x = array_ops.placeholder('int32')
    layer.apply(x)
    self.assertEqual(len(layer.get_losses_for(x)), 1)

  def testNameScopeIsConsistentWithVariableScope(self):
    # Github issue 13429.

    class MyLayer(base_layers.Layer):

      def build(self, input_shape):
        self.my_var = self.add_variable('my_var', (), dtypes.float32)
        self.built = True

      def call(self, inputs):
        return math_ops.multiply(inputs, self.my_var, name='my_op')

    def _gen_layer(x, name=None):
      layer = MyLayer(name=name)
      out = layer.apply(x)
      return layer, out

    # unnamed layer
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32, (), 'x')
      layer, op = _gen_layer(x)
      layer1, op1 = _gen_layer(op)
      layer2, op2 = _gen_layer(op1)

      self.assertEqual(layer.my_var.name, 'my_layer/my_var:0')
      self.assertEqual(op.name, 'my_layer/my_op:0')
      self.assertEqual(layer1.my_var.name, 'my_layer_1/my_var:0')
      self.assertEqual(op1.name, 'my_layer_1/my_op:0')
      self.assertEqual(layer2.my_var.name, 'my_layer_2/my_var:0')
      self.assertEqual(op2.name, 'my_layer_2/my_op:0')
    # name starts from zero
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32, (), 'x')
      layer, op = _gen_layer(x, name='name')
      layer1, op1 = _gen_layer(op, name='name_1')
      layer2, op2 = _gen_layer(op1, name='name_2')

      self.assertEqual(layer.my_var.name, 'name/my_var:0')
      self.assertEqual(op.name, 'name/my_op:0')
      self.assertEqual(layer1.my_var.name, 'name_1/my_var:0')
      self.assertEqual(op1.name, 'name_1/my_op:0')
      self.assertEqual(layer2.my_var.name, 'name_2/my_var:0')
      self.assertEqual(op2.name, 'name_2/my_op:0')
    # name starts from one
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32, (), 'x')
      layer, op = _gen_layer(x, name='name_1')
      layer1, op1 = _gen_layer(op, name='name_2')
      layer2, op2 = _gen_layer(op1, name='name_3')

      self.assertEqual(layer.my_var.name, 'name_1/my_var:0')
      self.assertEqual(op.name, 'name_1/my_op:0')
      self.assertEqual(layer1.my_var.name, 'name_2/my_var:0')
      self.assertEqual(op1.name, 'name_2/my_op:0')
      self.assertEqual(layer2.my_var.name, 'name_3/my_var:0')
      self.assertEqual(op2.name, 'name_3/my_op:0')

  def testVariablesAreLiftedFromFunctionBuildingGraphs(self):
    class MyLayer(base_layers.Layer):

      def build(self, input_shape):
        self.my_var = self.add_variable('my_var', (), dtypes.float32)
        self.built = True

      def call(self, inputs):
        return inputs

    outer_graph = ops.get_default_graph()
    function_building_graph = ops.Graph()
    function_building_graph._building_function = True
    with outer_graph.as_default():
      with function_building_graph.as_default():
        layer = MyLayer()
        # Create a variable by invoking build through __call__ and assert that
        # it is both tracked and lifted into the outer graph.
        inputs = array_ops.placeholder(dtypes.float32, (), 'inputs')
        layer.apply(inputs)
        self.assertEqual(len(layer.variables), 1)
        self.assertEqual(len(layer.trainable_variables), 1)
        self.assertEqual(layer.variables[0].graph, outer_graph)

  @test_util.run_deprecated_v1
  def testGetUpdateFor(self):

    class MyLayer(base_layers.Layer):

      def build(self, input_shape):
        self.a = self.add_variable('a',
                                   (),
                                   dtypes.float32,
                                   trainable=False)
        self.b = self.add_variable('b',
                                   (),
                                   dtypes.float32,
                                   trainable=False)
        self.add_update(state_ops.assign_add(self.a, 1., name='b_update'))
        self.built = True

      def call(self, inputs):
        self.add_update(state_ops.assign_add(self.a, inputs, name='a_update'),
                        inputs=True)
        return inputs + 1

    layer = MyLayer()
    inputs = array_ops.placeholder(dtypes.float32, (), 'inputs')
    intermediate_inputs = inputs + 1
    outputs = layer.apply(intermediate_inputs)

    self.assertEqual(len(layer.updates), 2)
    self.assertEqual(len(layer.get_updates_for(None)), 1)
    self.assertEqual(len(layer.get_updates_for([inputs])), 1)
    self.assertEqual(len(layer.get_updates_for([intermediate_inputs])), 1)
    self.assertEqual(len(layer.get_updates_for([outputs])), 0)

    # Call same layer on new input, creating one more conditional update
    inputs = array_ops.placeholder(dtypes.float32, (), 'inputs')
    intermediate_inputs = inputs + 1
    outputs = layer.apply(intermediate_inputs)

    self.assertEqual(len(layer.updates), 3)
    self.assertEqual(len(layer.get_updates_for(None)), 1)
    # Check that we are successfully filtering out irrelevant updates
    self.assertEqual(len(layer.get_updates_for([inputs])), 1)
    self.assertEqual(len(layer.get_updates_for([intermediate_inputs])), 1)
    self.assertEqual(len(layer.get_updates_for([outputs])), 0)

  @test_util.run_deprecated_v1
  def testGetLossesFor(self):

    class MyLayer(base_layers.Layer):

      def build(self, input_shape):
        self.a = self.add_variable('a',
                                   (),
                                   dtypes.float32,
                                   trainable=False)
        self.b = self.add_variable('b',
                                   (),
                                   dtypes.float32,
                                   trainable=False)
        self.add_loss(self.a)
        self.built = True

      def call(self, inputs):
        self.add_loss(inputs, inputs=True)
        return inputs + 1

    layer = MyLayer()
    inputs = array_ops.placeholder(dtypes.float32, (), 'inputs')
    intermediate_inputs = inputs + 1
    outputs = layer.apply(intermediate_inputs)

    self.assertEqual(len(layer.losses), 2)
    self.assertEqual(len(layer.get_losses_for(None)), 1)
    self.assertEqual(len(layer.get_losses_for([inputs])), 1)
    self.assertEqual(len(layer.get_losses_for([intermediate_inputs])), 1)
    self.assertEqual(len(layer.get_losses_for([outputs])), 0)

    # Call same layer on new input, creating one more conditional loss
    inputs = array_ops.placeholder(dtypes.float32, (), 'inputs')
    intermediate_inputs = inputs + 1
    outputs = layer.apply(intermediate_inputs)

    self.assertEqual(len(layer.losses), 3)
    self.assertEqual(len(layer.get_losses_for(None)), 1)
    # Check that we are successfully filtering out irrelevant losses
    self.assertEqual(len(layer.get_losses_for([inputs])), 1)
    self.assertEqual(len(layer.get_losses_for([intermediate_inputs])), 1)
    self.assertEqual(len(layer.get_losses_for([outputs])), 0)


class IdentityLayer(base_layers.Layer):
  """A layer returns the identity of it's input."""

  def call(self, inputs):
    return inputs


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class DTypeTest(test.TestCase, parameterized.TestCase):

  def _const(self, dtype):
    return array_ops.constant(1, dtype=dtype)

  def test_dtype_inferred_from_input(self):
    # Test with Tensor input
    layer = IdentityLayer()
    self.assertIsNone(layer.dtype)
    layer(self._const('float64'))
    self.assertEqual(layer.dtype, 'float64')

    # Test with Numpy input
    layer = IdentityLayer()
    self.assertIsNone(layer.dtype)
    layer(np.array(1., dtype='float64'))
    self.assertEqual(layer.dtype, 'float64')

    # Test with integer input
    layer = IdentityLayer()
    self.assertIsNone(layer.dtype)
    layer(self._const('int32'))
    self.assertEqual(layer.dtype, 'int32')

    # Test layer dtype doesn't change when passed a new dtype
    layer = IdentityLayer()
    self.assertIsNone(layer.dtype)
    layer(self._const('float64'))
    self.assertEqual(layer.dtype, 'float64')
    layer(self._const('float16'))
    self.assertEqual(layer.dtype, 'float64')

    # Test layer dtype inferred from first input
    layer = IdentityLayer()
    layer([self._const('float32'), self._const('float64')])
    self.assertEqual(layer.dtype, 'float32')

  def test_passing_dtype_to_constructor(self):
    layer = IdentityLayer(dtype='float64')
    layer(self._const('float32'))
    self.assertEqual(layer.dtype, 'float64')

    layer = IdentityLayer(dtype='int32')
    layer(self._const('float32'))
    self.assertEqual(layer.dtype, 'int32')

    layer = IdentityLayer(dtype=dtypes.float64)
    layer(self._const('float32'))
    self.assertEqual(layer.dtype, 'float64')

  def test_inputs_not_casted(self):
    layer = IdentityLayer(dtype='float32')
    self.assertEqual(layer(self._const('float64')).dtype, 'float64')


if __name__ == '__main__':
  test.main()
