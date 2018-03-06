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

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import base as base_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


class BaseLayerTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testLayerProperties(self):
    layer = base_layers.Layer(name='my_layer')
    self.assertEqual(layer.variables, [])
    self.assertEqual(layer.trainable_variables, [])
    self.assertEqual(layer.non_trainable_variables, [])
    if context.in_graph_mode():
      # updates, losses only supported in GRAPH mode
      self.assertEqual(layer.updates, [])
      self.assertEqual(layer.losses, [])
    self.assertEqual(layer.built, False)
    layer = base_layers.Layer(name='my_layer', trainable=False)
    self.assertEqual(layer.trainable, False)

  @test_util.run_in_graph_and_eager_modes()
  def testAddWeight(self):
    layer = base_layers.Layer(name='my_layer')

    # Test basic variable creation.
    variable = layer.add_variable(
        'my_var', [2, 2], initializer=init_ops.zeros_initializer())
    self.assertEqual(variable.name, 'my_layer/my_var:0')
    self.assertEqual(layer.variables, [variable])
    self.assertEqual(layer.trainable_variables, [variable])
    self.assertEqual(layer.non_trainable_variables, [])
    if context.in_graph_mode():
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
    if context.in_graph_mode():
      self.assertEqual(
          len(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)), 1)

      # regularizers only supported in GRAPH mode.
      regularizer = lambda x: math_ops.reduce_sum(x) * 1e-3
      variable = layer.add_variable(
          'reg_var', [2, 2],
          initializer=init_ops.zeros_initializer(),
          regularizer=regularizer)
      self.assertEqual(len(layer.losses), 1)

  def testNoEagerActivityRegularizer(self):
    with context.eager_mode():
      with self.assertRaisesRegexp(ValueError, 'activity_regularizer'):
        core_layers.Dense(1, activity_regularizer=lambda *args, **kwargs: 0.)

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
      self.assertEqual([v.name for v in layer.variables],
                       ['my_layer/my_var:0'])

      # Creating a layer with no scope leads to lazy construction of
      # the scope at apply() time.  It uses scope "<current scope>/base_name"
      lazy_layer = MyLayer(_reuse=True)
      with variable_scope.variable_scope('new_scope'):
        with variable_scope.variable_scope('my_layer'):
          variable_scope.get_variable('my_var', [2, 2])

        # Smoke test: it runs.
        lazy_layer.apply(inputs)
        # The variables were created outside of the Layer, and
        # reuse=True, so the Layer does not own them and they are not
        # stored in its collection.
        self.assertEqual(lazy_layer.variables, [])
        self.assertEqual(lazy_layer._scope.name, 'new_scope/my_layer')

      # Creating a layer with no scope leads to lazy construction of
      # the scope at apply() time. If 'scope' argument is passed to
      # apply(), it uses that scope when accessing variables.
      lazy_layer = MyLayer(_reuse=True)
      with variable_scope.variable_scope('new_scope') as new_scope:
        variable_scope.get_variable('my_var', [2, 2])

        # Smoke test: it runs.
        lazy_layer.apply(inputs, scope=new_scope)
        # The variables were created outside of the Layer, and
        # reuse=True, so the Layer does not own them and they are not
        # stored in its collection.
        self.assertEqual(lazy_layer.variables, [])
        self.assertEqual(lazy_layer._scope.name, 'new_scope')

      # Checking for graph equality is only done in GRAPH mode.
      with ops.Graph().as_default():
        inputs_ng = random_ops.random_uniform((5,), seed=1)
        with self.assertRaisesRegexp(ValueError, r'graph are not the same'):
          layer.apply(inputs_ng)

  @test_util.run_in_graph_and_eager_modes()
  def testCall(self):

    class MyLayer(base_layers.Layer):

      def call(self, inputs):
        return math_ops.square(inputs)

    layer = MyLayer(name='my_layer')
    inputs = random_ops.random_uniform((5,), seed=1)
    outputs = layer.apply(inputs)
    self.assertEqual(layer.built, True)
    if context.in_graph_mode():
      # op is only supported in GRAPH mode
      self.assertEqual(outputs.op.name, 'my_layer/Square')

  def testFirstCallCanCreateVariablesButSecondCanNotWhenBuildEmpty(self):
    # Note that this test is only run in Graph mode since with EAGER mode we can
    # still create a new variable on second call.

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
    self.assertEqual([v.name
                      for v in layer.variables], ['my_layer/my_var:0'])
    with self.assertRaisesRegexp(ValueError,
                                 'my_layer/this_will_break_on_second_call'):
      layer.apply(inputs)
    # The list of variables hasn't changed.
    self.assertEqual([v.name
                      for v in layer.variables], ['my_layer/my_var:0'])

  @test_util.run_in_graph_and_eager_modes()
  def testDeepCopy(self):

    class MyLayer(base_layers.Layer):

      def call(self, inputs):
        return math_ops.square(inputs)

    layer = MyLayer(name='my_layer')
    layer._private_tensor = random_ops.random_uniform(())
    inputs = random_ops.random_uniform((5,), seed=1)
    outputs = layer.apply(inputs)
    self.assertEqual(layer.built, True)
    if context.in_graph_mode():
      # op only supported in GRAPH mode.
      self.assertEqual(outputs.op.name, 'my_layer/Square')

    layer_copy = copy.deepcopy(layer)
    self.assertEqual(layer_copy.name, layer.name)
    self.assertEqual(layer_copy._scope.name, layer._scope.name)
    self.assertEqual(layer_copy._graph, layer._graph)
    self.assertEqual(layer_copy._private_tensor, layer._private_tensor)

  @test_util.run_in_graph_and_eager_modes()
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

  @test_util.run_in_graph_and_eager_modes()
  def testInputSpecNdimCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = base_layers.InputSpec(ndim=2)

      def call(self, inputs):
        return inputs

    if context.in_graph_mode():
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

  @test_util.run_in_graph_and_eager_modes()
  def testInputSpecMinNdimCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = base_layers.InputSpec(min_ndim=2)

      def call(self, inputs):
        return inputs

    if context.in_graph_mode():
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

  @test_util.run_in_graph_and_eager_modes()
  def testInputSpecMaxNdimCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = base_layers.InputSpec(max_ndim=2)

      def call(self, inputs):
        return inputs

    if context.in_graph_mode():
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

  @test_util.run_in_graph_and_eager_modes()
  def testInputSpecDtypeCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = base_layers.InputSpec(dtype='float32')

      def call(self, inputs):
        return inputs

    layer = CustomerLayer()
    with self.assertRaisesRegexp(ValueError, r'expected dtype=float32'):
      layer.apply(constant_op.constant(1, dtype=dtypes.int32))

    # Works
    layer = CustomerLayer()
    layer.apply(constant_op.constant(1.0, dtype=dtypes.float32))

  @test_util.run_in_graph_and_eager_modes()
  def testInputSpecAxesCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = base_layers.InputSpec(axes={-1: 2})

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

  @test_util.run_in_graph_and_eager_modes()
  def testInputSpecShapeCheck(self):

    class CustomerLayer(base_layers.Layer):

      def __init__(self):
        super(CustomerLayer, self).__init__()
        self.input_spec = base_layers.InputSpec(shape=(None, 3))

      def call(self, inputs):
        return inputs

    layer = CustomerLayer()
    with self.assertRaisesRegexp(ValueError, r'expected shape'):
      layer.apply(constant_op.constant([[1, 2]]))

    # Works
    layer = CustomerLayer()
    layer.apply(constant_op.constant([[1, 2, 3], [4, 5, 6]]))

  @test_util.run_in_graph_and_eager_modes()
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
    if context.in_graph_mode():
      layer.apply(array_ops.placeholder('int32'))
      layer.apply(array_ops.placeholder('int32', shape=(2, 3)))

  @test_util.run_in_graph_and_eager_modes()
  def test_count_params(self):
    dense = core_layers.Dense(16)
    dense.build((None, 4))
    self.assertEqual(dense.count_params(), 16 * 4 + 16)

    dense = core_layers.Dense(16)
    with self.assertRaises(ValueError):
      dense.count_params()

  @test_util.run_in_graph_and_eager_modes()
  def testDictInputOutput(self):

    class DictLayer(base_layers.Layer):

      def call(self, inputs):
        return {'l' + key: inputs[key] for key in inputs}

    layer = DictLayer()
    if context.in_graph_mode():
      i1 = array_ops.placeholder('int32')
      i2 = array_ops.placeholder('float32')
      result = layer.apply({'abel': i1, 'ogits': i2})
      self.assertTrue(isinstance(result, dict))
      self.assertEqual(set(['label', 'logits']), set(result.keys()))
    else:
      i1 = constant_op.constant(3)
      i2 = constant_op.constant(4.0)
      result = layer.apply({'abel': i1, 'ogits': i2})
      self.assertTrue(isinstance(result, dict))
      self.assertEqual(set(['label', 'logits']), set(result.keys()))
      self.assertEqual(3, result['label'].numpy())
      self.assertEqual(4.0, result['logits'].numpy())

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

  def testLayerGraphSetInFirstApply(self):
    with ops.Graph().as_default():
      layer = core_layers.Dense(1)  # Graph at construction time is ignored
    with ops.Graph().as_default():
      layer.apply(constant_op.constant([[1]]))
      # layer is now bound to second Graph
    with ops.Graph().as_default(), self.assertRaisesRegexp(
        ValueError, 'Input graph and Layer graph are not the same'):
      layer.apply(constant_op.constant([[1]]))


if __name__ == '__main__':
  test.main()
