# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TensorFlow 2.0 layer behavior."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools as it
import sys
import traceback
from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class DynamicLayer1(base_layer.Layer):

  def __init__(self, dynamic=False, **kwargs):
    super(DynamicLayer1, self).__init__(dynamic=dynamic, **kwargs)

  def call(self, inputs):
    if math_ops.reduce_sum(inputs) > 0:
      return math_ops.sqrt(inputs)
    else:
      return math_ops.square(inputs)

  def compute_output_shape(self, input_shape):
    return input_shape


class DynamicLayer2(base_layer.Layer):

  def __init__(self, dynamic=False, **kwargs):
    super(DynamicLayer2, self).__init__(dynamic=dynamic, **kwargs)

  def call(self, inputs):
    samples = []
    for sample in inputs:
      samples.append(math_ops.square(sample))
    return array_ops.stack(samples, axis=0)

  def compute_output_shape(self, input_shape):
    return input_shape


class InvalidLayer(base_layer.Layer):

  def call(self, inputs):
    raise ValueError('You did something wrong!')


class BaseLayerTest(keras_parameterized.TestCase):

  @parameterized.parameters(DynamicLayer1, DynamicLayer2)
  def test_dynamic_layer_in_functional_model_in_graph_mode(self, layer_class):
    with context.graph_mode():
      inputs = keras.Input((3,))
      # Works when `dynamic=True` is declared.
      outputs = layer_class(dynamic=True)(inputs)
      model = keras.Model(inputs, outputs)
      self.assertEqual(model.dynamic, True)
      # But then you cannot run the model since you're in a graph scope.
      with self.assertRaisesRegexp(
          ValueError, 'You must enable eager execution'):
        model.compile(rmsprop.RMSprop(0.001), loss='mse')

      # Fails when `dynamic=True` not declared.
      with self.assertRaisesRegexp(
          TypeError, 'attempting to use Python control flow'):
        _ = layer_class()(inputs)

  @parameterized.parameters(DynamicLayer1, DynamicLayer2)
  def test_dynamic_layer_in_functional_model_in_eager_mode(self, layer_class):
    inputs = keras.Input((3,))
    # Fails when `dynamic=True` not declared.
    with self.assertRaisesRegexp(
        TypeError, 'attempting to use Python control flow'):
      _ = layer_class()(inputs)
    # Works when `dynamic=True` is declared.
    outputs = layer_class(dynamic=True)(inputs)
    model = keras.Model(inputs, outputs)
    self.assertEqual(model.dynamic, True)
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    self.assertEqual(model.run_eagerly, True)
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

  def test_nested_dynamic_layers_in_eager_mode(self):
    inputs = keras.Input((3,))
    outputs = DynamicLayer1(dynamic=True)(inputs)
    inner_model = keras.Model(inputs, outputs)
    self.assertEqual(inner_model.dynamic, True)

    inputs = keras.Input((3,))
    x = DynamicLayer2(dynamic=True)(inputs)
    outputs = inner_model(x)

    model = keras.Model(inputs, outputs)
    self.assertEqual(model.dynamic, True)
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    self.assertEqual(model.run_eagerly, True)
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

  def test_dynamic_layers_in_sequential_model(self):
    # Without input_shape argument
    model = keras.Sequential([DynamicLayer1(dynamic=True),
                              keras.layers.Dense(3),
                              DynamicLayer2(dynamic=True)])
    self.assertEqual(model.dynamic, True)
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    self.assertEqual(model.run_eagerly, True)
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

    # With input_shape argument
    model = keras.Sequential([DynamicLayer1(dynamic=True, input_shape=(3,)),
                              DynamicLayer2(dynamic=True)])
    self.assertEqual(model.dynamic, True)
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    self.assertEqual(model.run_eagerly, True)
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

  def test_dynamic_layers_in_subclassed_model(self):

    class MyModel(keras.Model):

      def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = DynamicLayer1(dynamic=True)

      def call(self, inputs):
        return self.layer1(inputs)

    model = MyModel()
    self.assertEqual(model.dynamic, True)
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    self.assertEqual(model.run_eagerly, True)
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

  def test_dynamic_subclassed_model_no_shape_inference(self):

    class MyModel(keras.Model):

      def __init__(self):
        super(MyModel, self).__init__(dynamic=True)
        self.layer1 = keras.layers.Dense(3)
        self.layer2 = keras.layers.Dense(3)

      def call(self, inputs):
        if math_ops.reduce_sum(inputs) > 0:
          return self.layer1(inputs)
        else:
          return self.layer2(inputs)

    model = MyModel()
    self.assertEqual(model.dynamic, True)
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    self.assertEqual(model.run_eagerly, True)
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))
    self.assertEqual(model.outputs, [None])

  def test_dynamic_subclassed_model_with_shape_inference(self):

    class MyModel(keras.Model):

      def __init__(self):
        super(MyModel, self).__init__(dynamic=True)
        self.layer1 = keras.layers.Dense(3)
        self.layer2 = keras.layers.Dense(3)

      def call(self, inputs):
        if math_ops.reduce_sum(inputs) > 0:
          return self.layer1(inputs)
        else:
          return self.layer2(inputs)

      def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(
            tuple(input_shape[:-1].as_list()) + (3,))

    model = MyModel()
    self.assertEqual(model.dynamic, True)
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))
    self.assertEqual(model.outputs[0].shape.as_list(), [None, 3])

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_forward_pass(self):
    inputs = keras.Input((3,))
    with self.assertRaisesRegexp(ValueError, 'You did something wrong!'):
      _ = InvalidLayer()(inputs)

  @keras_parameterized.run_with_all_model_types
  @test_util.run_in_graph_and_eager_modes
  def test_build_with_numpy_data(self):
    model_layers = [
        keras.layers.Dense(3, activation='relu', kernel_initializer='ones'),
        keras.layers.Dense(1, activation='sigmoid', kernel_initializer='ones')
    ]
    model = testing_utils.get_model_from_layers(model_layers, input_shape=(4,))
    model(np.zeros((2, 4), dtype='float32'))
    self.assertTrue(model.built)

  @test_util.run_in_graph_and_eager_modes
  def test_default_add_weight(self):

    class TestLayer(keras.layers.Layer):

      def __init__(self):
        super(TestLayer, self).__init__()
        self.default_weight = self.add_weight()
        self.weight_without_name = self.add_weight(shape=(3, 4))
        self.regularized_weight_without_name = self.add_weight(
            shape=(3, 4), regularizer='l2')

    layer = TestLayer()
    self.assertEqual(layer.default_weight.shape.as_list(), [])
    self.assertEqual(layer.weight_without_name.shape.as_list(), [3, 4])
    self.assertEqual(layer.default_weight.dtype.name, 'float32')
    self.assertEqual(layer.weight_without_name.dtype.name, 'float32')
    self.assertEqual(len(layer.losses), 1)
    if not context.executing_eagerly():
      # Cannot access tensor.name in eager execution.
      self.assertTrue('Variable_2/Regularizer' in layer.losses[0].name)

  def test_learning_phase_freezing_for_layers(self):
    # This test is only meant to run in graph functions mode (ambient eager).
    # In forced eager, `model.predict` ignores the global learning phase
    # and just uses training=False. TODO(fchollet): consider unifying the
    # behaviors.

    class LearningPhaseLayer(keras.layers.Layer):

      def call(self, inputs):
        return keras.backend.in_train_phase(
            lambda: array_ops.ones_like(inputs),
            lambda: array_ops.zeros_like(inputs))

    def get_learning_phase_value():
      model = keras.models.Sequential([LearningPhaseLayer(input_shape=(1,))])
      return np.sum(model.predict(np.ones((1, 1))))

    self.assertEqual(get_learning_phase_value(), 0)

    # Test scope.
    with keras.backend.learning_phase_scope(1):
      self.assertEqual(get_learning_phase_value(), 1)

    # The effects of the scope end after exiting it.
    self.assertEqual(get_learning_phase_value(), 0)

    # Test setting.
    keras.backend.set_learning_phase(1)
    self.assertEqual(get_learning_phase_value(), 1)
    keras.backend.set_learning_phase(0)
    self.assertEqual(get_learning_phase_value(), 0)

  # Cannot be enabled with `run_eagerly=True`, see b/123904578
  @test_util.run_all_in_graph_and_eager_modes
  def test_layer_can_return_variable(self):

    class ComputeSum(keras.layers.Layer):

      def __init__(self):
        super(ComputeSum, self).__init__()
        self.total = variables.Variable(
            initial_value=array_ops.zeros((1, 1)), trainable=False)
        if not context.executing_eagerly():
          keras.backend.get_session().run(self.total.initializer)

      def call(self, inputs):
        self.total.assign_add(inputs)
        return self.total

    inputs = keras.Input(shape=(1,))
    model = keras.Model(inputs, ComputeSum()(inputs))
    model.predict(np.ones((1, 1)))

  def _get_layer_with_training_arg(self):

    class TrainingLayer(keras.layers.Layer):
      """A layer with a `training` argument in a defuned `call`."""

      @def_function.function
      def call(self, inputs, training=None):
        if training is None:
          training = keras.backend.learning_phase()
        return tf_utils.smart_cond(training,
                                   lambda: array_ops.ones_like(inputs),
                                   lambda: array_ops.zeros_like(inputs))

    return TrainingLayer()

  @keras_parameterized.run_with_all_model_types
  # b/124459427: can't test with `run_eagerly=True` for now.
  @test_util.run_in_graph_and_eager_modes
  def test_training_arg_in_defun(self):
    layer = self._get_layer_with_training_arg()
    model = testing_utils.get_model_from_layers([layer], input_shape=(1,))
    model.compile(rmsprop.RMSprop(0.),
                  loss='mae')
    history = model.fit(np.zeros((1, 1)), np.zeros((1, 1)))
    self.assertEqual(history.history['loss'][0], 1.)
    loss = model.evaluate(np.zeros((1, 1)), np.zeros((1, 1)))
    self.assertEqual(loss, 0.)

    # Test that the argument injection performed in `call` is not active
    # when the argument is passed explicitly.
    layer = self._get_layer_with_training_arg()
    inputs = keras.Input(shape=(1,))
    # Pass `training` by name
    outputs = layer(inputs, training=False)
    model = keras.Model(inputs, outputs)
    model.compile(rmsprop.RMSprop(0.),
                  loss='mae')
    history = model.fit(np.zeros((1, 1)), np.zeros((1, 1)))
    self.assertEqual(history.history['loss'][0], 0.)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_raw_variable_assignment(self):

    class RawVariableLayer(keras.layers.Layer):

      def __init__(self, **kwargs):
        super(RawVariableLayer, self).__init__(**kwargs)
        # Test variables in nested structure.
        self.var_list = [variables.Variable(1.), {'a': variables.Variable(2.)}]

      def call(self, inputs):
        return inputs * self.var_list[0] * self.var_list[1]['a']

    model = testing_utils.get_model_from_layers([RawVariableLayer()],
                                                input_shape=(10,))
    model.compile('sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())
    x, y = np.ones((10, 10)), np.ones((10, 10))
    # Checks that variables get initialized.
    model.fit(x, y, batch_size=2, epochs=2)


class SymbolicSupportTest(test.TestCase):

  def test_using_symbolic_tensors_with_tf_ops(self):
    # Single-input.
    x = keras.Input((3,))
    y = math_ops.square(x)
    self.assertEqual(y.graph, keras.backend.get_graph())

    # Multi-inputs.
    x1, x2 = keras.Input((3,)), keras.Input((3,))
    y = array_ops.concat([x1, x2], axis=1)
    self.assertEqual(y.graph, keras.backend.get_graph())

    # Mixing Keras symbolic tensors and graph tensors from the same graph works.
    with keras.backend.get_graph().as_default():
      x1 = keras.Input((3,))
    x2 = keras.Input((3,))
    y = math_ops.matmul(x1, x2)
    self.assertEqual(y.graph, keras.backend.get_graph())

    # Creating same op type (matmul) multiple times in the Keras graph works.
    x1 = keras.Input((3,))
    x2 = keras.Input((3,))
    y = math_ops.matmul(x1, x2)
    self.assertEqual(y.graph, keras.backend.get_graph())

  def test_mixing_eager_and_graph_tensors(self):
    with ops.Graph().as_default():
      x1 = array_ops.ones((3, 3))
    x2 = array_ops.ones((3, 3))
    self.assertIsInstance(x2, ops.EagerTensor)
    with self.assertRaisesRegexp(TypeError, 'Graph tensors'):
      math_ops.matmul(x1, x2)

  def test_mixing_numpy_arrays_and_graph_tensors(self):
    with ops.Graph().as_default():
      x1 = array_ops.ones((3, 3))
    x2 = np.ones((3, 3), dtype='float32')
    with self.assertRaisesRegexp(TypeError, 'Graph tensors'):
      math_ops.matmul(x1, x2)

  @test_util.run_in_graph_and_eager_modes
  def test_mixing_keras_symbolic_tensors_and_eager_tensors(self):
    x1 = keras.Input((3,))
    x2 = array_ops.ones((3, 3))
    y = math_ops.matmul(x1, x2)
    self.assertEqual(y.graph, keras.backend.get_graph())
    fn = keras.backend.function(inputs=[x1], outputs=[y])
    x_val = np.random.random((3, 3))
    y_val = np.ones((3, 3))
    self.assertAllClose(fn([x_val])[0],
                        np.matmul(x_val, y_val),
                        atol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def test_mixing_keras_symbolic_tensors_and_numpy_arrays(self):
    x1 = keras.Input((3,))
    x2 = np.ones((3, 3), dtype='float32')
    y = math_ops.matmul(x1, x2)
    self.assertEqual(y.graph, keras.backend.get_graph())
    fn = keras.backend.function(inputs=[x1], outputs=[y])
    x_val = np.random.random((3, 3))
    y_val = np.ones((3, 3))
    self.assertAllClose(fn([x_val])[0],
                        np.matmul(x_val, y_val),
                        atol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def test_reraising_exception(self):
    # When layer is not dynamic, we have some pattern matching during exception
    # handling to detect when the user is trying to use python control flow.
    # When an exception is thrown but the pattern doesn't match, we want to
    # preserve the originating stack trace. An early implementation of this
    # logic lost the stack trace. We test the correct behavior here.

    class TypeErrorLayer(base_layer.Layer):

      def call(self, inputs):
        def easily_identifiable_name():
          raise TypeError('Non-matching TypeError message.')
        easily_identifiable_name()

    inputs = keras.Input((3,))

    try:
      _ = TypeErrorLayer()(inputs)
    except TypeError:
      tb = traceback.extract_tb(sys.exc_info()[2])
      last_entry = tb[-1]
      function_name = last_entry[2]
      self.assertEqual(function_name, 'easily_identifiable_name')


@test_util.run_all_in_graph_and_eager_modes
class NestedTrackingTest(test.TestCase):

  def test_nested_layer_variable_tracking(self):
    # Test that variables from nested sublayers are
    # being tracked by subclassed layers.

    class MyLayer(keras.layers.Layer):

      def __init__(self):
        super(MyLayer, self).__init__()
        self.dense1 = keras.layers.Dense(1)
        self.dense2 = keras.layers.BatchNormalization()

      def build(self, input_shape):
        self.v1 = self.add_weight('v1', shape=input_shape[1:].as_list())
        self.v2 = variables.Variable(
            name='v2',
            initial_value=np.zeros(input_shape[1:].as_list(), dtype='float32'),
            trainable=False)

      def call(self, inputs):
        x = self.dense1(inputs) + self.dense2(inputs)
        return x + self.v1 + self.v2

    layer = MyLayer()
    inputs = keras.Input((1,))
    _ = layer(inputs)

    self.assertEqual(len(layer.weights), 8)
    self.assertEqual(len(layer.trainable_weights), 5)
    self.assertEqual(len(layer.non_trainable_weights), 3)

    layer.dense1.trainable = False
    self.assertEqual(len(layer.weights), 8)
    self.assertEqual(len(layer.trainable_weights), 3)
    self.assertEqual(len(layer.non_trainable_weights), 5)

    layer.trainable = False
    self.assertEqual(len(layer.weights), 8)
    self.assertEqual(len(layer.trainable_weights), 0)
    self.assertEqual(len(layer.non_trainable_weights), 8)
    self.assertEqual(
        set([layer.dense1, layer.dense2, layer.v1, layer.v2]),
        set([obj for unused_name, obj in layer._checkpoint_dependencies]))

  def test_nested_layer_updates_losses_tracking(self):
    # Test that updates and losses from nested sublayers are
    # being tracked by subclassed layers.

    class UpdateAndLossLayer(keras.layers.Layer):

      def build(self, _):
        self.v1 = self.add_weight('v1', shape=())

      def call(self, inputs):
        self.add_loss(math_ops.reduce_sum(inputs))
        self.add_update(state_ops.assign_add(self.v1, 1))
        return inputs + 1

    class MyLayer(keras.layers.Layer):

      def build(self, _):
        self.v1 = self.add_weight('v1', shape=())

      def __init__(self):
        super(MyLayer, self).__init__()
        self.ul1 = UpdateAndLossLayer()
        self.ul2 = UpdateAndLossLayer()

      def call(self, inputs):
        self.add_loss(math_ops.reduce_sum(inputs))
        self.add_update(state_ops.assign_add(self.v1, 1))
        x = self.ul1(inputs)
        return self.ul2(x)

    layer = MyLayer()

    if context.executing_eagerly():
      inputs = array_ops.ones((3, 1))
      _ = layer(inputs)
      self.assertEqual(len(layer.losses), 3)
      self.assertLen(layer.get_losses_for(None), 3)
    else:
      inputs = keras.Input((1,))
      _ = layer(inputs)
      self.assertEqual(len(layer.losses), 3)
      self.assertEqual(len(layer.updates), 3)
      self.assertLen(layer.get_losses_for(None), 3)

  def test_attribute_reassignment(self):
    l = keras.layers.Layer()
    l.a = keras.layers.Layer()
    l.a = []
    l.a = variables.Variable(1.)
    l.a = keras.layers.Layer()
    last_assignment = keras.layers.Layer()
    l.a = last_assignment
    l.b = variables.Variable(1.)
    del l.b
    l.c = keras.layers.Layer()
    del l.c
    l.d = last_assignment
    del l.d
    self.assertEqual([last_assignment], l._layers)
    self.assertEqual([], l.trainable_weights)
    self.assertEqual([], l.non_trainable_weights)
    self.assertEqual([], l.weights)
    del l.a
    self.assertEqual([], l._layers)

  def test_assign_op_not_tracked_as_variable(self):

    class LayerWithAssignAttr(keras.layers.Layer):

      def build(self, input_shape):
        self.v = variables.Variable(1.)
        self.v_assign = self.v.assign_add(2.)

    layer = LayerWithAssignAttr()
    layer.build((10, 10))

    self.assertEqual([layer.v], layer.variables)


@test_util.run_all_in_graph_and_eager_modes
class NameScopingTest(keras_parameterized.TestCase):

  def test_name_scope_layer(self):
    x = keras.backend.placeholder(shape=(10, 10))
    layer = keras.layers.Dense(10, name='MyName')
    layer(x)
    self.assertEqual(layer.bias.name, 'MyName/bias:0')
    self.assertEqual(layer.kernel.name, 'MyName/kernel:0')

  def test_name_scope_sublayer(self):
    x = keras.backend.placeholder(shape=(10, 10))
    layer = keras.layers.Dense(
        10, activation=keras.layers.ReLU(name='MyAct'), name='MyName2')
    y = layer(x)
    self.assertEqual(layer.bias.name, 'MyName2/bias:0')
    self.assertEqual(layer.kernel.name, 'MyName2/kernel:0')
    self.assertEqual(y.name, 'MyName2/MyAct/Relu:0')

  def test_name_scope_tf_tensor(self):
    x = ops.convert_to_tensor(np.ones((10, 10)))
    layer = keras.layers.Dense(
        10, activation=keras.layers.ReLU(name='MyAct'), name='MyName3')
    layer(x)
    self.assertEqual(layer.bias.name, 'MyName3/bias:0')
    self.assertEqual(layer.kernel.name, 'MyName3/kernel:0')


_LAYERS_TO_TEST = [
    (keras.layers.Dense, (1,), collections.OrderedDict(units=[1])),
    (keras.layers.Activation, (2, 2),
     collections.OrderedDict(activation=['relu'])),
    (keras.layers.Dropout, (16,), collections.OrderedDict(rate=[0.25])),
    (keras.layers.BatchNormalization, (8, 8, 3), collections.OrderedDict(
        axis=[3], center=[True, False], scale=[True, False])),
    (keras.layers.Conv1D, (8, 8), collections.OrderedDict(
        filters=[1], kernel_size=[1, 3], strides=[1, 2],
        padding=['valid', 'same'], use_bias=[True, False],
        kernel_regularizer=[None, 'l2'])),
    (keras.layers.Conv2D, (8, 8, 3), collections.OrderedDict(
        filters=[1], kernel_size=[1, 3], strides=[1, 2],
        padding=['valid', 'same'], use_bias=[True, False],
        kernel_regularizer=[None, 'l2'])),
    (keras.layers.LSTM, (8, 8), collections.OrderedDict(
        units=[1],
        activation=[None, 'relu'],
        kernel_regularizer=[None, 'l2'],
        dropout=[0, 0.5],
        stateful=[True, False],
        unroll=[True, False])),
]

OUTPUT_TEST_CASES = []
for layer_type, inp_shape, arg_dict in _LAYERS_TO_TEST:
  arg_combinations = [[(k, i) for i in v] for k, v in arg_dict.items()]  # pylint: disable=g-complex-comprehension
  for args in it.product(*arg_combinations):
    name = '_{}_{}'.format(
        layer_type.__name__, '_'.join('{}_{}'.format(k, v) for k, v in args))
    OUTPUT_TEST_CASES.append(
        (name, layer_type, inp_shape, {k: v for k, v in args}))


class OutputTypeTest(keras_parameterized.TestCase):
  """Test that layers and models produce the correct tensor types."""

  # In v1 graph there are only symbolic tensors.
  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  @parameterized.named_parameters(*OUTPUT_TEST_CASES)
  def test_layer_outputs(self, layer_to_test, input_shape, layer_kwargs):
    layer = layer_to_test(**layer_kwargs)

    input_data = np.ones(shape=(2,) + input_shape, dtype=np.float32)
    layer_result = layer(input_data)

    inp = keras.layers.Input(shape=input_shape, batch_size=2)
    model = keras.models.Model(inp, layer_to_test(**layer_kwargs)(inp))
    model_result = model(input_data)

    for x in [layer_result, model_result]:
      if not isinstance(x, ops.Tensor):
        raise ValueError('Tensor or EagerTensor expected, got type {}'
                         .format(type(x)))

      if isinstance(x, ops.EagerTensor) != context.executing_eagerly():
        expected_type = (ops.EagerTensor if context.executing_eagerly()
                         else ops.Tensor)
        raise ValueError('Expected type {}, got type {}'
                         .format(expected_type, type(x)))


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
