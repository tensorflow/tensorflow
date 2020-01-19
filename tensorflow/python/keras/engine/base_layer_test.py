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

import copy
import os
import sys
import traceback

import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.layers import core as legacy_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.summary import summary_iterator
from tensorflow.python.util import nest


class DynamicLayer(base_layer.Layer):

  def __init__(self, dynamic=False, **kwargs):
    super(DynamicLayer, self).__init__(dynamic=dynamic, **kwargs)

  def call(self, inputs):
    samples = tensor_array_ops.TensorArray(
        dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
    for idx, sample in enumerate(inputs):
      samples = samples.write(idx, math_ops.square(sample))
    return samples.stack()

  def compute_output_shape(self, input_shape):
    return input_shape


class InvalidLayer(base_layer.Layer):

  def call(self, inputs):
    raise ValueError('You did something wrong!')


class BaseLayerTest(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  def test_dynamic_layer(self):
    model = testing_utils.get_model_from_layers([DynamicLayer(dynamic=True)],
                                                input_shape=(3,))
    self.assertEqual(model.dynamic, True)
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    self.assertEqual(model.run_eagerly, True)
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

  @keras_parameterized.run_with_all_model_types
  def test_dynamic_layer_error(self):
    with self.assertRaisesRegexp(TypeError,
                                 'attempting to use Python control flow'):
      model = testing_utils.get_model_from_layers([DynamicLayer()],
                                                  input_shape=(3,))
      model.compile(rmsprop.RMSprop(0.001), loss='mse')
      model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

  @keras_parameterized.run_with_all_model_types
  def test_dynamic_layer_error_running_in_graph_mode(self):
    with context.graph_mode():
      model = testing_utils.get_model_from_layers([DynamicLayer(dynamic=True)],
                                                  input_shape=(3,))
      self.assertEqual(model.dynamic, True)
      # But then you cannot run the model since you're in a graph scope.
      with self.assertRaisesRegexp(
          ValueError, 'You must enable eager execution'):
        model.compile(rmsprop.RMSprop(0.001), loss='mse')

  def test_manual_compute_output_shape(self):
    class BuildCounter(keras.layers.Layer):

      def __init__(self, *args, **kwargs):  # pylint: disable=redefined-outer-name
        super(BuildCounter, self).__init__(*args, **kwargs)
        self.build_counter = 0

      def build(self, input_shape):
        self.build_counter += 1

      def call(self, inputs):
        return inputs

    with context.eager_mode():
      layer = BuildCounter(dtype=dtypes.float64)
      output_shape = layer.compute_output_shape((None, 10))
      self.assertEqual(layer.build_counter, 1)
      self.assertEqual(output_shape.as_list(), [None, 10])
      output_signature = layer.compute_output_signature(
          tensor_spec.TensorSpec(dtype=dtypes.float64, shape=[None, 10]))
      self.assertEqual(layer.build_counter, 1)
      self.assertEqual(output_signature.dtype, dtypes.float64)
      self.assertEqual(output_signature.shape.as_list(), [None, 10])
      layer(np.ones((5, 10)))
      self.assertEqual(layer.build_counter, 1)

  def test_eager_switch_case_input(self):
    with context.eager_mode():
      task = keras.Input(shape=(), dtype=dtypes.int32)
      control_flow_ops.switch_case(
          task[0], [lambda: constant_op.constant(1.0) for _ in range(10)])

  def test_dynamic_layer_with_deferred_sequential_model(self):
    model = keras.Sequential(
        [DynamicLayer(dynamic=True),
         keras.layers.Dense(3)])
    self.assertEqual(model.dynamic, True)
    model.compile(rmsprop.RMSprop(0.001), loss='mse')
    self.assertEqual(model.run_eagerly, True)
    model.train_on_batch(np.random.random((2, 3)), np.random.random((2, 3)))

  def test_nested_dynamic_layers_in_eager_mode(self):
    inputs = keras.Input((3,))
    outputs = DynamicLayer(dynamic=True)(inputs)
    inner_model = keras.Model(inputs, outputs)
    self.assertEqual(inner_model.dynamic, True)

    inputs = keras.Input((3,))
    x = DynamicLayer(dynamic=True)(inputs)
    outputs = inner_model(x)

    model = keras.Model(inputs, outputs)
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

  def test_deepcopy(self):
    with context.eager_mode():
      bias_reg = lambda x: 1e-3 * math_ops.reduce_sum(x)
      layer = keras.layers.Conv2D(32, (3, 3), bias_regularizer=bias_reg)
      # Call the Layer on data to generate regularize losses.
      layer(array_ops.ones((1, 10, 10, 3)))
      self.assertLen(layer.losses, 1)
      new_layer = copy.deepcopy(layer)
      self.assertEqual(new_layer.bias_regularizer, bias_reg)
      self.assertEqual(layer.get_config(), new_layer.get_config())

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_forward_pass(self):
    inputs = keras.Input((3,))
    with self.assertRaisesRegexp(ValueError, 'You did something wrong!'):
      _ = InvalidLayer()(inputs)

  def test_no_legacy_model(self):
    inputs = keras.Input((1,))
    legacy_dense_0 = legacy_core.Dense(1, name='legacy_dense_0')
    legacy_dense_1 = legacy_core.Dense(1, name='legacy_dense_1')

    layer = legacy_dense_0(inputs)
    layer = keras.layers.Dense(1)(layer)
    layer = legacy_dense_1(layer)

    expected_regex = (r'The following are legacy tf\.layers\.Layers:\n  '
                      '{}\n  {}'.format(legacy_dense_0, legacy_dense_1))

    with self.assertRaisesRegexp(TypeError, expected_regex):
      _ = keras.models.Model(inputs=[inputs], outputs=[layer])

    model = keras.models.Model(inputs=[inputs], outputs=[inputs])
    with self.assertRaisesRegexp(TypeError, expected_regex):
      model._insert_layers([legacy_dense_0, legacy_dense_1])

  def test_no_legacy_sequential(self):
    layers = [
        keras.layers.Dense(1),
        legacy_core.Dense(1, name='legacy_dense_0')
    ]

    expected_regex = r'legacy tf\.layers\.Layers:\n  {}'.format(layers[1])
    with self.assertRaisesRegexp(TypeError, expected_regex):
      _ = keras.models.Sequential(layers)

    with self.assertRaisesRegexp(TypeError, expected_regex):
      _ = keras.models.Sequential([keras.layers.Input(shape=(4,))] + layers)

    model = keras.models.Sequential()
    with self.assertRaisesRegexp(TypeError, expected_regex):
      for l in layers:
        model.add(l)

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

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_learning_phase_freezing_for_layers(self):
    class LearningPhaseLayer(keras.layers.Layer):

      def call(self, inputs):
        return keras.backend.in_train_phase(
            lambda: array_ops.ones_like(inputs),
            lambda: array_ops.zeros_like(inputs))

    def get_learning_phase_value():
      model = keras.models.Sequential([LearningPhaseLayer(input_shape=(1,))])
      model._run_eagerly = testing_utils.should_run_eagerly()
      model._experimental_run_tf_function = (
          testing_utils.should_run_tf_function())
      return np.sum(model(np.ones((1, 1))))

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

  @keras_parameterized.run_all_keras_modes
  def test_learning_phase_freezing_for_layers_in_predict(self):
    if not (testing_utils.should_run_eagerly() or
            testing_utils.should_run_tf_function()):
      self.skipTest('Predict fails to override the outer learning phase in'
                    'the FuncGraph path.')

    class LearningPhaseLayer(keras.layers.Layer):

      def call(self, inputs):
        return keras.backend.in_train_phase(
            lambda: array_ops.ones_like(inputs),
            lambda: array_ops.zeros_like(inputs))

    def get_learning_phase_value():
      model = keras.models.Sequential([LearningPhaseLayer(input_shape=(1,))])
      model._run_eagerly = testing_utils.should_run_eagerly()
      model._experimental_run_tf_function = (
          testing_utils.should_run_tf_function())
      return np.sum(model.predict(np.ones((1, 1))))

    self.assertEqual(get_learning_phase_value(), 0)

    # Test scope.
    with keras.backend.learning_phase_scope(1):
      self.assertEqual(get_learning_phase_value(), 0)

    # The effects of the scope end after exiting it.
    self.assertEqual(get_learning_phase_value(), 0)

    # Test setting.
    keras.backend.set_learning_phase(1)
    self.assertEqual(get_learning_phase_value(), 0)
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
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    x, y = np.ones((10, 10)), np.ones((10, 10))
    # Checks that variables get initialized.
    model.fit(x, y, batch_size=2, epochs=2)

  @test_util.run_in_graph_and_eager_modes
  def test_layer_names(self):
    inputs = keras.layers.Input(shape=[2])
    add1 = inputs + inputs
    add2 = keras.layers.Add()([inputs, inputs])
    add3 = inputs + inputs
    add4 = keras.layers.Add()([inputs, inputs])
    model = keras.models.Model(
        inputs=[inputs], outputs=[add1, add2, add3, add4])
    actual_names = [l.name for l in model.layers]
    graph_names = [
        'input_1', 'tf_op_layer_AddV2', 'add', 'tf_op_layer_AddV2_1', 'add_1'
    ]
    eager_names = [
        'input_1', 'tf_op_layer_add', 'add', 'tf_op_layer_add_2', 'add_1'
    ]
    for actual, eager, graph in zip(actual_names, graph_names, eager_names):
      self.assertIn(actual, {eager, graph})

  def test_add_trainable_weight_on_frozen_layer(self):

    class TestLayer(keras.layers.Layer):

      def build(self, input_shape):
        self.w = self.add_weight(shape=(), trainable=True)

      def call(self, inputs):
        return self.w * inputs

    layer = TestLayer()
    layer.trainable = False
    layer.build(None)
    layer.trainable = True
    self.assertListEqual(layer.trainable_weights, [layer.w])

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_passing_initial_weights_values(self):
    kernel_value = np.random.random((10, 2))
    layer_with_weights = keras.layers.Dense(
        2, use_bias=False, weights=[kernel_value])

    model = testing_utils.get_model_from_layers([layer_with_weights],
                                                input_shape=(10,))
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    inputs = np.random.random((3, 10))
    out = model.predict(inputs)
    self.assertAllClose(model.layers[-1].get_weights()[0], kernel_value)
    self.assertAllClose(out, np.dot(inputs, kernel_value))

  @test_util.run_in_graph_and_eager_modes
  def test_set_weights_and_get_weights(self):
    layer = keras.layers.Dense(2)
    layer.build((None, 10))
    kernel = np.random.random((10, 2))
    bias = np.random.random((2,))
    layer.set_weights([kernel, bias])
    weights = layer.get_weights()
    self.assertEqual(len(weights), 2)
    self.assertAllClose(weights[0], kernel)
    self.assertAllClose(weights[1], bias)
    with self.assertRaisesRegexp(
        ValueError, 'but the layer was expecting 2 weights'):
      layer.set_weights([1, 2, 3])
    with self.assertRaisesRegexp(
        ValueError, 'not compatible with provided weight shape'):
      layer.set_weights([kernel.T, bias])

  def test_get_config_error(self):

    class MyLayer(keras.layers.Layer):

      def __init__(self, my_kwarg='default', **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.my_kwarg = my_kwarg

    # `__init__` includes kwargs but `get_config` is not overridden, so
    # an error should be thrown:
    with self.assertRaises(NotImplementedError):
      MyLayer('custom').get_config()

    class MyLayerNew(keras.layers.Layer):

      def __init__(self, my_kwarg='default', **kwargs):
        super(MyLayerNew, self).__init__(**kwargs)
        self.my_kwarg = my_kwarg

      def get_config(self):
        config = super(MyLayerNew, self).get_config()
        config['my_kwarg'] = self.my_kwarg
        return config

    # Test to make sure that error is not raised if the method call is
    # from an overridden `get_config`:
    self.assertEqual(MyLayerNew('custom').get_config()['my_kwarg'], 'custom')

    class MyLayerNew2(keras.layers.Layer):

      def __init__(self, name='MyLayerName', dtype=None, **kwargs):  # pylint:disable=redefined-outer-name
        super(MyLayerNew2, self).__init__(name=name, dtype=dtype, **kwargs)

    # Check that if the kwargs in `__init__` are base layer constructor
    # arguments, no error is thrown:
    self.assertEqual(MyLayerNew2(name='New').get_config()['name'], 'New')

  @test_util.run_in_graph_and_eager_modes
  def test_count_params(self):
    dense = keras.layers.Dense(16)
    dense.build((None, 4))
    self.assertEqual(dense.count_params(), 16 * 4 + 16)

    dense = keras.layers.Dense(16)
    with self.assertRaisesRegexp(ValueError, 'call `count_params`'):
      dense.count_params()

    model = keras.Sequential(keras.layers.Dense(16))
    with self.assertRaisesRegexp(ValueError, 'call `count_params`'):
      model.count_params()

    dense = keras.layers.Dense(16, input_dim=4)
    model = keras.Sequential(dense)
    self.assertEqual(model.count_params(), 16 * 4 + 16)

  def test_super_not_called(self):

    class CustomLayerNotCallingSuper(keras.layers.Layer):

      def __init__(self):
        pass

    layer = CustomLayerNotCallingSuper()
    with self.assertRaisesRegexp(RuntimeError, 'You must call `super()'):
      layer(np.random.random((10, 2)))

  @test_util.run_in_graph_and_eager_modes
  def test_first_arg_not_called_inputs(self):
    x, y = array_ops.ones((10, 1)), array_ops.ones((10, 1))

    class ArgLayer(keras.layers.Layer):

      def call(self, x, y):
        return x + y

    layer = ArgLayer()
    out = self.evaluate(layer(x=x, y=y))
    self.assertAllClose(out, 2 * np.ones((10, 1)))

    class KwargLayer(keras.layers.Layer):

      def call(self, x=None, y=None):
        return x + y

    layer = KwargLayer()
    out = self.evaluate(layer(x=x, y=y))
    self.assertAllClose(out, 2 * np.ones((10, 1)))

    with self.assertRaisesRegexp(ValueError, 'must always be passed'):
      layer(y=y)

    class TFFunctionLayer(keras.layers.Layer):

      @def_function.function
      def call(self, x, y=None):
        if y is None:
          return x
        return x + y

    layer = TFFunctionLayer()
    out = self.evaluate(layer(x=x, y=y))
    self.assertAllClose(out, 2 * np.ones((10, 1)))

  def test_relu_layer_with_dtype(self):
    # Test case for GitHub issue 35430
    x = keras.Input(shape=(), name='x', dtype='int64')
    y = keras.layers.ReLU(max_value=100, dtype='int64')(x)


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
    except TypeError as e:
      if hasattr(e, 'ag_error_metadata'):
        self.assertIn('easily_identifiable_name', str(e))
        # See ErrorMetadataBase in autograph/pyct/errors.py
        function_name = e.ag_error_metadata.translated_stack[-1].function_name
      else:
        tb = traceback.extract_tb(sys.exc_info()[2])
        last_entry = tb[-1]
        function_name = last_entry[2]
      self.assertEqual(function_name, 'easily_identifiable_name')

  @test_util.run_in_graph_and_eager_modes
  def test_summaries_in_tf_function(self):
    if not context.executing_eagerly():
      return

    class MyLayer(keras.layers.Layer):

      def call(self, inputs):
        summary_ops_v2.scalar('mean', math_ops.reduce_mean(inputs))
        return inputs

    tmp_dir = self.get_temp_dir()
    writer = summary_ops_v2.create_file_writer_v2(tmp_dir)
    with writer.as_default(), summary_ops_v2.always_record_summaries():
      my_layer = MyLayer()
      x = array_ops.ones((10, 10))

      def my_fn(x):
        return my_layer(x)

      _ = my_fn(x)

    event_file = gfile.Glob(os.path.join(tmp_dir, 'events*'))
    self.assertLen(event_file, 1)
    event_file = event_file[0]
    tags = set()
    for e in summary_iterator.summary_iterator(event_file):
      for val in e.summary.value:
        tags.add(val.tag)
    self.assertEqual(set(['my_layer/mean']), tags)


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
        {id(v) for v in [layer.dense1, layer.dense2, layer.v1, layer.v2]},
        {id(v) for _, v in layer._checkpoint_dependencies})

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

  def test_layer_class_not_tracked_as_sublayer(self):
    # See https://github.com/tensorflow/tensorflow/issues/27431 for details.

    class LayerWithClassAttribute(keras.layers.Layer):

      def __init__(self):
        super(LayerWithClassAttribute, self).__init__()
        self.layer_fn = keras.layers.Dense

    layer = LayerWithClassAttribute()
    self.assertEmpty(layer.variables)
    self.assertEmpty(layer.submodules)

  def test_layer_call_fn_args(self):

    class NonDefunLayer(keras.layers.Layer):

      def call(self, inputs, a, mask, b=None, training=None):
        return inputs

    class DefunLayer(keras.layers.Layer):

      @def_function.function
      def call(self, x, mask, a, training=None, b=None):
        return x

    nondefun_layer = NonDefunLayer()
    self.assertEqual(nondefun_layer._call_fn_args,
                     ['inputs', 'a', 'mask', 'b', 'training'])
    defun_layer = DefunLayer()
    self.assertEqual(defun_layer._call_fn_args,
                     ['x', 'mask', 'a', 'training', 'b'])


@test_util.run_all_in_graph_and_eager_modes
class NameScopingTest(keras_parameterized.TestCase):

  def test_name_scope_layer(self):
    x = keras.backend.placeholder(shape=(10, 10))
    layer = keras.layers.Dense(10, name='MyName')
    layer(x)
    self.assertEqual(layer.bias.name, 'MyName/bias:0')
    self.assertEqual(layer.kernel.name, 'MyName/kernel:0')

  def test_name_scope_sublayer(self):

    class NameScopeTracker(keras.layers.Layer):

      def call(self, inputs):
        self.active_name_scope = ops.get_name_scope()
        return inputs

    x = keras.backend.placeholder(shape=(10, 10))
    sublayer = NameScopeTracker(name='Sublayer')
    layer = keras.layers.Dense(10, activation=sublayer, name='MyName2')
    layer(x)
    self.assertEqual(layer.bias.name, 'MyName2/bias:0')
    self.assertEqual(layer.kernel.name, 'MyName2/kernel:0')
    self.assertEqual(sublayer.active_name_scope, 'MyName2/Sublayer')

  def test_name_scope_tf_tensor(self):
    x = ops.convert_to_tensor(np.ones((10, 10)))
    layer = keras.layers.Dense(
        10, activation=keras.layers.ReLU(name='MyAct'), name='MyName3')
    layer(x)
    self.assertEqual(layer.bias.name, 'MyName3/bias:0')
    self.assertEqual(layer.kernel.name, 'MyName3/kernel:0')


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class AutographControlFlowTest(keras_parameterized.TestCase):

  def test_disabling_in_context_is_matched(self):

    test_obj = self

    class MyLayer(keras.layers.Layer):

      def call(self, inputs, training=None):
        with test_obj.assertRaisesRegex(TypeError, 'Tensor.*as.*bool'):
          if constant_op.constant(False):
            return inputs * 1.
        return inputs * 0.

    @def_function.function(autograph=False)
    def test_fn():
      return MyLayer()(constant_op.constant([[1., 2., 3.]]))

    test_fn()

  def test_if_training_pattern_output(self):

    class MyLayer(keras.layers.Layer):

      def call(self, inputs, training=None):
        if training:
          return inputs * 1.
        return inputs * 0.

    inputs = keras.Input((3,))
    outputs = MyLayer()(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    train_loss = model.train_on_batch(np.ones((2, 3)), np.ones((2, 3)))
    self.assertEqual(train_loss, 0.)
    test_loss = model.test_on_batch(np.ones((2, 3)), np.ones((2, 3)))
    self.assertEqual(test_loss, 1.)

  def test_if_training_pattern_loss(self):

    class MyLayer(keras.layers.Layer):

      def call(self, inputs, training=None):
        if training:
          loss = math_ops.reduce_sum(inputs)
        else:
          loss = 0.
        self.add_loss(loss)
        return inputs

    inputs = keras.Input((3,))
    outputs = MyLayer()(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    train_loss = model.train_on_batch(np.ones((2, 3)), np.ones((2, 3)))
    self.assertEqual(train_loss, 2 * 3)
    test_loss = model.test_on_batch(np.ones((2, 3)), np.ones((2, 3)))
    self.assertEqual(test_loss, 0)

  def test_if_training_pattern_metric(self):

    class MyLayer(keras.layers.Layer):

      def call(self, inputs, training=None):
        if training:
          metric = math_ops.reduce_sum(inputs)
        else:
          metric = 0.
        self.add_metric(metric, name='my_metric', aggregation='mean')
        return inputs

    inputs = keras.Input((3,))
    outputs = MyLayer()(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    for _ in range(3):
      _, train_metric = model.train_on_batch(np.ones((2, 3)),
                                             np.ones((2, 3)))

      self.assertEqual(train_metric, 2 * 3)
      _, test_metric = model.test_on_batch(np.ones((2, 3)),
                                           np.ones((2, 3)))
      self.assertEqual(test_metric, 0)

  def test_if_training_pattern_update(self):

    class MyLayer(keras.layers.Layer):

      def build(self, input_shape):
        self.counter = self.add_weight(
            shape=(), trainable=False, initializer='zeros')

      def call(self, inputs, training=None):
        if training:
          increment = 1.
        else:
          increment = 0.
        self.counter.assign_add(increment)
        return inputs

    inputs = keras.Input((3,))
    layer = MyLayer()
    outputs = layer(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(np.ones((2, 3)), np.ones((2, 3)))
    self.assertEqual(keras.backend.get_value(layer.counter), 1.)

  def test_conditional_updates_in_call(self):

    class MyLayer(keras.layers.Layer):

      def __init__(self):
        super(MyLayer,
              self).__init__(dynamic=testing_utils.should_run_eagerly())

      def build(self, input_shape):
        self.counter = self.add_weight(
            shape=(), trainable=False, initializer='zeros')

      def call(self, inputs, training=None):
        if training:
          z = math_ops.reduce_sum(inputs)
          self.add_update(lambda: self.counter.assign_add(z))
        return inputs

      def compute_output_shape(self, input_shape):
        return input_shape

    if testing_utils.should_run_eagerly():
      inputs = keras.Input((3,))
      layer = MyLayer()
      outputs = layer(inputs)
      model = keras.Model(inputs, outputs)
      model.compile(
          'sgd',
          'mse',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
      model.train_on_batch(np.ones((2, 3)), np.ones((2, 3)))
      self.assertEqual(keras.backend.get_value(layer.counter), 6.)
    else:
      # TODO(fchollet): support the same workflow in graph mode.
      with self.assertRaisesRegexp(RuntimeError,
                                   '`add_update` in a control flow branch'):
        layer = MyLayer()
        layer(keras.Input((3,)))
        _ = layer.updates

  def test_conditional_losses_in_call(self):

    class MyLayer(keras.layers.Layer):

      def __init__(self):
        super(MyLayer,
              self).__init__(dynamic=testing_utils.should_run_eagerly())

      def call(self, inputs, training=None):
        if training:
          self.add_loss(math_ops.reduce_sum(inputs))
        return inputs

      def compute_output_shape(self, input_shape):
        return input_shape

    if testing_utils.should_run_eagerly():
      inputs = keras.Input((3,))
      layer = MyLayer()
      outputs = layer(inputs)
      model = keras.Model(inputs, outputs)
      model.compile(
          'sgd',
          'mse',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
      loss = model.train_on_batch(np.ones((2, 3)), np.ones((2, 3)))
      self.assertEqual(loss, 2 * 3)
    else:
      with self.assertRaisesRegexp(RuntimeError,
                                   '`add_loss` in a control flow branch'):
        layer = MyLayer()(keras.Input((3,)))

  def test_conditional_callable_losses(self):
    model = keras.Sequential([
        keras.layers.Dense(
            1, kernel_regularizer=keras.regularizers.l2(1e-4), input_shape=(1,))
    ])
    model._run_eagerly = testing_utils.should_run_eagerly()
    model._experimental_run_tf_function = testing_utils.should_run_tf_function()

    def assert_graph(t):
      if not context.executing_eagerly():
        self.assertEqual(t.graph, ops.get_default_graph())

    @def_function.function
    def get_losses(t):
      if t < 0:
        return math_ops.reduce_sum(model.losses) * t
      else:
        return math_ops.reduce_sum(model.losses)

    assert_graph(get_losses(constant_op.constant(2.)))
    assert_graph(get_losses(constant_op.constant(0.5)))

  def test_conditional_metrics_in_call(self):

    class MyLayer(keras.layers.Layer):

      def __init__(self):
        super(MyLayer,
              self).__init__(dynamic=testing_utils.should_run_eagerly())

      def call(self, inputs, training=None):
        if training:
          self.add_metric(math_ops.reduce_sum(inputs),
                          name='sum',
                          aggregation='mean')
        return inputs

      def compute_output_shape(self, input_shape):
        return input_shape

    if testing_utils.should_run_eagerly():
      inputs = keras.Input((3,))
      layer = MyLayer()
      outputs = layer(inputs)
      model = keras.Model(inputs, outputs)
      model.compile(
          'sgd',
          'mse',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
      history = model.fit(np.ones((2, 3)), np.ones((2, 3)))
      self.assertEqual(history.history['sum'][-1], 2 * 3)
    else:
      # TODO(fchollet): support the same workflow in graph mode.
      with self.assertRaisesRegexp(RuntimeError,
                                   '`add_metric` in a control flow branch'):
        layer = MyLayer()(keras.Input((3,)))

  def test_conditional_activity_regularizer_in_call(self):

    class TestModel(keras.Model):

      def __init__(self):
        super(TestModel, self).__init__(
            name='test_model', dynamic=testing_utils.should_run_eagerly())
        self.layer = keras.layers.Dense(2, activity_regularizer='l2')

      def call(self, x, training=None):
        if math_ops.greater(math_ops.reduce_sum(x), 0.0):
          return self.layer(x)
        else:
          return self.layer(x)

    model = TestModel()
    model.compile(
        loss='mse',
        optimizer='sgd',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))

    if testing_utils.should_run_eagerly():
      model.fit(x, y, epochs=2, batch_size=5)
    else:
      with self.assertRaisesRegexp(
          RuntimeError, '`activity_regularizer` in a control flow branch'):
        model.fit(x, y, epochs=2, batch_size=5)

  def test_conditional_activity_regularizer_with_wrappers_in_call(self):

    class TestModel(keras.Model):

      def __init__(self):
        super(TestModel, self).__init__(
            name='test_model', dynamic=testing_utils.should_run_eagerly())
        self.layer = keras.layers.TimeDistributed(
            keras.layers.Dense(2, activity_regularizer='l2'),
            input_shape=(3, 4))

      def call(self, x, training=None):
        if math_ops.greater(math_ops.reduce_sum(x), 0.0):
          return self.layer(x)
        else:
          return self.layer(x)

    model = TestModel()
    model.compile(
        loss='mse',
        optimizer='sgd',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones(shape=(10, 3, 4))
    y = np.ones(shape=(10, 3, 2))

    if testing_utils.should_run_eagerly():
      model.fit(x, y, epochs=2, batch_size=5)
    else:
      with self.assertRaisesRegexp(
          RuntimeError, '`activity_regularizer` in a control flow branch'):
        model.fit(x, y, epochs=2, batch_size=5)


class AddLayer(keras.layers.Layer):
  """A layer which adds it's input to a variable.

  Useful for testing a layer with a variable
  """

  def build(self, _):
    self.v = self.add_weight('v', (), initializer='ones')
    self.built = True

  def call(self, inputs):
    return inputs + self.v


class IdentityLayer(keras.layers.Layer):
  """A layer that returns it's input.

  Useful for testing a layer without a variable.
  """

  def call(self, inputs):
    return inputs


@test_util.run_all_in_graph_and_eager_modes
class DTypeTest(keras_parameterized.TestCase):

  # This class only have tests relating to layer.dtype. Tests for dtype policies
  # are in mixed_precision/experimental/keras_test.py

  # TODO(reedwm): Maybe have a separate test file for input casting tests.

  def _const(self, dtype):
    return array_ops.constant(1, dtype=dtype)

  @testing_utils.enable_v2_dtype_behavior
  def test_dtype_defaults_to_floatx(self):
    layer = AddLayer()
    self.assertEqual(layer.dtype, 'float32')
    layer(self._const('float64'))
    self.assertEqual(layer.dtype, 'float32')  # dtype should not change

    try:
      backend.set_floatx('float64')
      layer = AddLayer()
      self.assertEqual(layer.dtype, 'float64')
    finally:
      backend.set_floatx('float32')

  @testing_utils.enable_v2_dtype_behavior
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

  @testing_utils.enable_v2_dtype_behavior
  def input_cast_to_dtype(self):
    layer = AddLayer()

    # Input should be cast to layer.dtype, so output should also be layer.dtype
    self.assertEqual(layer(self._const('float64')).dtype, 'float32')

    layer = AddLayer(dtype='float64')
    self.assertEqual(layer(self._const('float32')).dtype, 'float64')

    # Test inputs are not casted if layer.dtype is not floating-point
    layer = IdentityLayer(dtype='int32')
    self.assertEqual(layer(self._const('float64')).dtype, 'float64')

    # Test inputs are not casted if the inputs are not floating-point
    layer = IdentityLayer(dtype='float32')
    self.assertEqual(layer(self._const('int32')).dtype, 'int32')

    # Test Numpy arrays are casted
    layer = IdentityLayer(dtype='float64')
    self.assertEqual(layer(np.array(1, dtype='float32')).dtype, 'float64')

    # Test Python floats are casted
    layer = IdentityLayer(dtype='float64')
    self.assertEqual(layer(1.).dtype, 'float64')

  @testing_utils.enable_v2_dtype_behavior
  def multiple_inputs_cast_to_dtype(self):

    class MultiIdentityLayer(keras.layers.Layer):

      def call(self, inputs):
        return [array_ops.identity(x) for x in inputs]

    # Testing layer with default dtype of float32
    layer = MultiIdentityLayer()
    x, y = layer([self._const('float16'), self._const('float32')])
    self.assertEqual(x.dtype, 'float32')
    self.assertEqual(y.dtype, 'float32')

    # Test passing dtype to the constructor
    layer = MultiIdentityLayer(dtype='float64')
    x, y = layer([self._const('float16'), self._const('float32')])
    self.assertEqual(x.dtype, 'float64')
    self.assertEqual(y.dtype, 'float64')

    # Test several non-floating point types
    layer = MultiIdentityLayer(dtype='float64')
    x, y, z, w = layer([self._const('float16'), self._const('bool'),
                        self._const('float64'), self._constant('complex64')])
    self.assertEqual(x.dtype, 'float64')
    self.assertEqual(y.dtype, 'bool')
    self.assertEqual(z.dtype, 'float64')
    self.assertEqual(w.dtype, 'complex64')

  @testing_utils.enable_v2_dtype_behavior
  def test_extra_args_and_kwargs_not_casted(self):

    class IdentityLayerWithArgs(keras.layers.Layer):

      def call(self, inputs, *args, **kwargs):
        return nest.flatten([inputs, args, kwargs])

    layer = IdentityLayerWithArgs(dtype='float64')
    x, y, z = layer(self._const('float16'), self._const('float16'),
                    kwarg=self._const('float16'))
    self.assertEqual(x.dtype, 'float64')
    self.assertEqual(y.dtype, 'float16')
    self.assertEqual(z.dtype, 'float16')

  @testing_utils.enable_v2_dtype_behavior
  def test_layer_without_autocast(self):

    class IdentityLayerWithoutAutocast(IdentityLayer):

      def __init__(self, *args, **kwargs):
        kwargs['autocast'] = False
        super(IdentityLayerWithoutAutocast, self).__init__(*args, **kwargs)

    layer = IdentityLayerWithoutAutocast(dtype='float64')
    self.assertEqual(layer(self._const('float32')).dtype, 'float32')

  @testing_utils.enable_v2_dtype_behavior
  def test_dtype_warnings(self):
    # Test a layer warns when it casts inputs.
    layer = IdentityLayer()
    with test.mock.patch.object(tf_logging, 'warn') as mock_warn:
      layer(self._const('float64'))
      self.assertRegexpMatches(
          str(mock_warn.call_args),
          ".*from dtype float64 to the layer's dtype of float32.*"
          "The layer has dtype float32 because.*")

    # Test a layer does not warn a second time
    with test.mock.patch.object(tf_logging, 'warn') as mock_warn:
      layer(self._const('float64'))
      mock_warn.assert_not_called()

    # Test a new layer can warn even if a different layer already warned
    layer = IdentityLayer()
    with test.mock.patch.object(tf_logging, 'warn') as mock_warn:
      layer(self._const('float64'))
      self.assertRegexpMatches(
          str(mock_warn.call_args),
          ".*from dtype float64 to the layer's dtype of float32.*"
          "The layer has dtype float32 because.*")

    # Test a layer does not warn if a dtype is passed
    layer = IdentityLayer(dtype='float32')
    with test.mock.patch.object(tf_logging, 'warn') as mock_warn:
      layer(self._const('float64'))
      mock_warn.assert_not_called()

    # Test a layer does not warn if a Policy is set:
    with policy.policy_scope('float32'):
      layer = IdentityLayer()
      with test.mock.patch.object(tf_logging, 'warn') as mock_warn:
        layer(self._const('float64'))
        mock_warn.assert_not_called()

  @testing_utils.enable_v2_dtype_behavior
  def test_compute_output_signature(self):

    class IdentityLayerWithOutputShape(IdentityLayer):

      def compute_output_shape(self, input_shape):
        return input_shape

    layer = IdentityLayerWithOutputShape(dtype='float64')
    output_signature = layer.compute_output_signature(
        tensor_spec.TensorSpec(shape=(), dtype='float32'))
    self.assertEqual(output_signature.shape, ())
    self.assertEqual(output_signature.dtype, 'float64')

  @testing_utils.enable_v2_dtype_behavior
  def test_composite_tensors_input_casting(self):
    sparse = sparse_tensor.SparseTensor(
        indices=array_ops.constant([[0, 1], [2, 3]], dtype='int64'),
        values=array_ops.constant([0., 1.], dtype='float32'),
        dense_shape=array_ops.constant([4, 4], dtype='int64'))
    ragged = ragged_tensor.RaggedTensor.from_row_splits(
        values=array_ops.constant([1., 2., 3.], dtype='float32'),
        row_splits=array_ops.constant([0, 2, 2, 3], dtype='int64'))

    layer = IdentityLayer(dtype='float16')
    layer._supports_ragged_inputs = True

    for x in sparse, ragged:
      self.assertEqual(x.dtype, 'float32')
      y = layer(x)
      self.assertEqual(y.dtype, 'float16')
      self.assertEqual(type(x), type(y))

  def test_supports_ragged_inputs_attribute_error(self):
    with self.assertRaisesRegexp(ValueError,
                                 'does not support RaggedTensors'):
      ragged = ragged_tensor.RaggedTensor.from_row_splits(
          values=array_ops.constant([1., 2., 3.], dtype='float32'),
          row_splits=array_ops.constant([0, 2, 2, 3], dtype='int64'))
      model = keras.Sequential([
          keras.layers.InputLayer(input_shape=(None,), ragged=True),
          IdentityLayer()])
      model.compile(rmsprop.RMSprop(0.001), loss='mse')
      model.train_on_batch(ragged)

  @testing_utils.enable_v2_dtype_behavior
  def test_passing_non_tensor(self):
    layer = IdentityLayer()
    x = object()
    y = layer(x)  # Layer should not cast 'x', as it's not a tensor
    self.assertIs(x, y)

  @testing_utils.disable_v2_dtype_behavior
  def test_v1_behavior(self):
    # Test dtype defaults to None and inferred from input
    layer = IdentityLayer()
    self.assertIsNone(layer.dtype)
    layer(self._const('float64'))
    self.assertEqual(layer.dtype, 'float64')

    # Test layer does not cast to dtype
    self.assertEqual(layer(self._const('float32')).dtype, 'float32')


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
