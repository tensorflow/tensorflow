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

import sys
import traceback
from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.optimizer_v2 import rmsprop
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
    else:
      inputs = keras.Input((1,))
      _ = layer(inputs)
      self.assertEqual(len(layer.losses), 3)
      self.assertEqual(len(layer.updates), 3)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
