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
# ==============================================================================
"""Tests for `models.py` (model cloning, mainly)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os

import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import metrics
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test
from tensorflow.python.training import adam


class TestModel(keras.Model):
  """A model subclass."""

  def __init__(self, n_outputs=4, trainable=True):
    """A test class with one dense layer and number of outputs as a variable."""
    super(TestModel, self).__init__()
    self.layer1 = keras.layers.Dense(n_outputs)
    self.n_outputs = resource_variable_ops.ResourceVariable(
        n_outputs, trainable=trainable)

  def call(self, x):
    return self.layer1(x)


def sequential_model(add_input_layer, include_input_shape=True):
  model = keras.models.Sequential()
  if add_input_layer:
    model.add(keras.layers.InputLayer(input_shape=(4,)))
    model.add(keras.layers.Dense(4))
  elif include_input_shape:
    model.add(keras.layers.Dense(4, input_shape=(4,)))
  else:
    model.add(keras.layers.Dense(4))
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Dense(4))
  return model


class TestModelCloning(test.TestCase):

  @test_util.run_v1_only('b/120545219')
  def test_clone_sequential_model(self):
    with self.cached_session():
      val_a = np.random.random((10, 4))
      val_out = np.random.random((10, 4))

      model = sequential_model(False)

    # Everything should work in a new session.
    keras.backend.clear_session()

    with self.cached_session():
      # With placeholder creation
      new_model = keras.models.clone_model(model)
      # update ops from batch norm needs to be included
      self.assertEqual(len(new_model.get_updates_for(new_model.inputs)), 2)
      new_model.compile('rmsprop', 'mse')
      new_model.train_on_batch(val_a, val_out)

      # On top of new tensor
      input_a = keras.Input(shape=(4,))
      new_model = keras.models.clone_model(model, input_tensors=input_a)
      self.assertEqual(len(new_model.get_updates_for(new_model.inputs)), 2)
      new_model.compile('rmsprop', 'mse')
      new_model.train_on_batch(val_a, val_out)

      # On top of new, non-Keras tensor
      input_a = keras.backend.variable(val_a)
      new_model = keras.models.clone_model(model, input_tensors=input_a)
      self.assertEqual(len(new_model.get_updates_for(new_model.inputs)), 2)
      new_model.compile('rmsprop', 'mse')
      new_model.train_on_batch(None, val_out)

  @test_util.run_v1_only('b/120545219')
  def test_clone_sequential_model_input_layer(self):

    def test_input_layer(include_inputs):
      with self.cached_session():
        val_a = np.random.random((10, 4))
        model = sequential_model(include_inputs, include_inputs)
        # Sanity check
        self.assertEqual(
            isinstance(model._layers[0], keras.layers.InputLayer),
            include_inputs)
        self.assertEqual(model._is_graph_network, include_inputs)

      keras.backend.clear_session()
      with self.cached_session():
        # With placeholder creation -- clone model should have an InputLayer
        # if the original model has one.
        new_model = keras.models.clone_model(model)
        self.assertEqual(
            isinstance(new_model._layers[0], keras.layers.InputLayer),
            include_inputs)
        self.assertEqual(new_model._is_graph_network, model._is_graph_network)

        # On top of new tensor  -- clone model should always have an InputLayer.
        input_a = keras.Input(shape=(4,))
        new_model = keras.models.clone_model(model, input_tensors=input_a)
        self.assertIsInstance(new_model._layers[0], keras.layers.InputLayer)
        self.assertTrue(new_model._is_graph_network)

        # On top of new, non-Keras tensor  -- clone model should always have an
        # InputLayer.
        input_a = keras.backend.variable(val_a)
        new_model = keras.models.clone_model(model, input_tensors=input_a)
        self.assertIsInstance(new_model._layers[0], keras.layers.InputLayer)
        self.assertTrue(new_model._is_graph_network)

    test_input_layer(True)
    test_input_layer(False)

  @test_util.run_v1_only('b/120545219')
  def test_clone_functional_model(self):
    with self.cached_session():
      val_a = np.random.random((10, 4))
      val_b = np.random.random((10, 4))
      val_out = np.random.random((10, 4))

      input_a = keras.Input(shape=(4,))
      input_b = keras.Input(shape=(4,))
      dense_1 = keras.layers.Dense(4,)
      dense_2 = keras.layers.Dense(4,)

      x_a = dense_1(input_a)
      x_a = keras.layers.Dropout(0.5)(x_a)
      x_a = keras.layers.BatchNormalization()(x_a)
      x_b = dense_1(input_b)
      x_a = dense_2(x_a)
      outputs = keras.layers.add([x_a, x_b])
      model = keras.models.Model([input_a, input_b], outputs)

    # Everything should work in a new session.
    keras.backend.clear_session()

    with self.cached_session():
      # With placeholder creation
      new_model = keras.models.clone_model(model)
      self.assertEqual(len(new_model.get_updates_for(new_model.inputs)), 2)
      new_model.compile('rmsprop', 'mse')
      new_model.train_on_batch([val_a, val_b], val_out)

      # On top of new tensors
      input_a = keras.Input(shape=(4,), name='a')
      input_b = keras.Input(shape=(4,), name='b')
      new_model = keras.models.clone_model(
          model, input_tensors=[input_a, input_b])
      self.assertEqual(len(new_model.get_updates_for(new_model.inputs)), 2)
      new_model.compile('rmsprop', 'mse')
      new_model.train_on_batch([val_a, val_b], val_out)

      # On top of new, non-Keras tensors
      input_a = keras.backend.variable(val_a)
      input_b = keras.backend.variable(val_b)
      new_model = keras.models.clone_model(
          model, input_tensors=[input_a, input_b])
      self.assertEqual(len(new_model.get_updates_for(new_model.inputs)), 2)
      new_model.compile('rmsprop', 'mse')
      new_model.train_on_batch(None, val_out)

  @test_util.run_in_graph_and_eager_modes
  def test_clone_functional_model_with_masking(self):
    with self.cached_session():
      x = np.array([[[1], [1]], [[0], [0]]])
      inputs = keras.Input((2, 1))
      outputs = keras.layers.Masking(mask_value=0)(inputs)
      outputs = keras.layers.TimeDistributed(
          keras.layers.Dense(1, kernel_initializer='one'))(outputs)
      model = keras.Model(inputs, outputs)

      model = keras.models.clone_model(model)
      model.compile(loss='mse', optimizer=adam.AdamOptimizer(0.01))
      y = np.array([[[1], [1]], [[1], [1]]])
      loss = model.train_on_batch(x, y)
      self.assertEqual(float(loss), 0.)

  def test_model_cloning_invalid_use_cases(self):
    seq_model = keras.models.Sequential()
    seq_model.add(keras.layers.Dense(4, input_shape=(4,)))

    x = keras.Input((4,))
    y = keras.layers.Dense(4)(x)
    fn_model = keras.models.Model(x, y)

    with self.assertRaises(ValueError):
      keras.models._clone_functional_model(seq_model)
    with self.assertRaises(ValueError):
      keras.models._clone_functional_model(None)
    with self.assertRaises(ValueError):
      keras.models._clone_sequential_model(fn_model)

    with self.assertRaises(ValueError):
      keras.models._clone_sequential_model(seq_model, input_tensors=[x, x])
    with self.assertRaises(ValueError):
      keras.models._clone_sequential_model(seq_model, input_tensors=y)

  def test_functional_cloning_does_not_create_unnecessary_placeholders(self):
    with ops.Graph().as_default():
      x = keras.Input((4,))
      y = keras.layers.Dense(4)(x)
      model = keras.models.Model(x, y)
    graph = ops.Graph()
    with graph.as_default():
      x = array_ops.ones((10, 4))
      _ = keras.models.clone_model(model, input_tensors=[x])
      has_placeholder = _has_placeholder(graph)
      self.assertFalse(has_placeholder)

  def test_sequential_cloning_does_not_create_unnecessary_placeholders(self):
    with ops.Graph().as_default():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(4, input_shape=(4,)))
    graph = ops.Graph()
    with graph.as_default():
      x = array_ops.ones((10, 4))
      _ = keras.models.clone_model(model, input_tensors=[x])
      has_placeholder = _has_placeholder(graph)
      self.assertFalse(has_placeholder)


def _has_placeholder(graph):
  ops_types = [op.type for op in graph.get_operations()]
  return any('Placeholder' in s for s in ops_types)


class CheckpointingTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_optimizer_dependency(self):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, input_shape=(4,)))
    opt = adam.AdamOptimizer(0.01)
    model.compile(optimizer=opt, loss='mse')
    model.fit(x=np.array([[1., 2., 3., 4.]]), y=[1.], epochs=2)
    save_prefix = os.path.join(self.get_temp_dir(), 'ckpt')
    beta1_power, _ = opt._get_beta_accumulators()
    self.evaluate(beta1_power.assign(12.))
    model.save_weights(save_prefix)
    self.evaluate(beta1_power.assign(13.))
    model.load_weights(save_prefix)
    self.assertEqual(12., self.evaluate(beta1_power))


class TestModelBackend(test.TestCase):

  def test_model_backend_float64_use_cases(self):
    # Test case for GitHub issue 19318
    floatx = keras.backend.floatx()
    keras.backend.set_floatx('float64')

    x = keras.Input((5,))
    y = keras.layers.Dense(1)(x)
    model = keras.models.Model(x, y)
    model.compile('rmsprop', 'mse')

    keras.backend.set_floatx(floatx)


class TestModelDeepCopy(test.TestCase):

  def test_deep_copy_eager_mode_trainable(self):
    with context.eager_mode():
      x = random_ops.random_normal((32, 4))
      model = TestModel(trainable=True)
      model(x)  # Initialize Variables.
      model_copy = copy.deepcopy(model)
      self.assertEqual(len(model_copy.trainable_variables), 3)
      model_copy.n_outputs.assign(1200)
      self.assertFalse(
          np.allclose(model_copy.n_outputs.numpy(),
                      model.n_outputs.numpy()))

  def test_deep_copy_eager_mode_not_trainable(self):
    with context.eager_mode():
      x = random_ops.random_normal((32, 4))
      model = TestModel(trainable=False)
      model(x)
      model_copy = copy.deepcopy(model)
      self.assertEqual(len(model_copy.trainable_variables), 2)

      weights = model_copy.get_weights()
      weights = [w * 4 for w in weights]
      model_copy.set_weights(weights)
      self.assertFalse(
          np.allclose(model.get_weights()[0],
                      model_copy.get_weights()[0]))


@test_util.run_v1_only('b/120545219')
class TestCloneAndBuildModel(test.TestCase):

  def test_clone_and_build_non_compiled_model(self):
    with self.cached_session():
      inp = np.random.random((10, 4))
      out = np.random.random((10, 4))

      model = keras.models.Sequential()
      model.add(keras.layers.Dense(4, input_shape=(4,)))
      model.add(keras.layers.BatchNormalization())
      model.add(keras.layers.Dropout(0.5))
      model.add(keras.layers.Dense(4))

    # Everything should work in a new session.
    keras.backend.clear_session()

    with self.cached_session():
      with self.assertRaisesRegexp(ValueError, 'has not been compiled'):
        models.clone_and_build_model(model, compile_clone=True)

      # With placeholder creation
      new_model = models.clone_and_build_model(model, compile_clone=False)
      with self.assertRaisesRegexp(RuntimeError, 'must compile'):
        new_model.evaluate(inp, out)
      with self.assertRaisesRegexp(RuntimeError, 'must compile'):
        new_model.train_on_batch(inp, out)
      new_model.compile('rmsprop', 'mse')
      new_model.train_on_batch(inp, out)

      # Create new tensors for inputs and targets
      input_a = keras.Input(shape=(4,))
      target_a = keras.Input(shape=(4,))
      new_model = models.clone_and_build_model(model, input_tensors=input_a,
                                               target_tensors=[target_a],
                                               compile_clone=False)
      with self.assertRaisesRegexp(RuntimeError, 'must compile'):
        new_model.evaluate(inp, out)
      with self.assertRaisesRegexp(RuntimeError, 'must compile'):
        new_model.train_on_batch(inp, out)
      new_model.compile('rmsprop', 'mse')
      new_model.train_on_batch(inp, out)

  def _assert_same_compile_params(self, model):
    """Assert that two models have the same compile parameters."""

    self.assertEqual('mse', model.loss)
    self.assertTrue(
        isinstance(model.optimizer,
                   (keras.optimizers.RMSprop,
                    keras.optimizer_v2.rmsprop.RMSprop)))
    self.assertEqual(['acc', metrics.categorical_accuracy],
                     model._compile_metrics)

  def _clone_and_build_test_helper(self, model, is_subclassed=False):
    inp = np.random.random((10, 4))
    out = np.random.random((10, 4))

    # Everything should work in a new session.
    keras.backend.clear_session()

    with self.cached_session():
      # With placeholder creation
      new_model = models.clone_and_build_model(
          model, compile_clone=True, in_place_reset=is_subclassed)

      self._assert_same_compile_params(new_model)
      new_model.train_on_batch(inp, out)
      new_model.evaluate(inp, out)

      # Create new tensors for inputs and targets
      input_a = keras.Input(shape=(4,), name='a')
      new_model = models.clone_and_build_model(
          model, input_tensors=input_a, compile_clone=True,
          in_place_reset=is_subclassed)
      self._assert_same_compile_params(new_model)
      new_model.train_on_batch(inp, out)
      new_model.evaluate(inp, out)

      target_a = keras.Input(shape=(4,), name='b')
      new_model = models.clone_and_build_model(
          model, input_tensors=input_a, target_tensors=[target_a],
          compile_clone=True, in_place_reset=is_subclassed)
      self._assert_same_compile_params(new_model)
      new_model.train_on_batch(inp, out)
      new_model.evaluate(inp, out)

  def test_clone_and_build_compiled_sequential_model(self):
    with self.cached_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(4, input_shape=(4,)))
      model.add(keras.layers.BatchNormalization())
      model.add(keras.layers.Dropout(0.5))
      model.add(keras.layers.Dense(4))
      model.compile('rmsprop', 'mse',
                    metrics=['acc', metrics.categorical_accuracy])

    self._clone_and_build_test_helper(model)

  def test_clone_and_build_functional_model(self):
    with self.cached_session():
      input_a = keras.Input(shape=(4,))
      dense_1 = keras.layers.Dense(4,)
      dense_2 = keras.layers.Dense(4,)

      x_a = dense_1(input_a)
      x_a = keras.layers.Dropout(0.5)(x_a)
      x_a = keras.layers.BatchNormalization()(x_a)
      x_a = dense_2(x_a)
      model = keras.models.Model(input_a, x_a)
      model.compile('rmsprop', 'mse',
                    metrics=['acc', metrics.categorical_accuracy])

    self._clone_and_build_test_helper(model)

  def test_clone_and_build_subclassed_model(self):
    class SubclassedModel(keras.Model):

      def __init__(self):
        super(SubclassedModel, self).__init__()
        self.layer1 = keras.layers.Dense(4)
        self.layer2 = keras.layers.Dense(4)

      def call(self, inp):
        out = self.layer1(inp)
        out = keras.layers.BatchNormalization()(out)
        out = keras.layers.Dropout(0.5)(out)
        out = self.layer2(out)
        return out

    with self.cached_session():
      model = SubclassedModel()
      model.compile('rmsprop', 'mse',
                    metrics=['acc', metrics.categorical_accuracy])
    self._clone_and_build_test_helper(model, True)

  def assert_optimizer_iterations_increases(self, optimizer):
    with self.cached_session():
      input_a = keras.Input(shape=(4,))
      dense_1 = keras.layers.Dense(4,)
      dense_2 = keras.layers.Dense(4,)

      x_a = dense_1(input_a)
      x_a = keras.layers.Dropout(0.5)(x_a)
      x_a = keras.layers.BatchNormalization()(x_a)
      x_a = dense_2(x_a)
      model = keras.models.Model(input_a, x_a)
      model.compile(optimizer, 'mse',
                    metrics=['acc', metrics.categorical_accuracy])

      global_step = keras.backend.variable(123, dtype=dtypes.int64)
      clone_model = models.clone_and_build_model(
          model, compile_clone=True, optimizer_iterations=global_step)

      inp = np.random.random((10, 4))
      out = np.random.random((10, 4))
      clone_model.train_on_batch(inp, out)

      self.assertEqual(K.eval(global_step), 124)

  def test_replace_tf_optimizer_iterations_variable(self):
    self.assert_optimizer_iterations_increases(adam.AdamOptimizer(0.01))

  def test_replace_keras_optimizer_iterations_variable(self):
    self.assert_optimizer_iterations_increases(optimizers.Adam())

  def test_clone_and_build_sequential_model_without_inputs_defined(self):
    with self.cached_session():
      model = sequential_model(False, False)
      model.compile('rmsprop', 'mse',
                    metrics=['acc', metrics.categorical_accuracy])
    self._clone_and_build_test_helper(model, False)

    with self.cached_session():
      inp = np.random.random((10, 4))
      out = np.random.random((10, 4))
      model.train_on_batch(inp, out)
    self._clone_and_build_test_helper(model, False)


if __name__ == '__main__':
  test.main()
