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

import functools
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import metrics
from tensorflow.python.keras import models
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
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


def _get_layers(input_shape=(4,), add_input_layer=False):
  if add_input_layer:
    model_layers = [keras.layers.InputLayer(input_shape=input_shape),
                    keras.layers.Dense(4)]
  elif input_shape:
    model_layers = [keras.layers.Dense(4, input_shape=input_shape)]
  else:
    model_layers = [keras.layers.Dense(4)]

  model_layers += [
      keras.layers.BatchNormalization(),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(4)]

  return model_layers


def _get_model(input_shape=(4,)):
  model_layers = _get_layers(input_shape=None, add_input_layer=False)
  return testing_utils.get_model_from_layers(
      model_layers, input_shape=input_shape)


class TestModelCloning(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters([
      {'testcase_name': 'has_input_layer',
       'input_shape': (4,),
       'add_input_layer': True,
       'share_weights': False},
      {'testcase_name': 'no_input_layer',
       'input_shape': None,
       'add_input_layer': False,
       'share_weights': False},
      {'testcase_name': 'has_input_layer_share_weights',
       'input_shape': (4,),
       'add_input_layer': True,
       'share_weights': True},
      {'testcase_name': 'no_input_layer_share_weights',
       'input_shape': None,
       'add_input_layer': False,
       'share_weights': True},
  ])
  def test_clone_sequential_model(
      self, input_shape, add_input_layer, share_weights):

    if share_weights:
      clone_fn = functools.partial(
          keras.models._clone_sequential_model, layer_fn=models.share_weights)
    else:
      clone_fn = keras.models.clone_model

    val_a = np.random.random((10, 4))
    model = models.Sequential(_get_layers(input_shape, add_input_layer))
    # Sanity check
    self.assertEqual(
        isinstance(model._layers[0], keras.layers.InputLayer),
        add_input_layer)
    self.assertEqual(model._is_graph_network, add_input_layer)

    # With placeholder creation -- clone model should have an InputLayer
    # if the original model has one.
    new_model = clone_fn(model)
    self.assertEqual(
        isinstance(new_model._layers[0], keras.layers.InputLayer),
        add_input_layer)
    self.assertEqual(new_model._is_graph_network, model._is_graph_network)
    if input_shape and not ops.executing_eagerly_outside_functions():
      # update ops from batch norm needs to be included
      self.assertGreaterEqual(len(new_model.updates), 2)

    # On top of new tensor  -- clone model should always have an InputLayer.
    input_a = keras.Input(shape=(4,))
    new_model = clone_fn(model, input_tensors=input_a)
    self.assertIsInstance(new_model._layers[0], keras.layers.InputLayer)
    self.assertTrue(new_model._is_graph_network)

    # On top of new, non-Keras tensor  -- clone model should always have an
    # InputLayer.
    if not context.executing_eagerly():
      # TODO(b/121277734):Skip Eager contexts, as Input() layers raise an error
      # saying they should not be used with EagerTensors
      input_a = keras.backend.variable(val_a)
      new_model = clone_fn(model, input_tensors=input_a)
      self.assertIsInstance(new_model._layers[0], keras.layers.InputLayer)
      self.assertTrue(new_model._is_graph_network)

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters([
      {'testcase_name': 'clone_weights', 'share_weights': False},
      {'testcase_name': 'share_weights', 'share_weights': True},
  ])
  def test_clone_functional_model(self, share_weights):
    if share_weights:
      clone_fn = functools.partial(
          keras.models._clone_functional_model, layer_fn=models.share_weights)
    else:
      clone_fn = keras.models.clone_model

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

    # With placeholder creation
    new_model = clone_fn(model)
    if not ops.executing_eagerly_outside_functions():
      self.assertGreaterEqual(len(new_model.updates), 2)
    new_model.compile(
        testing_utils.get_v2_optimizer('rmsprop'),
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    new_model.train_on_batch([val_a, val_b], val_out)

    # On top of new tensors
    input_a = keras.Input(shape=(4,), name='a')
    input_b = keras.Input(shape=(4,), name='b')
    new_model = keras.models.clone_model(
        model, input_tensors=[input_a, input_b])
    if not ops.executing_eagerly_outside_functions():
      self.assertLen(new_model.updates, 2)
    new_model.compile(
        testing_utils.get_v2_optimizer('rmsprop'),
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    new_model.train_on_batch([val_a, val_b], val_out)

    # On top of new, non-Keras tensors
    if not context.executing_eagerly():
      # TODO(b/121277734):Skip Eager contexts, as Input() layers raise an error
      # saying they should not be used with EagerTensors
      input_a = keras.backend.variable(val_a)
      input_b = keras.backend.variable(val_b)
      new_model = clone_fn(model, input_tensors=[input_a, input_b])
      self.assertGreaterEqual(len(new_model.updates), 2)
      new_model.compile(
          testing_utils.get_v2_optimizer('rmsprop'),
          'mse',
          run_eagerly=testing_utils.should_run_eagerly())
      new_model.train_on_batch(None, val_out)

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters([
      {'testcase_name': 'clone_weights', 'share_weights': False},
      {'testcase_name': 'share_weights', 'share_weights': True},
  ])
  def test_clone_functional_with_masking(self, share_weights):
    if share_weights:
      clone_fn = functools.partial(
          keras.models._clone_functional_model, layer_fn=models.share_weights)
    else:
      clone_fn = keras.models.clone_model

    x = np.array([[[1.], [1.]], [[0.], [0.]]])
    inputs = keras.Input((2, 1))
    outputs = keras.layers.Masking(mask_value=0)(inputs)
    outputs = keras.layers.TimeDistributed(
        keras.layers.Dense(1, kernel_initializer='one'))(outputs)
    model = keras.Model(inputs, outputs)

    model = clone_fn(model)
    model.compile(
        loss='mse',
        optimizer=testing_utils.get_v2_optimizer('adam'),
        run_eagerly=testing_utils.should_run_eagerly())
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

  def test_functional_cloning_with_tensor_kwarg(self):
    """Test that cloning works with models that use Tensor kwargs."""

    class LayerWithTensorKwarg(keras.layers.Layer):

      def call(self, inputs, tensor=None):
        if tensor is not None:
          return inputs * math_ops.cast(tensor, dtypes.float32)
        else:
          return inputs

    inputs = keras.layers.Input(shape=(3))
    t = array_ops.sequence_mask(array_ops.shape(inputs)[1])
    model = keras.models.Model(inputs, LayerWithTensorKwarg()(inputs, t))
    model.add_loss(math_ops.reduce_sum(model.outputs))

    input_arr = np.random.random((1, 3)).astype(np.float32)
    with ops.Graph().as_default():
      with self.session() as sess:
        clone = keras.models.clone_model(model)
        self.assertLen(clone.losses, 1)

        loss = sess.run(clone.losses[0], feed_dict={clone.input: input_arr})
        self.assertAllClose(np.sum(input_arr), loss)


def _has_placeholder(graph):
  ops_types = [op.type for op in graph.get_operations()]
  return any('Placeholder' in s for s in ops_types)


class CheckpointingTests(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_optimizer_dependency(self):
    model = _get_model()
    opt = adam.AdamOptimizer(.01)
    model.compile(
        optimizer=opt,
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly())

    model.fit(
        x=np.array([[1., 2., 3., 4.]]),
        y=np.array([[1., 1., 1., 1.]]),
        epochs=2)
    save_prefix = os.path.join(self.get_temp_dir(), 'ckpt')
    beta1_power, _ = opt._get_beta_accumulators()
    self.evaluate(beta1_power.assign(12.))
    model.save_weights(save_prefix)
    self.evaluate(beta1_power.assign(13.))
    model.load_weights(save_prefix)
    self.assertEqual(12., self.evaluate(beta1_power))


@keras_parameterized.run_all_keras_modes
class TestModelBackend(keras_parameterized.TestCase):

  def test_model_backend_float64_use_cases(self):
    # Test case for GitHub issue 19318
    floatx = keras.backend.floatx()
    keras.backend.set_floatx('float64')

    x = keras.Input((5,))
    y = keras.layers.Dense(1)(x)
    model = keras.models.Model(x, y)
    model.compile(
        testing_utils.get_v2_optimizer('rmsprop'),
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())

    keras.backend.set_floatx(floatx)


class TestCloneAndBuildModel(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_clone_and_build_non_compiled_model(self):
    inp = np.random.random((10, 4))
    out = np.random.random((10, 4))

    model = _get_model()

    with self.assertRaisesRegexp(ValueError, 'has not been compiled'):
      models.clone_and_build_model(model, compile_clone=True)

    is_subclassed = (testing_utils.get_model_type() == 'subclass')
    # With placeholder creation
    new_model = models.clone_and_build_model(
        model, compile_clone=False, in_place_reset=is_subclassed)
    with self.assertRaisesRegexp(RuntimeError, 'must compile'):
      new_model.evaluate(inp, out)
    with self.assertRaisesRegexp(RuntimeError, 'must compile'):
      new_model.train_on_batch(inp, out)
    new_model.compile(
        testing_utils.get_v2_optimizer('rmsprop'),
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    new_model.train_on_batch(inp, out)

    # Create new tensors for inputs.
    input_a = keras.Input(shape=(4,))
    new_model = models.clone_and_build_model(
        model,
        input_tensors=input_a,
        compile_clone=False,
        in_place_reset=is_subclassed)
    with self.assertRaisesRegexp(RuntimeError, 'must compile'):
      new_model.evaluate(inp, out)
    with self.assertRaisesRegexp(RuntimeError, 'must compile'):
      new_model.train_on_batch(inp, out)
    new_model.compile(
        testing_utils.get_v2_optimizer('rmsprop'),
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    new_model.train_on_batch(inp, out)

  def _assert_same_compile_params(self, model):
    """Assert that two models have the same compile parameters."""

    self.assertEqual('mse', model.loss)
    self.assertTrue(
        isinstance(model.optimizer,
                   (keras.optimizers.RMSprop,
                    keras.optimizer_v2.rmsprop.RMSprop)))

  def _clone_and_build_test_helper(self, model, model_type):
    inp = np.random.random((10, 4))
    out = np.random.random((10, 4))

    is_subclassed = (model_type == 'subclass')

    # With placeholder creation
    new_model = models.clone_and_build_model(
        model, compile_clone=True, in_place_reset=is_subclassed)

    self._assert_same_compile_params(new_model)
    new_model.train_on_batch(inp, out)
    new_model.evaluate(inp, out)

    # Create new tensors for inputs.
    input_a = keras.Input(shape=(4,), name='a')
    new_model = models.clone_and_build_model(
        model, input_tensors=input_a, compile_clone=True,
        in_place_reset=is_subclassed)
    self._assert_same_compile_params(new_model)
    new_model.train_on_batch(inp, out)
    new_model.evaluate(inp, out)

    new_model = models.clone_and_build_model(
        model,
        input_tensors=input_a,
        target_tensors=None,
        compile_clone=True,
        in_place_reset=is_subclassed)
    self._assert_same_compile_params(new_model)
    new_model.train_on_batch(inp, out)
    new_model.evaluate(inp, out)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_clone_and_build_compiled(self):
    model = _get_model()
    model.compile(
        testing_utils.get_v2_optimizer('rmsprop'),
        'mse',
        metrics=['acc', metrics.categorical_accuracy],
        run_eagerly=testing_utils.should_run_eagerly())

    self._clone_and_build_test_helper(model, testing_utils.get_model_type())

  @keras_parameterized.run_all_keras_modes
  def test_clone_and_build_sequential_without_inputs_defined(self):
    model = models.Sequential(_get_layers(input_shape=None))
    model.compile(
        testing_utils.get_v2_optimizer('rmsprop'),
        'mse',
        metrics=['acc', metrics.categorical_accuracy],
        run_eagerly=testing_utils.should_run_eagerly())
    self._clone_and_build_test_helper(model, 'sequential')

    inp = np.random.random((10, 4))
    out = np.random.random((10, 4))
    model.train_on_batch(inp, out)
    self._clone_and_build_test_helper(model, 'sequential')

  def assert_optimizer_iterations_increases(self, optimizer):
    model = _get_model()
    model.compile(
        optimizer,
        'mse',
        metrics=['acc', metrics.categorical_accuracy],
        run_eagerly=testing_utils.should_run_eagerly())

    global_step = keras.backend.variable(123, dtype=dtypes.int64)
    clone_model = models.clone_and_build_model(
        model, compile_clone=True, optimizer_iterations=global_step,
        in_place_reset=(testing_utils.get_model_type() == 'subclass'))

    inp = np.random.random((10, 4))
    out = np.random.random((10, 4))
    clone_model.train_on_batch(inp, out)

    self.assertEqual(K.eval(global_step), 124)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_replace_tf_optimizer_iterations_variable(self):
    if context.executing_eagerly():
      self.skipTest('v1 optimizers not supported with eager.')
    self.assert_optimizer_iterations_increases(adam.AdamOptimizer(0.01))

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_replace_keras_optimizer_iterations_variable(self):
    self.assert_optimizer_iterations_increases('adam')

  def test_clone_optimizer_in_different_graph(self):
    with ops.Graph().as_default():
      with self.session():
        model = testing_utils.get_small_sequential_mlp(3, 4)
        optimizer = keras.optimizer_v2.adam.Adam()
        model.compile(
            optimizer, 'mse', metrics=['acc', metrics.categorical_accuracy],
            )
        model.fit(
            x=np.array([[1., 2., 3., 4.]]),
            y=np.array([[1., 1., 1., 1.]]),
            epochs=1)
        optimizer_config = optimizer.get_config()
    with ops.Graph().as_default():
      with self.session():
        with self.assertRaisesRegexp(ValueError,
                                     'Cannot use the given session'):
          models.clone_and_build_model(model, compile_clone=True)
        # The optimizer_config object allows the model to be cloned in a
        # different graph.
        models.clone_and_build_model(model, compile_clone=True,
                                     optimizer_config=optimizer_config)


if __name__ == '__main__':
  test.main()
