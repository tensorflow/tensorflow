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
"""Tests for normalization layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import normalization
from tensorflow.python.keras.layers import normalization_v2
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


class BatchNormalizationTest(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  def test_basic_batchnorm(self):
    testing_utils.layer_test(
        keras.layers.BatchNormalization,
        kwargs={
            'momentum': 0.9,
            'epsilon': 0.1,
            'gamma_regularizer': keras.regularizers.l2(0.01),
            'beta_regularizer': keras.regularizers.l2(0.01)
        },
        input_shape=(3, 4, 2))
    testing_utils.layer_test(
        keras.layers.BatchNormalization,
        kwargs={
            'gamma_initializer': 'ones',
            'beta_initializer': 'ones',
            'moving_mean_initializer': 'zeros',
            'moving_variance_initializer': 'ones'
        },
        input_shape=(3, 4, 2))
    testing_utils.layer_test(
        keras.layers.BatchNormalization,
        kwargs={'scale': False,
                'center': False},
        input_shape=(3, 3))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_batchnorm_weights(self):
    layer = keras.layers.BatchNormalization(scale=False, center=False)
    layer.build((None, 3, 4))
    self.assertEqual(len(layer.trainable_weights), 0)
    self.assertEqual(len(layer.weights), 2)

    layer = keras.layers.BatchNormalization()
    layer.build((None, 3, 4))
    self.assertEqual(len(layer.trainable_weights), 2)
    self.assertEqual(len(layer.weights), 4)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_batchnorm_regularization(self):
    layer = keras.layers.BatchNormalization(
        gamma_regularizer='l1', beta_regularizer='l1')
    layer.build((None, 3, 4))
    self.assertEqual(len(layer.losses), 2)
    max_norm = keras.constraints.max_norm
    layer = keras.layers.BatchNormalization(
        gamma_constraint=max_norm, beta_constraint=max_norm)
    layer.build((None, 3, 4))
    self.assertEqual(layer.gamma.constraint, max_norm)
    self.assertEqual(layer.beta.constraint, max_norm)

  @keras_parameterized.run_all_keras_modes
  def test_batchnorm_convnet(self):
    if test.is_gpu_available(cuda_only=True):
      with self.session(use_gpu=True):
        model = keras.models.Sequential()
        norm = keras.layers.BatchNormalization(
            axis=1, input_shape=(3, 4, 4), momentum=0.8)
        model.add(norm)
        model.compile(
            loss='mse',
            optimizer=gradient_descent.GradientDescentOptimizer(0.01),
            run_eagerly=testing_utils.should_run_eagerly())

        # centered on 5.0, variance 10.0
        x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 3, 4, 4))
        model.fit(x, x, epochs=4, verbose=0)
        out = model.predict(x)
        out -= np.reshape(keras.backend.eval(norm.beta), (1, 3, 1, 1))
        out /= np.reshape(keras.backend.eval(norm.gamma), (1, 3, 1, 1))

        np.testing.assert_allclose(np.mean(out, axis=(0, 2, 3)), 0.0, atol=1e-1)
        np.testing.assert_allclose(np.std(out, axis=(0, 2, 3)), 1.0, atol=1e-1)

  @keras_parameterized.run_all_keras_modes
  def test_batchnorm_convnet_channel_last(self):
    model = keras.models.Sequential()
    norm = keras.layers.BatchNormalization(
        axis=-1, input_shape=(4, 4, 3), momentum=0.8)
    model.add(norm)
    model.compile(
        loss='mse',
        optimizer=gradient_descent.GradientDescentOptimizer(0.01),
        run_eagerly=testing_utils.should_run_eagerly())

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 4, 4, 3))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)
    out -= np.reshape(keras.backend.eval(norm.beta), (1, 1, 1, 3))
    out /= np.reshape(keras.backend.eval(norm.gamma), (1, 1, 1, 3))

    np.testing.assert_allclose(np.mean(out, axis=(0, 1, 2)), 0.0, atol=1e-1)
    np.testing.assert_allclose(np.std(out, axis=(0, 1, 2)), 1.0, atol=1e-1)

  @keras_parameterized.run_all_keras_modes
  def test_batchnorm_correctness(self):
    _run_batchnorm_correctness_test(
        normalization.BatchNormalization, dtype='float32')
    _run_batchnorm_correctness_test(
        normalization_v2.BatchNormalization, dtype='float32')

  @keras_parameterized.run_all_keras_modes
  def test_batchnorm_float16(self):
    _run_batchnorm_correctness_test(
        normalization.BatchNormalization, dtype='float16')
    _run_batchnorm_correctness_test(
        normalization_v2.BatchNormalization, dtype='float16')

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  @testing_utils.enable_v2_dtype_behavior
  def test_batchnorm_mixed_precision(self):
    norm = keras.layers.BatchNormalization(
        axis=-1,
        input_shape=(4, 4, 3),
        momentum=0.8,
        dtype=policy.Policy('mixed_float16'))
    x = np.random.normal(size=(10, 4, 4, 3))
    y = norm(x)
    self.assertEqual(y.dtype, 'float16')
    self.assertEqual(norm.beta.dtype.base_dtype, 'float32')
    self.assertEqual(norm.gamma.dtype.base_dtype, 'float32')

  @combinations.generate(combinations.combine(mode=['graph', 'eager'],
                                              fused=[True, False]))
  @testing_utils.enable_v2_dtype_behavior
  def test_batchnorm_mixed_precision_does_not_overflow(self, fused):
    norm = keras.layers.BatchNormalization(
        axis=-1,
        input_shape=(1, 1, 1),
        fused=fused,
        dtype=policy.Policy('mixed_float16'))
    x = np.array([-1000., 1000.]).reshape((2, 1, 1, 1))
    y = norm(x, training=True)
    expected_y = np.array([-1.0, 1.0]).reshape((2, 1, 1, 1))
    self.assertAllClose(keras.backend.eval(y), expected_y)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_batchnorm_non_trainable_with_fit(self):
    # We use the same data shape for all the data we use in this test.
    # This will prevent any used tf.functions from retracing.
    # This helps us verify that changing trainable and recompiling really
    # does update the training loop, rather than a different data shape
    # triggering a retrace.
    data_shape = (100, 3)

    inputs = keras.Input((3,))
    bn = normalization_v2.BatchNormalization()
    outputs = bn(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(
        'rmsprop',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    model.fit(np.random.random(data_shape), np.random.random(data_shape))

    test_data = np.random.random(data_shape)
    test_targets = np.random.random(data_shape)
    test_loss = model.evaluate(test_data, test_targets)

    bn.trainable = False
    model.compile(
        'rmsprop',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    train_loss = model.train_on_batch(test_data, test_targets)
    self.assertAlmostEqual(test_loss, train_loss)

  def test_eager_batchnorm_in_custom_model_call_with_tf_function(self):

    class MyModel(keras.Model):

      def __init__(self):
        super(MyModel, self).__init__()
        self.bn = keras.layers.BatchNormalization()

      @def_function.function()
      def call(self, x, training):
        return self.bn(x, training=training)

    with context.eager_mode():
      model = MyModel()

      for _ in range(10):
        x = constant_op.constant(0.5, shape=[1, 1])
        model(x, training=True)

      # Make sure the moving mean and variance have been updated
      self.assertAllClose(model.bn.moving_mean.numpy(), [0.047], atol=3e-3)
      self.assertAllClose(model.bn.moving_variance.numpy(), [0.9], atol=3e-2)


class BatchNormalizationV1Test(keras_parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_v1_fused_attribute(self):
    norm = normalization.BatchNormalization()
    inp = keras.layers.Input((4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, True)

    norm = normalization.BatchNormalization(fused=False)
    self.assertEqual(norm.fused, False)
    inp = keras.layers.Input(shape=(4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, False)

    norm = normalization.BatchNormalization(virtual_batch_size=2)
    self.assertEqual(norm.fused, True)
    inp = keras.layers.Input(shape=(2, 2, 2))
    norm(inp)
    self.assertEqual(norm.fused, False)


class BatchNormalizationV2Test(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  def test_basic_batchnorm_v2(self):
    testing_utils.layer_test(
        normalization_v2.BatchNormalization,
        kwargs={'fused': True},
        input_shape=(3, 3, 3, 3))
    testing_utils.layer_test(
        normalization_v2.BatchNormalization,
        kwargs={'fused': None},
        input_shape=(3, 3, 3))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_v2_fused_attribute(self):
    norm = normalization_v2.BatchNormalization()
    self.assertEqual(norm.fused, None)
    inp = keras.layers.Input(shape=(4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, True)

    norm = normalization_v2.BatchNormalization()
    self.assertEqual(norm.fused, None)
    inp = keras.layers.Input(shape=(4, 4))
    norm(inp)
    self.assertEqual(norm.fused, False)

    norm = normalization_v2.BatchNormalization(virtual_batch_size=2)
    self.assertEqual(norm.fused, False)
    inp = keras.layers.Input(shape=(4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, False)

    norm = normalization_v2.BatchNormalization(fused=False)
    self.assertEqual(norm.fused, False)
    inp = keras.layers.Input(shape=(4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, False)

    norm = normalization_v2.BatchNormalization(fused=True, axis=[3])
    self.assertEqual(norm.fused, True)
    inp = keras.layers.Input(shape=(4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, True)

    with self.assertRaisesRegex(ValueError, 'fused.*renorm'):
      normalization_v2.BatchNormalization(fused=True, renorm=True)

    with self.assertRaisesRegex(ValueError, 'fused.*when axis is 1 or 3'):
      normalization_v2.BatchNormalization(fused=True, axis=2)

    with self.assertRaisesRegex(ValueError, 'fused.*when axis is 1 or 3'):
      normalization_v2.BatchNormalization(fused=True, axis=[1, 3])

    with self.assertRaisesRegex(ValueError, 'fused.*virtual_batch_size'):
      normalization_v2.BatchNormalization(fused=True, virtual_batch_size=2)

    with self.assertRaisesRegex(ValueError, 'fused.*adjustment'):
      normalization_v2.BatchNormalization(fused=True,
                                          adjustment=lambda _: (1, 0))

    norm = normalization_v2.BatchNormalization(fused=True)
    self.assertEqual(norm.fused, True)
    inp = keras.layers.Input(shape=(4, 4))
    with self.assertRaisesRegex(ValueError, '4D input tensors'):
      norm(inp)

  def test_updates_in_wrap_function(self):

    def my_func():
      layer = normalization.BatchNormalization()
      x = array_ops.ones((10, 1))
      y = layer(x, training=True)
      # Updates should be tracked in a `wrap_function`.
      self.assertLen(layer.updates, 2)
      return y

    wrapped_fn = wrap_function.wrap_function(my_func, [])
    wrapped_fn()

  @keras_parameterized.run_all_keras_modes
  def test_basic_batchnorm_v2_none_shape_and_virtual_batch_size(self):
    # Test case for GitHub issue for 32380
    norm = normalization_v2.BatchNormalization(virtual_batch_size=8)
    inp = keras.layers.Input(shape=(None, None, 3))
    _ = norm(inp)


def _run_batchnorm_correctness_test(layer, dtype='float32', fused=False):
  model = keras.models.Sequential()
  model.add(keras.Input(shape=(2, 2, 2), dtype=dtype))
  norm = layer(momentum=0.8, fused=fused)
  model.add(norm)
  if dtype == 'float16':
    # Keras models require float32 losses.
    model.add(keras.layers.Lambda(lambda x: keras.backend.cast(x, 'float32')))
  model.compile(
      loss='mse',
      optimizer=gradient_descent.GradientDescentOptimizer(0.01),
      run_eagerly=testing_utils.should_run_eagerly())

  # centered on 5.0, variance 10.0
  x = (np.random.normal(loc=5.0, scale=10.0, size=(1000, 2, 2, 2))
       .astype(dtype))
  model.fit(x, x, epochs=4, verbose=0)
  out = model.predict(x)
  out -= keras.backend.eval(norm.beta)
  out /= keras.backend.eval(norm.gamma)

  np.testing.assert_allclose(out.mean(), 0.0, atol=2e-1)
  np.testing.assert_allclose(out.std(), 1.0, atol=2e-1)


@parameterized.parameters(
    [normalization.BatchNormalization, normalization_v2.BatchNormalization])
class NormalizationLayersGraphModeOnlyTest(
    test.TestCase, parameterized.TestCase):

  def test_shared_batchnorm(self, layer):
    """Test that a BN layer can be shared across different data streams."""
    with self.cached_session():
      # Test single layer reuse
      bn = layer()
      x1 = keras.layers.Input(shape=(10,))
      _ = bn(x1)

      x2 = keras.layers.Input(shape=(10,))
      y2 = bn(x2)

      x = np.random.normal(loc=5.0, scale=10.0, size=(2, 10))
      model = keras.models.Model(x2, y2)

      model.compile(gradient_descent.GradientDescentOptimizer(0.01), 'mse')
      model.train_on_batch(x, x)

      # Test model-level reuse
      x3 = keras.layers.Input(shape=(10,))
      y3 = model(x3)
      new_model = keras.models.Model(x3, y3, name='new_model')

      new_model.compile(gradient_descent.GradientDescentOptimizer(0.01), 'mse')
      new_model.train_on_batch(x, x)

  def test_that_trainable_disables_updates(self, layer):
    with self.cached_session():
      val_a = np.random.random((10, 4))
      val_out = np.random.random((10, 4))

      a = keras.layers.Input(shape=(4,))
      layer = layer(input_shape=(4,))
      b = layer(a)
      model = keras.models.Model(a, b)

      model.trainable = False
      model.compile(gradient_descent.GradientDescentOptimizer(0.01), 'mse')

      x1 = model.predict(val_a)
      model.train_on_batch(val_a, val_out)
      x2 = model.predict(val_a)
      self.assertAllClose(x1, x2, atol=1e-7)

      model.trainable = True
      model.compile(gradient_descent.GradientDescentOptimizer(0.01), 'mse')

      model.train_on_batch(val_a, val_out)
      x2 = model.predict(val_a)
      assert np.abs(np.sum(x1 - x2)) > 1e-5

      layer.trainable = False
      model.compile(gradient_descent.GradientDescentOptimizer(0.01), 'mse')

      x1 = model.predict(val_a)
      model.train_on_batch(val_a, val_out)
      x2 = model.predict(val_a)
      self.assertAllClose(x1, x2, atol=1e-7)

  def test_batchnorm_trainable(self, layer):
    """Tests that batchnorm layer is trainable when learning phase is enabled.

    Computes mean and std for current inputs then
    applies batch normalization using them.

    Args:
      layer: Either V1 or V2 of BatchNormalization layer.
    """
    # TODO(fchollet): enable in all execution modes when issue with
    # learning phase setting is resolved.
    with ops.Graph().as_default(), self.cached_session():
      bn_mean = 0.5
      bn_std = 10.
      val_a = np.expand_dims(np.arange(10.), axis=1)

      def get_model(bn_mean, bn_std):
        inp = keras.layers.Input(shape=(1,))
        x = layer()(inp)
        model1 = keras.models.Model(inp, x)
        model1.set_weights([
            np.array([1.]),
            np.array([0.]),
            np.array([bn_mean]),
            np.array([bn_std**2])
        ])
        return model1

      # Simulates training-mode with trainable layer.
      # Should use mini-batch statistics.
      with keras.backend.learning_phase_scope(1):
        model = get_model(bn_mean, bn_std)
        model.compile(loss='mse', optimizer='rmsprop')
        out = model.predict(val_a)
        self.assertAllClose(
            (val_a - np.mean(val_a)) / np.std(val_a), out, atol=1e-3)


def _run_layernorm_correctness_test(layer, dtype='float32'):
  model = keras.models.Sequential()
  model.add(keras.layers.Lambda(lambda x: math_ops.cast(x, dtype='float16')))
  norm = layer(input_shape=(2, 2, 2), dtype=dtype)
  model.add(norm)
  model.compile(
      loss='mse',
      optimizer=gradient_descent.GradientDescentOptimizer(0.01),
      run_eagerly=testing_utils.should_run_eagerly())

  # centered on 5.0, variance 10.0
  x = (np.random.normal(loc=5.0, scale=10.0, size=(1000, 2, 2, 2))
       .astype(dtype))
  model.fit(x, x, epochs=4, verbose=0)
  out = model.predict(x)
  out -= keras.backend.eval(norm.beta)
  out /= keras.backend.eval(norm.gamma)

  np.testing.assert_allclose(out.mean(), 0.0, atol=1e-1)
  np.testing.assert_allclose(out.std(), 1.0, atol=1e-1)


class LayerNormalizationTest(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  def test_basic_layernorm(self):
    testing_utils.layer_test(
        keras.layers.LayerNormalization,
        kwargs={
            'gamma_regularizer': keras.regularizers.l2(0.01),
            'beta_regularizer': keras.regularizers.l2(0.01)
        },
        input_shape=(3, 4, 2))
    testing_utils.layer_test(
        keras.layers.LayerNormalization,
        kwargs={
            'gamma_initializer': 'ones',
            'beta_initializer': 'ones',
        },
        input_shape=(3, 4, 2))
    testing_utils.layer_test(
        keras.layers.LayerNormalization,
        kwargs={'scale': False,
                'center': False},
        input_shape=(3, 3))
    testing_utils.layer_test(
        keras.layers.LayerNormalization,
        kwargs={'axis': (-3, -2, -1)},
        input_shape=(2, 8, 8, 3))

  @keras_parameterized.run_all_keras_modes
  def test_non_fused_layernorm(self):
    testing_utils.layer_test(
        keras.layers.LayerNormalization,
        kwargs={'axis': -2},
        input_shape=(3, 4, 2))
    testing_utils.layer_test(
        keras.layers.LayerNormalization,
        kwargs={'axis': (-3, -2)},
        input_shape=(2, 8, 8, 3))
    testing_utils.layer_test(
        keras.layers.LayerNormalization,
        kwargs={'axis': (-3, -1)},
        input_shape=(2, 8, 8, 3))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_layernorm_weights(self):
    layer = keras.layers.LayerNormalization(scale=False, center=False)
    layer.build((None, 3, 4))
    self.assertEqual(len(layer.trainable_weights), 0)
    self.assertEqual(len(layer.weights), 0)

    layer = keras.layers.LayerNormalization()
    layer.build((None, 3, 4))
    self.assertEqual(len(layer.trainable_weights), 2)
    self.assertEqual(len(layer.weights), 2)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_layernorm_regularization(self):
    layer = keras.layers.LayerNormalization(
        gamma_regularizer='l1', beta_regularizer='l1')
    layer.build((None, 3, 4))
    self.assertEqual(len(layer.losses), 2)
    max_norm = keras.constraints.max_norm
    layer = keras.layers.LayerNormalization(
        gamma_constraint=max_norm, beta_constraint=max_norm)
    layer.build((None, 3, 4))
    self.assertEqual(layer.gamma.constraint, max_norm)
    self.assertEqual(layer.beta.constraint, max_norm)

  @keras_parameterized.run_all_keras_modes
  def test_layernorm_convnet_channel_last(self):
    model = keras.models.Sequential()
    norm = keras.layers.LayerNormalization(input_shape=(4, 4, 3))
    model.add(norm)
    model.compile(
        loss='mse',
        optimizer=gradient_descent.GradientDescentOptimizer(0.01),
        run_eagerly=testing_utils.should_run_eagerly())

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 4, 4, 3))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)
    out -= np.reshape(keras.backend.eval(norm.beta), (1, 1, 1, 3))
    out /= np.reshape(keras.backend.eval(norm.gamma), (1, 1, 1, 3))

    np.testing.assert_allclose(np.mean(out, axis=(0, 1, 2)), 0.0, atol=1e-1)
    np.testing.assert_allclose(np.std(out, axis=(0, 1, 2)), 1.0, atol=1e-1)

  @keras_parameterized.run_all_keras_modes
  def test_layernorm_correctness(self):
    _run_layernorm_correctness_test(
        normalization.LayerNormalization, dtype='float32')

  @keras_parameterized.run_all_keras_modes
  def test_layernorm_mixed_precision(self):
    _run_layernorm_correctness_test(
        normalization.LayerNormalization, dtype='float16')

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testIncorrectAxisType(self):
    with self.assertRaisesRegex(TypeError,
                                r'Expected an int or a list/tuple of ints'):
      _ = normalization.LayerNormalization(axis={'axis': -1})

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInvalidAxis(self):
    with self.assertRaisesRegex(ValueError, r'Invalid axis: 3'):
      layer_norm = normalization.LayerNormalization(axis=3)
      layer_norm.build(input_shape=(2, 2, 2))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testDuplicateAxis(self):
    with self.assertRaisesRegex(ValueError, r'Duplicate axis:'):
      layer_norm = normalization.LayerNormalization(axis=[-1, -1])
      layer_norm.build(input_shape=(2, 2, 2))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testFusedAttr(self):
    layer_norm = normalization.LayerNormalization(axis=[-2, -1])
    layer_norm.build(input_shape=(2, 2, 2))
    self.assertEqual(layer_norm._fused, True)


class LayerNormalizationNumericsTest(keras_parameterized.TestCase):
  """Tests LayerNormalization has correct and numerically stable outputs."""

  def _expected_layer_norm(self, x, beta, gamma, batch_input_shape, axis,
                           epsilon):
    """Returns the layer norm, which is computed using NumPy."""
    broadcast_shape = [batch_input_shape[i] if i in axis else 1
                       for i in range(len(batch_input_shape))]
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    expected = (x - mean) / np.sqrt(var + epsilon)
    expected *= np.reshape(gamma, broadcast_shape)
    expected += np.reshape(beta, broadcast_shape)
    return expected

  def _test_forward_pass(self, batch_input_shape, axis, fp64_tol=1e-14,
                         fp32_tol=1e-6, fp16_tol=1e-2):
    """Tests the forward pass of layer normalization.

    Args:
      batch_input_shape: The input shape that will be used to test, including
        the batch dimension.
      axis: A list of axises to normalize. Will be passed to the `axis` argument
        of LayerNormalization.
      fp64_tol: The relative and absolute tolerance for float64.
      fp32_tol: The relative and absolute tolerance for float32.
      fp16_tol: The relative and absolute tolerance for float16.
    """
    param_shape = [batch_input_shape[i] for i in axis]
    param_elems = 1
    for dim in param_shape:
      param_elems *= dim
    beta = np.arange(param_elems, dtype='float64').reshape(param_shape)
    gamma = np.arange(1, param_elems + 1, dtype='float64').reshape(param_shape)
    x = np.random.normal(size=batch_input_shape)

    for epsilon in 1e-12, 1e-3:
      expected = self._expected_layer_norm(x, beta, gamma, batch_input_shape,
                                           axis, epsilon)
      for dtype in 'float64', 'float32', 'float16':
        norm = normalization.LayerNormalization(
            axis=axis, dtype=dtype, batch_input_shape=batch_input_shape,
            epsilon=epsilon, beta_initializer=keras.initializers.constant(beta),
            gamma_initializer=keras.initializers.constant(gamma))
        y = norm(keras.backend.cast(x, dtype))
        actual = keras.backend.eval(y)

        if dtype == 'float64':
          tol = fp64_tol
        elif dtype == 'float32':
          tol = fp32_tol
        else:
          assert dtype == 'float16'
          tol = fp16_tol

        # We use absolute tolerances in addition to relative tolerances, because
        # some of the values are very close to zero.
        self.assertAllClose(expected, actual, rtol=tol, atol=tol)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_forward(self):
    # For numeric stability, we ensure the axis's dimension(s) have at least 4
    # elements.
    self._test_forward_pass((4, 3), (0,))
    self._test_forward_pass((3, 4), (1,))
    self._test_forward_pass((4, 3, 2), (0,))
    self._test_forward_pass((2, 4, 2), (1,))
    self._test_forward_pass((2, 3, 4), (2,), fp16_tol=5e-2)
    self._test_forward_pass((2, 3, 2), (0, 2))
    self._test_forward_pass((2, 2, 2, 2), (1, 3))
    self._test_forward_pass((2, 2, 2, 2), (2, 3))
    self._test_forward_pass((2, 3, 4, 5), (3,))

  def _test_backward_pass(self, batch_input_shape, axis, fp64_tol=1e-5,
                          fp32_tol=1e-5, fp16_tol=2e-2):
    """Tests the backwards pass of layer normalization.

    Args:
      batch_input_shape: The input shape that will be used to test, including
        the batch dimension.
      axis: A list of axises to normalize. Will be passed to the `axis` argument
        of LayerNormalization.
      fp64_tol: The relative and absolute tolerance for float64.
      fp32_tol: The relative and absolute tolerance for float32.
      fp16_tol: The relative and absolute tolerance for float16.
    """
    param_shape = [batch_input_shape[i] for i in axis]
    param_elems = 1
    for dim in param_shape:
      param_elems *= dim
    beta = np.arange(param_elems, dtype='float64').reshape(param_shape)
    gamma = np.arange(1, param_elems + 1, dtype='float64').reshape(param_shape)
    x = np.random.normal(size=batch_input_shape)

    for epsilon in 1e-12, 1e-3:
      # Float64 must come first in this list, as we use the float64 numerical
      # gradients to compare to the float32 and float16 symbolic gradients as
      # well. Computing float32/float16 numerical gradients is too numerically
      # unstable.
      for dtype in 'float64', 'float32', 'float16':
        norm = normalization.LayerNormalization(
            axis=axis, dtype=dtype, batch_input_shape=batch_input_shape,
            epsilon=epsilon, beta_initializer=keras.initializers.constant(beta),
            gamma_initializer=keras.initializers.constant(gamma))
        norm.build(x.shape)

        # pylint: disable=cell-var-from-loop
        def forward_fn(x, beta, gamma):
          # We must monkey-patch the attributes of `norm` with the function
          # arguments, so that the gradient checker will properly compute their
          # gradients. The gradient checker computes gradients with respect to
          # the input arguments of `f`.
          with test.mock.patch.object(norm, 'beta', beta):
            with test.mock.patch.object(norm, 'gamma', gamma):
              return norm(x)
        # pylint: enable=cell-var-from-loop
        results = gradient_checker_v2.compute_gradient(
            forward_fn, [keras.backend.cast(x, dtype), norm.beta, norm.gamma])
        ([x_grad_t, beta_grad_t, gamma_grad_t],
         [x_grad_n, beta_grad_n, gamma_grad_n]) = results

        if dtype == 'float64':
          # We use the float64 numeric gradients as the reference, to compare
          # against the symbolic gradients for all dtypes.
          x_grad_ref = x_grad_n
          beta_grad_ref = beta_grad_n
          gamma_grad_ref = gamma_grad_n
          tol = fp64_tol
        elif dtype == 'float32':
          tol = fp32_tol
        else:
          assert dtype == 'float16'
          tol = fp16_tol

        # We use absolute tolerances in addition to relative tolerances, because
        # some of the values are very close to zero.
        self.assertAllClose(x_grad_t, x_grad_ref, rtol=tol, atol=tol)
        self.assertAllClose(beta_grad_t, beta_grad_ref, rtol=tol, atol=tol)
        self.assertAllClose(gamma_grad_t, gamma_grad_ref, rtol=tol, atol=tol)

  # The gradient_checker_v2 does not work properly with LayerNorm in graph mode.
  @tf_test_util.run_v2_only
  def test_backward(self):
    # For numeric stability, we ensure the axis's dimension(s) have at least 4
    # elements.
    self._test_backward_pass((4, 3), (0,))
    self._test_backward_pass((2, 4, 2), (1,))
    self._test_backward_pass((2, 3, 4), (2,))
    self._test_backward_pass((2, 3, 2), (0, 2), fp64_tol=5e-4, fp32_tol=5e-4)
    self._test_backward_pass((2, 2, 2, 2), (1, 3))
    self._test_backward_pass((2, 2, 2, 2), (2, 3))


if __name__ == '__main__':
  test.main()
