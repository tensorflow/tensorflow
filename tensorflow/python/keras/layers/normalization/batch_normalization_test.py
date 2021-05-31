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

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers.normalization import batch_normalization
from tensorflow.python.keras.layers.normalization import batch_normalization_v1
from tensorflow.python.ops import array_ops
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
    testing_utils.layer_test(
        keras.layers.BatchNormalization,
        kwargs={
            'gamma_initializer': 'ones',
            'beta_initializer': 'ones',
            'moving_mean_initializer': 'zeros',
            'moving_variance_initializer': 'ones'
        },
        input_shape=(3, 2, 4, 2))

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
      with self.session():
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
        batch_normalization_v1.BatchNormalization, dtype='float32')
    _run_batchnorm_correctness_test(
        batch_normalization.BatchNormalization, dtype='float32')

  @keras_parameterized.run_all_keras_modes
  def test_batchnorm_float16(self):
    _run_batchnorm_correctness_test(
        batch_normalization_v1.BatchNormalization, dtype='float16')
    _run_batchnorm_correctness_test(
        batch_normalization.BatchNormalization, dtype='float16')

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  @testing_utils.enable_v2_dtype_behavior
  def test_batchnorm_mixed_precision(self):
    norm = keras.layers.BatchNormalization(
        axis=-1,
        input_shape=(4, 4, 3),
        momentum=0.8,
        dtype='mixed_float16')
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
        dtype='mixed_float16')
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
    bn = batch_normalization.BatchNormalization()
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

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_eager_batchnorm_in_custom_model_call_with_tf_function(self):

    class MyModel(keras.Model):

      def __init__(self):
        super(MyModel, self).__init__()
        self.bn = keras.layers.BatchNormalization()

      @def_function.function()
      def call(self, x, training):
        return self.bn(x, training=training)

    model = MyModel()

    for _ in range(10):
      x = constant_op.constant(0.5, shape=[1, 1])
      model(x, training=True)

    # Make sure the moving mean and variance have been updated
    self.assertAllClose(model.bn.moving_mean.numpy(), [0.047], atol=3e-3)
    self.assertAllClose(model.bn.moving_variance.numpy(), [0.9], atol=3e-2)

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_bessels_correction(self):
    # Bessel's correction is currently only used in the fused case. In the
    # future, it may be used in the nonfused case as well.

    x = constant_op.constant([0., 2.], shape=[2, 1, 1, 1])
    layer = batch_normalization.BatchNormalization(
        momentum=0.5, moving_variance_initializer='zeros')
    layer(x, training=True)
    self.assertTrue(layer.fused)
    # Since fused is used, Bessel's correction is used. The variance of [0, 2]
    # is 2 with Bessel's correction. Since the momentum is 0.5, the variance is
    # 2 * 0.5 == 1.
    self.assertAllEqual(self.evaluate(layer.moving_variance), [1.])

    x = constant_op.constant([0., 2.], shape=[2, 1, 1, 1, 1])
    layer = batch_normalization.BatchNormalization(
        momentum=0.5, moving_variance_initializer='zeros')
    layer(x, training=True)
    self.assertTrue(layer.fused)
    # Since fused is used, Bessel's correction is used. The variance of [0, 2]
    # is 2 with Bessel's correction. Since the momentum is 0.5, the variance is
    # 2 * 0.5 == 1.
    self.assertAllEqual(self.evaluate(layer.moving_variance), [1.])


class BatchNormalizationV1Test(keras_parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_v1_fused_attribute(self):
    norm = batch_normalization_v1.BatchNormalization()
    inp = keras.layers.Input((4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, True)

    norm = batch_normalization_v1.BatchNormalization(fused=False)
    self.assertEqual(norm.fused, False)
    inp = keras.layers.Input(shape=(4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, False)

    norm = batch_normalization_v1.BatchNormalization(virtual_batch_size=2)
    self.assertEqual(norm.fused, True)
    inp = keras.layers.Input(shape=(2, 2, 2))
    norm(inp)
    self.assertEqual(norm.fused, False)


class BatchNormalizationV2Test(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  def test_basic_batchnorm_v2(self):
    testing_utils.layer_test(
        batch_normalization.BatchNormalization,
        kwargs={'fused': True},
        input_shape=(3, 3, 3, 3))
    testing_utils.layer_test(
        batch_normalization.BatchNormalization,
        kwargs={'fused': None},
        input_shape=(3, 3, 3))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_v2_fused_attribute(self):
    norm = batch_normalization.BatchNormalization()
    self.assertIsNone(norm.fused)
    inp = keras.layers.Input(shape=(4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, True)

    norm = batch_normalization.BatchNormalization()
    self.assertIsNone(norm.fused)
    inp = keras.layers.Input(shape=(4, 4))
    norm(inp)
    self.assertEqual(norm.fused, False)

    norm = batch_normalization.BatchNormalization()
    self.assertIsNone(norm.fused)
    inp = keras.layers.Input(shape=(4, 4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, True)

    norm = batch_normalization.BatchNormalization(virtual_batch_size=2)
    self.assertEqual(norm.fused, False)
    inp = keras.layers.Input(shape=(4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, False)

    norm = batch_normalization.BatchNormalization(fused=False)
    self.assertEqual(norm.fused, False)
    inp = keras.layers.Input(shape=(4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, False)

    norm = batch_normalization.BatchNormalization(fused=True, axis=[3])
    self.assertEqual(norm.fused, True)
    inp = keras.layers.Input(shape=(4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, True)

    with self.assertRaisesRegex(ValueError, 'fused.*renorm'):
      batch_normalization.BatchNormalization(fused=True, renorm=True)

    with self.assertRaisesRegex(ValueError, 'fused.*when axis is 1 or 3'):
      batch_normalization.BatchNormalization(fused=True, axis=2)

    with self.assertRaisesRegex(ValueError, 'fused.*when axis is 1 or 3'):
      batch_normalization.BatchNormalization(fused=True, axis=[1, 3])

    with self.assertRaisesRegex(ValueError, 'fused.*virtual_batch_size'):
      batch_normalization.BatchNormalization(fused=True, virtual_batch_size=2)

    with self.assertRaisesRegex(ValueError, 'fused.*adjustment'):
      batch_normalization.BatchNormalization(
          fused=True, adjustment=lambda _: (1, 0))

    norm = batch_normalization.BatchNormalization(fused=True)
    self.assertEqual(norm.fused, True)
    inp = keras.layers.Input(shape=(4, 4))
    with self.assertRaisesRegex(ValueError, '4D or 5D input tensors'):
      norm(inp)

  def test_updates_in_wrap_function(self):

    def my_func():
      layer = batch_normalization_v1.BatchNormalization()
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
    norm = batch_normalization.BatchNormalization(virtual_batch_size=8)
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


@parameterized.parameters([
    batch_normalization_v1.BatchNormalization,
    batch_normalization.BatchNormalization
])
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


if __name__ == '__main__':
  test.main()
