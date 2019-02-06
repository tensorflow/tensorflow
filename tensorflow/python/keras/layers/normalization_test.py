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

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import normalization
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

  @tf_test_util.run_in_graph_and_eager_modes
  def test_batchnorm_weights(self):
    layer = keras.layers.BatchNormalization(scale=False, center=False)
    layer.build((None, 3, 4))
    self.assertEqual(len(layer.trainable_weights), 0)
    self.assertEqual(len(layer.weights), 2)

    layer = keras.layers.BatchNormalization()
    layer.build((None, 3, 4))
    self.assertEqual(len(layer.trainable_weights), 2)
    self.assertEqual(len(layer.weights), 4)

  @tf_test_util.run_in_graph_and_eager_modes
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
        model.compile(loss='mse',
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
    model.compile(loss='mse',
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
        normalization.BatchNormalization, dtype='float32', fused=True)
    _run_batchnorm_correctness_test(
        normalization.BatchNormalization, dtype='float32', fused=False)

  @keras_parameterized.run_all_keras_modes
  def test_batchnorm_mixed_precision(self):
    _run_batchnorm_correctness_test(
        normalization.BatchNormalization, dtype='float16')
    _run_batchnorm_correctness_test(
        normalization.BatchNormalization, dtype='float16', fused=True)
    _run_batchnorm_correctness_test(
        normalization.BatchNormalization, dtype='float16', fused=False)


class BatchNormalizationV1Test(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes
  def test_v1_fused_attribute(self):
    norm = normalization.BatchNormalizationV1()
    inp = keras.layers.Input((4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, True)

    norm = normalization.BatchNormalizationV1(fused=False)
    self.assertEqual(norm.fused, False)
    inp = keras.layers.Input(shape=(4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, False)

    norm = normalization.BatchNormalizationV1(virtual_batch_size=2)
    self.assertEqual(norm.fused, True)
    inp = keras.layers.Input(shape=(2, 2, 2))
    norm(inp)
    self.assertEqual(norm.fused, False)


class BatchNormalizationV2Test(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  def test_basic_batchnorm_v2(self):
    testing_utils.layer_test(
        normalization.BatchNormalizationV2,
        kwargs={'fused': True},
        input_shape=(3, 3, 3, 3))
    testing_utils.layer_test(
        normalization.BatchNormalizationV2,
        kwargs={'fused': None},
        input_shape=(3, 3, 3))

  @tf_test_util.run_in_graph_and_eager_modes
  def test_v2_fused_attribute(self):
    norm = normalization.BatchNormalizationV2()
    self.assertEqual(norm.fused, None)
    inp = keras.layers.Input(shape=(4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, True)

    norm = normalization.BatchNormalizationV2()
    self.assertEqual(norm.fused, None)
    inp = keras.layers.Input(shape=(4, 4))
    norm(inp)
    self.assertEqual(norm.fused, False)

    norm = normalization.BatchNormalizationV2(virtual_batch_size=2)
    self.assertEqual(norm.fused, False)
    inp = keras.layers.Input(shape=(4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, False)

    norm = normalization.BatchNormalizationV2(fused=False)
    self.assertEqual(norm.fused, False)
    inp = keras.layers.Input(shape=(4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, False)

    norm = normalization.BatchNormalizationV2(fused=True, axis=[3])
    self.assertEqual(norm.fused, True)
    inp = keras.layers.Input(shape=(4, 4, 4))
    norm(inp)
    self.assertEqual(norm.fused, True)

    with self.assertRaisesRegexp(ValueError, 'fused.*renorm'):
      normalization.BatchNormalizationV2(fused=True, renorm=True)

    with self.assertRaisesRegexp(ValueError, 'fused.*when axis is 1 or 3'):
      normalization.BatchNormalizationV2(fused=True, axis=2)

    with self.assertRaisesRegexp(ValueError, 'fused.*when axis is 1 or 3'):
      normalization.BatchNormalizationV2(fused=True, axis=[1, 3])

    with self.assertRaisesRegexp(ValueError, 'fused.*virtual_batch_size'):
      normalization.BatchNormalizationV2(fused=True, virtual_batch_size=2)

    with self.assertRaisesRegexp(ValueError, 'fused.*adjustment'):
      normalization.BatchNormalizationV2(fused=True,
                                         adjustment=lambda _: (1, 0))

    norm = normalization.BatchNormalizationV2(fused=True)
    self.assertEqual(norm.fused, True)
    inp = keras.layers.Input(shape=(4, 4))
    with self.assertRaisesRegexp(ValueError, '4D input tensors'):
      norm(inp)


def _run_batchnorm_correctness_test(layer, dtype='float32', fused=False):
  model = keras.models.Sequential()
  norm = layer(input_shape=(2, 2, 2), momentum=0.8, fused=fused)
  model.add(norm)
  model.compile(loss='mse',
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


class NormalizationLayersGraphModeOnlyTest(test.TestCase):

  def test_shared_batchnorm(self):
    """Test that a BN layer can be shared across different data streams.
    """
    with self.cached_session():
      # Test single layer reuse
      bn = keras.layers.BatchNormalization()
      x1 = keras.layers.Input(shape=(10,))
      _ = bn(x1)

      x2 = keras.layers.Input(shape=(10,))
      y2 = bn(x2)

      x = np.random.normal(loc=5.0, scale=10.0, size=(2, 10))
      model = keras.models.Model(x2, y2)

      model.compile(gradient_descent.GradientDescentOptimizer(0.01), 'mse')
      model.train_on_batch(x, x)

      self.assertEqual(len(bn.updates), 4)
      self.assertEqual(len(model.updates), 2)
      self.assertEqual(len(model.get_updates_for(x1)), 0)
      self.assertEqual(len(model.get_updates_for(x2)), 2)

      # Test model-level reuse
      x3 = keras.layers.Input(shape=(10,))
      y3 = model(x3)
      new_model = keras.models.Model(x3, y3, name='new_model')

      self.assertEqual(len(new_model.updates), 2)
      self.assertEqual(len(model.updates), 4)
      self.assertEqual(len(new_model.get_updates_for(x3)), 2)
      new_model.compile(gradient_descent.GradientDescentOptimizer(0.01), 'mse')
      new_model.train_on_batch(x, x)

  def test_that_trainable_disables_updates(self):
    with self.cached_session():
      val_a = np.random.random((10, 4))
      val_out = np.random.random((10, 4))

      a = keras.layers.Input(shape=(4,))
      layer = keras.layers.BatchNormalization(input_shape=(4,))
      b = layer(a)
      model = keras.models.Model(a, b)

      model.trainable = False
      assert not model.updates

      model.compile(gradient_descent.GradientDescentOptimizer(0.01), 'mse')
      assert not model.updates

      x1 = model.predict(val_a)
      model.train_on_batch(val_a, val_out)
      x2 = model.predict(val_a)
      self.assertAllClose(x1, x2, atol=1e-7)

      model.trainable = True
      model.compile(gradient_descent.GradientDescentOptimizer(0.01), 'mse')
      assert model.updates

      model.train_on_batch(val_a, val_out)
      x2 = model.predict(val_a)
      assert np.abs(np.sum(x1 - x2)) > 1e-5

      layer.trainable = False
      model.compile(gradient_descent.GradientDescentOptimizer(0.01), 'mse')
      assert not model.updates

      x1 = model.predict(val_a)
      model.train_on_batch(val_a, val_out)
      x2 = model.predict(val_a)
      self.assertAllClose(x1, x2, atol=1e-7)

  @tf_test_util.run_deprecated_v1
  def test_batchnorm_trainable(self):
    """Tests that batchnorm layer is trainable when learning phase is enabled.

    Computes mean and std for current inputs then
    applies batch normalization using them.
    """
    # TODO(fchollet): enable in all execution modes when issue with
    # learning phase setting is resolved.
    with self.cached_session():
      bn_mean = 0.5
      bn_std = 10.
      val_a = np.expand_dims(np.arange(10.), axis=1)

      def get_model(bn_mean, bn_std):
        inp = keras.layers.Input(shape=(1,))
        x = keras.layers.BatchNormalization()(inp)
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
  norm = layer(input_shape=(2, 2, 2))
  model.add(norm)
  model.compile(loss='mse',
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

  @tf_test_util.run_in_graph_and_eager_modes
  def test_layernorm_weights(self):
    layer = keras.layers.LayerNormalization(scale=False, center=False)
    layer.build((None, 3, 4))
    self.assertEqual(len(layer.trainable_weights), 0)
    self.assertEqual(len(layer.weights), 0)

    layer = keras.layers.LayerNormalization()
    layer.build((None, 3, 4))
    self.assertEqual(len(layer.trainable_weights), 2)
    self.assertEqual(len(layer.weights), 2)

  @tf_test_util.run_in_graph_and_eager_modes
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
  def test_layernorm_convnet(self):
    if test.is_gpu_available(cuda_only=True):
      with self.session(use_gpu=True):
        model = keras.models.Sequential()
        norm = keras.layers.LayerNormalization(
            input_shape=(3, 4, 4), params_axis=1)
        model.add(norm)
        model.compile(loss='mse',
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
  def test_layernorm_convnet_channel_last(self):
    model = keras.models.Sequential()
    norm = keras.layers.LayerNormalization(input_shape=(4, 4, 3))
    model.add(norm)
    model.compile(loss='mse',
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

  def doOutputTest(self,
                   input_shape,
                   tol=1e-5,
                   norm_axis=None,
                   params_axis=-1,
                   dtype=None):
    ndim = len(input_shape)
    if norm_axis is None:
      moments_axis = range(1, ndim)
    elif isinstance(norm_axis, int):
      if norm_axis < 0:
        moments_axis = [norm_axis + ndim]
      else:
        moments_axis = [norm_axis]
    else:
      moments_axis = []
      for dim in norm_axis:
        if dim < 0:
          dim = dim + ndim
        moments_axis.append(dim)

    moments_axis = tuple(moments_axis)
    expected_shape = []
    for i in range(ndim):
      if i not in moments_axis:
        expected_shape.append(input_shape[i])

    expected_mean = np.zeros(expected_shape)
    expected_var = np.ones(expected_shape)
    for mu in [0.0, 1e2]:
      for sigma in [1.0, 0.1]:
        inputs = np.random.randn(*input_shape) * sigma + mu
        inputs_t = constant_op.constant(inputs, shape=input_shape)
        layer = normalization.LayerNormalization(
            norm_axis=norm_axis, params_axis=params_axis, dtype=dtype)
        outputs = layer(inputs_t)
        beta = layer.beta
        gamma = layer.gamma
        for weight in layer.weights:
          self.evaluate(weight.initializer)
        outputs = self.evaluate(outputs)
        beta = self.evaluate(beta)
        gamma = self.evaluate(gamma)

        # The mean and variance of the output should be close to 0 and 1
        # respectively.

        # Make sure that there are no NaNs
        self.assertFalse(np.isnan(outputs).any())
        mean = np.mean(outputs, axis=moments_axis)
        var = np.var(outputs, axis=moments_axis)
        # Layer-norm implemented in numpy
        eps = 1e-12
        expected_out = (
            (gamma * (inputs - np.mean(
                inputs, axis=moments_axis, keepdims=True)) /
             np.sqrt(eps + np.var(
                 inputs, axis=moments_axis, keepdims=True))) + beta)
        self.assertAllClose(expected_mean, mean, atol=tol, rtol=tol)
        self.assertAllClose(expected_var, var, atol=tol)
        # The full computation gets a bigger tolerance
        self.assertAllClose(expected_out, outputs, atol=5 * tol)

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutput2DInput(self):
    self.doOutputTest((10, 300))
    self.doOutputTest((10, 300), norm_axis=[0])
    self.doOutputTest((10, 300), params_axis=[0, 1])

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutput2DInputDegenerateNormAxis(self):
    with self.assertRaisesRegexp(ValueError, r'Invalid axis: 2'):
      self.doOutputTest((10, 300), norm_axis=2)

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutput4DInput(self):
    self.doOutputTest((100, 10, 10, 3))

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutput4DInputNormOnInnermostAxis(self):
    # Equivalent tests
    shape = (100, 10, 10, 3)
    self.doOutputTest(
        shape, norm_axis=list(range(3, len(shape))), tol=1e-4, dtype='float64')
    self.doOutputTest(shape, norm_axis=-1, tol=1e-4, dtype='float64')

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutputSmallInput(self):
    self.doOutputTest((10, 10, 10, 30))

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutputSmallInputNormOnInnermostAxis(self):
    self.doOutputTest((10, 10, 10, 30), norm_axis=3)

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutputSmallInputNormOnMixedAxes(self):
    self.doOutputTest((10, 10, 10, 30), norm_axis=[0, 3])
    self.doOutputTest((10, 10, 10, 30), params_axis=[-2, -1])
    self.doOutputTest((10, 10, 10, 30), norm_axis=[0, 3],
                      params_axis=[-3, -2, -1])

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutputBigInput(self):
    self.doOutputTest((1, 100, 100, 1))
    self.doOutputTest((1, 100, 100, 1), norm_axis=[1, 2])
    self.doOutputTest((1, 100, 100, 1), norm_axis=[1, 2],
                      params_axis=[-2, -1])


if __name__ == '__main__':
  test.main()
