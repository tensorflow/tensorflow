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

import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers.normalization import layer_normalization
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


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
        layer_normalization.LayerNormalization, dtype='float32')

  @keras_parameterized.run_all_keras_modes
  def test_layernorm_mixed_precision(self):
    _run_layernorm_correctness_test(
        layer_normalization.LayerNormalization, dtype='float16')

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testIncorrectAxisType(self):
    with self.assertRaisesRegex(TypeError,
                                r'Expected an int or a list/tuple of ints'):
      _ = layer_normalization.LayerNormalization(axis={'axis': -1})

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInvalidAxis(self):
    with self.assertRaisesRegex(ValueError, r'Invalid axis: 3'):
      layer_norm = layer_normalization.LayerNormalization(axis=3)
      layer_norm.build(input_shape=(2, 2, 2))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testDuplicateAxis(self):
    with self.assertRaisesRegex(ValueError, r'Duplicate axis:'):
      layer_norm = layer_normalization.LayerNormalization(axis=[-1, -1])
      layer_norm.build(input_shape=(2, 2, 2))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testFusedAttr(self):
    layer_norm = layer_normalization.LayerNormalization(axis=[-2, -1])
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
    """Tests the forward pass of layer layer_normalization.

    Args:
      batch_input_shape: The input shape that will be used to test, including
        the batch dimension.
      axis: A list of axises to normalize. Will be passed to the `axis` argument
        of Layerlayer_normalization.
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
        norm = layer_normalization.LayerNormalization(
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
    """Tests the backwards pass of layer layer_normalization.

    Args:
      batch_input_shape: The input shape that will be used to test, including
        the batch dimension.
      axis: A list of axises to normalize. Will be passed to the `axis` argument
        of Layerlayer_normalization.
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
        norm = layer_normalization.LayerNormalization(
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
  @testing_utils.run_v2_only
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
