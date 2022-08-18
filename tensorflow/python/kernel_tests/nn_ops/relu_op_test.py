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
"""Tests for Relu and ReluGrad."""

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


def _elu_grad_grad(activation):
  if activation < 0:
    return np.exp(activation)
  return 0


class ReluTest(test.TestCase):

  def _npRelu(self, np_features):
    return np.maximum(np_features, np.zeros(np_features.shape))

  def testNpRelu(self):
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 0.0], [0.1, 0.0, 0.5, 0.0, 0.9]]),
        self._npRelu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                     0.9]])))

  def _testRelu(self, np_features):
    np_relu = self._npRelu(np_features)
    tf_relu = nn_ops.relu(np_features)
    self.assertAllClose(np_relu, tf_relu)
    self.assertShapeEqual(np_relu, tf_relu)

  def testNumbersCPU(self):
    for t in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testRelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNumbersGPU(self):
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [np.float16, np.float32, np.float64]:
      self._testRelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testReluInt8x4GoodShape(self):
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest("No GPU available")
    inputs = np.array([[-50, 7, 23, 0], [-1, -5, 6, 11]])
    np_relu = self._npRelu(inputs)
    tf_relu = nn_ops.relu(constant_op.constant(inputs, dtypes.qint8))
    self.assertAllClose(np_relu, tf_relu)
    self.assertShapeEqual(np_relu, tf_relu)

  @test_util.disable_xla("b/123338077")  # Passes with XLA
  def testReluInt8x4BadShape(self):
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest("No GPU available")
    inputs = constant_op.constant(
        np.array([[-50, 7, 23], [0, 1, -5], [6, -2, 11]]), dtypes.qint8)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Tensor size must be a multiple of 4 for Relu<qint8>. Got 9"):
      self.evaluate(nn_ops.relu(inputs))

    inputs = constant_op.constant(
        np.array([1, -2, 3, -4, 5, -6, 7, -8, 9, -8, 7, -6, 5, -4, 3, -2, 1]),
        dtypes.qint8)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Tensor size must be a multiple of 4 for Relu<qint8>. Got 17"):
      self.evaluate(nn_ops.relu(inputs))

  def testNoElement(self):
    self._testRelu(np.array([[], []], dtype=np.float32))

  @test_util.disable_xla("b/157978028: Does not yet pass with XLA")
  def testNaNPropagation(self):
    for t in [np.float16, np.float32, np.float64]:
      self._testRelu(np.array([-1, np.nan, 1, np.nan]).astype(t))

  # The gradient test for ReLU is a bit tricky as the derivative is not well
  # defined at around zero and we want to avoid that in terms of input values.
  def testGradientFloat32(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(
          nn_ops.relu, [x], delta=1.0 / 1024))
    self.assertLess(err, 1e-6)

  # The gradient test for ReLU is a bit tricky as the derivative is not well
  # defined at around zero and we want to avoid that in terms of input values.
  def testGradientFloat16(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float16,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu, [x]))
    self.assertLess(err, 1e-6)

  def testGradientFloat64(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(
          nn_ops.relu, [x], delta=1.0 / 1024))
    self.assertLess(err, 1e-15)

  def testGradGradFloat32(self):
    with self.cached_session():

      def f(x):
        assert x.dtype == dtypes.float32
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.relu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x], delta=1.0 / 1024))
    self.assertLess(err, 1e-4)

  def testGradGradFloat64(self):
    with self.cached_session():

      def f(x):
        assert x.dtype == dtypes.float64
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.relu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x], delta=1.0 / 1024))
    self.assertLess(err, 1e-10)

  def testGradientScalar(self):
    x = variables.Variable(100.)

    def loss():
      return nn_ops.relu(x)**2

    optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.25)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(optimizer.minimize(loss))
    self.assertAllClose(x.read_value(), 50.0)

  def testGradientNoElement(self):
    with self.cached_session():

      def f(x):
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.relu(x)
        return tape.gradient(y, x)

      x = np.asarray([[], []], dtype=np.float32)
      z = list(gradient_checker_v2.compute_gradient(f, [x]))[0][0]
      self.assertAllEqual(z, np.reshape(x, (0, 0)))


class Relu6Test(test.TestCase):

  def _npRelu6(self, np_features):
    sixes = np.copy(np_features)
    sixes.fill(6.0)
    return np.minimum(
        np.maximum(np_features, np.zeros(np_features.shape)), sixes)

  def testNpRelu6(self):
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 6.0], [0.1, 0.0, 6.0, 0.0, 0.9]]),
        self._npRelu6(
            np.array([[-0.9, 0.7, -0.5, 0.3, 6.0], [0.1, -0.3, 6.5, -0.7,
                                                    0.9]])))

  def _testRelu6(self, np_features):
    np_relu6 = self._npRelu6(np_features)
    tf_relu6 = nn_ops.relu6(np_features)
    self.assertAllClose(np_relu6, tf_relu6)
    self.assertShapeEqual(np_relu6, tf_relu6)

  def testNumbersCPU(self):
    for t in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testRelu6(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNumbersGPU(self):
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [np.float16, np.float64, np.double]:
      self._testRelu6(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  @test_util.disable_xla("b/157978028: Does not yet pass with XLA")
  def testNaNPropagation(self):
    for t in [np.float16, np.float32, np.float64]:
      self._testRelu6(np.array([-1, np.nan, 1, 7, np.nan]).astype(t))

  # The gradient test for ReLU6 is a bit tricky as the derivative is
  # not well defined at around zero and six and we want to avoid that
  # in terms of input values.
  def testGradientFloat32(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu6, [x]))
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu6, [x]))
    self.assertLess(err, 1e-10)


class LeakyReluTest(test.TestCase):

  def _npLeakyRelu(self, np_features, alpha=0.1):
    return np.maximum(np_features, alpha * np_features)

  def testNpLeakyRelu(self):
    self.assertAllClose(
        np.array([[-0.09, 0.7, -0.05, 0.3, -0.01],
                  [0.1, -0.03, 0.5, -0.07, 0.9]]),
        self._npLeakyRelu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                     0.9]]),
            alpha=0.1))

  def _testLeakyRelu(self, np_features, alpha):
    np_leaky_relu = self._npLeakyRelu(np_features, alpha)
    tf_leaky_relu = nn_ops.leaky_relu(np_features, alpha)
    self.assertAllCloseAccordingToType(np_leaky_relu, tf_leaky_relu)
    self.assertShapeEqual(np_leaky_relu, tf_leaky_relu)

  def testNumbersCPU(self):
    for t in [
        np.int32, np.int64, np.float16, np.float32, np.float64,
        dtypes.bfloat16.as_numpy_dtype
    ]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testLeakyRelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
            alpha=0.2)

  def testNumbersGPU(self):
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [np.float16, np.float32, np.float64]:
      self._testLeakyRelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          alpha=0.1)

  def testNaNPropagation(self):
    for t in [np.float16, np.float32, np.float64]:
      self._testLeakyRelu(np.array([-1, np.nan, 1, np.nan]).astype(t),
                          alpha=0.2)

  # The gradient test for Leaky ReLU is a bit tricky as the derivative is not
  # well defined at around zero and we want to avoid that in terms of input
  # values.
  def testGradientFloat32(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.leaky_relu, [x]))
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.leaky_relu, [x]))
    self.assertLess(err, 1e-10)

  def testGradGradFloat32(self):
    with self.cached_session():

      def f(x):
        assert x.dtype == dtypes.float32
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.leaky_relu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-4)

  def testGradGradFloat64(self):
    with self.cached_session():

      def f(x):
        assert x.dtype == dtypes.float64
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.leaky_relu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-10)

  def testGradientScalar(self):
    x = variables.Variable(-100.)

    def loss():
      return nn_ops.leaky_relu(x, 0.05)**2

    optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.2)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(optimizer.minimize(loss))
    self.assertAllClose(x.read_value(), -99.9)

  def testUnexpectedAlphaValue(self):
    self.assertAllClose(
        np.array([[-9.0, 0.7, -5.0, 0.3, -0.1], [0.1, -3.0, 0.5, -27.0, 0.9]]),
        nn_ops.leaky_relu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.01],
                      [0.1, -0.3, 0.5, -2.7, 0.9]]),
            alpha=10))
    self.assertAllClose(
        np.array([[9.0, 0.7, 5.0, 0.3, 0.1], [0.1, 3.0, 0.5, 27.0, 0.9]]),
        nn_ops.leaky_relu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.01],
                      [0.1, -0.3, 0.5, -2.7, 0.9]]),
            alpha=-10))


class EluTest(test.TestCase):

  def _npElu(self, np_features):
    return np.where(np_features < 0, np.exp(np_features) - 1, np_features)

  def testNpElu(self):
    self.assertAllClose(
        np.array([[-0.59343034025, 0.7, -0.39346934028, 0.3, -0.09516258196],
                  [0.1, -0.25918177931, 0.5, -0.5034146962, 0.9]]),
        self._npElu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                     0.9]])))

  def _testElu(self, np_features):
    np_elu = self._npElu(np_features)
    tf_elu = nn_ops.elu(np_features)
    self.assertAllCloseAccordingToType(np_elu, tf_elu)
    self.assertShapeEqual(np_elu, tf_elu)

  def testNumbersCPU(self):
    for t in [np.float16, np.float32, np.float64]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testElu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNumbersGPU(self):
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [np.float16, np.float32, np.float64]:
      self._testElu(np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNaNPropagation(self):
    for t in [np.float16, np.float32, np.float64]:
      self._testElu(np.array([-1, np.nan, 1, np.nan]).astype(t))

  def testGradientFloat32(self):
    with self.cached_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = np.asarray(x_val, dtype=np.float32, order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.elu, [x]))
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    with self.cached_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = np.asarray(x_val, dtype=np.float64, order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.elu, [x]))
    self.assertLess(err, 1e-6)

  def testGradGrad(self):
    with self.cached_session():

      def f(x):
        with backprop.GradientTape(persistent=True) as tape:
          tape.watch(x)
          y = nn_ops.elu(x)
          dy = tape.gradient(y, x)
        return tape.gradient(dy, x)

      for x in [-1., -0.5, 0.5, 1.]:
        got = self.evaluate(f(constant_op.constant(x)))
        want = _elu_grad_grad(x)
        err = np.abs(got - want)
        self.assertLess(err, 1e-4)

  def testGradGradFloat32(self):
    with self.cached_session():

      def f(x):
        assert x.dtype == dtypes.float32
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.elu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-4)

  def testGradGradFloat64(self):
    with self.cached_session():

      def f(x):
        assert x.dtype == dtypes.float64
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.elu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-6)


class SeluTest(test.TestCase):

  def _npSelu(self, np_features):
    scale = 1.0507009873554804934193349852946
    scale_alpha = 1.7580993408473768599402175208123
    return np.where(np_features < 0, scale_alpha * (np.exp(np_features) - 1),
                    scale * np_features)

  def testNpSelu(self):
    self.assertAllClose(
        np.array([[-1.0433095, 0.73549069, -0.6917582, 0.3152103, -0.16730527],
                  [0.1050701, -0.45566732, 0.5253505, -0.88505305, 0.9456309]]),
        self._npSelu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                     0.9]])))

  def _testSelu(self, np_features):
    np_selu = self._npSelu(np_features)
    tf_selu = nn_ops.selu(np_features)
    self.assertAllCloseAccordingToType(np_selu, tf_selu)
    self.assertShapeEqual(np_selu, tf_selu)

  def testNumbers(self):
    for t in [np.float16, np.float32, np.float64]:
      self._testSelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))
      # Force executed on CPU in case GPU kernels are available.
      with ops.device("/device:CPU:0"):
        self._testSelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testGradientFloat32(self):
    with self.cached_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = np.asarray(x_val, dtype=np.float32, order="F")
      err = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(
          nn_ops.selu, [x], delta=1.0 / 1024))
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    with self.cached_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = np.asarray(x_val, dtype=np.float64, order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.selu, [x]))
    self.assertLess(err, 1e-6)

  def testGradGradFloat32(self):
    with self.cached_session():

      def f(x):
        assert x.dtype == dtypes.float32
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.selu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x], delta=1.0 / 1024))
    self.assertLess(err, 1e-4)

  def testGradGradFloat64(self):
    with self.cached_session():

      def f(x):
        assert x.dtype == dtypes.float64
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.selu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-6)


class CreluTest(test.TestCase):

  def testCreluShape(self):
    f = random_ops.random_normal([50, 5, 7, 10])
    t = nn_ops.crelu(f)
    self.assertEqual([50, 5, 7, 20], t.get_shape())

  def _testCrelu(self, np_features):
    np_relu = np.maximum(np_features, np.zeros_like(np_features))
    np_neg_relu = np.maximum(-np_features, np.zeros_like(np_features))
    np_crelu = np.concatenate((np_relu, np_neg_relu),
                              len(np_features.shape) - 1)

    tf_crelu = nn_ops.crelu(np_features)

    self.assertAllClose(np_crelu, tf_crelu)
    self.assertShapeEqual(np_crelu, tf_crelu)

  def testNumbersCPU(self):
    for t in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testCrelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNumbersGPU(self):
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [np.float16, np.float32, np.float64]:
      self._testCrelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNumbersWithAxis0(self):
    tf_crelu = nn_ops.crelu(
        np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]), axis=0)
    np_crelu = np.array([[0, 7, 0, 3, 0], [1, 0, 5, 0, 9], [9, 0, 5, 0, 1],
                         [0, 3, 0, 7, 0]])
    self.assertAllEqual(np_crelu, tf_crelu)

  def testNumbersWithAxis1(self):
    tf_crelu = nn_ops.crelu(
        np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]), axis=1)
    np_crelu = np.array([[0, 7, 0, 3, 0, 9, 0, 5, 0, 1],
                         [1, 0, 5, 0, 9, 0, 3, 0, 7, 0]])
    self.assertAllEqual(np_crelu, tf_crelu)


if __name__ == "__main__":
  test.main()
