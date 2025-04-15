# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for miscellaneous functionality in tensorflow.ops.nn on WeakTensor."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weak_tensor_ops  # pylint: disable=unused-import
from tensorflow.python.ops import weak_tensor_test_util
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test as test_lib


_get_weak_tensor = weak_tensor_test_util.get_weak_tensor


@test_util.run_all_in_graph_and_eager_modes
class LogSoftmaxTest(test_lib.TestCase, parameterized.TestCase):

  def _log_softmax(self, x):
    assert len(x.shape) == 2
    m = x.max(1)[:, np.newaxis]
    u = x - m
    return u - np.log(np.sum(np.exp(u), 1, keepdims=True))

  def testLogSoftmax(self):
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)
    y_np = self._log_softmax(x_np)
    x_wt = _get_weak_tensor(x_np)
    y_wt = nn_ops.log_softmax_v2(x_wt)
    y_tf_np = self.evaluate(y_wt)
    eps = 1e-3

    self.assertIsInstance(y_wt, weak_tensor.WeakTensor)
    self.assertAllClose(y_tf_np, y_np, eps)

  def testLogSoftmaxAxes(self):
    arr = _get_weak_tensor(np.linspace(0.0, 1, 12).reshape(3, 4))
    x_neg_axis = nn_ops.log_softmax_v2(arr, axis=-2)
    y_pos_axis = nn_ops.log_softmax_v2(arr, axis=0)
    z_gt_axis = nn_ops.log_softmax_v2(arr, axis=0)
    x_neg_axis_tf = self.evaluate(x_neg_axis)
    y_pos_axis_tf = self.evaluate(y_pos_axis)
    z_gt_axis_tf = self.evaluate(z_gt_axis)
    eps = 1e-3
    self.assertAllClose(x_neg_axis_tf, y_pos_axis_tf, eps)
    self.assertAllClose(y_pos_axis_tf, z_gt_axis_tf, eps)

  @parameterized.parameters(((5, 10),), ((2, 3, 4),))
  def testGradient(self, x_shape):
    x_np = np.random.randn(*x_shape).astype(np.float64)
    with self.cached_session():
      x_tf = _get_weak_tensor(x_np)
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          nn_ops.log_softmax_v2, [x_tf])
      self.assertAllClose(theoretical, numerical)


class ReluTest(test_lib.TestCase):

  def test(self):
    np.random.seed(1)  # Make it reproducible.
    x = np.random.randn(3, 4).astype(np.float32)
    x_wt = _get_weak_tensor(x)
    y_wt = nn_ops.relu(x_wt)
    y = np.maximum(x, 0.0)
    z = self.evaluate(y_wt)

    self.assertIsInstance(y_wt, weak_tensor.WeakTensor)
    self.assertAllEqual(y, z)

  @test_util.disable_xla(
      "This test relies on undefined behavior that XLA does not replicate")
  @test_util.run_deprecated_v1
  def testNaNs(self):
    # Test that relu(nan) = nan for various sizes.
    for i in range(18):
      x = np.zeros(i) + np.nan
      # TODO(b/178335491): This is broken on GPU today.
      with self.cached_session(use_gpu=False):
        z = nn_ops.relu(_get_weak_tensor(x)).eval()
        self.assertTrue(np.isnan(z).all())


class LeakyReluTest(test_lib.TestCase):

  def testRange(self):
    batch_size = 3
    height, width = 4, 4
    np.random.seed(1)  # Make it reproducible.
    inputs = np.random.uniform(size=(batch_size, height, width,
                                     3)).astype(np.float32)
    inputs = _get_weak_tensor(inputs)

    outputs = nn_ops.leaky_relu(inputs)
    self.assertEqual(inputs.shape, outputs.shape)
    self.assertIsInstance(outputs, weak_tensor.WeakTensor)

    inputs, outputs = self.evaluate([inputs, outputs])

    self.assertGreaterEqual(outputs.min(), 0.0)
    self.assertLessEqual(outputs.max(), 1.0)
    self.assertAllClose(inputs, outputs)

  @test_util.run_deprecated_v1
  def testValues(self):
    for dtype in [np.int32, np.int64, np.float32, np.float64]:
      values = _get_weak_tensor(np.array([-2, -1, 0, 1, 2], dtype=dtype))
      outputs = nn_ops.leaky_relu(values)
      self.assertIsInstance(outputs, weak_tensor.WeakTensor)

      outputs = self.evaluate(outputs)

      tol = 2e-3 if dtype == np.float16 else 1e-6
      self.assertAllClose(
          outputs, [-0.4, -0.2, 0.0, 1.0, 2.0], rtol=tol, atol=tol)

  @test_util.run_deprecated_v1
  def testName(self):
    values = _get_weak_tensor(np.array([-2, -1, 0, 1, 2], dtype=np.float64))
    outputs_with_name_set = nn_ops.leaky_relu(values, name="test_relu_op")
    self.assertEqual(outputs_with_name_set.name, "test_relu_op:0")
    self.assertIsInstance(outputs_with_name_set, weak_tensor.WeakTensor)

    outputs_without_name_set = nn_ops.leaky_relu(values)
    self.assertEqual(outputs_without_name_set.name, "LeakyRelu:0")
    self.assertIsInstance(outputs_without_name_set, weak_tensor.WeakTensor)


class GeluTest(test_lib.TestCase):

  def test(self):

    def gelu(x, approximate=False):
      if approximate:
        return 0.5 * x * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
      else:
        from scipy.stats import norm  # pylint: disable=g-import-not-at-top
        return x * norm.cdf(x)

    np.random.seed(1)  # Make it reproducible.
    x = np.random.randn(3, 4).astype(np.float32)
    x_wt = weak_tensor.WeakTensor(x)
    y = gelu(x)
    y_wt = nn_ops.gelu(x_wt)
    z = self.evaluate(y_wt)
    self.assertIsInstance(y_wt, weak_tensor.WeakTensor)
    self.assertAllClose(y, z)

    y = gelu(x, True)
    y_wt = nn_ops.gelu(x_wt, True)
    z = self.evaluate(y_wt)
    self.assertIsInstance(y_wt, weak_tensor.WeakTensor)
    self.assertAllClose(y, z)


@test_util.run_all_in_graph_and_eager_modes
class SwishTest(test_lib.TestCase):

  def testValues(self):
    np_values = np.array(
        [np.linspace(-7.0, 0.0, 100),
         np.linspace(0.0, 7.0, 100)],
        dtype=np.float32)
    tf_values = _get_weak_tensor(np_values)
    actual_tf_outputs = nn_impl.swish(tf_values)
    self.assertIsInstance(actual_tf_outputs, weak_tensor.WeakTensor)
    expected_tf_outputs = tf_values * math_ops.sigmoid(tf_values)

    actual_outputs, expected_outputs = self.evaluate(
        [actual_tf_outputs, expected_tf_outputs])

    self.assertAllClose(actual_outputs, expected_outputs)

  def testValuesWithBeta(self):
    np_values = np.array(
        [np.linspace(-7.0, 0.0, 100),
         np.linspace(0.0, 7.0, 100)],
        dtype=np.float32)
    tf_values = _get_weak_tensor(np_values)
    actual_tf_outputs = nn_impl.swish(tf_values, beta=0.5)
    self.assertIsInstance(actual_tf_outputs, weak_tensor.WeakTensor)
    expected_tf_outputs = tf_values * math_ops.sigmoid(0.5 * tf_values)

    actual_outputs, expected_outputs = self.evaluate(
        [actual_tf_outputs, expected_tf_outputs])

    self.assertAllClose(actual_outputs, expected_outputs)

  def testGradients(self):
    shape = [5, 3, 4]
    sigma = 5
    input_values = np.random.randn(*shape) * sigma
    x_tf = _get_weak_tensor(input_values)
    with self.cached_session():
      def f(x):  # pylint: disable=invalid-name
        return nn_impl.swish(x)

      theoretical, numerical = gradient_checker_v2.compute_gradient(
          f, [x_tf])
      self.assertAllClose(theoretical, numerical)

  def testGradientsWithBeta(self):
    shape = [5, 3, 4]
    sigma = 5
    input_values = np.random.randn(*shape) * sigma
    x_tf = _get_weak_tensor(input_values)
    with self.cached_session():
      def f(x):  # pylint: disable=invalid-name
        return nn_impl.swish(x, beta=0.5)

      theoretical, numerical = gradient_checker_v2.compute_gradient(
          f, [x_tf])
      self.assertAllClose(theoretical, numerical)

if __name__ == "__main__":
  ops.set_dtype_conversion_mode("all")
  test_lib.main()
