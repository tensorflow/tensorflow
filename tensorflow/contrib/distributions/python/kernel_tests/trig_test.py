# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Trigonometric functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import trig
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gradients_impl
from tensorflow.python.platform import test


class SinhTest(test.TestCase):

  def test_versus_numpy_scalar_positive_x(self):
    x = 1.23
    with self.test_session():
      self.assertAllClose(np.sinh(x), trig.sinh(x).eval())

  def test_versus_numpy_multidim_x(self):
    x = [[0., -1.], [0.5, -2.]]
    with self.test_session():
      self.assertAllClose(np.sinh(x), trig.sinh(x).eval())


class ArcSinhTest(test.TestCase):
  """Test arcsinh.

  Note that many tests below were run over a range of values using np.logspace.
  The accuracy was highly dependent on the exact values used within that range,
  and whether our approximation matched with numpy.  For that reason, we used
  1000 values over every range, which ensures we hit most of the "tricky" values
  """

  def _assert_all_finite(self, values):
    self.assertAllEqual(np.ones_like(values).astype(np.bool),
                        np.isfinite(values))

  def test_versus_numpy_scalar_positive_x(self):
    x = 1.23
    with self.test_session():
      self.assertAllClose(np.arcsinh(x), trig.arcsinh(x).eval())

  def test_versus_numpy_at_zero(self):
    # Zero is especially difficult.
    with self.test_session() as sess:
      x = constant_op.constant(0.)
      y = trig.arcsinh(x)
      grad = gradients_impl.gradients(y, x)[0]
      x_, y_, grad_ = sess.run([x, y, grad])
      self.assertAllClose(np.arcsinh(x_), y_)
      self._assert_all_finite(y_)
      self._assert_all_finite(grad_)

  def test_versus_numpy_multidim_x(self):
    with self.test_session() as sess:
      x = constant_op.constant([[0., -1.], [0.5, -2.]])
      y = trig.arcsinh(x)
      grad = gradients_impl.gradients(y, x)[0]
      x_, y_, grad_ = sess.run([x, y, grad])
      self.assertAllClose(np.arcsinh(x_), y_)
      self._assert_all_finite(y_)
      self._assert_all_finite(grad_)

  def test_versus_numpy_positive_values_32_bit(self):
    with self.test_session() as sess:
      # Larger than 38 is Inf in float32.
      x = constant_op.constant(np.logspace(0, 38, num=2000).astype(np.float32))
      y = trig.arcsinh(x)
      grad = gradients_impl.gradients(y, x)[0]
      x_, y_, grad_ = sess.run([x, y, grad])
      self.assertAllClose(
          np.arcsinh(x_),  # numpy does this in 64bit
          y_,
          rtol=1e-6)
      self._assert_all_finite(y_)
      self._assert_all_finite(grad_)

  def test_versus_numpy_moderate_negative_values_32_bit(self):
    # The moderate negative values were the most difficult to get close to
    # numpy.
    with self.test_session() as sess:
      x = constant_op.constant(-np.logspace(0, 10, num=1000).astype(np.float32))
      y = trig.arcsinh(x)
      grad = gradients_impl.gradients(y, x)[0]
      x_, y_, grad_ = sess.run([x, y, grad])
      self.assertAllClose(
          np.arcsinh(x_),  # numpy does this in 64bit
          y_,
          rtol=1e-4)
      self._assert_all_finite(y_)
      self._assert_all_finite(grad_)

  def test_versus_numpy_extreme_negative_values_32_bit(self):
    # For these extreme values arcsinh uses the approximation 1 / (2 * x), and
    # 1 / 10^38 = 0 in 32bit...so stop at 10^37.
    with self.test_session() as sess:
      x = constant_op.constant(
          -np.logspace(10, 37, num=1000).astype(np.float32))
      y = trig.arcsinh(x)
      grad = gradients_impl.gradients(y, x)[0]
      x_, y_, grad_ = sess.run([x, y, grad])
      self.assertAllClose(
          np.arcsinh(x_),  # numpy does this in 64bit
          y_,
          rtol=1e-6)
      self._assert_all_finite(y_)
      self._assert_all_finite(grad_)

  def test_versus_numpy_positive_values_64_bit(self):
    with self.test_session() as sess:
      x = constant_op.constant(np.logspace(0, 200, num=1000))
      y = trig.arcsinh(x)
      grad = gradients_impl.gradients(y, x)[0]
      x_, y_, grad_ = sess.run([x, y, grad])
      self.assertAllClose(
          np.arcsinh(x_),
          y_,
          rtol=1e-6)
      self._assert_all_finite(y_)
      self._assert_all_finite(grad_)

  def test_versus_numpy_negative_values_64_bit(self):
    with self.test_session() as sess:
      x = constant_op.constant(-np.logspace(0, 200, num=1000))
      y = trig.arcsinh(x)
      grad = gradients_impl.gradients(y, x)[0]
      x_, y_, grad_ = sess.run([x, y, grad])
      self.assertAllClose(
          np.arcsinh(x_),
          y_,
          rtol=1e-5)
      self._assert_all_finite(y_)
      self._assert_all_finite(grad_)

  def test_arcsinh_is_inverse_to_sinh_near_zero(self):
    sinh = trig.sinh
    arcsinh = trig.arcsinh
    with self.test_session() as sess:
      x = np.linspace(-1.1, 1.1, num=1000).astype(np.float32)
      arcsinh_x = arcsinh(x)
      sinh_arcsinh_x = sinh(arcsinh_x)
      arcsinh_sinh_arcsinh_x = arcsinh(sinh_arcsinh_x)

      arcsinh_x_, sinh_arcsinh_x_, arcsinh_sinh_arcsinh_x_ = sess.run(
          [arcsinh_x, sinh_arcsinh_x, arcsinh_sinh_arcsinh_x])

    self.assertAllClose(x, sinh_arcsinh_x_)
    self.assertAllClose(arcsinh_x_, arcsinh_sinh_arcsinh_x_)

  def test_arcsinh_is_inverse_to_sinh_where_x_is_very_small(self):
    sinh = trig.sinh
    arcsinh = trig.arcsinh
    # Exact same cutoff as is in the code.
    very_small_cutoff = -0.01 / np.sqrt(np.finfo(np.float32).eps)
    with self.test_session() as sess:
      x = -np.logspace(
          np.log(-very_small_cutoff),
          np.log(-1000 * very_small_cutoff),
          num=1000).astype(np.float32)
      arcsinh_x = arcsinh(x)
      sinh_arcsinh_x = sinh(arcsinh_x)
      arcsinh_sinh_arcsinh_x = arcsinh(sinh_arcsinh_x)

      arcsinh_x_, sinh_arcsinh_x_, arcsinh_sinh_arcsinh_x_ = sess.run(
          [arcsinh_x, sinh_arcsinh_x, arcsinh_sinh_arcsinh_x])

    self.assertAllClose(x, sinh_arcsinh_x_, rtol=1e-5)
    self.assertAllClose(arcsinh_x_, arcsinh_sinh_arcsinh_x_)

  def test_arcsinh_is_inverse_to_sinh_where_x_is_moderate_or_big(self):
    sinh = trig.sinh
    arcsinh = trig.arcsinh
    very_big_cutoff = np.sqrt(np.finfo(np.float32).max)
    with self.test_session() as sess:
      x = np.linspace(1., very_big_cutoff, num=1000).astype(np.float32)
      arcsinh_x = arcsinh(x)
      sinh_arcsinh_x = sinh(arcsinh_x)
      arcsinh_sinh_arcsinh_x = arcsinh(sinh_arcsinh_x)

      arcsinh_x_, sinh_arcsinh_x_, arcsinh_sinh_arcsinh_x_ = sess.run(
          [arcsinh_x, sinh_arcsinh_x, arcsinh_sinh_arcsinh_x])

    self.assertAllClose(x, sinh_arcsinh_x_, rtol=1e-5)
    self.assertAllClose(arcsinh_x_, arcsinh_sinh_arcsinh_x_)

  def test_arcsinh_is_inverse_to_sinh_where_x_is_very_big(self):
    sinh = trig.sinh
    arcsinh = trig.arcsinh
    very_big_cutoff = np.sqrt(np.finfo(np.float32).max)
    with self.test_session() as sess:
      x = np.logspace(
          np.log(very_big_cutoff),
          5 + np.log(very_big_cutoff), num=1000).astype(np.float32)
      arcsinh_x = arcsinh(x)
      sinh_arcsinh_x = sinh(arcsinh_x)
      arcsinh_sinh_arcsinh_x = arcsinh(sinh_arcsinh_x)

      arcsinh_x_, sinh_arcsinh_x_, arcsinh_sinh_arcsinh_x_ = sess.run(
          [arcsinh_x, sinh_arcsinh_x, arcsinh_sinh_arcsinh_x])

    self.assertAllClose(x, sinh_arcsinh_x_, rtol=1e-5)
    self.assertAllClose(arcsinh_x_, arcsinh_sinh_arcsinh_x_)


class CoshTest(test.TestCase):

  def test_versus_numpy_scalar_positive_x(self):
    x = 1.23
    with self.test_session():
      self.assertAllClose(np.cosh(x), trig.cosh(x).eval())

  def test_versus_numpy_multidim_x(self):
    x = [[0., -1.], [0.5, -2.]]
    with self.test_session():
      self.assertAllClose(np.cosh(x), trig.cosh(x).eval())


class LogCoshTest(test.TestCase):

  def _assert_all_finite(self, values):
    self.assertAllEqual(np.ones_like(values).astype(np.bool),
                        np.isfinite(values))

  def test_versus_numpy_scalar_32bit(self):
    with self.test_session() as sess:
      x_64 = np.linspace(-200, 200, 2000)
      x = constant_op.constant(x_64.astype(np.float32))

      y = trig.log_cosh(x)
      grad = gradients_impl.gradients(y, x)[0]
      y_, grad_ = sess.run([y, grad])

      self.assertAllClose(np.log(np.cosh(x_64)), y_)
      self._assert_all_finite(y_)
      self._assert_all_finite(grad_)


if __name__ == "__main__":
  test.main()
