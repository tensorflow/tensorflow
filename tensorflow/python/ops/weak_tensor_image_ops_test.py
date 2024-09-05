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
"""Tests for tensorflow.ops.image_ops on WeakTensor."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import weak_tensor_ops  # pylint: disable=unused-import
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.platform import test

_get_weak_tensor = weak_tensor_test_util.get_weak_tensor


class AdjustBrightnessTest(test.TestCase):

  def _testBrightness(self, x_np, y_np, delta, tol=1e-6):
    with self.cached_session():
      x = _get_weak_tensor(x_np, shape=x_np.shape)
      y = image_ops.adjust_brightness(x, delta)
      y_tf = self.evaluate(y)

      self.assertIsInstance(y, WeakTensor)
      self.assertAllClose(y_tf, y_np, tol)

  def testPositiveDeltaFloat32(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float32).reshape(x_shape) / 255.0

    y_data = [10, 15, 23, 64, 145, 236, 47, 18, 244, 100, 265, 11]
    y_np = np.array(y_data, dtype=np.float32).reshape(x_shape) / 255.0

    self._testBrightness(x_np, y_np, delta=10.0 / 255.0)

  def testPositiveDeltaFloat64(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float64).reshape(x_shape) / 255.0

    y_data = [10, 15, 23, 64, 145, 236, 47, 18, 244, 100, 265, 11]
    y_np = np.array(y_data, dtype=np.float64).reshape(x_shape) / 255.0

    self._testBrightness(x_np, y_np, delta=10.0 / 255.0, tol=1e-3)


class AdjustGamma(test.TestCase):

  def test_adjust_gamma_less_zero_float32(self):
    """White image should be returned for gamma equal to zero."""
    with self.cached_session():
      x_data = np.random.uniform(0, 1.0, (8, 8))
      x_np = np.array(x_data, dtype=np.float32)

      x = _get_weak_tensor(x_np, shape=x_np.shape)

      err_msg = "Gamma should be a non-negative real number"
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError), err_msg):
        image_ops.adjust_gamma(x, gamma=-1)

  def test_adjust_gamma_less_zero_tensor(self):
    """White image should be returned for gamma equal to zero."""
    with self.cached_session():
      x_data = np.random.uniform(0, 1.0, (8, 8))
      x_np = np.array(x_data, dtype=np.float32)

      x = _get_weak_tensor(x_np, shape=x_np.shape)
      y = constant_op.constant(-1.0, dtype=dtypes.float32)

      err_msg = "Gamma should be a non-negative real number"
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError), err_msg):
        image = image_ops.adjust_gamma(x, gamma=y)
        self.evaluate(image)

  def _test_adjust_gamma_float32(self, gamma):
    """Verifying the output with expected results for gamma correction for float32 images."""
    with self.cached_session():
      x_np = np.random.uniform(0, 1.0, (8, 8))
      x = _get_weak_tensor(x_np, shape=x_np.shape)
      y = image_ops.adjust_gamma(x, gamma=gamma)
      y_tf = self.evaluate(y)

      self.assertIsInstance(y, WeakTensor)
      y_np = np.clip(np.power(x_np, gamma), 0, 1.0)

      self.assertAllClose(y_tf, y_np, 1e-6)

  def test_adjust_gamma_one_float32(self):
    """Same image should be returned for gamma equal to one."""
    self._test_adjust_gamma_float32(1.0)

  def test_adjust_gamma_less_one_float32(self):
    """Verifying the output with expected results for gamma correction with gamma equal to half for float32 images."""
    self._test_adjust_gamma_float32(0.5)

  def test_adjust_gamma_greater_one_float32(self):
    """Verifying the output with expected results for gamma correction with gamma equal to two for float32 images."""
    self._test_adjust_gamma_float32(1.0)

  def test_adjust_gamma_zero_float32(self):
    """White image should be returned for gamma equal to zero for float32 images."""
    self._test_adjust_gamma_float32(0.0)


if __name__ == "__main__":
  ops.set_dtype_conversion_mode("all")
  test.main()
