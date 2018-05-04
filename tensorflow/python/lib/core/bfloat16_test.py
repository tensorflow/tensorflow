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

"""Test cases for the bfloat16 Python type."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

# pylint: disable=unused-import,g-bad-import-order
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


bfloat16 = pywrap_tensorflow.TF_bfloat16_type()


class Bfloat16Test(test.TestCase):

  def float_values(self):
    """Returns values that should round trip exactly to float and back."""
    epsilon = float.fromhex("1.0p-7")
    return [
        0.0, 1.0, -1, 0.5, -0.5, epsilon, 1.0 + epsilon, 1.0 - epsilon,
        -1.0 - epsilon, -1.0 + epsilon, 3.5, 42.0, 255.0, 256.0,
        float("inf"), float("-inf"), float("nan")]

  def _assertFloatIdentical(self, v, w):
    if math.isnan(v):
      self.assertTrue(math.isnan(w))
    else:
      self.assertEqual(v, w)

  def testRoundTripToFloat(self):
    for v in self.float_values():
      self._assertFloatIdentical(v, float(bfloat16(v)))

  def testRoundTripToInt(self):
    for v in [-256, -255, -34, -2, -1, 0, 1, 2, 10, 47, 128, 255, 256, 512]:
      self.assertEqual(v, int(bfloat16(v)))

  def testStr(self):
    self.assertEqual("0", str(bfloat16(0.0)))
    self.assertEqual("1", str(bfloat16(1.0)))
    self.assertEqual("-3.5", str(bfloat16(-3.5)))
    self.assertEqual("0.0078125", str(bfloat16(float.fromhex("1.0p-7"))))
    self.assertEqual("inf", str(bfloat16(float("inf"))))
    self.assertEqual("-inf", str(bfloat16(float("-inf"))))
    self.assertEqual("nan", str(bfloat16(float("nan"))))

  def testRepr(self):
    self.assertEqual("bfloat16(0)", repr(bfloat16(0)))
    self.assertEqual("bfloat16(1)", repr(bfloat16(1)))
    self.assertEqual("bfloat16(-3.5)", repr(bfloat16(-3.5)))
    self.assertEqual("bfloat16(0.0078125)",
                     repr(bfloat16(float.fromhex("1.0p-7"))))
    self.assertEqual("bfloat16(inf)", repr(bfloat16(float("inf"))))
    self.assertEqual("bfloat16(-inf)", repr(bfloat16(float("-inf"))))
    self.assertEqual("bfloat16(nan)", repr(bfloat16(float("nan"))))

  def testHash(self):
    self.assertEqual(0, hash(bfloat16(0.0)))
    self.assertEqual(0x3f80, hash(bfloat16(1.0)))
    self.assertEqual(0x7fc0, hash(bfloat16(float("nan"))))

  # Tests for Python operations
  def testNegate(self):
    for v in self.float_values():
      self._assertFloatIdentical(-v, float(-bfloat16(v)))

  def testAdd(self):
    self._assertFloatIdentical(0, float(bfloat16(0) + bfloat16(0)))
    self._assertFloatIdentical(1, float(bfloat16(1) + bfloat16(0)))
    self._assertFloatIdentical(0, float(bfloat16(1) + bfloat16(-1)))
    self._assertFloatIdentical(5.5, float(bfloat16(2) + bfloat16(3.5)))
    self._assertFloatIdentical(1.25, float(bfloat16(3.5) + bfloat16(-2.25)))
    self._assertFloatIdentical(float("inf"),
                               float(bfloat16(float("inf")) + bfloat16(-2.25)))
    self._assertFloatIdentical(float("-inf"),
                               float(bfloat16(float("-inf")) + bfloat16(-2.25)))
    self.assertTrue(math.isnan(float(bfloat16(3.5) + bfloat16(float("nan")))))

  def testSub(self):
    self._assertFloatIdentical(0, float(bfloat16(0) - bfloat16(0)))
    self._assertFloatIdentical(1, float(bfloat16(1) - bfloat16(0)))
    self._assertFloatIdentical(2, float(bfloat16(1) - bfloat16(-1)))
    self._assertFloatIdentical(-1.5, float(bfloat16(2) - bfloat16(3.5)))
    self._assertFloatIdentical(5.75, float(bfloat16(3.5) - bfloat16(-2.25)))
    self._assertFloatIdentical(float("-inf"),
                               float(bfloat16(-2.25) - bfloat16(float("inf"))))
    self._assertFloatIdentical(float("inf"),
                               float(bfloat16(-2.25) - bfloat16(float("-inf"))))
    self.assertTrue(math.isnan(float(bfloat16(3.5) - bfloat16(float("nan")))))

  def testMul(self):
    self._assertFloatIdentical(0, float(bfloat16(0) * bfloat16(0)))
    self._assertFloatIdentical(0, float(bfloat16(1) * bfloat16(0)))
    self._assertFloatIdentical(-1, float(bfloat16(1) * bfloat16(-1)))
    self._assertFloatIdentical(-7.875, float(bfloat16(3.5) * bfloat16(-2.25)))
    self._assertFloatIdentical(float("-inf"),
                               float(bfloat16(float("inf")) * bfloat16(-2.25)))
    self._assertFloatIdentical(float("inf"),
                               float(bfloat16(float("-inf")) * bfloat16(-2.25)))
    self.assertTrue(math.isnan(float(bfloat16(3.5) * bfloat16(float("nan")))))

  def testDiv(self):
    self.assertTrue(math.isnan(float(bfloat16(0) / bfloat16(0))))
    self._assertFloatIdentical(float("inf"), float(bfloat16(1) / bfloat16(0)))
    self._assertFloatIdentical(-1, float(bfloat16(1) / bfloat16(-1)))
    self._assertFloatIdentical(-1.75, float(bfloat16(3.5) / bfloat16(-2)))
    self._assertFloatIdentical(float("-inf"),
                               float(bfloat16(float("inf")) / bfloat16(-2.25)))
    self._assertFloatIdentical(float("inf"),
                               float(bfloat16(float("-inf")) / bfloat16(-2.25)))
    self.assertTrue(math.isnan(float(bfloat16(3.5) / bfloat16(float("nan")))))

  def testLess(self):
    for v in self.float_values():
      for w in self.float_values():
        self.assertEqual(v < w, bfloat16(v) < bfloat16(w))

  def testLessEqual(self):
    for v in self.float_values():
      for w in self.float_values():
        self.assertEqual(v <= w, bfloat16(v) <= bfloat16(w))

  def testGreater(self):
    for v in self.float_values():
      for w in self.float_values():
        self.assertEqual(v > w, bfloat16(v) > bfloat16(w))

  def testGreaterEqual(self):
    for v in self.float_values():
      for w in self.float_values():
        self.assertEqual(v >= w, bfloat16(v) >= bfloat16(w))

  def testEqual(self):
    for v in self.float_values():
      for w in self.float_values():
        self.assertEqual(v == w, bfloat16(v) == bfloat16(w))

  def testNotEqual(self):
    for v in self.float_values():
      for w in self.float_values():
        self.assertEqual(v != w, bfloat16(v) != bfloat16(w))

  def testNan(self):
    a = np.isnan(bfloat16(float("nan")))
    self.assertTrue(a)
    np.testing.assert_allclose(np.array([1.0, a]), np.array([1.0, a]))

    a = np.array(
        [bfloat16(1.34375),
         bfloat16(1.4375),
         bfloat16(float("nan"))],
        dtype=dtypes.bfloat16.as_numpy_dtype)
    b = np.array(
        [bfloat16(1.3359375),
         bfloat16(1.4375),
         bfloat16(float("nan"))],
        dtype=dtypes.bfloat16.as_numpy_dtype)
    np.testing.assert_allclose(
        a, b, rtol=0.1, atol=0.1, equal_nan=True, err_msg="", verbose=True)


class Bfloat16NumPyTest(test.TestCase):

  def testDtype(self):
    self.assertEqual(bfloat16, np.dtype(bfloat16))

  def testArray(self):
    x = np.array([[1, 2, 3]], dtype=bfloat16)
    self.assertEqual(bfloat16, x.dtype)
    self.assertEqual("[[bfloat16(1) bfloat16(2) bfloat16(3)]]", str(x))
    self.assertAllEqual(x, x)
    self.assertAllClose(x, x)
    self.assertTrue((x == x).all())

  def testComparisons(self):
    x = np.array([401408, 7, -32], dtype=np.float32)
    bx = x.astype(bfloat16)
    y = np.array([82432, 7, 0], dtype=np.float32)
    by = y.astype(bfloat16)
    self.assertAllEqual(x == y, bx == by)
    self.assertAllEqual(x != y, bx != by)
    self.assertAllEqual(x < y, bx < by)
    self.assertAllEqual(x > y, bx > by)
    self.assertAllEqual(x <= y, bx <= by)
    self.assertAllEqual(x >= y, bx >= by)

  def testEqual2(self):
    a = np.array([401408], bfloat16)
    b = np.array([82432], bfloat16)
    self.assertFalse(a.__eq__(b))

  def testCasts(self):
    for dtype in [
        np.float16, np.float32, np.float64, np.int32, np.int64,
        np.complex64, np.complex128]:
      x = np.array([[1, 2, 3]], dtype=dtype)
      y = x.astype(bfloat16)
      z = y.astype(dtype)
      self.assertTrue(np.all(x == y))
      self.assertEqual(bfloat16, y.dtype)
      self.assertTrue(np.all(x == z))
      self.assertEqual(dtype, z.dtype)

  def testConformNumpyComplex(self):
    for dtype in [np.complex64, np.complex128]:
      x = np.array([1.1, 2.2 + 2.2j, 3.3], dtype=dtype)
      y_np = x.astype(np.float32)
      y_tf = x.astype(bfloat16)
      self.assertAllClose(y_np, y_tf, atol=2e-2)

      z_np = y_np.astype(dtype)
      z_tf = y_tf.astype(dtype)
      self.assertAllClose(z_np, z_tf, atol=2e-2)

  def testAdd(self):
    x = np.array([[1, 2, 3]], dtype=bfloat16)
    y = np.array([[4, 5, 6]], dtype=bfloat16)
    self.assertAllClose(np.array([[5, 7, 9]]), x + y)

  def testLogSumExp(self):
    x = np.array([[1, 2, 3]], dtype=np.float32)
    y = np.array([[4, 5, 6]], dtype=np.float32)
    self.assertAllClose(np.logaddexp(x, y),
                        np.logaddexp(x.astype(bfloat16), y.astype(bfloat16)),
                        atol=2e-2)


if __name__ == "__main__":
  test.main()
