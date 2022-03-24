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
"""Tests for XLA JIT compiler."""

import unittest

import numpy as np
import six

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest


def nhwc_to_format(x, data_format):
  """Converts a numpy array from NHWC format to `data_format`."""
  rank = len(x.shape)
  if data_format == "NCHW":
    return np.transpose(x, [0, rank - 1] + list(range(1, rank - 1)))
  elif data_format == "NHWC":
    return x
  else:
    raise ValueError("Unknown format {}".format(data_format))


class UnaryOpsTest(xla_test.XLATestCase):
  """Test cases for unary operators."""

  def _assertOpOutputMatchesExpected(self,
                                     op,
                                     inp,
                                     expected,
                                     equality_test=None,
                                     rtol=1e-3,
                                     atol=1e-5):
    """Verifies that 'op' produces 'expected' when fed input 'inp' .

    Args:
      op: operator to test
      inp: numpy input array to use as input to 'op'.
      expected: numpy array representing the expected output of 'op'.
      equality_test: either None, or a function that tests two numpy arrays for
        equality. If None, self.assertAllClose is used.
      rtol: relative tolerance for equality test.
      atol: absolute tolerance for equality test.
    """
    with self.session() as session:
      with self.test_scope():
        pinp = array_ops.placeholder(
            dtypes.as_dtype(inp.dtype), inp.shape, name="a")
        output = op(pinp)
      result = session.run(output, {pinp: inp})
      if equality_test is None:
        self.assertEqual(output.dtype, expected.dtype)
        self.assertAllCloseAccordingToType(
            expected, result, rtol=rtol, atol=atol, bfloat16_rtol=0.03)
      else:
        equality_test(result, expected, rtol=rtol, atol=atol)

  def ListsAreClose(self, result, expected, rtol, atol):
    """Tests closeness of two lists of floats."""
    self.assertEqual(len(result), len(expected))
    for i in range(len(result)):
      self.assertAllClose(result[i], expected[i], rtol, atol)

  def AssertCloseAndSorted(self, result, expected, rtol, atol):
    """Tests that result and expeted are both close and sorted."""
    self.assertAllClose(result, expected, rtol, atol)
    self.assertAllEqual(np.sort(result), result)

  def AssertAllEqual(self, result, expected, rtol, atol):
    """Tests that result and expeted are exactly equal."""
    self.assertAllEqual(result, expected)

  def testAllTypeOps(self):
    for dtype in self.numeric_types - {np.int8, np.uint8}:
      self._assertOpOutputMatchesExpected(
          array_ops.diag, np.array([1, 2, 3, 4], dtype=dtype),
          np.array(
              [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]],
              dtype=dtype))
      self._assertOpOutputMatchesExpected(
          array_ops.diag_part,
          np.arange(36).reshape([2, 3, 2, 3]).astype(dtype),
          np.array([[0, 7, 14], [21, 28, 35]], dtype=dtype))
      self._assertOpOutputMatchesExpected(
          array_ops.diag, np.array([[1, 2], [3, 4]], dtype=dtype),
          np.array(
              [[[[1, 0], [0, 0]], [[0, 2], [0, 0]]], [[[0, 0], [3, 0]],
                                                      [[0, 0], [0, 4]]]],
              dtype=dtype))

      self._assertOpOutputMatchesExpected(
          array_ops.identity,
          np.array([[-1, 1]], dtype=dtype),
          expected=np.array([[-1, 1]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          array_ops.prevent_gradient,
          np.array([[-1, 1]], dtype=dtype),
          expected=np.array([[-1, 1]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          array_ops.squeeze,
          np.array([[[[[]]]]], dtype=dtype),
          expected=np.array([], dtype=dtype))
      self._assertOpOutputMatchesExpected(
          array_ops.squeeze,
          np.array([[[1], [2]]], dtype=dtype),
          expected=np.array([1, 2], dtype=dtype))
      self._assertOpOutputMatchesExpected(
          array_ops.squeeze,
          np.array([[[1]], [[2]]], dtype=dtype),
          expected=np.array([1, 2], dtype=dtype))
      self._assertOpOutputMatchesExpected(
          array_ops.squeeze,
          np.array([[[1, 2], [3, 4]]], dtype=dtype),
          expected=np.array([[1, 2], [3, 4]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          array_ops.stop_gradient,
          np.array([[-1, 1]], dtype=dtype),
          expected=np.array([[-1, 1]], dtype=dtype))

  def testLog(self):
    for dtype in self.float_types - {dtypes.bfloat16.as_numpy_dtype}:
      tol = 1e-4 if dtype == np.float32 else 1e-9
      # pylint: disable=invalid-unary-operand-type
      x = np.linspace(-np.e, np.e, num=1000, dtype=dtype)
      self._assertOpOutputMatchesExpected(
          math_ops.log, x, expected=np.log(x), atol=tol, rtol=tol)

      x = np.linspace(0., np.e * 1e-30, num=1000, dtype=dtype)
      self._assertOpOutputMatchesExpected(
          math_ops.log, x, expected=np.log(x), atol=tol, rtol=tol)

      x = np.linspace(0., np.pi * 1e30, num=1000, dtype=dtype)
      self._assertOpOutputMatchesExpected(
          math_ops.log, x, expected=np.log(x), atol=tol, rtol=tol)

  def testSin(self):
    for dtype in self.float_types - {dtypes.bfloat16.as_numpy_dtype}:
      tol = 1e-6 if dtype == np.float32 else 1e-12

      x = np.linspace(-4 * np.e, 4 * np.e, num=1000, dtype=dtype)
      self._assertOpOutputMatchesExpected(
          math_ops.sin, x, expected=np.sin(x), rtol=tol, atol=tol)

      x = np.linspace(0., np.e * 1e-30, num=1000, dtype=dtype)
      self._assertOpOutputMatchesExpected(
          math_ops.sin, x, expected=np.sin(x), rtol=tol, atol=tol)

      if dtype == np.float64:
        x = np.linspace(0., np.e * 1e8, num=1000, dtype=dtype)
        self._assertOpOutputMatchesExpected(
            math_ops.sin, x, expected=np.sin(x), rtol=tol, atol=1e-5)

  def testCos(self):
    for dtype in self.float_types - {dtypes.bfloat16.as_numpy_dtype}:
      tol = 1e-6 if dtype == np.float32 else 1e-12

      x = np.linspace(-4 * np.e, 4 * np.e, num=1000, dtype=dtype)
      self._assertOpOutputMatchesExpected(
          math_ops.cos, x, expected=np.cos(x), rtol=tol, atol=tol)

      x = np.linspace(0., np.e * 1e-30, num=1000, dtype=dtype)
      self._assertOpOutputMatchesExpected(
          math_ops.cos, x, expected=np.cos(x), rtol=tol, atol=tol)

      if dtype == np.float64:
        x = np.linspace(0., np.e * 1e8, num=1000, dtype=dtype)
        self._assertOpOutputMatchesExpected(
            math_ops.cos, x, expected=np.cos(x), rtol=tol, atol=1e-5)

  def testFloatOps(self):
    for dtype in self.float_types:
      x = np.arange(-0.90, 0.90, 0.25)
      self._assertOpOutputMatchesExpected(
          math_ops.acos, x.astype(dtype), expected=np.arccos(x).astype(dtype))
      self._assertOpOutputMatchesExpected(
          math_ops.asin, x.astype(dtype), expected=np.arcsin(x).astype(dtype))
      x = np.arange(-3, 3).reshape(1, 3, 2)
      self._assertOpOutputMatchesExpected(
          math_ops.atan, x.astype(dtype), expected=np.arctan(x).astype(dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.acosh,
          np.array([1, 2, 3, 4], dtype=dtype),
          expected=np.array(
              [0, 1.3169579, 1.76274717, 2.06343707], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.asinh,
          np.array([1, 2, 3, 4], dtype=dtype),
          expected=np.array(
              [0.88137359, 1.44363548, 1.81844646, 2.09471255], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.atanh,
          np.array([0.1, 0.2, 0.3, 0.4], dtype=dtype),
          expected=np.array(
              [0.10033535, 0.20273255, 0.3095196, 0.42364893], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.ceil,
          np.array([[-1.7, 1.2]], dtype=dtype),
          expected=np.array([[-1, 2]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.cosh,
          np.array([1, 2, 3, 4], dtype=dtype),
          expected=np.array(
              [1.54308063, 3.76219569, 10.067662, 27.30823284], dtype=dtype))

      # Disable float16 testing for now
      if dtype != np.float16:
        x = np.arange(-10, 10, 1).astype(dtype)
        with self.session() as session:
          erf_x = session.run(math_ops.erf(x))
          erfc_x = session.run(math_ops.erfc(x))

        self._assertOpOutputMatchesExpected(math_ops.erf, x, expected=erf_x)
        self._assertOpOutputMatchesExpected(math_ops.erfc, x, expected=erfc_x)

      self._assertOpOutputMatchesExpected(
          math_ops.exp,
          np.array([[-1, 1]], dtype=dtype),
          expected=np.array([[0.36787945, 2.7182817]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.expm1,
          np.array([[-1, 1]], dtype=dtype),
          expected=np.array([[-0.63212056, 1.71828183]], dtype=dtype),
          rtol=1e-5)

      self._assertOpOutputMatchesExpected(
          math_ops.floor,
          np.array([[-1.7, 1.2]], dtype=dtype),
          expected=np.array([[-2, 1]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.is_finite,
          np.array([[np.NINF, -2, -1, 0, 0.5, 1, 2, np.inf, np.nan]],
                   dtype=dtype),
          expected=np.array([[0, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=np.bool_))

      # Tests for tf.nn ops.
      self._assertOpOutputMatchesExpected(
          nn_ops.l2_loss, np.array([[[]]], dtype=dtype), expected=dtype(0))

      self._assertOpOutputMatchesExpected(nn_ops.l2_loss, dtype(4), dtype(8))

      self._assertOpOutputMatchesExpected(
          nn_ops.l2_loss, np.array([[-2, 4]], dtype=dtype), expected=dtype(10))

      self._assertOpOutputMatchesExpected(
          math_ops.reciprocal,
          np.array([[1, 2]], dtype=dtype),
          expected=np.array([[1, 0.5]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.log,
          np.array([[1, 2]], dtype=dtype),
          expected=np.array([[0, 0.69314718]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.sin,
          np.array([[1, 2]], dtype=dtype),
          expected=np.array([[0.841478, 0.909302]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.cos,
          np.array([[1, 2]], dtype=dtype),
          expected=np.array([[0.540297, -0.41614]], dtype=dtype))

      # Confirm that log1p will remain precise across a range of small values.
      self._assertOpOutputMatchesExpected(
          math_ops.log1p,
          np.array([[1e-14, 1e-15, 0.6, 2] + [x * 1e-5 for x in range(1, 20)]],
                   dtype=dtype),
          expected=np.log1p(
              np.array(
                  [[1e-14, 1e-15, 0.6, 2] + [x * 1e-5 for x in range(1, 20)]],
                  dtype=dtype)).astype(dtype),
          rtol=1e-15 if dtype == np.float64 else 1e-4,
          atol=1e-15 if dtype == np.float64 else 1e-4)

      self._assertOpOutputMatchesExpected(
          math_ops.rint,
          np.array(
              [[-1.7, 1.2, 4.0, 0.0], [-3.5, -2.5, -1.5, -0.5],
               [0.5, 1.5, 2.5, 3.5]],
              dtype=dtype),
          expected=np.array(
              [[-2, 1, 4, 0], [-4, -2, -2, 0], [0, 2, 2, 4]], dtype=dtype))
      self._assertOpOutputMatchesExpected(
          math_ops.round,
          np.array(
              [[-1.7, 1.2, 4.0, 0.0], [-3.5, -2.5, -1.5, -0.5],
               [0.5, 1.5, 2.5, 3.5]],
              dtype=dtype),
          expected=np.array(
              [[-2, 1, 4, 0], [-4, -2, -2, 0], [0, 2, 2, 4]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.rsqrt,
          np.array([[4, 16]], dtype=dtype),
          expected=np.array([[0.5, 0.25]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.sigmoid,
          np.array([[1, 1, 1, 1], [1, 2, 3, 4]], dtype=dtype),
          expected=np.array(
              [[0.7310586, 0.7310586, 0.7310586, 0.7310586],
               [0.7310586, 0.880797, 0.95257413, 0.98201376]],
              dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.sigmoid,
          np.array([-300, -150, 0, 150, 300], dtype=dtype),
          expected=np.array([0, 0, 0.5, 1, 1], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.sinh,
          np.array([1, 2, 3, 4], dtype=dtype),
          expected=np.array(
              [1.17520119, 3.62686041, 10.01787493, 27.2899172], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.sqrt,
          np.array([[4, 9]], dtype=dtype),
          expected=np.array([[2, 3]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.tan,
          np.array([1, 2, 3, 4], dtype=dtype),
          expected=np.array(
              [1.55740772, -2.18503986, -0.14254654, 1.15782128], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.tanh,
          np.array([[1, 2, 3, 4], [np.inf, -np.inf, np.nan, 20],
                    [19, -19, 22, -22]],
                   dtype=dtype),
          expected=np.array(
              [[0.76159418, 0.96402758, 0.99505478, 0.99932933],
               [1.0, -1.0, np.nan, 1.0], [1.0, -1.0, 1.0, -1.0]],
              dtype=dtype))

      self._assertOpOutputMatchesExpected(
          nn_ops.log_softmax,
          np.array([[1, 1, 1, 1], [1, 2, 3, 4]], dtype=dtype),
          expected=np.array(
              [[-1.3862944, -1.3862944, -1.3862944, -1.3862944],
               [-3.4401896, -2.4401896, -1.4401897, -0.44018969]],
              dtype=dtype))

      self._assertOpOutputMatchesExpected(
          nn_ops.elu,
          np.array([[-1, 0, 1, -1e-6]], dtype=dtype),
          expected=np.array([[-0.63212056, 0, 1, -9.999995e-07]], dtype=dtype),
          rtol=1e-5,
          atol=1e-6)

      self._assertOpOutputMatchesExpected(
          nn_ops.selu,
          np.array([[-1, 0, 1, -1e-5]], dtype=dtype),
          expected=np.array(
              [[-1.11133074, 0., 1.05070099, -1.758090550379974e-05]],
              dtype=dtype),
          rtol=1e-5,
          atol=1e-6)

      self._assertOpOutputMatchesExpected(
          nn_ops.relu,
          np.array([[-1, 1]], dtype=dtype),
          expected=np.array([[0, 1]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          nn_ops.relu6,
          np.array([[-0.05, 6.05, 5]], dtype=dtype),
          expected=np.array([[0, 6, 5]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          nn_ops.leaky_relu,
          np.array([[-2, -1, 0, 1, 2]], dtype=dtype),
          expected=np.array([[-0.4, -0.2, 0.0, 1.0, 2.0]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          nn_ops.softmax,
          np.array([1, 2, 3, 4], dtype=dtype),
          expected=np.array([0.032058604, 0.087144323, 0.23688284, 0.64391428],
                            dtype=dtype))

      self._assertOpOutputMatchesExpected(
          nn_ops.softmax,
          np.array([[1, 1, 1, 1], [1, 2, 3, 4]], dtype=dtype),
          expected=np.array(
              [[0.25, 0.25, 0.25, 0.25],
               [0.032058604, 0.087144323, 0.23688284, 0.64391428]],
              dtype=dtype))

      self._assertOpOutputMatchesExpected(
          nn_ops.softmax,
          np.array([[[1, 1], [1, 1]], [[1, 2], [3, 4]]], dtype=dtype),
          expected=np.array(
              [[[0.5, 0.5], [0.5, 0.5]],
               [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]],
              dtype=dtype))

      self._assertOpOutputMatchesExpected(
          nn_ops.softsign,
          np.array([[-2, -1, 0, 1, 2]], dtype=dtype),
          expected=np.array(
              [[-0.66666669, -0.5, 0, 0.5, 0.66666669]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.sign,
          np.array([[-2.0, -1.0, -0.0, +0.0, 1.0, 2.0,
                     float("nan")]],
                   dtype=dtype),
          expected=np.array([[-1.0, -1.0, -0.0, +0.0, 1.0, 1.0,
                              float("nan")]],
                            dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.is_finite,
          np.array([[42, float("inf"), -123], [float("nan"), 0, -0.0]],
                   dtype=dtype),
          expected=np.array([[True, False, True], [False, True, True]],
                            dtype=np.bool_))

      self._assertOpOutputMatchesExpected(
          math_ops.lgamma,
          np.array(0.5, dtype=dtype),
          expected=np.array(np.log(np.pi) / 2, dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.lgamma,
          np.array(
              [[1, 2, 3], [4, 5, 6], [1 / 2, 3 / 2, 5 / 2],
               [-3 / 2, -7 / 2, -11 / 2]],
              dtype=dtype),
          expected=np.array(
              [
                  [0, 0, np.log(2.0)],
                  [np.log(6.0), np.log(24.0),
                   np.log(120)],
                  [
                      np.log(np.pi) / 2,
                      np.log(np.pi) / 2 - np.log(2),
                      np.log(np.pi) / 2 - np.log(4) + np.log(3)
                  ],
                  [
                      np.log(np.pi) / 2 - np.log(3) + np.log(4),
                      np.log(np.pi) / 2 - np.log(105) + np.log(16),
                      np.log(np.pi) / 2 - np.log(10395) + np.log(64),
                  ],
              ],
              dtype=dtype))

      # The actual result is complex. Take the real part.
      self._assertOpOutputMatchesExpected(
          math_ops.lgamma,
          np.array([-1 / 2, -5 / 2, -9 / 2], dtype=dtype),
          expected=np.array(
              [
                  np.log(np.pi) / 2 + np.log(2),
                  np.log(np.pi) / 2 - np.log(15) + np.log(8),
                  np.log(np.pi) / 2 - np.log(945) + np.log(32),
              ],
              dtype=dtype),
          atol=1e-4)

      self._assertOpOutputMatchesExpected(
          math_ops.digamma,
          np.array(
              [[1.0, 0.5, 1 / 3.0], [0.25, 1 / 6.0, 0.125], [2.0, 3.0, 4.0],
               [6.0, 8.0, 9.0]],
              dtype=dtype),
          expected=np.array(
              [
                  [
                      -np.euler_gamma, -2 * np.log(2) - np.euler_gamma,
                      -np.pi / 2 / np.sqrt(3) - 3 * np.log(3) / 2 -
                      np.euler_gamma
                  ],
                  [
                      -np.pi / 2 - 3 * np.log(2) - np.euler_gamma,
                      -np.pi * np.sqrt(3) / 2 - 2 * np.log(2) -
                      3 * np.log(3) / 2 - np.euler_gamma,
                      -np.pi / 2 - 4 * np.log(2) -
                      (np.pi + np.log(2 + np.sqrt(2)) - np.log(2 - np.sqrt(2)))
                      / np.sqrt(2) - np.euler_gamma
                  ],
                  [
                      1 - np.euler_gamma, 1.5 - np.euler_gamma,
                      11 / 6.0 - np.euler_gamma
                  ],
                  [
                      137 / 60.0 - np.euler_gamma, 363 / 140.0 - np.euler_gamma,
                      761 / 280.0 - np.euler_gamma
                  ],
              ],
              dtype=dtype))

  def testSigmoidNumericalStability(self):
    for dtype in self.float_types:
      if dtype != np.float16:
        self._assertOpOutputMatchesExpected(
            lambda x: math_ops.sigmoid(x) / math_ops.log1p(math_ops.exp(x)),
            np.array([-40, 40], dtype=dtype),
            expected=np.array([1.0, 0.025], dtype=dtype))

  def testQuantizeAndDequantize(self):
    for dtype in self.float_types:

      def quantize_and_dequantize_v2(x):
        return array_ops.quantize_and_dequantize(
            x, -127, 127, signed_input=True, num_bits=8)

      def quantize_and_dequantize_v3(x):
        return array_ops.quantize_and_dequantize_v3(
            x, -127, 127, num_bits=8, signed_input=True, range_given=False)

      def quantize_and_dequantize_v4(x):
        return array_ops.quantize_and_dequantize_v2(
            x, -127, 127, signed_input=True, num_bits=8)

      test_fns = (quantize_and_dequantize_v2, quantize_and_dequantize_v3,
                  quantize_and_dequantize_v4)
      for test_fn in test_fns:
        self._assertOpOutputMatchesExpected(
            test_fn,
            np.array([-1, -0.5, 0, 0.3], dtype=dtype),
            expected=np.array([-1., -0.5, 0., 0.296875], dtype=dtype))

      def quantize_and_dequantize_v2_round_half_up(x):
        return array_ops.quantize_and_dequantize(
            x,
            -1,
            1.0,
            signed_input=True,
            num_bits=8,
            range_given=True,
            round_mode="HALF_UP")

      self._assertOpOutputMatchesExpected(
          quantize_and_dequantize_v2_round_half_up,
          np.array([-0.8, -0.5, 0, 0.3, 0.8, -2, 33], dtype=dtype),
          expected=np.array([
              -102.0 / 127,
              -63.0 / 127,
              0,
              38.0 / 127,
              102.0 / 127,
              -128.0 / 127,
              1,
          ],
                            dtype=dtype))

      def quantize_and_dequantize_v2_round_half_to_even(x):
        return array_ops.quantize_and_dequantize(
            x,
            -1.0,
            1.0,
            signed_input=True,
            num_bits=8,
            range_given=True,
            round_mode="HALF_TO_EVEN")

      self._assertOpOutputMatchesExpected(
          quantize_and_dequantize_v2_round_half_to_even,
          np.array(
              [
                  -0.8,
                  # The -0.5 should become -63.5 after scaling and with
                  # rounding this should become -64. But with the test
                  # unary_ops_test_cpu_ondemand, this fails as the result
                  # before scaling becomes -63.499996 and gets rounded to -63.
                  # TODO(sreenik): Some one more familiar with this test needs
                  # to take a look and resolve this. This works on all other
                  # variations of the platform like cpu, and gpu.
                  # -0.5,
                  0,
                  0.3,
                  0.8,
                  -2,
                  33
              ],
              dtype=dtype),
          expected=np.array(
              [
                  -102.0 / 127,
                  # -64.0 / 127,
                  0,
                  38.0 / 127,
                  102.0 / 127,
                  -128.0 / 127,
                  1,
              ],
              dtype=dtype))

  def testComplexOps(self):
    for dtype in self.complex_types:

      self._assertOpOutputMatchesExpected(
          math_ops.acosh,
          np.array([0.1, 0.2j, 0.3 - 0.1j, 0.4 + 0.5j], dtype=dtype),
          expected=np.arccosh(
              np.array([0.1, 0.2j, 0.3 - 0.1j, 0.4 + 0.5j], dtype=dtype)))

      self._assertOpOutputMatchesExpected(
          math_ops.asinh,
          np.array([0.1, 0.2j, 0.3 - 0.1j, 0.4 + 0.5j], dtype=dtype),
          expected=np.arcsinh(
              np.array([0.1, 0.2j, 0.3 - 0.1j, 0.4 + 0.5j], dtype=dtype)))

      self._assertOpOutputMatchesExpected(
          math_ops.atanh,
          np.array([0.1, 0.2j, 0.3 - 0.1j, 0.4 + 0.5j], dtype=dtype),
          expected=np.arctanh(
              np.array([0.1, 0.2j, 0.3 - 0.1j, 0.4 + 0.5j], dtype=dtype)))

      self._assertOpOutputMatchesExpected(
          math_ops.cosh,
          np.array([1j, 2 - 3j, 3, 4 + 2j], dtype=dtype),
          expected=np.cosh(np.array([1j, 2 - 3j, 3, 4 + 2j], dtype=dtype)))

      self._assertOpOutputMatchesExpected(
          math_ops.sinh,
          np.array([1, 2j, 2 - 3j, 4 + 5j], dtype=dtype),
          expected=np.sinh(np.array([1, 2j, 2 - 3j, 4 + 5j], dtype=dtype)))

      self._assertOpOutputMatchesExpected(
          math_ops.exp,
          np.array([[-1 + 2j, 3j, 2 - 3j]], dtype=dtype),
          expected=np.exp(np.array([[-1 + 2j, 3j, 2 - 3j]], dtype=dtype)))

      self._assertOpOutputMatchesExpected(
          math_ops.expm1,
          np.array([[-1 + 2j, 3j, 2 - 3j]], dtype=dtype),
          expected=np.expm1(np.array([[-1 + 2j, 3j, 2 - 3j]], dtype=dtype)),
          rtol=1e-6,
          atol=1e-6)

      # For real part close to zero, or imaginary part close to a multiple of
      # pi.

      self._assertOpOutputMatchesExpected(
          math_ops.expm1,
          np.array([[1e-11 + 1j, -1e-11 - 1j, 1. + 1e-11j,
                     -1. - 1e-11j, 1e-13j + 1e-13j]], dtype=dtype),
          # TODO(srvasude): Use numpy as the source of truth after we depend on
          # latest numpy with this pull request:
          # https://github.com/numpy/numpy/pull/15110.
          # The numbers below were generated by scipy.special.expm1.
          expected=np.array([[
              -4.59697694e-01+8.41470985e-01j,
              -4.59697694e-01-8.41470985e-01j,
              1.71828183e+00+2.71828183e-11j,
              -6.32120559e-01-3.67879441e-12j,
              -2.00000000e-26+2.00000000e-13j]], dtype=dtype),
          rtol=1e-09,
          atol=1e-20)

      self._assertOpOutputMatchesExpected(
          math_ops.reciprocal,
          np.array([[1, 2j, 2 + 3j]], dtype=dtype),
          expected=1.0 / np.array([[1, 2j, 2 + 3j]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.log,
          np.array([[5j, 3 - 2j]], dtype=dtype),
          expected=np.log(np.array([[5j, 3 - 2j]], dtype=dtype)))

      self._assertOpOutputMatchesExpected(
          math_ops.sin,
          np.array([[5j, 3 - 2j]], dtype=dtype),
          expected=np.sin(np.array([[5j, 3 - 2j]], dtype=dtype)))

      self._assertOpOutputMatchesExpected(
          math_ops.cos,
          np.array([[5j, 3 - 2j]], dtype=dtype),
          expected=np.cos(np.array([[5j, 3 - 2j]], dtype=dtype)))

      self._assertOpOutputMatchesExpected(
          math_ops.log1p,
          np.array([[1e-14, 1e-15j, 0.6 - 0.3j]], dtype=dtype),
          expected=np.log1p(
              np.array([[1e-14, 1e-15j, 0.6 - 0.3j]], dtype=dtype)),
          rtol=1e-4,
          atol=1e-6)

      val = np.array([1, 2j, 2 - 3j, 4 + 5j], dtype=dtype)
      self._assertOpOutputMatchesExpected(
          math_ops.rsqrt, val, expected=1 / np.sqrt(val))

      self._assertOpOutputMatchesExpected(
          math_ops.sigmoid, val, expected=1 / (1 + np.exp(-val)))

      self._assertOpOutputMatchesExpected(
          math_ops.sqrt, val, expected=np.sqrt(val))

      self._assertOpOutputMatchesExpected(
          math_ops.tanh,
          np.array([1, 2j, 2 - 3j, 4 + 5j], dtype=dtype),
          expected=np.tanh(np.array([1, 2j, 2 - 3j, 4 + 5j], dtype=dtype)))

      self._assertOpOutputMatchesExpected(
          math_ops.tan,
          np.array([1, 2j, 2 - 3j, 4 + 5j], dtype=dtype),
          expected=np.tan(np.array([1, 2j, 2 - 3j, 4 + 5j], dtype=dtype)))

      ctypes = {np.complex64: np.float32, np.complex128: np.float64}
      self._assertOpOutputMatchesExpected(
          math_ops.abs,
          np.array([[3 - 4j, -1j, np.inf]], dtype=dtype),
          expected=np.array([[5, 1, np.inf]], dtype=ctypes[dtype]))

      self._assertOpOutputMatchesExpected(
          math_ops.negative,
          np.array([[-1 + 2j, -3j]], dtype=dtype),
          expected=np.array([[1 - 2j, 3j]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.square,
          np.array([[-2 - 3j, 3 + 4j, 5j]], dtype=dtype),
          expected=np.array([[-2 - 3j, 3 + 4j, 5j]], dtype=dtype)**2)

      self._assertOpOutputMatchesExpected(
          array_ops.zeros_like,
          np.array([[4j, 3 - 2j], [2, -1j]], dtype=dtype),
          expected=np.array([[0, 0], [0, 0]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          array_ops.ones_like,
          np.array([[-4j, 3 + 2j], [2, -1j]], dtype=dtype),
          expected=np.array([[1, 1], [1, 1]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.angle,
          np.array([1 + 3j, -4 + 7j, 2.7, -3j], dtype=dtype),
          expected=np.angle(np.array([1 + 3j, -4 + 7j, 2.7, -3j], dtype=dtype)))

      self._assertOpOutputMatchesExpected(
          math_ops.conj,
          np.array([1 + 3j, -4 + 7j, 2.7, -3j], dtype=dtype),
          expected=np.array([1 - 3j, -4 - 7j, 2.7, 3j], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.imag,
          np.array([1 + 3j, -4 + 7j, 2.7, -3j], dtype=dtype),
          expected=np.array([3, 7, 0, -3], dtype=ctypes[dtype]))

      self._assertOpOutputMatchesExpected(
          math_ops.real,
          np.array([1 + 3j, -4 + 7j, 2.7, -3j], dtype=dtype),
          expected=np.array([1, -4, 2.7, 0], dtype=ctypes[dtype]))

  def testIntOps(self):
    for dtype in self.int_types:
      self._assertOpOutputMatchesExpected(
          bitwise_ops.invert,
          np.array([0, -1, 1, 16, 42], dtype=dtype),
          expected=np.array([-1, 0, -2, -17, -43], dtype=dtype))

      # Test population_count for array inputs.
      raw_inputs = [
          0, 1, -1, 3, -3, 5, -5, 14, -14, 127, 128, 255, 256, 65535, 65536,
          2**31 - 1, 2**31, 2**32 - 1, 2**32, -2**32 + 1, -2**32, -2**63 + 1,
          2**63 - 1
      ]
      # Only choose inputs which fit in the int dtype.
      raw_inputs = list(
          filter(lambda x: np.iinfo(dtype).min <= x <= np.iinfo(dtype).max,
                 raw_inputs))
      inputs = np.array(raw_inputs, dtype=dtype)

      def count_bits(x):
        return sum(bin(z).count("1") for z in six.iterbytes(x.tobytes()))

      truth = [count_bits(x) for x in inputs]
      self._assertOpOutputMatchesExpected(
          bitwise_ops.population_count,
          inputs,
          expected=np.array(truth, dtype=np.uint8),
          equality_test=self.AssertAllEqual)

      # Test population_count for scalar inputs.
      for raw_inp in raw_inputs:
        inp = dtype(raw_inp)
        truth = count_bits(inp)
        self._assertOpOutputMatchesExpected(
            bitwise_ops.population_count,
            inp,
            expected=np.uint8(truth),
            equality_test=self.AssertAllEqual)

  def testNumericOps(self):
    for dtype in self.numeric_types - {np.int8, np.uint8}:
      self._assertOpOutputMatchesExpected(
          math_ops.abs,
          np.array([[2, -1]], dtype=dtype),
          expected=np.array([[2, 1]], dtype=np.real(dtype(0)).dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.negative,
          np.array([[-1, 1]], dtype=dtype),
          expected=np.array([[1, -1]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          math_ops.square,
          np.array([[-2, 3]], dtype=dtype),
          expected=np.array([[4, 9]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          array_ops.zeros_like,
          np.array([[4, 3], [2, 1]], dtype=dtype),
          expected=np.array([[0, 0], [0, 0]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          array_ops.ones_like,
          np.array([[4, 3], [2, 1]], dtype=dtype),
          expected=np.array([[1, 1], [1, 1]], dtype=dtype))

  # TODO(phawkins): these tests fail unless fastmath optimizations
  # are disabled. Use more robust IsInf/IsNaN detection and enable these
  # tests.
  @unittest.skip("test case fails in fast-math mode")
  def testIsInfAndIsNan(self):
    for dtype in self.float_types:
      self._assertOpOutputMatchesExpected(
          math_ops.is_inf,
          np.array([[np.NINF, -2, -1, 0, 0.5, 1, 2, np.inf, np.nan]],
                   dtype=dtype),
          expected=np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.bool_))
      self._assertOpOutputMatchesExpected(
          math_ops.is_nan,
          np.array([[np.NINF, -2, -1, 0, 0.5, 1, 2, np.inf, np.nan]],
                   dtype=dtype),
          expected=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.bool_))
      self._assertOpOutputMatchesExpected(
          math_ops.sign,
          np.array([[np.nan]], dtype=dtype),
          expected=np.array([[0.0]], dtype=dtype))

  def testLogicalOps(self):
    self._assertOpOutputMatchesExpected(
        math_ops.logical_not,
        np.array([[True, False], [False, True]], dtype=np.bool_),
        expected=np.array([[False, True], [True, False]], dtype=np.bool_))

  def testBiasAddGrad(self):
    self._assertOpOutputMatchesExpected(
        gen_nn_ops.bias_add_grad,
        np.array([[1., 2.], [3., 4.]], dtype=np.float32),
        expected=np.array([4., 6.], dtype=np.float32))

    self._assertOpOutputMatchesExpected(
        lambda x: gen_nn_ops.bias_add_grad(x, data_format="NCHW"),
        np.array(
            [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float32),
        expected=np.array([14., 22.], dtype=np.float32))

  def testCast(self):
    shapes = [[], [4], [2, 3], [2, 0, 4]]
    types = {
        dtypes.bool, dtypes.float32, dtypes.float64, dtypes.complex64,
        dtypes.int32, dtypes.int64, dtypes.uint32, dtypes.uint64
    }
    for src_type in types:
      for dst_type in types:
        src_np_dtype = src_type.as_numpy_dtype
        dst_np_dtype = dst_type.as_numpy_dtype

        for shape in shapes:
          src = np.arange(np.prod(shape)).astype(src_np_dtype)

          if src_type in self.complex_tf_types:
            src += (np.arange(np.prod(shape)) * 2j).astype(src_np_dtype)
          src = src.reshape(shape)
          dst = src.astype(dst_np_dtype)
          self._assertOpOutputMatchesExpected(
              lambda x, dst_type=dst_type: math_ops.cast(x, dst_type),
              src,
              expected=dst)

        # Check special values.
        if src_type.is_integer:
          imin = np.iinfo(src_np_dtype).min
          imax = np.iinfo(src_np_dtype).max
          src = np.array([imin, imax, 0, 1, -1], dtype=src_np_dtype)
        elif src_type in self.float_tf_types:
          if dst_type.is_integer:
            imin = np.iinfo(dst_np_dtype).min
            imax = np.iinfo(dst_np_dtype).max // 2
            src = np.array([imin, imax, 0, 1], dtype=src_np_dtype)
          elif dst_type in self.float_tf_types:
            fmin = np.finfo(dst_np_dtype).min
            fmax = np.finfo(dst_np_dtype).max
            tiny = np.finfo(dst_np_dtype).tiny
            eps = np.finfo(dst_np_dtype).eps
            src = np.array(
                [fmin, fmax, np.nan, eps, -eps, tiny, -tiny, np.inf, -np.inf],
                dtype=src_np_dtype)
        dst = src.astype(dst_np_dtype)
        self._assertOpOutputMatchesExpected(
            lambda x, dst_type=dst_type: math_ops.cast(x, dst_type),
            src,
            expected=dst)

  def testBitcast(self):
    self._assertOpOutputMatchesExpected(
        lambda x: array_ops.bitcast(x, dtypes.int32),
        np.array([1, 0x3f800000], np.int32),
        expected=np.array([1, 0x3f800000], np.int32))
    self._assertOpOutputMatchesExpected(
        lambda x: array_ops.bitcast(x, dtypes.float32),
        np.array([1, 0x3f800000], np.int32),
        expected=np.array([1e-45, 1.0], np.float32))
    self._assertOpOutputMatchesExpected(
        lambda x: array_ops.bitcast(x, dtypes.int32),
        np.array([1e-45, 1.0], np.float32),
        expected=np.array([1, 0x3f800000], np.int32))
    if np.int64 in self.numeric_types:
      self._assertOpOutputMatchesExpected(
          lambda x: array_ops.bitcast(x, dtypes.int64),
          np.array([1, 0x100000003f800000], np.uint64),
          expected=np.array([1, 0x100000003f800000], np.int64))
      self._assertOpOutputMatchesExpected(
          lambda x: array_ops.bitcast(x, dtypes.uint64),
          np.array([1, 0x100000003f800000], np.int64),
          expected=np.array([1, 0x100000003f800000], np.uint64))

  def testBitcastInt8ToFloat(self):
    self._assertOpOutputMatchesExpected(
        lambda x: array_ops.bitcast(x, dtypes.float32),
        np.array([[1, 0, 0, 0], [0xd0, 0x0f, 0x49, 0x40]], np.int8),
        expected=np.array([1e-45, 3.14159], np.float32))
    self._assertOpOutputMatchesExpected(
        lambda x: array_ops.bitcast(x, dtypes.np.int8),
        np.array([1e-45, 3.14159], np.float32),
        expected=np.array([[1, 0, 0, 0], [0xd0, 0x0f, 0x49, 0x40]], np.int8))

  def testInvertPermutation(self):
    for np_dtype in [np.int32, np.int64]:
      self._assertOpOutputMatchesExpected(
          array_ops.invert_permutation,
          np.array([1, 2, 0], np_dtype),
          expected=np.array([2, 0, 1], dtype=np_dtype))

  def testInvertPermutationTwiceIsNoop(self):

    def invert_twice(x):
      return array_ops.invert_permutation(array_ops.invert_permutation(x))

    for np_dtype in [np.int32, np.int64]:
      self._assertOpOutputMatchesExpected(
          invert_twice,
          np.array([1, 2, 0], np_dtype),
          expected=np.array([1, 2, 0], dtype=np_dtype))

  def testRank(self):
    rank_op = lambda x: array_ops.rank_internal(x, optimize=False)
    for dtype in self.numeric_types:
      self._assertOpOutputMatchesExpected(
          rank_op, dtype(7), expected=np.int32(0))
      self._assertOpOutputMatchesExpected(
          rank_op, np.array([[], []], dtype=dtype), expected=np.int32(2))
      self._assertOpOutputMatchesExpected(
          rank_op, np.array([-1, 1], dtype=dtype), expected=np.int32(1))
      self._assertOpOutputMatchesExpected(
          rank_op, np.array([[-1, 1]], dtype=dtype), expected=np.int32(2))
      self._assertOpOutputMatchesExpected(
          rank_op,
          np.array([[-1], [1], [4]], dtype=dtype),
          expected=np.int32(2))

  def testShape(self):
    shape_op = lambda x: array_ops.shape_internal(x, optimize=False)
    for dtype in self.numeric_types:
      self._assertOpOutputMatchesExpected(
          shape_op, dtype(7), expected=np.array([], dtype=np.int32))
      self._assertOpOutputMatchesExpected(
          shape_op,
          np.array([[], []], dtype=dtype),
          expected=np.array([2, 0], dtype=np.int32))
      self._assertOpOutputMatchesExpected(
          shape_op,
          np.array([-1, 1], dtype=dtype),
          expected=np.array([2], dtype=np.int32))
      self._assertOpOutputMatchesExpected(
          shape_op,
          np.array([[-1, 1]], dtype=dtype),
          expected=np.array([1, 2], dtype=np.int32))
      self._assertOpOutputMatchesExpected(
          shape_op,
          np.array([[-1], [1], [4]], dtype=dtype),
          expected=np.array([3, 1], dtype=np.int32))

  def testSize(self):
    size_op = lambda x: array_ops.size_internal(x, optimize=False)
    for dtype in self.numeric_types:
      self._assertOpOutputMatchesExpected(
          size_op, dtype(7), expected=np.int32(1))
      self._assertOpOutputMatchesExpected(
          size_op, np.array([[], []], dtype=dtype), expected=np.int32(0))
      self._assertOpOutputMatchesExpected(
          size_op, np.array([-1, 1], dtype=dtype), expected=np.int32(2))
      self._assertOpOutputMatchesExpected(
          size_op, np.array([[-1, 1]], dtype=dtype), expected=np.int32(2))
      self._assertOpOutputMatchesExpected(
          size_op,
          np.array([[-1], [1], [4]], dtype=dtype),
          expected=np.int32(3))

  def testSizeWithInt64OutType(self):

    def size_op(x):
      return array_ops.size_internal(x, optimize=False, out_type=np.int64)

    for dtype in self.numeric_types:
      self._assertOpOutputMatchesExpected(
          size_op,
          np.array([[-1], [1], [4]], dtype=dtype),
          expected=np.int64(3))

  def testUnpack(self):
    self._assertOpOutputMatchesExpected(
        array_ops.unstack,
        np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32),
        expected=[
            np.array([1., 2.], dtype=np.float32),
            np.array([3., 4.], dtype=np.float32),
            np.array([5., 6.], dtype=np.float32),
        ],
        equality_test=self.ListsAreClose)

    self._assertOpOutputMatchesExpected(
        lambda x: array_ops.unstack(x, axis=1),
        np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32),
        expected=[
            np.array([1., 3., 5.], dtype=np.float32),
            np.array([2., 4., 6.], dtype=np.float32),
        ],
        equality_test=self.ListsAreClose)

  def testDepthToSpace(self):

    def make_op(data_format):

      def op(x):
        return array_ops.depth_to_space(
            x, block_size=2, data_format=data_format)

      return op

    for dtype in self.numeric_types:
      for data_format in ["NCHW", "NHWC"]:
        self._assertOpOutputMatchesExpected(
            make_op(data_format),
            nhwc_to_format(
                np.array([[[[1, 2, 3, 4]]]], dtype=dtype), data_format),
            expected=nhwc_to_format(
                np.array([[[[1], [2]], [[3], [4]]]], dtype=dtype), data_format))

        self._assertOpOutputMatchesExpected(
            make_op(data_format),
            nhwc_to_format(
                np.array(
                    [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]], dtype=dtype),
                data_format),
            expected=nhwc_to_format(
                np.array(
                    [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]],
                    dtype=dtype), data_format))

        self._assertOpOutputMatchesExpected(
            make_op(data_format),
            nhwc_to_format(
                np.array(
                    [[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12],
                                                     [13, 14, 15, 16]]]],
                    dtype=dtype), data_format),
            expected=nhwc_to_format(
                np.array(
                    [[[[1], [2], [5], [6]], [[3], [4], [7], [8]],
                      [[9], [10], [13], [14]], [[11], [12], [15], [16]]]],
                    dtype=dtype), data_format))

      self._assertOpOutputMatchesExpected(
          make_op("NCHW_VECT_C"),
          np.arange(32, dtype=dtype).reshape((1, 8, 1, 1, 4)),
          expected=np.array([[[[[0, 1, 2, 3], [8, 9, 10, 11]],
                               [[16, 17, 18, 19], [24, 25, 26, 27]]],
                              [[[4, 5, 6, 7], [12, 13, 14, 15]],
                               [[20, 21, 22, 23], [28, 29, 30, 31]]]]],
                            dtype=dtype))

  def testSpaceToDepth(self):

    def make_op(data_format):

      def op(x):
        return array_ops.space_to_depth(
            x, block_size=2, data_format=data_format)

      return op

    for dtype in self.numeric_types:
      for data_format in ["NCHW", "NHWC"]:
        self._assertOpOutputMatchesExpected(
            make_op(data_format),
            nhwc_to_format(
                np.array([[[[1], [2]], [[3], [4]]]], dtype=dtype), data_format),
            expected=nhwc_to_format(
                np.array([[[[1, 2, 3, 4]]]], dtype=dtype), data_format))

        self._assertOpOutputMatchesExpected(
            make_op(data_format),
            nhwc_to_format(
                np.array(
                    [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]],
                    dtype=dtype), data_format),
            expected=nhwc_to_format(
                np.array(
                    [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]], dtype=dtype),
                data_format))

        self._assertOpOutputMatchesExpected(
            make_op(data_format),
            nhwc_to_format(
                np.array(
                    [[[[1], [2], [5], [6]], [[3], [4], [7], [8]],
                      [[9], [10], [13], [14]], [[11], [12], [15], [16]]]],
                    dtype=dtype), data_format),
            expected=nhwc_to_format(
                np.array(
                    [[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12],
                                                     [13, 14, 15, 16]]]],
                    dtype=dtype), data_format))

      self._assertOpOutputMatchesExpected(
          make_op("NCHW_VECT_C"),
          np.arange(32, dtype=dtype).reshape((1, 2, 2, 2, 4)),
          expected=np.array(
              [[[[[0, 1, 2, 3]]], [[[16, 17, 18, 19]]], [[[4, 5, 6, 7]]],
                [[[20, 21, 22, 23]]], [[[8, 9, 10, 11]]], [[[24, 25, 26, 27]]],
                [[[12, 13, 14, 15]]], [[[28, 29, 30, 31]]]]],
              dtype=dtype))

  def _assertSoftplusMatchesExpected(self,
                                     features,
                                     dtype,
                                     equality_test=None,
                                     rtol=1e-6,
                                     atol=9.1e-6):
    features = np.array(features, dtype=dtype)
    zero = np.asarray(0).astype(dtype)
    expected = np.logaddexp(zero, features).astype(dtype)
    self._assertOpOutputMatchesExpected(
        nn_ops.softplus,
        features,
        expected=expected,
        equality_test=equality_test,
        rtol=rtol,
        atol=atol)

  def testSoftplus(self):
    for dtype in self.float_types & {dtypes.float32, dtypes.float64}:
      self._assertSoftplusMatchesExpected([[-2, 0, 8]], dtype)
      self._assertSoftplusMatchesExpected(
          [[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]], dtype)
      if dtype == dtypes.bfloat16.as_numpy_dtype:
        log_eps = np.log(np.finfo(np.float32).eps)
      else:
        log_eps = np.log(np.finfo(dtype).eps)
      one = dtype(1)
      ten = dtype(10)
      self._assertSoftplusMatchesExpected([
          log_eps, log_eps - one, log_eps + one, log_eps - ten, log_eps + ten,
          -log_eps, -log_eps - one, -log_eps + one, -log_eps - ten,
          -log_eps + ten
      ], dtype)

      self._assertSoftplusMatchesExpected(
          [0.69302183, 0.69324386],
          dtype,
          equality_test=self.AssertCloseAndSorted,
          rtol=9e-5,
          atol=9e-5)

  def testToBool(self):
    for dtype in self.numeric_types - self.complex_types:
      self._assertOpOutputMatchesExpected(
          gen_functional_ops.to_bool,
          np.array(5, dtype=dtype),
          expected=np.array(True))

      self._assertOpOutputMatchesExpected(
          gen_functional_ops.to_bool,
          np.array(0, dtype=dtype),
          expected=np.array(False))

      self._assertOpOutputMatchesExpected(
          gen_functional_ops.to_bool,
          np.array([], dtype=dtype),
          expected=np.array(False))

      self._assertOpOutputMatchesExpected(
          gen_functional_ops.to_bool,
          np.array([1, 2, 3], dtype=dtype),
          expected=np.array(True))


if __name__ == "__main__":
  googletest.main()
