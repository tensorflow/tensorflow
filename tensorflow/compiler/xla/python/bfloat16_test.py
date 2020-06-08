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

import collections
import itertools
import math

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from tensorflow.compiler.xla.python import xla_client

bfloat16 = xla_client.bfloat16


def numpy_assert_allclose(a, b, **kwargs):
  a = a.astype(np.float32) if a.dtype == bfloat16 else a
  b = b.astype(np.float32) if b.dtype == bfloat16 else b
  return np.testing.assert_allclose(a, b, **kwargs)


epsilon = float.fromhex("1.0p-7")

# Values that should round trip exactly to float and back.
FLOAT_VALUES = [
    0.0, 1.0, -1, 0.5, -0.5, epsilon, 1.0 + epsilon, 1.0 - epsilon,
    -1.0 - epsilon, -1.0 + epsilon, 3.5, 42.0, 255.0, 256.0,
    float("inf"),
    float("-inf"),
    float("nan")
]


class Bfloat16Test(parameterized.TestCase):
  """Tests the non-numpy Python methods of the bfloat16 type."""

  def testRoundTripToFloat(self):
    for v in FLOAT_VALUES:
      np.testing.assert_equal(v, float(bfloat16(v)))

  def testRoundTripNumpyTypes(self):
    for dtype in [np.float16, np.float32, np.float64]:
      np.testing.assert_equal(-3.75, dtype(bfloat16(dtype(-3.75))))
      np.testing.assert_equal(1.5, float(bfloat16(dtype(1.5))))
      np.testing.assert_equal(4.5, dtype(bfloat16(np.array(4.5, dtype))))
      np.testing.assert_equal(
          np.array([2, 5, -1], bfloat16), bfloat16(np.array([2, 5, -1], dtype)))

  def testRoundTripToInt(self):
    for v in [-256, -255, -34, -2, -1, 0, 1, 2, 10, 47, 128, 255, 256, 512]:
      self.assertEqual(v, int(bfloat16(v)))

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(({
      "testcase_name": "_" + dtype.__name__,
      "dtype": dtype
  } for dtype in [bfloat16, np.float16, np.float32, np.float64]))
  def testRoundTripToNumpy(self, dtype):
    for v in FLOAT_VALUES:
      np.testing.assert_equal(v, bfloat16(dtype(v)))
      np.testing.assert_equal(v, dtype(bfloat16(dtype(v))))
      np.testing.assert_equal(v, dtype(bfloat16(np.array(v, dtype))))
    if dtype != bfloat16:
      np.testing.assert_equal(
          np.array(FLOAT_VALUES, dtype),
          bfloat16(np.array(FLOAT_VALUES, dtype)).astype(dtype))

  def testStr(self):
    self.assertEqual("0", str(bfloat16(0.0)))
    self.assertEqual("1", str(bfloat16(1.0)))
    self.assertEqual("-3.5", str(bfloat16(-3.5)))
    self.assertEqual("0.0078125", str(bfloat16(float.fromhex("1.0p-7"))))
    self.assertEqual("inf", str(bfloat16(float("inf"))))
    self.assertEqual("-inf", str(bfloat16(float("-inf"))))
    self.assertEqual("nan", str(bfloat16(float("nan"))))

  def testRepr(self):
    self.assertEqual("0", repr(bfloat16(0)))
    self.assertEqual("1", repr(bfloat16(1)))
    self.assertEqual("-3.5", repr(bfloat16(-3.5)))
    self.assertEqual("0.0078125", repr(bfloat16(float.fromhex("1.0p-7"))))
    self.assertEqual("inf", repr(bfloat16(float("inf"))))
    self.assertEqual("-inf", repr(bfloat16(float("-inf"))))
    self.assertEqual("nan", repr(bfloat16(float("nan"))))

  def testHash(self):
    self.assertEqual(0, hash(bfloat16(0.0)))
    self.assertEqual(0x3f80, hash(bfloat16(1.0)))
    self.assertEqual(0x7fc0, hash(bfloat16(float("nan"))))

  # Tests for Python operations
  def testNegate(self):
    for v in FLOAT_VALUES:
      np.testing.assert_equal(-v, float(-bfloat16(v)))

  def testAdd(self):
    np.testing.assert_equal(0, float(bfloat16(0) + bfloat16(0)))
    np.testing.assert_equal(1, float(bfloat16(1) + bfloat16(0)))
    np.testing.assert_equal(0, float(bfloat16(1) + bfloat16(-1)))
    np.testing.assert_equal(5.5, float(bfloat16(2) + bfloat16(3.5)))
    np.testing.assert_equal(1.25, float(bfloat16(3.5) + bfloat16(-2.25)))
    np.testing.assert_equal(
        float("inf"), float(bfloat16(float("inf")) + bfloat16(-2.25)))
    np.testing.assert_equal(
        float("-inf"), float(bfloat16(float("-inf")) + bfloat16(-2.25)))
    self.assertTrue(math.isnan(float(bfloat16(3.5) + bfloat16(float("nan")))))

    # Test type promotion against Numpy scalar values.
    self.assertEqual(np.float32, type(bfloat16(3.5) + np.float16(2.25)))
    self.assertEqual(np.float32, type(np.float16(3.5) + bfloat16(2.25)))
    self.assertEqual(np.float32, type(bfloat16(3.5) + np.float32(2.25)))
    self.assertEqual(np.float32, type(np.float32(3.5) + bfloat16(2.25)))
    self.assertEqual(np.float64, type(bfloat16(3.5) + np.float64(2.25)))
    self.assertEqual(np.float64, type(np.float64(3.5) + bfloat16(2.25)))
    self.assertEqual(np.float64, type(bfloat16(3.5) + float(2.25)))
    self.assertEqual(np.float64, type(float(3.5) + bfloat16(2.25)))
    self.assertEqual(np.float32,
                     type(bfloat16(3.5) + np.array(2.25, np.float32)))
    self.assertEqual(np.float32,
                     type(np.array(3.5, np.float32) + bfloat16(2.25)))

  def testSub(self):
    np.testing.assert_equal(0, float(bfloat16(0) - bfloat16(0)))
    np.testing.assert_equal(1, float(bfloat16(1) - bfloat16(0)))
    np.testing.assert_equal(2, float(bfloat16(1) - bfloat16(-1)))
    np.testing.assert_equal(-1.5, float(bfloat16(2) - bfloat16(3.5)))
    np.testing.assert_equal(5.75, float(bfloat16(3.5) - bfloat16(-2.25)))
    np.testing.assert_equal(
        float("-inf"), float(bfloat16(-2.25) - bfloat16(float("inf"))))
    np.testing.assert_equal(
        float("inf"), float(bfloat16(-2.25) - bfloat16(float("-inf"))))
    self.assertTrue(math.isnan(float(bfloat16(3.5) - bfloat16(float("nan")))))

  def testMul(self):
    np.testing.assert_equal(0, float(bfloat16(0) * bfloat16(0)))
    np.testing.assert_equal(0, float(bfloat16(1) * bfloat16(0)))
    np.testing.assert_equal(-1, float(bfloat16(1) * bfloat16(-1)))
    np.testing.assert_equal(-7.875, float(bfloat16(3.5) * bfloat16(-2.25)))
    np.testing.assert_equal(
        float("-inf"), float(bfloat16(float("inf")) * bfloat16(-2.25)))
    np.testing.assert_equal(
        float("inf"), float(bfloat16(float("-inf")) * bfloat16(-2.25)))
    self.assertTrue(math.isnan(float(bfloat16(3.5) * bfloat16(float("nan")))))

  def testDiv(self):
    self.assertTrue(math.isnan(float(bfloat16(0) / bfloat16(0))))
    np.testing.assert_equal(float("inf"), float(bfloat16(1) / bfloat16(0)))
    np.testing.assert_equal(-1, float(bfloat16(1) / bfloat16(-1)))
    np.testing.assert_equal(-1.75, float(bfloat16(3.5) / bfloat16(-2)))
    np.testing.assert_equal(
        float("-inf"), float(bfloat16(float("inf")) / bfloat16(-2.25)))
    np.testing.assert_equal(
        float("inf"), float(bfloat16(float("-inf")) / bfloat16(-2.25)))
    self.assertTrue(math.isnan(float(bfloat16(3.5) / bfloat16(float("nan")))))

  def testLess(self):
    for v in FLOAT_VALUES:
      for w in FLOAT_VALUES:
        self.assertEqual(v < w, bfloat16(v) < bfloat16(w))

  def testLessEqual(self):
    for v in FLOAT_VALUES:
      for w in FLOAT_VALUES:
        self.assertEqual(v <= w, bfloat16(v) <= bfloat16(w))

  def testGreater(self):
    for v in FLOAT_VALUES:
      for w in FLOAT_VALUES:
        self.assertEqual(v > w, bfloat16(v) > bfloat16(w))

  def testGreaterEqual(self):
    for v in FLOAT_VALUES:
      for w in FLOAT_VALUES:
        self.assertEqual(v >= w, bfloat16(v) >= bfloat16(w))

  def testEqual(self):
    for v in FLOAT_VALUES:
      for w in FLOAT_VALUES:
        self.assertEqual(v == w, bfloat16(v) == bfloat16(w))

  def testNotEqual(self):
    for v in FLOAT_VALUES:
      for w in FLOAT_VALUES:
        self.assertEqual(v != w, bfloat16(v) != bfloat16(w))

  def testNan(self):
    a = np.isnan(bfloat16(float("nan")))
    self.assertTrue(a)
    numpy_assert_allclose(np.array([1.0, a]), np.array([1.0, a]))

    a = np.array([bfloat16(1.34375),
                  bfloat16(1.4375),
                  bfloat16(float("nan"))],
                 dtype=bfloat16)
    b = np.array(
        [bfloat16(1.3359375),
         bfloat16(1.4375),
         bfloat16(float("nan"))],
        dtype=bfloat16)
    numpy_assert_allclose(
        a, b, rtol=0.1, atol=0.1, equal_nan=True, err_msg="", verbose=True)


BinaryOp = collections.namedtuple("BinaryOp", ["op"])

UNARY_UFUNCS = [
    np.negative, np.positive, np.absolute, np.fabs, np.rint, np.sign,
    np.conjugate, np.exp, np.exp2, np.expm1, np.log, np.log10, np.log1p,
    np.log2, np.sqrt, np.square, np.cbrt, np.reciprocal, np.sin, np.cos, np.tan,
    np.arcsin, np.arccos, np.arctan, np.sinh, np.cosh, np.tanh, np.arcsinh,
    np.arccosh, np.arctanh, np.deg2rad, np.rad2deg, np.floor, np.ceil, np.trunc
]

BINARY_UFUNCS = [
    np.add, np.subtract, np.multiply, np.divide, np.logaddexp, np.logaddexp2,
    np.floor_divide, np.power, np.remainder, np.fmod, np.heaviside, np.arctan2,
    np.hypot, np.maximum, np.minimum, np.fmax, np.fmin, np.copysign
]

BINARY_PREDICATE_UFUNCS = [
    np.equal, np.not_equal, np.less, np.greater, np.less_equal,
    np.greater_equal, np.logical_and, np.logical_or, np.logical_xor
]


class Bfloat16NumPyTest(parameterized.TestCase):
  """Tests the NumPy integration of the bfloat16 type."""

  def testDtype(self):
    self.assertEqual(bfloat16, np.dtype(bfloat16))

  def testArray(self):
    x = np.array([[1, 2, 3]], dtype=bfloat16)
    self.assertEqual(bfloat16, x.dtype)
    self.assertEqual("[[1 2 3]]", str(x))
    np.testing.assert_equal(x, x)
    numpy_assert_allclose(x, x)
    self.assertTrue((x == x).all())

  def testComparisons(self):
    x = np.array([401408, 7, -32], dtype=np.float32)
    bx = x.astype(bfloat16)
    y = np.array([82432, 7, 0], dtype=np.float32)
    by = y.astype(bfloat16)
    np.testing.assert_equal(x == y, bx == by)
    np.testing.assert_equal(x != y, bx != by)
    np.testing.assert_equal(x < y, bx < by)
    np.testing.assert_equal(x > y, bx > by)
    np.testing.assert_equal(x <= y, bx <= by)
    np.testing.assert_equal(x >= y, bx >= by)

  def testEqual2(self):
    a = np.array([401408], bfloat16)
    b = np.array([82432], bfloat16)
    self.assertFalse(a.__eq__(b))

  def testCasts(self):
    for dtype in [
        np.float16, np.float32, np.float64, np.int8, np.int16, np.int32,
        np.int64, np.complex64, np.complex128, np.uint8, np.uint16, np.uint32,
        np.uint64
    ]:
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
      numpy_assert_allclose(y_np, y_tf, atol=2e-2)

      z_np = y_np.astype(dtype)
      z_tf = y_tf.astype(dtype)
      numpy_assert_allclose(z_np, z_tf, atol=2e-2)

  def testArange(self):
    np.testing.assert_equal(
        np.arange(100, dtype=np.float32).astype(bfloat16),
        np.arange(100, dtype=bfloat16))
    np.testing.assert_equal(
        np.arange(-10.5, 7.8, 0.5, dtype=np.float32).astype(bfloat16),
        np.arange(-10.5, 7.8, 0.5, dtype=bfloat16))
    np.testing.assert_equal(
        np.arange(-0., -7., -0.25, dtype=np.float32).astype(bfloat16),
        np.arange(-0., -7., -0.25, dtype=bfloat16))
    np.testing.assert_equal(
        np.arange(-16384., 16384., 64., dtype=np.float32).astype(bfloat16),
        np.arange(-16384., 16384., 64., dtype=bfloat16))

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(({
      "testcase_name": "_" + op.__name__,
      "op": op
  } for op in UNARY_UFUNCS))
  def testUnaryUfunc(self, op):
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7, 10).astype(bfloat16)
    numpy_assert_allclose(
        op(x).astype(np.float32), op(x.astype(np.float32)), rtol=1e-2)

  @parameterized.named_parameters(({
      "testcase_name": "_" + op.__name__,
      "op": op
  } for op in BINARY_UFUNCS))
  def testBinaryUfunc(self, op):
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7, 10).astype(bfloat16)
    y = rng.randn(4, 1, 7, 10).astype(bfloat16)
    numpy_assert_allclose(
        op(x, y).astype(np.float32),
        op(x.astype(np.float32), y.astype(np.float32)),
        rtol=1e-2)

  @parameterized.named_parameters(({
      "testcase_name": "_" + op.__name__,
      "op": op
  } for op in BINARY_PREDICATE_UFUNCS))
  def testBinaryPredicateUfunc(self, op):
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7).astype(bfloat16)
    y = rng.randn(4, 1, 7).astype(bfloat16)
    np.testing.assert_equal(
        op(x, y), op(x.astype(np.float32), y.astype(np.float32)))

  @parameterized.named_parameters(({
      "testcase_name": "_" + op.__name__,
      "op": op
  } for op in [np.isfinite, np.isinf, np.isnan, np.signbit, np.logical_not]))
  def testPredicateUfunc(self, op):
    rng = np.random.RandomState(seed=42)
    shape = (3, 7, 10)
    posinf_flips = rng.rand(*shape) < 0.1
    neginf_flips = rng.rand(*shape) < 0.1
    nan_flips = rng.rand(*shape) < 0.1
    vals = rng.randn(*shape)
    vals = np.where(posinf_flips, np.inf, vals)
    vals = np.where(neginf_flips, -np.inf, vals)
    vals = np.where(nan_flips, np.nan, vals)
    vals = vals.astype(bfloat16)
    np.testing.assert_equal(op(vals), op(vals.astype(np.float32)))

  def testDivmod(self):
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7).astype(bfloat16)
    y = rng.randn(4, 1, 7).astype(bfloat16)
    o1, o2 = np.divmod(x, y)
    e1, e2 = np.divmod(x.astype(np.float32), y.astype(np.float32))
    numpy_assert_allclose(o1, e1, rtol=1e-2)
    numpy_assert_allclose(o2, e2, rtol=1e-2)

  def testModf(self):
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7).astype(bfloat16)
    o1, o2 = np.modf(x)
    e1, e2 = np.modf(x.astype(np.float32))
    numpy_assert_allclose(o1.astype(np.float32), e1, rtol=1e-2)
    numpy_assert_allclose(o2.astype(np.float32), e2, rtol=1e-2)

  def testLdexp(self):
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7).astype(bfloat16)
    y = rng.randint(-50, 50, (1, 7))
    numpy_assert_allclose(
        np.ldexp(x, y).astype(np.float32),
        np.ldexp(x.astype(np.float32), y),
        rtol=1e-2,
        atol=1e-6)

  def testFrexp(self):
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7).astype(bfloat16)
    mant1, exp1 = np.frexp(x)
    mant2, exp2 = np.frexp(x.astype(np.float32))
    np.testing.assert_equal(exp1, exp2)
    numpy_assert_allclose(mant1, mant2, rtol=1e-2)

  def testNextAfter(self):
    one = np.array(1., dtype=bfloat16)
    two = np.array(2., dtype=bfloat16)
    zero = np.array(0., dtype=bfloat16)
    nan = np.array(np.nan, dtype=bfloat16)
    np.testing.assert_equal(np.nextafter(one, two) - one, epsilon)
    np.testing.assert_equal(np.nextafter(one, zero) - one, -epsilon / 2)
    np.testing.assert_equal(np.isnan(np.nextafter(nan, one)), True)
    np.testing.assert_equal(np.isnan(np.nextafter(one, nan)), True)
    np.testing.assert_equal(np.nextafter(one, one), one)
    smallest_denormal = float.fromhex("1.0p-133")
    np.testing.assert_equal(np.nextafter(zero, one), smallest_denormal)
    np.testing.assert_equal(np.nextafter(zero, -one), -smallest_denormal)
    for a, b in itertools.permutations([0., -0., nan], 2):
      np.testing.assert_equal(
          np.nextafter(
              np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)),
          np.nextafter(
              np.array(a, dtype=bfloat16), np.array(b, dtype=bfloat16)))


if __name__ == "__main__":
  absltest.main()
