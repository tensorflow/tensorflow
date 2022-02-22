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

import collections
import copy
import itertools
import math
import sys
from typing import Type

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

# pylint: disable=unused-import,g-bad-import-order
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.core import _pywrap_bfloat16
from tensorflow.python.platform import test

bfloat16 = _pywrap_bfloat16.TF_bfloat16_type()


def numpy_assert_allclose(a, b, **kwargs):
  a = a.astype(np.float32) if a.dtype == bfloat16 else a
  b = b.astype(np.float32) if b.dtype == bfloat16 else b
  return np.testing.assert_allclose(a, b, **kwargs)


def numpy_promote_types(a: Type[np.generic],
                        b: Type[np.generic]) -> Type[np.generic]:
  if a == bfloat16 and b == bfloat16:
    return bfloat16
  if a == bfloat16:
    a = np.float32
  if b == bfloat16:
    b = np.float32
  return np.promote_types(a, b)


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
    for dtype in [np.float16, np.float32, np.float64, np.longdouble]:
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
  } for dtype in [bfloat16, np.float16, np.float32, np.float64, np.longdouble]))
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

  def testHashZero(self):
    """Tests that negative zero and zero hash to the same value."""
    self.assertEqual(hash(bfloat16(-0.0)), hash(bfloat16(0.0)))

  @parameterized.parameters(np.extract(np.isfinite(FLOAT_VALUES), FLOAT_VALUES))
  def testHashNumbers(self, value):
    self.assertEqual(hash(value), hash(bfloat16(value)), str(value))

  @parameterized.named_parameters(("PositiveNan", bfloat16(float("nan"))),
                                  ("NegativeNan", bfloat16(float("-nan"))))
  def testHashNan(self, nan):
    nan_hash = hash(nan)
    nan_object_hash = object.__hash__(nan)
    # The hash of a NaN is either 0 or a hash of the object pointer.
    self.assertIn(nan_hash, (sys.hash_info.nan, nan_object_hash), str(nan))

  def testHashInf(self):
    self.assertEqual(sys.hash_info.inf, hash(bfloat16(float("inf"))), "inf")
    self.assertEqual(-sys.hash_info.inf, hash(bfloat16(float("-inf"))), "-inf")

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

  def testAddScalarTypePromotion(self):
    """Tests type promotion against Numpy scalar values."""
    types = [bfloat16, np.float16, np.float32, np.float64, np.longdouble]
    for lhs_type in types:
      for rhs_type in types:
        expected_type = numpy_promote_types(lhs_type, rhs_type)
        actual_type = type(lhs_type(3.5) + rhs_type(2.25))
        self.assertEqual(expected_type, actual_type)

  def testAddArrayTypePromotion(self):
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

  def testSort(self):
    values_to_sort = np.float32(FLOAT_VALUES)
    sorted_f32 = np.sort(values_to_sort)
    sorted_bf16 = np.sort(values_to_sort.astype(bfloat16))  # pylint: disable=too-many-function-args
    np.testing.assert_equal(sorted_f32, np.float32(sorted_bf16))

  def testArgmax(self):
    values_to_sort = np.float32(bfloat16(np.float32(FLOAT_VALUES)))
    argmax_f32 = np.argmax(values_to_sort)
    argmax_bf16 = np.argmax(values_to_sort.astype(bfloat16))  # pylint: disable=too-many-function-args
    np.testing.assert_equal(argmax_f32, argmax_bf16)

  def testArgmaxOnNan(self):
    """Ensures we return the right thing for multiple NaNs."""
    one_with_nans = np.array(
        [1.0, float("nan"), float("nan")], dtype=np.float32)
    np.testing.assert_equal(
        np.argmax(one_with_nans.astype(bfloat16)), np.argmax(one_with_nans))

  def testArgmaxOnNegativeInfinity(self):
    """Ensures we return the right thing for negative infinities."""
    inf = np.array([float("-inf")], dtype=np.float32)
    np.testing.assert_equal(np.argmax(inf.astype(bfloat16)), np.argmax(inf))

  def testArgmin(self):
    values_to_sort = np.float32(bfloat16(np.float32(FLOAT_VALUES)))
    argmin_f32 = np.argmin(values_to_sort)
    argmin_bf16 = np.argmin(values_to_sort.astype(bfloat16))  # pylint: disable=too-many-function-args
    np.testing.assert_equal(argmin_f32, argmin_bf16)

  def testArgminOnNan(self):
    """Ensures we return the right thing for multiple NaNs."""
    one_with_nans = np.array(
        [1.0, float("nan"), float("nan")], dtype=np.float32)
    np.testing.assert_equal(
        np.argmin(one_with_nans.astype(bfloat16)), np.argmin(one_with_nans))

  def testArgminOnPositiveInfinity(self):
    """Ensures we return the right thing for positive infinities."""
    inf = np.array([float("inf")], dtype=np.float32)
    np.testing.assert_equal(np.argmin(inf.astype(bfloat16)), np.argmin(inf))

  def testDtypeFromString(self):
    assert np.dtype("bfloat16") == np.dtype(bfloat16)


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

  def testDeepCopyDoesNotAlterHash(self):
    # For context, see https://github.com/google/jax/issues/4651. If the hash
    # value of the type descriptor is not initialized correctly, a deep copy
    # can change the type hash.
    dtype = np.dtype(bfloat16)
    h = hash(dtype)
    _ = copy.deepcopy(dtype)
    self.assertEqual(h, hash(dtype))

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

  def testCanCast(self):
    allowed_casts = [
        (np.bool_, bfloat16),
        (np.int8, bfloat16),
        (np.uint8, bfloat16),
        (bfloat16, np.float32),
        (bfloat16, np.float64),
        (bfloat16, np.longdouble),
        (bfloat16, np.complex64),
        (bfloat16, np.complex128),
        (bfloat16, np.clongdouble),
    ]
    all_dtypes = [
        np.float16, np.float32, np.float64, np.longdouble, np.int8, np.int16,
        np.int32, np.int64, np.complex64, np.complex128, np.clongdouble,
        np.uint8, np.uint16, np.uint32, np.uint64, np.intc, np.int_,
        np.longlong, np.uintc, np.ulonglong
    ]
    for d in all_dtypes:
      self.assertEqual((bfloat16, d) in allowed_casts, np.can_cast(bfloat16, d))
      self.assertEqual((d, bfloat16) in allowed_casts, np.can_cast(d, bfloat16))

  def testCasts(self):
    for dtype in [
        np.float16, np.float32, np.float64, np.longdouble, np.int8, np.int16,
        np.int32, np.int64, np.complex64, np.complex128, np.clongdouble,
        np.uint8, np.uint16, np.uint32, np.uint64, np.intc, np.int_,
        np.longlong, np.uintc, np.ulonglong
    ]:
      x = np.array([[1, 2, 3]], dtype=dtype)
      y = x.astype(bfloat16)
      z = y.astype(dtype)
      self.assertTrue(np.all(x == y))
      self.assertEqual(bfloat16, y.dtype)
      self.assertTrue(np.all(x == z))
      self.assertEqual(dtype, z.dtype)

  def testConformNumpyComplex(self):
    for dtype in [np.complex64, np.complex128, np.clongdouble]:
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

  @parameterized.parameters(list(range(1, 128)))
  def testCopySign(self, nan_payload):
    inf_bits = 0x7f80
    nan_bits = inf_bits | nan_payload
    little_endian_uint16 = np.dtype(np.uint16).newbyteorder("L")
    little_endian_bfloat = np.dtype(bfloat16).newbyteorder("L")
    nan = little_endian_uint16.type(nan_bits).view(little_endian_bfloat)
    nan_with_sign = np.copysign(nan, bfloat16(-1))
    nan_with_sign_bits = nan_with_sign.view(little_endian_uint16)
    np.testing.assert_equal(nan_bits | (1 << 15), nan_with_sign_bits)

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

  def testSpacing(self):
    # Sweep a variety of binades to see that spacing gives the proper ULP.
    # All subnormals have a fixed distance of 2^-133.
    with self.subTest(name="Subnormals"):
      for i in range(-133, -126):
        power_of_two = bfloat16(2.0**i)
        distance = float.fromhex("0x1p-133")
        np.testing.assert_equal(np.spacing(power_of_two), distance)
        np.testing.assert_equal(np.spacing(-power_of_two), -distance)
    # Normals have a distance which depends on their binade.
    with self.subTest(name="Normals"):
      for i in range(-126, 127):
        power_of_two = bfloat16(2.0**i)
        distance = epsilon * power_of_two
        np.testing.assert_equal(np.spacing(power_of_two), distance)
        np.testing.assert_equal(np.spacing(-power_of_two), -distance)
    inf = bfloat16(float("inf"))
    nan = bfloat16(float("nan"))
    # Check that spacing agrees with arithmetic involving nextafter.
    with self.subTest(name="NextAfter"):
      for x in FLOAT_VALUES:
        x_bfloat16 = bfloat16(x)
        spacing = np.spacing(x_bfloat16)
        toward = np.copysign(inf, x_bfloat16)
        nextup = np.nextafter(x_bfloat16, toward)
        np.testing.assert_equal(spacing, nextup - x_bfloat16)
    # Check that spacing for special values gives the correct answer.
    with self.subTest(name="NonFinite"):
      np.testing.assert_equal(np.spacing(nan), np.spacing(np.float32(nan)))
      np.testing.assert_equal(np.spacing(inf), np.spacing(np.float32(inf)))


if __name__ == "__main__":
  absltest.main()
