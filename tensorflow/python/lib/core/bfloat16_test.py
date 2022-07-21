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
"""Test cases for the bfloat16,float8_e4m3b11 Python types."""

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
float8_e4m3b11 = _pywrap_bfloat16.TF_float8_e4m3b11_type()


def numpy_assert_allclose(a, b, float_type, **kwargs):
  a = a.astype(np.float32) if a.dtype == float_type else a
  b = b.astype(np.float32) if b.dtype == float_type else b
  return np.testing.assert_allclose(a, b, **kwargs)


def numpy_promote_types(
    a: Type[np.generic], b: Type[np.generic], float_type: Type[np.generic],
    next_largest_fp_type: Type[np.generic]) -> Type[np.generic]:
  if a == float_type and b == float_type:
    return float_type
  if a == float_type:
    a = next_largest_fp_type
  if b == float_type:
    b = next_largest_fp_type
  return np.promote_types(a, b)


def truncate(x, float_type):
  if isinstance(x, np.ndarray):
    return x.astype(float_type).astype(np.float32)
  else:
    return type(x)(float_type(x))


def test_binary_operation(a, b, op, float_type):
  a = float_type(a)
  b = float_type(b)
  expected = op(np.float32(a), np.float32(b))
  result = op(a, b)
  if math.isnan(expected):
    if not math.isnan(result):
      raise AssertionError("%s expected to be nan." % repr(result))
  else:
    np.testing.assert_equal(
        truncate(expected, float_type=float_type), float(result))


epsilon = {
    bfloat16: float.fromhex("1.0p-7"),
    float8_e4m3b11: float.fromhex("1.0p-3"),
}

# Values that should round trip exactly to float and back.
FLOAT_VALUES = {}
FLOAT_VALUES[bfloat16] = [
    0.0, 1.0, -1, 0.5, -0.5, epsilon[bfloat16], 1.0 + epsilon[bfloat16],
    1.0 - epsilon[bfloat16], -1.0 - epsilon[bfloat16], -1.0 + epsilon[bfloat16],
    3.5, 4, 5, 7,
    float("inf"),
    float("-inf"),
    float("nan")
]

FLOAT_VALUES[float8_e4m3b11] = [
    0.0,
    1.0,
    -1,
    0.5,
    -0.5,
    epsilon[float8_e4m3b11],
    1.0 + epsilon[float8_e4m3b11],
    1.0 - epsilon[float8_e4m3b11],
    -1.0 - epsilon[float8_e4m3b11],
    -1.0 + epsilon[float8_e4m3b11],
    3.5,
    4,
    5,
    7,
    float(30),  # max float
    float(-30),  # min float
    float("nan")
]


# pylint: disable=g-complex-comprehension
@parameterized.named_parameters(({
    "testcase_name": "_" + dtype.__name__,
    "float_type": dtype
} for dtype in [bfloat16, float8_e4m3b11]))
class CustomFloatTest(parameterized.TestCase):
  """Tests the non-numpy Python methods of the custom float type."""

  def testRoundTripToFloat(self, float_type):
    for v in FLOAT_VALUES[float_type]:
      np.testing.assert_equal(v, float(float_type(v)))

  def testRoundTripNumpyTypes(self, float_type):
    for dtype in [np.float16, np.float32, np.float64, np.longdouble]:
      np.testing.assert_equal(-3.75, dtype(float_type(dtype(-3.75))))
      np.testing.assert_equal(1.5, float(float_type(dtype(1.5))))
      np.testing.assert_equal(4.5, dtype(float_type(np.array(4.5, dtype))))
      np.testing.assert_equal(
          np.array([2, 5, -1], float_type),
          float_type(np.array([2, 5, -1], dtype)))

  def testRoundTripToInt(self, float_type):
    for v in {
        bfloat16: [
            -256, -255, -34, -2, -1, 0, 1, 2, 10, 47, 128, 255, 256, 512
        ],
        float8_e4m3b11: list(range(-30, 30, 2)) + list(range(-15, 15, 2)),
    }[float_type]:
      self.assertEqual(v, int(float_type(v)))

  def testRoundTripToNumpy(self, float_type):
    for dtype in [
        float_type, np.float16, np.float32, np.float64, np.longdouble
    ]:
      with self.subTest(dtype.__name__):
        for v in FLOAT_VALUES[float_type]:
          np.testing.assert_equal(v, float_type(dtype(v)))
          np.testing.assert_equal(v, dtype(float_type(dtype(v))))
          np.testing.assert_equal(v, dtype(float_type(np.array(v, dtype))))
        if dtype != float_type:
          np.testing.assert_equal(
              np.array(FLOAT_VALUES[float_type], dtype),
              float_type(np.array(FLOAT_VALUES[float_type],
                                  dtype)).astype(dtype))

  def testStr(self, float_type):
    for value in [
        0.0, 1.0, -3.5,
        float.fromhex("1.0p-7"),
        float("inf"),
        float("-inf"),
        float("nan")
    ]:
      self.assertEqual("%.6g" % float(float_type(value)),
                       str(float_type(value)))

  def testRepr(self, float_type):
    for value in [
        0.0, 1.0, -3.5,
        float.fromhex("1.0p-7"),
        float("inf"),
        float("-inf"),
        float("nan")
    ]:
      self.assertEqual("%.6g" % float(float_type(value)),
                       repr(float_type(value)))

  def testItem(self, float_type):
    self.assertIsInstance(float_type(0).item(), float)

  def testHashZero(self, float_type):
    """Tests that negative zero and zero hash to the same value."""
    self.assertEqual(hash(float_type(-0.0)), hash(float_type(0.0)))

  def testHashNumbers(self, float_type):
    for value in np.extract(
        np.isfinite(FLOAT_VALUES[float_type]), FLOAT_VALUES[float_type]):
      with self.subTest(value):
        self.assertEqual(hash(value), hash(float_type(value)), str(value))

  def testHashNan(self, float_type):
    for name, nan in [("PositiveNan", float_type(float("nan"))),
                      ("NegativeNan", float_type(float("-nan")))]:
      with self.subTest(name):
        nan_hash = hash(nan)
        nan_object_hash = object.__hash__(nan)
        # The hash of a NaN is either 0 or a hash of the object pointer.
        self.assertIn(nan_hash, (sys.hash_info.nan, nan_object_hash), str(nan))

  def testHashInf(self, float_type):
    if float_type == float8_e4m3b11:
      self.skipTest("Not supported")  # no inf for e4m3b11
    self.assertEqual(sys.hash_info.inf, hash(float_type(float("inf"))), "inf")
    self.assertEqual(-sys.hash_info.inf, hash(float_type(float("-inf"))),
                     "-inf")

  # Tests for Python operations
  def testNegate(self, float_type):
    for v in FLOAT_VALUES[float_type]:
      np.testing.assert_equal(
          float(float_type(-float(float_type(v)))), float(-float_type(v)))

  def testAdd(self, float_type):
    for a, b in [(0, 0), (1, 0), (1, -1), (2, 3.5), (3.5, -2.25),
                 (float("inf"), -2.25), (float("-inf"), -2.25),
                 (3.5, float("nan"))]:
      test_binary_operation(a, b, op=lambda a, b: a + b, float_type=float_type)

  def testAddScalarTypePromotion(self, float_type):
    """Tests type promotion against Numpy scalar values."""
    types = [float_type, np.float16, np.float32, np.float64, np.longdouble]
    for lhs_type in types:
      for rhs_type in types:
        expected_type = numpy_promote_types(
            lhs_type,
            rhs_type,
            float_type=float_type,
            next_largest_fp_type={
                bfloat16: np.float32,
                float8_e4m3b11: np.float32,
            }[float_type])
        actual_type = type(lhs_type(3.5) + rhs_type(2.25))
        self.assertEqual(expected_type, actual_type)

  def testAddArrayTypePromotion(self, float_type):
    self.assertEqual(np.float32,
                     type(float_type(3.5) + np.array(2.25, np.float32)))
    self.assertEqual(np.float32,
                     type(np.array(3.5, np.float32) + float_type(2.25)))

  def testSub(self, float_type):
    for a, b in [(0, 0), (1, 0), (1, -1), (2, 3.5), (3.5, -2.25),
                 (-2.25, float("inf")), (-2.25, float("-inf")),
                 (3.5, float("nan"))]:
      test_binary_operation(a, b, op=lambda a, b: a - b, float_type=float_type)

  def testMul(self, float_type):
    for a, b in [(0, 0), (1, 0), (1, -1), (3.5, -2.25), (float("inf"), -2.25),
                 (float("-inf"), -2.25), (3.5, float("nan"))]:
      test_binary_operation(a, b, op=lambda a, b: a * b, float_type=float_type)

  def testDiv(self, float_type):
    for a, b in [(0, 0), (1, 0), (1, -1), (2, 3.5), (3.5, -2.25),
                 (float("inf"), -2.25), (float("-inf"), -2.25),
                 (3.5, float("nan"))]:
      test_binary_operation(a, b, op=lambda a, b: a / b, float_type=float_type)

  def testLess(self, float_type):
    for v in FLOAT_VALUES[float_type]:
      for w in FLOAT_VALUES[float_type]:
        self.assertEqual(v < w, float_type(v) < float_type(w))

  def testLessEqual(self, float_type):
    for v in FLOAT_VALUES[float_type]:
      for w in FLOAT_VALUES[float_type]:
        self.assertEqual(v <= w, float_type(v) <= float_type(w))

  def testGreater(self, float_type):
    for v in FLOAT_VALUES[float_type]:
      for w in FLOAT_VALUES[float_type]:
        self.assertEqual(v > w, float_type(v) > float_type(w))

  def testGreaterEqual(self, float_type):
    for v in FLOAT_VALUES[float_type]:
      for w in FLOAT_VALUES[float_type]:
        self.assertEqual(v >= w, float_type(v) >= float_type(w))

  def testEqual(self, float_type):
    for v in FLOAT_VALUES[float_type]:
      for w in FLOAT_VALUES[float_type]:
        self.assertEqual(v == w, float_type(v) == float_type(w))

  def testNotEqual(self, float_type):
    for v in FLOAT_VALUES[float_type]:
      for w in FLOAT_VALUES[float_type]:
        self.assertEqual(v != w, float_type(v) != float_type(w))

  def testNan(self, float_type):
    a = np.isnan(float_type(float("nan")))
    self.assertTrue(a)
    numpy_assert_allclose(
        np.array([1.0, a]), np.array([1.0, a]), float_type=float_type)

    a = np.array(
        [float_type(1.34375),
         float_type(1.4375),
         float_type(float("nan"))],
        dtype=float_type)
    b = np.array(
        [float_type(1.3359375),
         float_type(1.4375),
         float_type(float("nan"))],
        dtype=float_type)
    numpy_assert_allclose(
        a,
        b,
        rtol=0.1,
        atol=0.1,
        equal_nan=True,
        err_msg="",
        verbose=True,
        float_type=float_type)

  def testSort(self, float_type):
    values_to_sort = np.float32(FLOAT_VALUES[float_type])
    sorted_f32 = np.sort(values_to_sort)
    sorted_bf16 = np.sort(values_to_sort.astype(float_type))  # pylint: disable=too-many-function-args
    np.testing.assert_equal(sorted_f32, np.float32(sorted_bf16))

  def testArgmax(self, float_type):
    values_to_sort = np.float32(
        float_type(np.float32(FLOAT_VALUES[float_type])))
    argmax_f32 = np.argmax(values_to_sort)
    argmax_bf16 = np.argmax(values_to_sort.astype(float_type))  # pylint: disable=too-many-function-args
    np.testing.assert_equal(argmax_f32, argmax_bf16)

  def testArgmaxOnNan(self, float_type):
    """Ensures we return the right thing for multiple NaNs."""
    one_with_nans = np.array(
        [1.0, float("nan"), float("nan")], dtype=np.float32)
    np.testing.assert_equal(
        np.argmax(one_with_nans.astype(float_type)), np.argmax(one_with_nans))

  def testArgmaxOnNegativeInfinity(self, float_type):
    """Ensures we return the right thing for negative infinities."""
    inf = np.array([float("-inf")], dtype=np.float32)
    np.testing.assert_equal(np.argmax(inf.astype(float_type)), np.argmax(inf))

  def testArgmin(self, float_type):
    values_to_sort = np.float32(
        float_type(np.float32(FLOAT_VALUES[float_type])))
    argmin_f32 = np.argmin(values_to_sort)
    argmin_bf16 = np.argmin(values_to_sort.astype(float_type))  # pylint: disable=too-many-function-args
    np.testing.assert_equal(argmin_f32, argmin_bf16)

  def testArgminOnNan(self, float_type):
    """Ensures we return the right thing for multiple NaNs."""
    one_with_nans = np.array(
        [1.0, float("nan"), float("nan")], dtype=np.float32)
    np.testing.assert_equal(
        np.argmin(one_with_nans.astype(float_type)), np.argmin(one_with_nans))

  def testArgminOnPositiveInfinity(self, float_type):
    """Ensures we return the right thing for positive infinities."""
    inf = np.array([float("inf")], dtype=np.float32)
    np.testing.assert_equal(np.argmin(inf.astype(float_type)), np.argmin(inf))

  def testDtypeFromString(self, float_type):
    assert np.dtype(float_type.__name__) == np.dtype(float_type)


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


# pylint: disable=g-complex-comprehension
@parameterized.named_parameters(({
    "testcase_name": "_" + dtype.__name__,
    "float_type": dtype
} for dtype in [bfloat16, float8_e4m3b11]))
class CustomFloatNumPyTest(parameterized.TestCase):
  """Tests the NumPy integration of the float8_e4m3b11 type."""

  def testDtype(self, float_type):
    self.assertEqual(float_type, np.dtype(float_type))

  def testDeepCopyDoesNotAlterHash(self, float_type):
    # For context, see https://github.com/google/jax/issues/4651. If the hash
    # value of the type descriptor is not initialized correctly, a deep copy
    # can change the type hash.
    dtype = np.dtype(float_type)
    h = hash(dtype)
    _ = copy.deepcopy(dtype)
    self.assertEqual(h, hash(dtype))

  def testArray(self, float_type):
    x = np.array([[1, 2, 3]], dtype=float_type)
    self.assertEqual(float_type, x.dtype)
    self.assertEqual("[[1 2 3]]", str(x))
    np.testing.assert_equal(x, x)
    numpy_assert_allclose(x, x, float_type=float_type)
    self.assertTrue((x == x).all())

  def testComparisons(self, float_type):
    x = np.array([30, 7, -30], dtype=np.float32)
    bx = x.astype(float_type)
    y = np.array([17, 7, 0], dtype=np.float32)
    by = y.astype(float_type)
    np.testing.assert_equal(x == y, bx == by)
    np.testing.assert_equal(x != y, bx != by)
    np.testing.assert_equal(x < y, bx < by)
    np.testing.assert_equal(x > y, bx > by)
    np.testing.assert_equal(x <= y, bx <= by)
    np.testing.assert_equal(x >= y, bx >= by)

  def testEqual2(self, float_type):
    if float_type == float8_e4m3b11:
      self.skipTest("Not supported")  # out of range.
    a = np.array([401408], float_type)
    b = np.array([82432], float_type)
    self.assertFalse(a.__eq__(b))

  def testCanCast(self, float_type):
    allowed_casts = [
        (np.bool_, float_type),
        (np.int8, float_type),
        (np.uint8, float_type),
        (float_type, np.float32),
        (float_type, np.float64),
        (float_type, np.longdouble),
        (float_type, np.complex64),
        (float_type, np.complex128),
        (float_type, np.clongdouble),
    ]
    all_dtypes = [
        np.float16, np.float32, np.float64, np.longdouble, np.int8, np.int16,
        np.int32, np.int64, np.complex64, np.complex128, np.clongdouble,
        np.uint8, np.uint16, np.uint32, np.uint64, np.intc, np.int_,
        np.longlong, np.uintc, np.ulonglong
    ]
    for d in all_dtypes:
      with self.subTest(d.__name__):
        self.assertEqual((float_type, d) in allowed_casts,
                         np.can_cast(float_type, d))
        self.assertEqual((d, float_type) in allowed_casts,
                         np.can_cast(d, float_type))

  def testCasts(self, float_type):
    for dtype in [
        np.float16, np.float32, np.float64, np.longdouble, np.int8, np.int16,
        np.int32, np.int64, np.complex64, np.complex128, np.clongdouble,
        np.uint8, np.uint16, np.uint32, np.uint64, np.intc, np.int_,
        np.longlong, np.uintc, np.ulonglong
    ]:
      x = np.array([[1, 2, 3]], dtype=dtype)
      y = x.astype(float_type)
      z = y.astype(dtype)
      self.assertTrue(np.all(x == y))
      self.assertEqual(float_type, y.dtype)
      self.assertTrue(np.all(x == z))
      self.assertEqual(dtype, z.dtype)

  def testConformNumpyComplex(self, float_type):
    for dtype in [np.complex64, np.complex128, np.clongdouble]:
      x = np.array([1.5, 2.5 + 2.j, 3.25], dtype=dtype)
      y_np = x.astype(np.float32)
      y_tf = x.astype(float_type)
      numpy_assert_allclose(y_np, y_tf, atol=2e-2, float_type=float_type)

      z_np = y_np.astype(dtype)
      z_tf = y_tf.astype(dtype)
      numpy_assert_allclose(z_np, z_tf, atol=2e-2, float_type=float_type)

  def testArange(self, float_type):
    np.testing.assert_equal(
        np.arange(100, dtype=np.float32).astype(float_type),
        np.arange(100, dtype=float_type))
    np.testing.assert_equal(
        np.arange(-16, 16, 1, dtype=np.float32).astype(float_type),
        np.arange(-16, 16, 1, dtype=float_type))
    np.testing.assert_equal(
        np.arange(-0., -7., -0.25, dtype=np.float32).astype(float_type),
        np.arange(-0., -7., -0.25, dtype=float_type))
    np.testing.assert_equal(
        np.arange(-30., 30., 2., dtype=np.float32).astype(float_type),
        np.arange(-30., 30., 2., dtype=float_type))

  def testUnaryUfunc(self, float_type):
    for op in UNARY_UFUNCS:
      with self.subTest(op.__name__):
        rng = np.random.RandomState(seed=42)
        x = rng.randn(3, 7, 10).astype(float_type)
        numpy_assert_allclose(
            op(x).astype(np.float32),
            truncate(op(x.astype(np.float32)), float_type=float_type),
            rtol=1e-4,
            float_type=float_type)

  def testBinaryUfunc(self, float_type):
    for op in BINARY_UFUNCS:
      with self.subTest(op.__name__):
        rng = np.random.RandomState(seed=42)
        x = rng.randn(3, 7, 10).astype(float_type)
        y = rng.randn(4, 1, 7, 10).astype(float_type)
        numpy_assert_allclose(
            op(x, y).astype(np.float32),
            truncate(
                op(x.astype(np.float32), y.astype(np.float32)),
                float_type=float_type),
            rtol=1e-4,
            float_type=float_type)

  def testBinaryPredicateUfunc(self, float_type):
    for op in BINARY_PREDICATE_UFUNCS:
      with self.subTest(op.__name__):
        rng = np.random.RandomState(seed=42)
        x = rng.randn(3, 7).astype(float_type)
        y = rng.randn(4, 1, 7).astype(float_type)
        np.testing.assert_equal(
            op(x, y), op(x.astype(np.float32), y.astype(np.float32)))

  def testPredicateUfunc(self, float_type):
    for op in [np.isfinite, np.isinf, np.isnan, np.signbit, np.logical_not]:
      with self.subTest(op.__name__):
        rng = np.random.RandomState(seed=42)
        shape = (3, 7, 10)
        posinf_flips = rng.rand(*shape) < 0.1
        neginf_flips = rng.rand(*shape) < 0.1
        nan_flips = rng.rand(*shape) < 0.1
        vals = rng.randn(*shape)
        vals = np.where(posinf_flips, np.inf, vals)
        vals = np.where(neginf_flips, -np.inf, vals)
        vals = np.where(nan_flips, np.nan, vals)
        vals = vals.astype(float_type)
        np.testing.assert_equal(op(vals), op(vals.astype(np.float32)))

  def testDivmod(self, float_type):
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7).astype(float_type)
    y = rng.randn(4, 1, 7).astype(float_type)
    o1, o2 = np.divmod(x, y)
    e1, e2 = np.divmod(x.astype(np.float32), y.astype(np.float32))
    numpy_assert_allclose(
        o1,
        truncate(e1, float_type=float_type),
        rtol=1e-2,
        float_type=float_type)
    numpy_assert_allclose(
        o2,
        truncate(e2, float_type=float_type),
        rtol=1e-2,
        float_type=float_type)

  def testModf(self, float_type):
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7).astype(float_type)
    o1, o2 = np.modf(x)
    e1, e2 = np.modf(x.astype(np.float32))
    numpy_assert_allclose(
        o1.astype(np.float32),
        truncate(e1, float_type=float_type),
        rtol=1e-2,
        float_type=float_type)
    numpy_assert_allclose(
        o2.astype(np.float32),
        truncate(e2, float_type=float_type),
        rtol=1e-2,
        float_type=float_type)

  def testLdexp(self, float_type):
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7).astype(float_type)
    y = rng.randint(-50, 50, (1, 7)).astype(np.int32)
    self.assertEqual(np.ldexp(x, y).dtype, x.dtype)
    numpy_assert_allclose(
        np.ldexp(x, y).astype(np.float32),
        truncate(np.ldexp(x.astype(np.float32), y), float_type=float_type),
        rtol=1e-2,
        atol=1e-6,
        float_type=float_type)

  def testFrexp(self, float_type):
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7).astype(float_type)
    mant1, exp1 = np.frexp(x)
    mant2, exp2 = np.frexp(x.astype(np.float32))
    np.testing.assert_equal(exp1, exp2)
    numpy_assert_allclose(mant1, mant2, rtol=1e-2, float_type=float_type)

  def testCopySign(self, float_type):
    if float_type == float8_e4m3b11:
      self.skipTest("Not supported")  # Nans don't have payload.
    for nan_payload in list(range(1, 128)):
      with self.subTest(nan_payload):
        one = np.array(1., dtype=float_type)
        inf_bits = 0x7f80
        two = np.array(2., dtype=float_type)
        nan_bits = inf_bits | nan_payload
        zero = np.array(0., dtype=float_type)
        little_endian_uint16 = np.dtype(np.uint16).newbyteorder("L")
        nan = np.array(np.nan, dtype=float_type)
        little_endian_bfloat = np.dtype(bfloat16).newbyteorder("L")
        np.testing.assert_equal(
            np.nextafter(one, two) - one, epsilon[float_type])
        nan = little_endian_uint16.type(nan_bits).view(little_endian_bfloat)
        np.testing.assert_equal(
            np.nextafter(one, zero) - one, -epsilon[float_type] / 2)
        nan_with_sign = np.copysign(nan, bfloat16(-1))
        nan_with_sign_bits = nan_with_sign.view(little_endian_uint16)
        np.testing.assert_equal(nan_bits | (1 << 15), nan_with_sign_bits)

  def testNextAfter(self, float_type):
    one = np.array(1., dtype=float_type)
    two = np.array(2., dtype=float_type)
    zero = np.array(0., dtype=float_type)
    nan = np.array(np.nan, dtype=float_type)
    np.testing.assert_equal(np.nextafter(one, two) - one, epsilon[float_type])
    np.testing.assert_equal(
        np.nextafter(one, zero) - one, -epsilon[float_type] / 2)
    np.testing.assert_equal(np.isnan(np.nextafter(nan, one)), True)
    np.testing.assert_equal(np.isnan(np.nextafter(one, nan)), True)
    np.testing.assert_equal(np.nextafter(one, one), one)
    smallest_denormal = {
        bfloat16: float.fromhex("1.0p-133"),
        float8_e4m3b11: float.fromhex("1.0p-13"),
    }[float_type]
    np.testing.assert_equal(np.nextafter(zero, one), smallest_denormal)
    np.testing.assert_equal(np.nextafter(zero, -one), -smallest_denormal)
    for a, b in itertools.permutations([0., nan], 2):
      np.testing.assert_equal(
          np.nextafter(
              np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)),
          np.nextafter(
              np.array(a, dtype=float_type), np.array(b, dtype=float_type)))

  def testSpacing(self, float_type):
    # Sweep a variety of binades to see that spacing gives the proper ULP.
    # All subnormals have a fixed distance of 2^-133.
    with self.subTest(name="Subnormals"):
      if float_type == float8_e4m3b11:
        self.skipTest("Not supported")
      for i in {
          float8_e4m3b11: range(-13, -10),
          bfloat16: range(-133, -126)
      }[float_type]:
        power_of_two = float_type(2.0**i)
        distance = {
            float8_e4m3b11: float.fromhex("0x1p-13"),
            bfloat16: float.fromhex("0x1p-133")
        }[float_type]
        np.testing.assert_equal(np.spacing(power_of_two), distance)
        np.testing.assert_equal(np.spacing(-power_of_two), -distance)
    # Normals have a distance which depends on their binade.
    with self.subTest(name="Normals"):
      for i in {
          float8_e4m3b11: range(-10, 4),
          bfloat16: range(-126, 127)
      }[float_type]:
        power_of_two = float_type(2.0**i)
        distance = epsilon[float_type] * power_of_two
        np.testing.assert_equal(np.spacing(power_of_two), distance)
        np.testing.assert_equal(np.spacing(-power_of_two), -distance)
    inf = float_type(float("inf"))
    nan = float_type(float("nan"))
    # Check that spacing agrees with arithmetic involving nextafter.
    with self.subTest(name="NextAfter"):
      for x in FLOAT_VALUES[float_type]:
        x_float_type = float_type(x)
        spacing = np.spacing(x_float_type)
        toward = np.copysign(inf, x_float_type)
        nextup = np.nextafter(x_float_type, toward)
        np.testing.assert_equal(spacing, nextup - x_float_type)
    # Check that spacing for special values gives the correct answer.
    with self.subTest(name="NonFinite"):
      np.testing.assert_equal(np.spacing(nan), np.spacing(np.float32(nan)))
      if float_type != float8_e4m3b11:  # inf not supported.
        np.testing.assert_equal(np.spacing(inf), np.spacing(np.float32(inf)))


if __name__ == "__main__":
  absltest.main()
