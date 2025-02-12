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
"""Tests for tensorflow.ops.tf.cast."""

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class CastOpTest(test.TestCase):

  def _toDataType(self, dtype):
    """Returns TensorFlow data type for numpy type."""
    if dtype == np.float32:
      return dtypes.float32
    elif dtype == np.float64:
      return dtypes.float64
    elif dtype == np.int32:
      return dtypes.int32
    elif dtype == np.int64:
      return dtypes.int64
    elif dtype == np.bool_:
      return dtypes.bool
    elif dtype == np.complex64:
      return dtypes.complex64
    elif dtype == np.complex128:
      return dtypes.complex128
    else:
      return None

  def _cast(self, x, dtype, use_gpu=False):
    with test_util.device(use_gpu):
      val = constant_op.constant(x, self._toDataType(np.array([x]).dtype))
      cast = math_ops.cast(val, self._toDataType(dtype), name="cast")
      return self.evaluate(cast)

  def _test(self, x, dtype, use_gpu=False):
    """Tests cast(x) to dtype behaves the same as numpy.astype."""
    np_ans = x.astype(dtype)
    tf_ans = self._cast(x, dtype, use_gpu)
    self.assertAllEqual(np_ans, tf_ans)

  def _testTypes(self, x, use_gpu=False):
    """Tests cast(x) to different tf."""
    if use_gpu:
      type_list = [
          np.float32, np.float64, np.int64, np.complex64, np.complex128
      ]
    else:
      type_list = [
          np.float32, np.float64, np.int32, np.int64, np.complex64,
          np.complex128
      ]
    for from_type in type_list:
      for to_type in type_list:
        self._test(x.astype(from_type), to_type, use_gpu)

    self._test(x.astype(np.bool_), np.float32, use_gpu)
    self._test(x.astype(np.uint8), np.float32, use_gpu)
    if not use_gpu:
      self._test(x.astype(np.bool_), np.int32, use_gpu)
      self._test(x.astype(np.int32), np.int32, use_gpu)

  def _testAll(self, x):
    self._testTypes(x, use_gpu=False)
    if x.dtype == np.float32 or x.dtype == np.float64:
      self._testTypes(x, use_gpu=True)

  def testBasic(self):
    self._testAll(np.arange(-10, 10).reshape(2, 10))
    self._testAll(np.linspace(-10, 10, 17))

  def testSmallValues(self):
    f4 = np.finfo(np.float32)
    f8 = np.finfo(np.float64)
    self._testAll(
        np.array([
            0, -1, 1, -f4.resolution, f4.resolution, f8.resolution,
            -f8.resolution
        ]))

  def testBfloat16(self):
    a = np.random.uniform(-100, 100, 100).astype(np.float32)
    with self.cached_session(use_gpu=False):
      b = math_ops.cast(math_ops.cast(a, dtypes.bfloat16), dtypes.float32)
      self.assertAllClose(a, self.evaluate(b), rtol=1 / 128.)
    with self.cached_session():
      b = math_ops.cast(math_ops.cast(a, dtypes.bfloat16), dtypes.float32)
      self.assertAllClose(a, self.evaluate(b), rtol=1 / 128.)

  def testFloat8(self):
    a = np.random.uniform(-100, 100, 100).astype(np.float32)
    for float8 in (dtypes.float8_e4m3fn, dtypes.float8_e5m2):
      # Including float8_e4m3fn should cover the float8 combinations without
      # loss of precision.
      for dtype in (dtypes.float64, dtypes.float32, dtypes.bfloat16,
                    dtypes.float16, dtypes.float8_e4m3fn):
        with self.cached_session(use_gpu=True):
          b = ops.convert_to_tensor(a, float8)
          c = math_ops.cast(math_ops.cast(b, dtype), float8)
          self.assertAllEqual(b, c)

  def testRandom(self):
    self._testAll(np.random.normal(0, 10, 210).reshape([2, 3, 5, 7]))
    self._testAll(np.random.normal(0, 1e6, 210).reshape([2, 3, 5, 7]))

  # Special values like int32max, int64min, inf, -inf, nan casted to
  # integer values in somewhat unexpected ways. And they behave
  # differently on CPU and GPU.
  def _compare(self, x, dst_dtype, expected, use_gpu=False):
    np.testing.assert_equal(
        self._cast(
            x, dst_dtype, use_gpu=use_gpu), dst_dtype(expected))

  def testIntToFloatBoundary(self):
    i4 = np.iinfo(np.int32)
    i8 = np.iinfo(np.int64)

    self._compare(i4.min, np.float32, i4.min, False)
    self._compare(i4.max, np.float32, i4.max, False)
    self._compare(i8.min, np.float32, i8.min, False)
    self._compare(i8.max, np.float32, i8.max, False)
    self._compare(i4.min, np.float64, i4.min, False)
    self._compare(i4.max, np.float64, i4.max, False)
    self._compare(i8.min, np.float64, i8.min, False)
    self._compare(i8.max, np.float64, i8.max, False)
    # NOTE: GPU does not support int32/int64 for casting.

  def testInfNan(self):
    self._compare(np.inf, np.float32, np.inf, False)
    self._compare(np.inf, np.float64, np.inf, False)
    self._compare(-np.inf, np.float32, -np.inf, False)
    self._compare(-np.inf, np.float64, -np.inf, False)
    self.assertAllEqual(np.isnan(self._cast(np.nan, np.float32, False)), True)
    self.assertAllEqual(np.isnan(self._cast(np.nan, np.float64, False)), True)

    self._compare(np.inf, np.float32, np.inf, True)
    self._compare(np.inf, np.float64, np.inf, True)
    self._compare(-np.inf, np.float32, -np.inf, True)
    self._compare(-np.inf, np.float64, -np.inf, True)
    self.assertAllEqual(np.isnan(self._cast(np.nan, np.float32, True)), True)
    self.assertAllEqual(np.isnan(self._cast(np.nan, np.float64, True)), True)

  def _OpError(self, x, dtype, err):
    with self.assertRaisesOpError(err):
      self.evaluate(math_ops.cast(x, dtype))

  def testNotImplemented(self):
    self._OpError(np.arange(0, 10), dtypes.string, "Cast.*int.*string.*")

  def testCastToTypeOfVariable(self):
    with self.cached_session():
      x = variables.Variable(5, dtype=dtypes.float32)
      y = variables.Variable(True, dtype=dtypes.bool)
      cast = math_ops.cast(y, x.dtype)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(cast))

  def testGradients(self):
    t = [dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]
    for src_t in t:
      for dst_t in t:
        with self.cached_session():
          x = constant_op.constant(1.0, src_t)

          def cast(x, dst_t=dst_t):
            x = array_ops.identity(x)
            x = math_ops.cast(x, dst_t)
            return x

          err = gradient_checker_v2.max_error(
              *gradient_checker_v2.compute_gradient(cast, [x]))
          self.assertLess(err, 1e-3)

  def testRefDtype(self):
    with context.graph_mode(), self.cached_session():
      x = gen_state_ops.variable(shape=[1], dtype=dtypes.float32)
      result = math_ops.cast(x, dtypes.float32)
      self.assertEqual(x.dtype, dtypes.float32_ref)
      self.assertEqual(result.dtype, dtypes.float32)


class SparseTensorCastTest(test.TestCase):

  def testCast(self):
    indices = constant_op.constant([[0], [1], [2]], dtypes.int64)
    values = constant_op.constant(np.array([1, 2, 3], np.int64))
    shape = constant_op.constant([3], dtypes.int64)
    st = sparse_tensor.SparseTensor(indices, values, shape)
    st_cast = math_ops.cast(st, dtypes.float32)

    self.assertAllEqual(st_cast.indices, [[0], [1], [2]])
    self.assertAllEqual(st_cast.values,
                        np.array([1, 2, 3], np.float32))
    self.assertAllEqual(st_cast.dense_shape, [3])


class SaturateCastTest(test.TestCase):

  def testSaturate(self):
    in_types = dtypes.float32,
    out_types = dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.float32
    for in_type in in_types:
      for out_type in out_types:
        lo, hi = in_type.min, in_type.max
        x = constant_op.constant(
            [lo, lo + 1, lo // 2, hi // 2, hi - 1, hi], dtype=in_type)
        y = math_ops.saturate_cast(x, dtype=out_type)
        self.assertEqual(y.dtype, out_type)
        x, y = self.evaluate([x, y])
        correct = np.maximum(out_type.min, np.minimum(out_type.max, x))
        self.assertAllEqual(correct, y)

  def testSaturateRealToComplex(self):
    in_types = (dtypes.float32, dtypes.float64)
    out_types = (dtypes.complex64, dtypes.complex128)
    for in_type in in_types:
      for out_type in out_types:
        lo, hi = in_type.min, in_type.max
        x = constant_op.constant([lo, lo + 1, lo // 2, hi // 2, hi - 1, hi],
                                 dtype=in_type)
        y = math_ops.saturate_cast(x, dtype=out_type)
        self.assertEqual(y.dtype, out_type)
        x, y = self.evaluate([x, y])
        correct = np.maximum(out_type.real_dtype.min,
                             np.minimum(out_type.real_dtype.max, x))
        self.assertAllEqual(correct, y)

  def testSaturateComplexToReal(self):
    in_types = (dtypes.complex64, dtypes.complex128)
    out_types = (dtypes.float32, dtypes.float64)
    for in_type in in_types:
      for out_type in out_types:
        lo, hi = in_type.real_dtype.min, in_type.real_dtype.max
        x = constant_op.constant([lo, lo + 1, lo // 2, hi // 2, hi - 1, hi],
                                 dtype=in_type)
        y = math_ops.saturate_cast(x, dtype=out_type)
        self.assertEqual(y.dtype, out_type)
        x, y = self.evaluate([x, y])
        correct = np.maximum(out_type.min, np.minimum(out_type.max, x))
        self.assertAllEqual(correct, y)

  def testSaturateAvoidsUndefinedBehavior(self):
    # E.g. float32 -> uint32, float32 cannot represent uint32 max, so trying
    # to clip to that range can introduce values still outside those
    # representable by uint32, which result in UB when followed by a cast.
    for in_type in [dtypes.float32, dtypes.float16, dtypes.bfloat16]:
      for out_type in [
          dtypes.int32,
          dtypes.uint32,
          dtypes.int16,
          dtypes.uint16,
          dtypes.int8,
          dtypes.uint8,
      ]:
        # Avoid casting issues by using comparisons instead of min/max to
        # enable proper type promotion.
        lo = (
            out_type.min
            if out_type.min > (in_type.min + 1)
            else (in_type.min + 1)
        )
        hi = (
            out_type.max
            if out_type.max < (in_type.max - 1)
            else (in_type.max - 1)
        )
        x = constant_op.constant(
            [lo - 1, lo, lo + 1, lo // 2, hi // 2, hi - 1, hi, hi + 1],
            dtype=in_type,
        )
        y = math_ops.saturate_cast(x, out_type)
        x, y = self.evaluate([x, y])

        # Ensure that we are at most one representable input element away from
        # the true answer.
        np_in_type = in_type.as_numpy_dtype
        expected = np.clip(x, out_type.min, out_type.max).astype(np_in_type)
        u = y.astype(np_in_type)
        v = expected
        self.assertTrue(np.all(v == np.nextafter(u, v)))

  @test_util.disable_xla("Clamp is not implemented for C128 in XLA")
  def testSaturateComplexToComplex(self):
    in_types = (dtypes.complex64, dtypes.complex128)
    out_types = (dtypes.complex64, dtypes.complex128)
    for in_type in in_types:
      for out_type in out_types:
        lo, hi = in_type.real_dtype.min, in_type.real_dtype.max
        x_real = constant_op.constant(
            [lo, lo + 1, lo // 2, hi // 2, hi - 1, hi],
            dtype=in_type.real_dtype)
        x = math_ops.complex(x_real, array_ops.transpose(x_real))
        y = math_ops.saturate_cast(x, dtype=out_type)
        self.assertEqual(y.dtype, out_type)
        x, y = self.evaluate([x, y])
        correct = np.maximum(
            out_type.real_dtype.min,
            np.minimum(out_type.real_dtype.max, np.real(x))) + 1j * np.maximum(
                out_type.real_dtype.min,
                np.minimum(out_type.real_dtype.max, np.imag(x)))
        self.assertAllEqual(correct, y)

  @test_util.with_forward_compatibility_horizons([2024, 11, 2])
  def testSaturateHandlesInfs(self):
    for in_type in [
        dtypes.float64,
        dtypes.float32,
        dtypes.float16,
        dtypes.bfloat16,
    ]:
      for out_type in [
          dtypes.float64,
          dtypes.float32,
          dtypes.float16,
          dtypes.bfloat16,
      ]:
        inf = float("inf")
        nan = float("nan")
        x = constant_op.constant([inf, -inf, nan, -nan], dtype=in_type)
        y = math_ops.saturate_cast(x, dtype=out_type)
        self.assertAllEqual(math_ops.is_finite(y), [True, True, False, False])


if __name__ == "__main__":
  test.main()
