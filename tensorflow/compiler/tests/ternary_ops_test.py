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
"""Test cases for ternary operators."""

from absl.testing import parameterized
import numpy as np
import scipy.special as sps

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class TernaryOpsTest(xla_test.XLATestCase, parameterized.TestCase):

  def _testTernary(self, op, a, b, c, expected, rtol=1e-3, atol=1e-6):
    with self.session() as session:
      with self.test_scope():
        pa = array_ops.placeholder(dtypes.as_dtype(a.dtype), a.shape, name="a")
        pb = array_ops.placeholder(dtypes.as_dtype(b.dtype), b.shape, name="b")
        pc = array_ops.placeholder(dtypes.as_dtype(c.dtype), c.shape, name="c")
        output = op(pa, pb, pc)
      result = session.run(output, {pa: a, pb: b, pc: c})
      self.assertAllClose(result, expected, rtol=rtol, atol=atol)
      return result

  @parameterized.parameters(
      {'start': 1, 'end': 2, 'num': 1},
      {'start': 1, 'end': 4, 'num': 3},
      {'start': 0, 'end': 41, 'num': 42})
  @test_util.disable_mlir_bridge(
      'TODO(b/156174708): Dynamic result types not supported')
  def testLinspace(self, start, end, num):
    expected = np.linspace(start, end, num, dtype=np.float32)
    result = self._testTernary(
        math_ops.linspace,
        np.float32(start),
        np.float32(end),
        np.int32(num),
        expected)
    # According to linspace spec, start has to be the first element and end has
    # to be last element.
    self.assertEqual(result[-1], expected[-1])
    self.assertEqual(result[0], expected[0])

  def testRange(self):
    for dtype in (self.int_types | self.float_types) - {np.uint8}:
      self._testTernary(
          math_ops.range,
          dtype(1),
          dtype(2),
          dtype(1),
          expected=np.array([1], dtype=dtype),
      )
      self._testTernary(
          math_ops.range,
          dtype(1),
          dtype(7),
          dtype(2),
          expected=np.array([1, 3, 5], dtype=dtype),
      )
    # The distance from start to limit doesn't fit in int64.
    self._testTernary(
        math_ops.range,
        np.int64(-(2**63)),
        np.int64(2**63 - 1),
        np.int64(2**62),
        expected=np.array([-(2**63), -(2**62), 0, 2**62], dtype=np.int64),
    )

  def _rangeOfMaskedSize(self, args, dtype=np.int32):
    # n is the number of true entries in mask, a size only known at run time,
    # and args maps it to the range's start, limit and delta.
    @def_function.function(jit_compile=True)
    def f(mask):
      indices = array_ops.reshape(array_ops.where(mask), [-1])
      n = array_ops.size(array_ops.gather(math_ops.range(6), indices))
      return gen_math_ops._range(
          *[math_ops.cast(a, dtype) for a in args(math_ops.cast(n, dtype))]
      )

    with self.session() as session:
      with self.device_scope():
        mask = array_ops.placeholder(dtypes.bool, [6])
        output = f(mask)
      return session.run(
          output, {mask: [True, False, True, False, True, False]}
      )

  def testRangeDynamicBounds(self):
    # Regression test for GitHub issue 128950: with a negative delta, the
    # output was sized from the wrong bounds of start and limit, and with a
    # dynamic start, its values began at start's bound. Here n is 3, bound 6.
    for dtype in (np.int32, np.int64, np.float32):
      self.assertAllEqual(
          self._rangeOfMaskedSize(lambda n: (n - 1, -1, -1), dtype), [2, 1, 0]
      )
      self.assertAllEqual(
          self._rangeOfMaskedSize(lambda n: (6, n, -1), dtype), [6, 5, 4]
      )
      self.assertAllEqual(
          self._rangeOfMaskedSize(lambda n: (n, 5, 1), dtype), [3, 4]
      )
      self.assertAllEqual(
          self._rangeOfMaskedSize(lambda n: (n, -n, -2), dtype), [3, 1, -1]
      )
      # The CPU kernel rejects runtime values in the wrong order for delta's
      # sign. XLA can't report that at run time, and returns an empty range.
      self.assertAllEqual(
          self._rangeOfMaskedSize(lambda n: (n, 1, 1), dtype), []
      )
      self.assertAllEqual(
          self._rangeOfMaskedSize(lambda n: (n, 4, -1), dtype), []
      )
    self.assertAllEqual(
        self._rangeOfMaskedSize(lambda n: (n, 10, 2), np.uint32), [3, 5, 7, 9]
    )

  def testSelect(self):
    for dtype in self.numeric_types:
      self._testTernary(
          array_ops.where,
          np.array(False),
          np.array(2, dtype=dtype),
          np.array(7, dtype=dtype),
          expected=np.array(7, dtype=dtype))

      self._testTernary(
          array_ops.where,
          np.array(True),
          np.array([1, 2, 3, 4], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([1, 2, 3, 4], dtype=dtype))

      self._testTernary(
          array_ops.where,
          np.array(False),
          np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          expected=np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype))

      self._testTernary(
          array_ops.where,
          np.array([0, 1, 1, 0], dtype=np.bool_),
          np.array([1, 2, 3, 4], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([5, 2, 3, 8], dtype=dtype))

      self._testTernary(
          array_ops.where,
          np.array([0, 1, 0], dtype=np.bool_),
          np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          expected=np.array([[7, 8], [3, 4], [11, 12]], dtype=dtype))

  def testSelectV2(self):
    for dtype in self.numeric_types:
      self._testTernary(
          array_ops.where_v2,
          np.array(False),
          np.array(2, dtype=dtype),
          np.array(7, dtype=dtype),
          expected=np.array(7, dtype=dtype))

      self._testTernary(
          array_ops.where_v2,
          np.array(True),
          np.array([1, 2, 3, 4], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([1, 2, 3, 4], dtype=dtype))

      self._testTernary(
          array_ops.where_v2,
          np.array(False),
          np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          expected=np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype))

      self._testTernary(
          array_ops.where_v2,
          np.array([0, 1, 1, 0], dtype=np.bool_),
          np.array([1, 2, 3, 4], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([5, 2, 3, 8], dtype=dtype))

      # Broadcast the condition
      self._testTernary(
          array_ops.where_v2,
          np.array([0, 1], dtype=np.bool_),
          np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          expected=np.array([[7, 2], [9, 4], [11, 6]], dtype=dtype))

      # Broadcast the then branch to the else
      self._testTernary(
          array_ops.where_v2,
          np.array([[0, 1], [1, 0], [1, 1]], dtype=np.bool_),
          np.array([[1, 2]], dtype=dtype),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          expected=np.array([[7, 2], [1, 10], [1, 2]], dtype=dtype))

      # Broadcast the else branch to the then
      self._testTernary(
          array_ops.where_v2,
          np.array([[1, 0], [0, 1], [0, 0]], dtype=np.bool_),
          np.array([[7, 8], [9, 10], [11, 12]], dtype=dtype),
          np.array([[1, 2]], dtype=dtype),
          expected=np.array([[7, 2], [1, 10], [1, 2]], dtype=dtype))

      # Broadcast the then/else branches to the condition
      self._testTernary(
          array_ops.where_v2,
          np.array([[1, 0], [0, 1], [1, 1]], dtype=np.bool_),
          np.array(7, dtype=dtype),
          np.array(8, dtype=dtype),
          expected=np.array([[7, 8], [8, 7], [7, 7]], dtype=dtype))
      self._testTernary(
          array_ops.where_v2,
          np.array([[1, 0], [0, 1], [0, 0]], dtype=np.bool_),
          np.array(7, dtype=dtype),
          np.array([8, 9], dtype=dtype),
          expected=np.array([[7, 9], [8, 7], [8, 9]], dtype=dtype))

  def testSlice(self):
    for dtype in self.numeric_types:
      self._testTernary(
          array_ops.slice,
          np.array([[], [], []], dtype=dtype),
          np.array([1, 0], dtype=np.int32),
          np.array([2, 0], dtype=np.int32),
          expected=np.array([[], []], dtype=dtype),
      )

      self._testTernary(
          array_ops.slice,
          np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype),
          np.array([0, 1], dtype=np.int32),
          np.array([2, 1], dtype=np.int32),
          expected=np.array([[2], [5]], dtype=dtype),
      )

  def testClipByValue(self):
    for dtype in (
        self.numeric_types - self.complex_types - self.unsigned_int_types
    ):
      test_cases = [
          (np.array([2, 4, 5], dtype=dtype), dtype(7)),  #
          (dtype(1), np.array([2, 4, 5], dtype=dtype)),  #
          (
              np.array([-2, 7, 7], dtype=dtype),
              np.array([-2, 9, 8], dtype=dtype),
          ),
      ]
      x = np.array([-2, 10, 6], dtype=dtype)
      for lower, upper in test_cases:
        self._testTernary(
            gen_math_ops._clip_by_value,
            x,
            lower,
            upper,
            expected=np.minimum(np.maximum(x, lower), upper))

  def testBetaincSanity(self):
    # This operation is only supported for float32 and float64.
    for dtype in self.numeric_types & {np.float32, np.float64}:
      # Sanity check a few identities:
      # - betainc(a, b, 0) == 0
      # - betainc(a, b, 1) == 1
      # - betainc(a, 1, x) == x ** a
      # Compare against the implementation in SciPy.
      a = np.array([.3, .4, .2, .2], dtype=dtype)
      b = np.array([1., 1., .4, .4], dtype=dtype)
      x = np.array([.3, .4, .0, .1], dtype=dtype)
      expected = sps.betainc(a, b, x)
      self._testTernary(
          math_ops.betainc, a, b, x, expected, rtol=5e-5, atol=6e-5
      )

  @parameterized.parameters(
      {
          'sigma': 1e15,
          'rtol': 1e-6,
          'atol': 1e-4
      },
      {
          'sigma': 30,
          'rtol': 1e-6,
          'atol': 2e-3
      },
      {
          'sigma': 1e-8,
          'rtol': 5e-4,
          'atol': 3e-4
      },
      {
          'sigma': 1e-16,
          'rtol': 1e-6,
          'atol': 2e-4
      },
  )
  def testBetainc(self, sigma, rtol, atol):
    # This operation is only supported for float32 and float64.
    for dtype in self.numeric_types & {np.float32, np.float64}:
      # Randomly generate a, b, x in the numerical domain of betainc.
      # Compare against the implementation in SciPy.
      a = np.abs(np.random.randn(10, 10) * sigma).astype(dtype)  # in (0, infty)
      b = np.abs(np.random.randn(10, 10) * sigma).astype(dtype)  # in (0, infty)
      x = np.random.rand(10, 10).astype(dtype)  # in (0, 1)
      expected = sps.betainc(a, b, x, dtype=dtype)
      self._testTernary(
          math_ops.betainc, a, b, x, expected, rtol=rtol, atol=atol)


if __name__ == "__main__":
  googletest.main()
