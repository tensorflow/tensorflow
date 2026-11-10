# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for IsotonicRegression under XLA (b/no XLA_CPU_JIT kernel)."""

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class IsotonicRegressionOpTest(xla_test.XLATestCase, parameterized.TestCase):

  def _assert_isotonic_close(self, expected, actual, dtype):
    if dtype in (
        dtypes.bfloat16,
        dtypes.half,
        np.float16,
        dtypes.bfloat16.as_numpy_dtype,
    ):
      self.assertAllCloseAccordingToType(expected, actual, rtol=1e-2, atol=1e-2)
    else:
      self.assertAllClose(expected, actual)

  def _verify_isotonic(self, x_np, decreasing=True, axis=-1):
    """Verifies XLA execution matches eager baseline for isotonic regression."""
    # Compute ground truth using non-XLA CPU eager execution.
    with ops.device("/device:CPU:0"):
      expected_y, expected_segments = nn_ops.isotonic_regression(
          x_np, decreasing=decreasing, axis=axis
      )
      expected_y_np = expected_y.numpy()
      expected_seg_np = expected_segments.numpy()

    # 1. Test via XLA session and test_scope / device_scope
    with self.session() as sess:
      x_ph = array_ops.placeholder(
          dtypes.as_dtype(x_np.dtype), shape=x_np.shape
      )
      with self.device_scope():
        out_y, out_segments = nn_ops.isotonic_regression(
            x_ph, decreasing=decreasing, axis=axis
        )
      res_y, res_segments = sess.run(
          [out_y, out_segments], feed_dict={x_ph: x_np}
      )
      self._assert_isotonic_close(expected_y_np, res_y, x_np.dtype)
      self.assertAllEqual(expected_seg_np, res_segments)

    # 2. Test via tf.function(jit_compile=True) if eager is active
    if context.executing_eagerly():
      @def_function.function(jit_compile=True)
      def jitted_fn(inp):
        return nn_ops.isotonic_regression(
            inp, decreasing=decreasing, axis=axis
        )

      with ops.device("/device:CPU:0"):
        jit_y, jit_segments = jitted_fn(constant_op.constant(x_np))
        self._assert_isotonic_close(expected_y_np, jit_y.numpy(), x_np.dtype)
        self.assertAllEqual(expected_seg_np, jit_segments.numpy())

  @parameterized.parameters(
      {"decreasing": True},
      {"decreasing": False},
  )
  def test1DIncreasingAndDecreasing(self, decreasing):
    x_inc = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    self._verify_isotonic(x_inc, decreasing=decreasing)

    x_dec = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    self._verify_isotonic(x_dec, decreasing=decreasing)

  @parameterized.parameters(
      {"decreasing": True},
      {"decreasing": False},
  )
  def test1DUnordered(self, decreasing):
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype=np.float32)
    self._verify_isotonic(x, decreasing=decreasing)

  @parameterized.parameters(
      {"decreasing": True},
      {"decreasing": False},
  )
  def testNegativeValues(self, decreasing):
    # Strictly negative inputs
    x_neg = np.array([-10.0, -8.0, -5.0, -2.0, -1.0], dtype=np.float32)
    self._verify_isotonic(x_neg, decreasing=decreasing)

    # Strictly negative reverse
    x_neg_rev = np.array([-1.0, -3.0, -7.0, -12.0], dtype=np.float32)
    self._verify_isotonic(x_neg_rev, decreasing=decreasing)

  @parameterized.parameters(
      {"decreasing": True},
      {"decreasing": False},
  )
  def testMixedPositiveAndNegative(self, decreasing):
    x_mixed = np.array(
        [-5.0, 3.0, -2.0, 8.0, 0.0, -1.0, 4.0], dtype=np.float32
    )
    self._verify_isotonic(x_mixed, decreasing=decreasing)

  @parameterized.parameters(
      {"decreasing": True},
      {"decreasing": False},
  )
  def testConstantAndAllZeros(self, decreasing):
    # All zeros
    x_zeros = np.zeros(6, dtype=np.float32)
    self._verify_isotonic(x_zeros, decreasing=decreasing)

    # Constant non-zero positive
    x_const_pos = np.full(5, 3.5, dtype=np.float32)
    self._verify_isotonic(x_const_pos, decreasing=decreasing)

    # Constant non-zero negative
    x_const_neg = np.full(5, -7.2, dtype=np.float32)
    self._verify_isotonic(x_const_neg, decreasing=decreasing)

  @parameterized.parameters(
      {"decreasing": True},
      {"decreasing": False},
  )
  def testDuplicatesAndTies(self, decreasing):
    # Multiple adjacent and non-adjacent duplicates
    x_ties = np.array(
        [3.0, 3.0, 1.0, 1.0, 4.0, 4.0, 2.0, 2.0], dtype=np.float32
    )
    self._verify_isotonic(x_ties, decreasing=decreasing)

  def testSingleElement(self):
    x_single = np.array([42.0], dtype=np.float32)
    self._verify_isotonic(x_single, decreasing=True)
    self._verify_isotonic(x_single, decreasing=False)

    x_single_neg = np.array([-13.5], dtype=np.float32)
    self._verify_isotonic(x_single_neg, decreasing=True)
    self._verify_isotonic(x_single_neg, decreasing=False)

  @parameterized.parameters(
      {"axis": -1, "decreasing": True},
      {"axis": -1, "decreasing": False},
      {"axis": 0, "decreasing": True},
      {"axis": 0, "decreasing": False},
      {"axis": 1, "decreasing": True},
      {"axis": 1, "decreasing": False},
  )
  def test2DBatch(self, axis, decreasing):
    x = np.array(
        [
            [3.0, 1.0, 2.0, -1.0],
            [1.0, 3.0, 4.0, 2.0],
            [-2.0, 0.0, -1.0, 5.0],
        ],
        dtype=np.float32,
    )
    self._verify_isotonic(x, decreasing=decreasing, axis=axis)

  @parameterized.parameters(
      {"axis": -1},
      {"axis": 0},
      {"axis": 1},
      {"axis": 2},
  )
  def test3DHigherRank(self, axis):
    np.random.seed(42)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    self._verify_isotonic(x, decreasing=True, axis=axis)
    self._verify_isotonic(x, decreasing=False, axis=axis)

  @parameterized.parameters(
      {"dtype": np.float32, "expected_out_dtype": np.float32},
      {"dtype": np.float64, "expected_out_dtype": np.float64},
      {"dtype": np.int32, "expected_out_dtype": np.float64},
      {"dtype": np.int64, "expected_out_dtype": np.float64},
      {"dtype": np.int16, "expected_out_dtype": np.float32},
      {"dtype": np.int8, "expected_out_dtype": np.float32},
  )
  def testDtypesAndPromotions(self, dtype, expected_out_dtype):
    x = np.array([[3, 1, 2], [1, 3, 4]], dtype=dtype)
    self._verify_isotonic(x, decreasing=True)
    self._verify_isotonic(x, decreasing=False)

  def testBfloat16AndHalf(self):
    x_half = np.array([[3.0, 1.0, 2.0], [1.0, 3.0, 4.0]], dtype=np.float16)
    self._verify_isotonic(x_half, decreasing=True)
    self._verify_isotonic(x_half, decreasing=False)

    x_bf16 = np.array(
        [[3.0, 1.0, 2.0], [1.0, 3.0, 4.0]],
        dtype=dtypes.bfloat16.as_numpy_dtype,
    )
    self._verify_isotonic(x_bf16, decreasing=True)
    self._verify_isotonic(x_bf16, decreasing=False)


if __name__ == "__main__":
  test.main()

