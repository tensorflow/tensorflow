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
"""Tests for tensorflow.ops.argmax_op."""
import functools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ArgMaxTest(test.TestCase, parameterized.TestCase):

  def _testArg(self,
               method,
               x,
               axis,
               expected_values,
               use_gpu=False,
               expected_err_re=None):
    with self.session(use_gpu=use_gpu):
      ans = method(x, axis=axis)
      if expected_err_re is None:
        tf_ans = self.evaluate(ans)
        # Defaults to int64 output.
        self.assertEqual(np.int64, tf_ans.dtype)
        self.assertAllEqual(tf_ans, expected_values)
        self.assertShapeEqual(expected_values, ans)
      else:
        with self.assertRaisesOpError(expected_err_re):
          self.evaluate(ans)

  def _testBothArg(self,
                   method,
                   x,
                   axis,
                   expected_values,
                   expected_err_re=None):
    self._testArg(method, x, axis, expected_values, True, expected_err_re)
    # Compilation time is too large with XLA/CPU autojit.
    if not test_util.is_xla_enabled():
      self._testArg(method, x, axis, expected_values, False, expected_err_re)

  def _testBasic(self, dtype):
    x = np.arange(200, dtype=np.float32).astype(dtype)
    np.random.shuffle(x)

    # Check that argmin and argmax match numpy along the primary axis
    self._testBothArg(math_ops.argmax, x, 0, x.argmax())
    self._testBothArg(math_ops.argmin, x, 0, x.argmin())

  def _testTieBreaking(self, dtype):
    x = np.zeros(200, dtype=dtype)

    # Check that argmin and argmax match numpy along the primary axis for
    # breaking ties.
    self._testBothArg(math_ops.argmax, x, 0, x.argmax())
    self._testBothArg(math_ops.argmin, x, 0, x.argmin())

    # Check that argmin and argmax match numpy along axis=1 for
    # breaking ties.
    x = np.array([[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]], dtype=dtype)
    self._testBothArg(math_ops.argmax, x, 1, x.argmax(axis=1))
    self._testBothArg(math_ops.argmin, x, 1, x.argmin(axis=1))

  def _testDim(self, dtype):
    shape = (3, 2, 4, 5, 6, 3, 7)
    x = np.arange(
        functools.reduce(lambda x, y: x * y, shape),
        dtype=np.float32).astype(dtype)
    np.random.shuffle(x)
    x = x.reshape(shape)

    # Check that argmin and argmax match numpy along all axes
    for axis in range(-7, 7):
      self._testBothArg(math_ops.argmax, x, axis, x.argmax(axis))
      self._testBothArg(math_ops.argmin, x, axis, x.argmin(axis))

  @parameterized.parameters(np.float16, np.float32, np.float64, np.int16,
                            np.int32, np.int64, np.bool_,
                            dtypes.bfloat16.as_numpy_dtype)
  def testTypes(self, dtype):
    self._testBasic(dtype,)
    self._testTieBreaking(dtype)
    self._testDim(dtype)

  def testFloatInt32Output(self):
    x = np.asarray(100 * np.random.randn(200), dtype=np.float32)
    expected_values = x.argmax()
    with self.session():
      ans = math_ops.argmax(x, axis=0, output_type=dtypes.int32)
      tf_ans = self.evaluate(ans)
      self.assertEqual(np.int32, tf_ans.dtype)
      # The values are equal when comparing int32 to int64 because
      # the values don't have a range that exceeds 32-bit integers.
      self.assertAllEqual(tf_ans, expected_values)
    expected_values = x.argmin()
    with self.session():
      ans = math_ops.argmin(x, axis=0, output_type=dtypes.int32)
      tf_ans = self.evaluate(ans)
      self.assertEqual(np.int32, tf_ans.dtype)
      self.assertAllEqual(tf_ans, expected_values)

  def testEmpty(self):
    with self.cached_session():
      for op in math_ops.argmin, math_ops.argmax:
        with self.assertRaisesOpError(
            r"Reduction axis 0 is empty in shape \[0\]"):
          op([], 0).eval()

  @test_util.run_in_graph_and_eager_modes
  def testNumpyIntegerAxis(self):
    """NumPy integers and 0-D NumPy arrays are valid `axis` values.

    Both expose a `dtype` attribute, so `tf.math.argmax` treated them as
    tensors and rejected them with a message claiming that even
    `int32`/`int64` are unsupported, while `tf.math.argmin` accepted them. See
    GitHub issue #129821.
    """
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    for op in (math_ops.argmax, math_ops.argmin):
      expected_axis_1 = op(x, axis=1)
      expected_axis_last = op(x, axis=-1)
      for axis in (
          np.int8(1),
          np.uint8(1),
          np.int16(1),
          np.int32(1),
          np.int64(1),
          np.array(1, dtype=np.int8),
          np.array(1, dtype=np.int32),
          np.array(1, dtype=np.int64),
      ):
        self.assertAllEqual(op(x, axis=axis), expected_axis_1)
      for axis in (np.int64(-1), np.array(-1), np.array(-1, dtype=np.int32)):
        self.assertAllEqual(op(x, axis=axis), expected_axis_last)
    # Tensor `axis` values take the `tf.DType` branch, not the NumPy one.
    expected_argmax_1 = math_ops.argmax(x, axis=1)
    for dt in (
        dtypes.int8,
        dtypes.uint8,
        dtypes.int16,
        dtypes.uint16,
        dtypes.int32,
    ):
      axis_t = array_ops.ones((), dtype=dt)
      self.assertAllEqual(math_ops.argmax(x, axis=axis_t), expected_argmax_1)

  @test_util.run_in_graph_and_eager_modes
  def testNonIntegerAxisIsStillRejected(self):
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    # `bool` is an `int` subclass, so `np.array(True).item()` would silently
    # become axis 1; 1-D arrays are not scalar axes either.
    for axis in (
        1.0,
        np.float32(1.0),
        np.array(1.0),
        np.bool_(True),
        np.array(True),
        np.array([1], dtype=np.int32),
    ):
      with self.assertRaisesRegex(
          TypeError, "axis must be int or Tensor with integer datatype"
      ):
        math_ops.argmax(x, axis=axis)
    for dt in (dtypes.float32, dtypes.bool):
      with self.assertRaisesRegex(
          TypeError, "axis tensor dtypes .* is not supported"
      ):
        math_ops.argmax(x, axis=array_ops.ones((), dtype=dt))

  @test_util.run_deprecated_v1
  def testDefaultAxis(self):
    with self.cached_session():
      for op in math_ops.argmin, math_ops.argmax:
        ans = op([1]).eval()
        self.assertAllEqual(ans, 0)

  @test_util.run_deprecated_v1
  def testOutputEmpty(self):
    with self.cached_session():
      for op in math_ops.argmin, math_ops.argmax:
        ret = op(array_ops.zeros(shape=[1, 0, 2]), axis=-1).eval()
        self.assertEqual(ret.shape, (1, 0))


if __name__ == "__main__":
  test.main()
