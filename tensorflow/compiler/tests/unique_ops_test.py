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
"""Tests for unique ops."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import googletest


class UniqueTest(xla_test.XLATestCase):

  def testNegativeAxis(self):
    """Verifies that an axis with negative index is converted to positive."""
    with self.session() as session:
      with self.test_scope():
        px = array_ops.placeholder(dtypes.float32, [2, 1, 1], name="x")
        axis = constant_op.constant([-1], dtype=dtypes.int32)
        output = gen_array_ops.unique_v2(px, axis)
      result = session.run(
          output, {px: np.array([[[-2.0]], [[10.0]]], dtype=np.float32)}
      )
      self.assertAllEqual(
          result.y, np.array([[[-2.0]], [[10.0]]], dtype=np.float32)
      )
      self.assertAllEqual(result.idx, np.array([0], dtype=np.int32))


class UniqueWithCountsTest(xla_test.XLATestCase):

  def _run_jit_compiled(self, op, x):
    # jit_compile=True turns a missing XLA kernel into an error. Without it, the
    # session soft-places the op on the regular TF kernel and the test passes
    # without exercising XLA.
    x = np.array(x)
    with self.session() as session:
      with self.test_scope():
        px = array_ops.placeholder(x.dtype, x.shape, name="x")
        output = def_function.function(op, jit_compile=True)(px)
      return session.run(output, {px: x})

  def _unique_with_counts(self, x, out_idx=dtypes.int32):
    return self._run_jit_compiled(
        lambda t: gen_array_ops.unique_with_counts(t, out_idx=out_idx), x
    )

  def _unique_with_counts_v2(self, x, axis):
    return self._run_jit_compiled(
        lambda t: gen_array_ops.unique_with_counts_v2(
            t, constant_op.constant(axis, dtype=dtypes.int32)
        ),
        x,
    )

  def testValuesKeepFirstOccurrenceOrder(self):
    for x, y, idx, count in [
        ([1, 2, 2, 3, 3, 3], [1, 2, 3], [0, 1, 1, 2, 2, 2], [1, 2, 3]),
        ([3, 1, 3, 2, 1, 3], [3, 1, 2], [0, 1, 0, 2, 1, 0], [3, 2, 1]),
        ([5, 5, 5], [5], [0, 0, 0], [3]),
        ([9, 8, 7], [9, 8, 7], [0, 1, 2], [1, 1, 1]),
        ([7], [7], [0], [1]),
    ]:
      with self.subTest(x=x):
        result = self._unique_with_counts(np.array(x, dtype=np.int32))
        self.assertAllEqual(result.y, y)
        self.assertAllEqual(result.idx, idx)
        self.assertAllEqual(result.count, count)

  def testNaNsAreDistinct(self):
    result = self._unique_with_counts(
        np.array([np.nan, 1.0, np.nan, 1.0], dtype=np.float32)
    )
    self.assertAllEqual(result.y, [np.nan, 1.0, np.nan])
    self.assertAllEqual(result.idx, [0, 1, 2, 1])
    self.assertAllEqual(result.count, [1, 2, 1])

  def testOutIdxInt64(self):
    result = self._unique_with_counts(
        np.array([3, 1, 3], dtype=np.int32), out_idx=dtypes.int64
    )
    self.assertEqual(result.idx.dtype, np.int64)
    self.assertEqual(result.count.dtype, np.int64)
    self.assertAllEqual(result.count, [2, 1])

  def testEmpty(self):
    for out_idx in [dtypes.int32, dtypes.int64]:
      with self.subTest(out_idx=out_idx):
        result = self._unique_with_counts(
            np.zeros([0], dtype=np.int32), out_idx=out_idx
        )
        self.assertEqual(result.y.shape, (0,))
        self.assertEqual(result.idx.shape, (0,))
        self.assertEqual(result.count.shape, (0,))
        self.assertEqual(result.idx.dtype, out_idx.as_numpy_dtype)
        self.assertEqual(result.count.dtype, out_idx.as_numpy_dtype)

  def testDynamicInputIsRejected(self):
    with self.assertRaisesRegex(
        errors.UnimplementedError,
        "UniqueWithCounts does not support inputs with a dynamic shape when "
        "compiled with XLA",
    ):
      self._run_jit_compiled(
          lambda t: gen_array_ops.unique_with_counts(
              array_ops.boolean_mask(t, t > 0)
          ),
          np.array([3, -1, 3, 2, -1, 3], dtype=np.int32),
      )

  def testUnknownShapeArgument(self):
    x = np.array([3, 1, 3], dtype=np.int32)
    with self.session() as session:
      with self.test_scope():
        px = array_ops.placeholder(dtypes.int32, [None], name="x")
        output = def_function.function(
            gen_array_ops.unique_with_counts, jit_compile=True
        )(px)
      result = session.run(output, {px: x})
    self.assertAllEqual(result.y, [3, 1])
    self.assertAllEqual(result.idx, [0, 1, 0])
    self.assertAllEqual(result.count, [2, 1])

  def testV2Axis(self):
    x = np.array([[1, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int32)
    for axis, y, idx, count in [
        ([0], [[1, 0, 0], [2, 0, 0]], [0, 0, 1], [2, 1]),
        ([1], [[1, 0], [1, 0], [2, 0]], [0, 1, 1], [1, 2]),
        ([-1], [[1, 0], [1, 0], [2, 0]], [0, 1, 1], [1, 2]),
    ]:
      with self.subTest(axis=axis):
        result = self._unique_with_counts_v2(x, axis)
        self.assertAllEqual(result.y, y)
        self.assertAllEqual(result.idx, idx)
        self.assertAllEqual(result.count, count)


if __name__ == "__main__":
  googletest.main()
