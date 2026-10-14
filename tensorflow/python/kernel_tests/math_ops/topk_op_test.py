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
"""Tests for TopK op."""

import itertools
import sys

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class TopKTest(test.TestCase):

  def _validateTopK(
      self,
      inputs,
      k,
      expected_values,
      expected_indices,
      sorted=True,
      index_type=dtypes.int32,
  ):  # pylint: disable=redefined-builtin
    np_expected_values = np.array(expected_values)
    np_expected_indices = np.array(expected_indices)
    with self.cached_session():
      values_op, indices_op = nn_ops.top_k(
          inputs, k, sorted=sorted, index_type=index_type
      )
      values, indices = self.evaluate([values_op, indices_op])

      self.assertEqual(indices.dtype, index_type)
      self.assertShapeEqual(np_expected_values, values_op)
      self.assertShapeEqual(np_expected_indices, indices_op)

      if sorted:
        self.assertAllClose(np_expected_values, values)
        # Do some special casing of equality of indices: if indices
        # are not the same, but values are floating type, ensure that
        # the values are within epsilon of each other.
        if not np.issubdtype(np_expected_values.dtype, np.floating) and \
            np_expected_values.dtype != dtypes.bfloat16.as_numpy_dtype:
          # Values are not floating point type; check indices exactly
          self.assertAllEqual(np_expected_indices, indices)
        else:
          # Values are floating point; indices may be swapped for
          # values near each other.
          indices_not_equal = np_expected_indices != indices
          if np.any(indices_not_equal):
            values_unsure = values[indices_not_equal]
            expected_values_unsure = expected_values[indices_not_equal]
            self.assertAllClose(expected_values_unsure, values_unsure)
      else:
        np_inputs = np.array(inputs)

        # Check that the indices are valid.
        for result_index, src_index in np.ndenumerate(indices):
          value = values[result_index]
          expected_value = np_inputs[result_index[0], src_index]
          np.testing.assert_almost_equal(value, expected_value)

        # Check that if two elements are equal, the lower-index element appears
        # first.
        shape = values.shape
        for batch_index in range(shape[0]):
          for index in range(shape[1] - 1):
            if np.isclose(values[batch_index, index],
                          values[batch_index, index + 1]):
              self.assertLess(indices[batch_index, index],
                              indices[batch_index, index + 1])

        # Now check the results, ignoring order.
        self.assertAllEqual(np.sort(np_expected_indices), np.sort(indices))
        self.assertAllClose(np.sort(np_expected_values), np.sort(values))

  def testTop1(self):
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.3, 0.2]]
    self._validateTopK(inputs, 1, [[0.4], [0.3]], [[3], [1]])

  def testTop2(self):
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.4, 0.2]]
    self._validateTopK(inputs, 2, [[0.4, 0.3], [0.4, 0.3]], [[3, 1], [2, 1]])

  def testOutputIndexType(self):
    for index_type in [dtypes.int16, dtypes.int32, dtypes.int64]:
      inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.4, 0.2]]
      self._validateTopK(
          inputs,
          2,
          [[0.4, 0.3], [0.4, 0.3]],
          [[3, 1], [2, 1]],
          index_type=index_type,
      )

  def testKType(self):
    for ktype in [dtypes.int32, dtypes.int64, dtypes.int16]:
      inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.4, 0.2]]
      self._validateTopK(
          inputs,
          constant_op.constant(2, dtype=ktype),
          [[0.4, 0.3], [0.4, 0.3]],
          [[3, 1], [2, 1]],
      )

  def testTop3(self):
    for k in range(3, 11, 2):
      for dim in range(512, 12288, 512):
        inputs = np.random.permutation(
            np.linspace(0, 100, dim, dtype=np.float64))
        indices = np.argsort(-inputs)[:k]
        values = -np.sort(-inputs)[:k]
        self._validateTopK(inputs, k, values, indices)

  def testTop1AllNan(self):
    inputs = [[np.nan, np.nan], [np.nan, np.nan]]
    self._validateTopK(inputs, 1, [[np.nan], [np.nan]], [[0], [0]])

  def testTop1WithNan(self) -> None:
    neg_nan = np.copysign(np.nan, -1.0)
    inputs = [
        [3.0, np.nan, 2.0],
        [1.0, 2.0, 3.0, np.nan],
        [np.nan, 5.0, np.nan],
        [1.0, 5.0, 2.0],
        # A leading +NaN is the maximum and short-circuits the scan.
        [np.nan, 1000.0, np.nan, 2.0],
        # A leading -NaN does not: the scan continues to the later +NaN.
        [neg_nan, 1.0, np.nan, 2.0],
    ]
    expected_indices = [1, 3, 0, 1, 0, 2]
    with self.cached_session():
      for row, expected_idx in zip(inputs, expected_indices):
        values, indices = self.evaluate(nn_ops.top_k(row, 1))
        self.assertAllEqual(indices, [expected_idx])
        if np.isnan(row[expected_idx]):
          self.assertTrue(np.isnan(values[0]))
        else:
          self.assertEqual(values[0], row[expected_idx])

      # The two leading-NaN rows again, batched with a finite row.
      batched = np.array(
          inputs[4:] + [[4.0, 3.0, 1000.0, 2.0]], dtype=np.float32
      )
      values, indices = self.evaluate(nn_ops.top_k(batched, 1))
      self.assertAllEqual(indices, [[0], [2], [2]])
      self.assertTrue(np.all(np.isnan(values[:2])))
      self.assertFalse(np.any(np.signbit(values[:2])))
      self.assertEqual(values[2, 0], 1000.0)

  def testTopKNanEdgeCases(self) -> None:
    # +NaN sorts first and -NaN last, as on GPU (radix sort on the key bits)
    # and in the XLA total order.
    neg_nan = np.copysign(np.nan, -1.0)
    pos_nan = np.nan
    inputs = [-np.inf, pos_nan, 0.0, np.inf, -0.0, neg_nan]
    with self.cached_session():
      values, indices = self.evaluate(nn_ops.top_k(inputs, 6))
      self.assertAllEqual(indices, [1, 3, 2, 4, 0, 5])
      self.assertTrue(np.isnan(values[0]))
      self.assertAllEqual(values[1:5], [np.inf, 0.0, -0.0, -np.inf])
      self.assertTrue(np.isnan(values[5]))

      all_nans = [pos_nan, neg_nan, pos_nan, neg_nan]
      for k, expected in ((1, [0]), (2, [0, 2]), (4, [0, 2, 1, 3])):
        vals, idxs = self.evaluate(nn_ops.top_k(all_nans, k))
        self.assertTrue(np.all(np.isnan(vals)))
        self.assertAllEqual(idxs, expected)

      # k == 1 prefers any non-NaN value over -NaN.
      _, idxs = self.evaluate(nn_ops.top_k([neg_nan, -np.inf, neg_nan], 1))
      self.assertAllEqual(idxs, [1])

  def testTopKNegatedNan(self) -> None:
    # Ascending sort is implemented as top_k(-x): negated NaNs must sort last.
    inputs = -np.array(
        [np.nan, 3.0, 1.0, np.nan, 2.0, np.nan, 0.5], dtype=np.float32
    )
    self.assertTrue(np.all(np.signbit(inputs[[0, 3, 5]])))
    with self.cached_session():
      values, indices = self.evaluate(nn_ops.top_k(inputs, 7))
      self.assertAllEqual(indices, [6, 2, 4, 1, 0, 3, 5])
      self.assertAllEqual(values[:4], [-0.5, -1.0, -2.0, -3.0])
      self.assertTrue(np.all(np.isnan(values[4:])))

  def testTopKIntegerTypes(self) -> None:
    inputs = np.array([[5, 2, 8, 8, 1], [9, 3, 7, 0, 4]], dtype=np.int32)
    with self.cached_session():
      values, indices = self.evaluate(nn_ops.top_k(inputs, 1))
      self.assertAllEqual(values, [[8], [9]])
      self.assertAllEqual(indices, [[2], [0]])

      values, indices = self.evaluate(nn_ops.top_k(inputs, 3))
      self.assertAllEqual(values, [[8, 8, 5], [9, 7, 4]])
      self.assertAllEqual(indices, [[2, 3, 0], [0, 2, 4]])

      values, indices = self.evaluate(nn_ops.top_k(inputs, 5))
      self.assertAllEqual(values, [[8, 8, 5, 2, 1], [9, 7, 4, 3, 0]])
      self.assertAllEqual(indices, [[2, 3, 0, 1, 4], [0, 2, 4, 1, 3]])

      # k == 1 keeps the lowest index on ties for other integer widths too,
      # including a tie after the running maximum has been updated.
      for dtype in (
          np.int8,
          np.uint8,
          np.int16,
          np.uint16,
          np.uint32,
          np.int64,
          np.uint64,
      ):
        with self.subTest(dtype=dtype):
          info = np.iinfo(dtype)
          row = np.array([[info.min, info.max, 0, info.max]], dtype=dtype)
          values, indices = self.evaluate(nn_ops.top_k(row, 1))
          self.assertAllEqual(indices, [[1]])
          self.assertAllEqual(values, [[info.max]])

  def testTopKNanDtypes(self) -> None:
    dtypes_to_test = (
        np.float64,
        np.float32,
        np.float16,
        dtypes.bfloat16.as_numpy_dtype,
    )
    index_types = (dtypes.int16, dtypes.int32, dtypes.int64)
    for dtype, index_type in itertools.product(dtypes_to_test, index_types):
      inputs = np.array(
          [np.nan, 3.0, 1.0, np.nan, 2.0, np.nan, 0.5], dtype=dtype
      )
      with self.subTest(dtype=dtype, index_type=index_type):
        with self.cached_session():
          for k, expected in (
              (1, [0]),
              (3, [0, 3, 5]),
              (7, [0, 3, 5, 1, 4, 2, 6]),
          ):
            values, indices = self.evaluate(
                nn_ops.top_k(inputs, k, index_type=index_type)
            )
            self.assertEqual(indices.dtype, index_type.as_numpy_dtype)
            self.assertAllEqual(indices, expected)
            self.assertTrue(np.all(np.isnan(values[:3])))
            self.assertAllClose(
                values[3:], [3.0, 2.0, 1.0, 0.5][: max(k - 3, 0)]
            )

          # +NaN first and -NaN last on the k == 1, heap, and full-sort paths;
          # a leading -NaN must not end the k == 1 scan early.
          neg_nan_row = np.array(
              [np.copysign(np.nan, -1.0), 1.0, np.nan, 2.0], dtype=dtype
          )
          for k, expected in ((1, [2]), (2, [2, 3]), (4, [2, 3, 1, 0])):
            values, indices = self.evaluate(
                nn_ops.top_k(neg_nan_row, k, index_type=index_type)
            )
            self.assertAllEqual(indices, expected)
            values = values.astype(np.float32)
            self.assertTrue(np.isnan(values[0]))
            self.assertFalse(np.signbit(values[0]))
            if k == neg_nan_row.size:  # Full sort: -NaN, with its sign, last.
              self.assertTrue(np.isnan(values[-1]))
              self.assertTrue(np.signbit(values[-1]))

  def testTopKBatchedWithNan(self) -> None:
    inputs = np.array(
        [
            [3.0, np.nan, 2.0, 1.0],
            [np.nan, 5.0, np.nan, 4.0],
            [1.0, 2.0, 3.0, np.nan],
            [4.0, 3.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    with self.cached_session():
      values, indices = self.evaluate(nn_ops.top_k(inputs, 1))
      self.assertTrue(np.isnan(values[0, 0]))
      self.assertEqual(indices[0, 0], 1)
      self.assertTrue(np.isnan(values[1, 0]))
      self.assertEqual(indices[1, 0], 0)
      self.assertTrue(np.isnan(values[2, 0]))
      self.assertEqual(indices[2, 0], 3)
      self.assertEqual(values[3, 0], 4.0)
      self.assertEqual(indices[3, 0], 0)

      values, indices = self.evaluate(nn_ops.top_k(inputs, 2))
      self.assertTrue(np.isnan(values[0, 0]))
      self.assertEqual(indices[0, 0], 1)
      self.assertEqual(values[0, 1], 3.0)
      self.assertEqual(indices[0, 1], 0)

      self.assertTrue(np.isnan(values[1, 0]))
      self.assertTrue(np.isnan(values[1, 1]))
      self.assertEqual(indices[1, 0], 0)
      self.assertEqual(indices[1, 1], 2)

  def testTopKLargeBatchWithNanMatchesReference(self) -> None:
    # Large enough to be split across several CPU shards. Pinned to CPU: the
    # GPU heap path used for these shapes is not covered by this change.
    rng = np.random.default_rng(1234)
    num_rows, num_cols = 256, 2048
    # Quarter steps create many ties (including -0.0 vs 0.0).
    inputs = (
        np.round(rng.standard_normal((num_rows, num_cols)) * 4) / 4
    ).astype(np.float32)
    nan_mask = rng.random((num_rows, num_cols)) < 0.05
    signs = np.where(rng.random((num_rows, num_cols)) < 0.5, -1.0, 1.0)
    inputs[nan_mask] = np.copysign(np.nan, signs[nan_mask])

    # Reference order: +NaN, then values descending, then -NaN; ties by index.
    is_nan = np.isnan(inputs)
    group = np.where(is_nan, np.where(np.signbit(inputs), 2, 0), 1)
    neg_values = np.where(is_nan, 0.0, -inputs)
    col = np.broadcast_to(np.arange(num_cols), inputs.shape)
    expected = np.stack(
        [np.lexsort((col[r], neg_values[r], group[r])) for r in range(num_rows)]
    )

    with self.cached_session(), ops.device("/cpu:0"):
      for k in (1, 37, num_cols):
        _, indices = self.evaluate(nn_ops.top_k(inputs, k))
        self.assertAllEqual(indices, expected[:, :k])

  def testTopKNanUnsorted(self) -> None:
    inputs = [np.nan, 3.0, 1.0, np.nan, 2.0, np.nan, 0.5]
    with self.cached_session():
      values, indices = self.evaluate(nn_ops.top_k(inputs, 3, sorted=False))
      self.assertTrue(np.all(np.isnan(values)))
      self.assertEqual(set(indices), {0, 3, 5})

      # k == 1 and k == num_cols take dedicated code paths.
      values, indices = self.evaluate(nn_ops.top_k(inputs, 1, sorted=False))
      self.assertTrue(np.isnan(values[0]))
      self.assertAllEqual(indices, [0])

      values, indices = self.evaluate(nn_ops.top_k(inputs, 7, sorted=False))
      self.assertEqual(sorted(indices), list(range(7)))
      self.assertEqual(np.sum(np.isnan(values)), 3)

  def _testLargeSort(self, dtype):
    b = 10
    n = 5000
    inputs = np.random.permutation(
        np.linspace(0, 100, b * n, dtype=dtype)).reshape(b, n)
    indices = np.argsort(-inputs, axis=1)
    values = -np.sort(-inputs, axis=1)
    self._validateTopK(inputs, n, values, indices)

  def testLargeSort(self):
    self._testLargeSort(np.float32)
    self._testLargeSort(np.float16)
    self._testLargeSort(dtypes.bfloat16.as_numpy_dtype)

  def _testLargeTopK(self, dtype):
    b = 10
    n = 5000
    k = n - 1
    inputs = np.random.permutation(
        np.linspace(0, 100, b * n, dtype=dtype)).reshape(b, n)
    indices = np.argsort(-inputs, axis=1)[:, :k]
    values = -np.sort(-inputs, axis=1)[:, :k]
    self._validateTopK(inputs, k, values, indices)

  def testLargeTopK(self):
    self._testLargeTopK(np.float32)
    self._testLargeTopK(np.float16)
    self._testLargeTopK(dtypes.bfloat16.as_numpy_dtype)

  def _testMediumTopK(self, dtype):
    b = 5
    n = 500
    k = 50
    inputs = np.random.permutation(
        np.linspace(0, 100, b * n, dtype=dtype)).reshape(b, n)
    indices = np.argsort(-inputs, axis=1)[:, :k]
    values = -np.sort(-inputs, axis=1)[:, :k]
    self._validateTopK(inputs, k, values, indices)

  def testMediumTopK(self):
    self._testMediumTopK(np.float32)
    self._testMediumTopK(np.float16)
    self._testMediumTopK(dtypes.bfloat16.as_numpy_dtype)

  def testStableSort(self):
    b = 5
    n = 500
    for k in [1, 5, 50, 500]:
      # Lots of repeated integers taking values in [0, 3]
      inputs = np.random.permutation(
          np.linspace(0, 3, b * n, dtype=np.int32)).reshape(b, n)
      # Use mergesort, a stable sort, to get the indices.
      indices = np.argsort(-inputs, axis=1, kind="mergesort")[:, :k]
      values = -np.sort(-inputs, axis=1)[:, :k]
      self._validateTopK(inputs, k, values, indices)

  def testTopAll(self):
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.3, 0.2]]
    self._validateTopK(inputs, 4, [[0.4, 0.3, 0.2, 0.1], [0.3, 0.3, 0.2, 0.1]],
                       [[3, 1, 2, 0], [1, 2, 3, 0]])

  def testTop3Unsorted(self):
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.4, 0.3, 0.2]]
    self._validateTopK(
        inputs,
        3, [[0.2, 0.3, 0.4], [0.2, 0.4, 0.3]], [[2, 1, 3], [3, 1, 2]],
        sorted=False)

  def testTop3Vector(self):
    inputs = [3, 6, 15, 18, 6, 12, 1, 17, 3, 0, 4, 19, 1, 6]
    self._validateTopK(inputs, 3, [19, 18, 17], [11, 3, 7])

  def testTensorK(self):
    inputs = [3, 6, 15, 18, 6, 12, 1, 17, 3, 0, 4, 19, 1, 6]
    k = constant_op.constant(3)
    self._validateTopK(inputs, k, [19, 18, 17], [11, 3, 7])

  def testTop3ZeroRows(self):
    inputs = np.zeros([0, 10], dtype=np.float32)
    self._validateTopK(inputs, 3, np.zeros([0, 3], dtype=np.float32),
                       np.zeros([0, 3], dtype=np.int32))

  def testKNegative(self):
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "Need k >= 0, got -7|non-negative",
    ):
      self.evaluate(nn_ops.top_k([[0.1, 0.2], [0.3, 0.4]], -7))

  def testKTooLarge(self):
    inputs = [[0.1, 0.2], [0.3, 0.4]]
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        r"must have last dimension >= k = 4|must have at least k",
    ):
      self.evaluate(nn_ops.top_k(inputs, 4))

  @test_util.run_deprecated_v1
  def testTopKGradients(self):
    with self.session() as sess:
      inputs = array_ops.placeholder(dtypes.float32, shape=[2, 5])
      values, _ = nn_ops.top_k(inputs, 3)
      grad = sess.run(
          gradients_impl.gradients(
              values, inputs, grad_ys=[[[1., 2., 3.], [4., 5., 6.]]]),
          feed_dict={inputs: [[2., -1., 1000., 3., 4.],
                              [1., 5., 2., 4., 3.]]})[0]
    self.assertEqual(
        grad.tolist(), [[0., 0., 1., 3., 2.], [0., 4., 0., 5., 6.]])


class TopKBenchmark(test.Benchmark):

  def benchmarkTopK(self):
    for (m, n, p, use_gpu) in itertools.product(
        [128],
        [10, 100, 1000, 10000, 100000],
        [0.001, 0.01, 0.5, 0.99, 1.0],
        [False, True]):
      k = int(p * n)
      if k == 0:
        continue
      name = "m_%d_n_%d_k_%g_use_gpu_%s" % (m, n, k, use_gpu)
      device = "/%s:0" % ("gpu" if use_gpu else "cpu")
      with ops.Graph().as_default():
        with ops.device(device):
          x = random_ops.random_uniform((m, n))
          v = resource_variable_ops.ResourceVariable(x)
          op = nn_ops.top_k(v, k)
        with session.Session() as sess:
          self.evaluate(v.initializer)
          r = self.run_op_benchmark(sess, op, min_iters=100, name=name)
          gb_processed_input = m * n / 1.0e9
          throughput = gb_processed_input / r["wall_time"]
          print("Benchmark: %s \t wall_time: %0.03g s \t "
                "Throughput: %0.03g GB/s" % (name, r["wall_time"], throughput))
          sys.stdout.flush()


if __name__ == "__main__":
  test.main()
