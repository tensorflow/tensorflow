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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import sys

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class TopKTest(test.TestCase):

  def _validateTopK(self,
                    inputs,
                    k,
                    expected_values,
                    expected_indices,
                    sorted=True):  # pylint: disable=redefined-builtin
    np_expected_values = np.array(expected_values)
    np_expected_indices = np.array(expected_indices)
    with self.test_session(use_gpu=True) as sess:
      values_op, indices_op = nn_ops.top_k(inputs, k, sorted=sorted)
      values, indices = sess.run([values_op, indices_op])

      self.assertShapeEqual(np_expected_values, values_op)
      self.assertShapeEqual(np_expected_indices, indices_op)

      if sorted:
        self.assertAllClose(np_expected_values, values)
        # Do some special casing of equality of indices: if indices
        # are not the same, but values are floating type, ensure that
        # the values are within epsilon of each other.
        if not np.issubdtype(np_expected_values.dtype, np.float):
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
          np.testing.utils.assert_almost_equal(value, expected_value)

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

  def testTop3(self):
    k = 5
    inputs = np.random.permutation(np.linspace(0, 100, 6140, dtype=np.float64))
    indices = np.argsort(-inputs)[:k]
    values = -np.sort(-inputs)[:k]
    self._validateTopK(inputs, k, values, indices)

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

  def testKNegative(self):
    inputs = [[0.1, 0.2], [0.3, 0.4]]
    with self.test_session(use_gpu=True):
      k = array_ops.placeholder(dtypes.int32)
      values, _ = nn_ops.top_k(inputs, k)
      with self.assertRaisesOpError("Need k >= 0, got -7"):
        values.eval(feed_dict={k: -7})

  def testKTooLarge(self):
    inputs = [[0.1, 0.2], [0.3, 0.4]]
    with self.assertRaisesRegexp(ValueError,
                                 r"must have last dimension >= k = 4"):
      nn_ops.top_k(inputs, 4)

  def testTopKGradients(self):
    with self.test_session(use_gpu=True) as sess:
      inputs = array_ops.placeholder(dtypes.int32, shape=[2, 5])
      values, _ = nn_ops.top_k(inputs, 3)
      grad = sess.run(
          gradients_impl.gradients(
              values, inputs, grad_ys=[[[1, 2, 3], [4, 5, 6]]]),
          feed_dict={inputs: [[2, -1, 1000, 3, 4], [1, 5, 2, 4, 3]]})[0]
    self.assertEqual(grad.tolist(), [[0, 0, 1, 3, 2], [0, 4, 0, 5, 6]])


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
          v.initializer.run()
          r = self.run_op_benchmark(sess, op, min_iters=100, name=name)
          gb_processed_input = m * n / 1.0e9
          throughput = gb_processed_input / r["wall_time"]
          print("Benchmark: %s \t wall_time: %0.03g s \t "
                "Throughput: %0.03g GB/s" % (name, r["wall_time"], throughput))
          sys.stdout.flush()


if __name__ == "__main__":
  test.main()
