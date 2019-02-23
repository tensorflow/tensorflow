# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SparseAdd."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import timeit

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
import tensorflow.python.ops.sparse_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


def _sparsify(x, thresh=0.5, index_dtype=np.int64):
  x[x < thresh] = 0

  non_zero = np.where(x)
  x_indices = np.vstack(non_zero).astype(index_dtype).T
  x_values = x[non_zero]
  x_shape = x.shape

  return sparse_tensor.SparseTensor(
      indices=x_indices, values=x_values, dense_shape=x_shape), len(x_values)


class SparseAddTest(test.TestCase):

  def _randomTensor(self, size, np_dtype, sparse=True):
    n, m = size
    x = np.random.randn(n, m).astype(np_dtype)
    return _sparsify(x) if sparse else x

  def _SparseTensorValue_3x3(self, negate=False):
    # [    1]
    # [2    ]
    # [3   4]
    # ...or its cwise negation, if `negate`
    ind = np.array([[0, 1], [1, 0], [2, 0], [2, 1]])
    val = np.array([1, 2, 3, 4])
    if negate:
      val = -np.array([1, 2, 3, 4])
    shape = np.array([3, 3])
    return sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(val, np.float32), np.array(shape, np.int64))

  def _SparseTensor_3x3(self, negate=False):
    return sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_3x3(negate))

  def _SparseTensor_3x3_v2(self):
    # [           1]
    # [-1.9        ]
    # [   3    -4.2]
    ind = np.array([[0, 1], [1, 0], [2, 0], [2, 1]])
    val = np.array([1, -1.9, 3, -4.2])
    shape = np.array([3, 3])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.float32),
        constant_op.constant(shape, dtypes.int64))

  def testAddSelf(self):
    with test_util.force_cpu():
      for sp_a in (self._SparseTensorValue_3x3(), self._SparseTensor_3x3()):
        for sp_b in (self._SparseTensorValue_3x3(), self._SparseTensor_3x3()):
          sp_sum = sparse_ops.sparse_add(sp_a, sp_b)
          self.assertAllEqual((3, 3), sp_sum.get_shape())

          sum_out = self.evaluate(sp_sum)

          self.assertEqual(sp_sum.dense_shape.get_shape(), [2])
          self.assertAllEqual(sum_out.indices, [[0, 1], [1, 0], [2, 0], [2, 1]])
          self.assertAllEqual(sum_out.values, [2, 4, 6, 8])
          self.assertAllEqual(sum_out.dense_shape, [3, 3])

  def testAddSelfAndNegation(self):
    with test_util.force_cpu():
      sp_a = self._SparseTensor_3x3()
      sp_b = self._SparseTensor_3x3(negate=True)

      sp_sum = sparse_ops.sparse_add(sp_a, sp_b, 0.1)
      sum_out = self.evaluate(sp_sum)

      self.assertEqual(sp_sum.dense_shape.get_shape(), [2])
      self.assertAllEqual(sum_out.indices, np.empty([0, 2]))
      self.assertAllEqual(sum_out.values, [])
      self.assertAllEqual(sum_out.dense_shape, [3, 3])

  def testSmallValuesShouldVanish(self):
    with test_util.force_cpu():
      sp_a = self._SparseTensor_3x3()
      sp_b = self._SparseTensor_3x3_v2()

      # sum:
      # [       2]
      # [.1      ]
      # [ 6   -.2]

      # two values should vanish: |.1| < .21, and |-.2| < .21
      sp_sum = sparse_ops.sparse_add(sp_a, sp_b, thresh=0.21)
      sum_out = self.evaluate(sp_sum)

      self.assertEqual(sp_sum.dense_shape.get_shape(), [2])
      self.assertAllEqual(sum_out.indices, [[0, 1], [2, 0]])
      self.assertAllEqual(sum_out.values, [2, 6])
      self.assertAllEqual(sum_out.dense_shape, [3, 3])

      # only .1 vanishes
      sp_sum = sparse_ops.sparse_add(sp_a, sp_b, thresh=0.11)
      sum_out = self.evaluate(sp_sum)

      self.assertEqual(sp_sum.dense_shape.get_shape(), [2])
      self.assertAllEqual(sum_out.indices, [[0, 1], [2, 0], [2, 1]])
      self.assertAllClose(sum_out.values, [2, 6, -.2])
      self.assertAllEqual(sum_out.dense_shape, [3, 3])

  @test_util.run_deprecated_v1
  def testGradients(self):
    np.random.seed(1618)  # Make it reproducible.
    with self.session(use_gpu=False):
      for n in [10, 31]:
        for m in [4, 17]:
          sp_a, nnz_a = self._randomTensor([n, m], np.float32)
          sp_b, nnz_b = self._randomTensor([n, m], np.float32)
          sp_sum = sparse_ops.sparse_add(sp_a, sp_b)
          nnz_sum = len(self.evaluate(sp_sum.values))

          err = gradient_checker.compute_gradient_error(
              [sp_a.values, sp_b.values], [(nnz_a,), (nnz_b,)], sp_sum.values,
              (nnz_sum,))
          self.assertLess(err, 1e-3)

  def testAddSparseDense(self):
    np.random.seed(1618)  # Make it reproducible.
    n, m = np.random.randint(30, size=2)
    for dtype in [np.float32, np.float64, np.int64, np.complex64]:
      for index_dtype in [np.int32, np.int64]:
        rand_vals_np = np.random.randn(n, m).astype(dtype)
        dense_np = np.random.randn(n, m).astype(dtype)

        with test_util.force_cpu():
          sparse, unused_nnz = _sparsify(rand_vals_np, index_dtype=index_dtype)
          s = self.evaluate(
              sparse_ops.sparse_add(sparse, constant_op.constant(dense_np)))
          self.assertAllEqual(dense_np + rand_vals_np, s)
          self.assertTrue(s.dtype == dtype)

          # check commutativity
          s = self.evaluate(
              sparse_ops.sparse_add(constant_op.constant(dense_np), sparse))
          self.assertAllEqual(dense_np + rand_vals_np, s)
          self.assertTrue(s.dtype == dtype)

  @test_util.run_deprecated_v1
  def testSparseTensorDenseAddGradients(self):
    np.random.seed(1618)  # Make it reproducible.
    n, m = np.random.randint(30, size=2)
    rand_vals_np = np.random.randn(n, m).astype(np.float32)
    dense_np = np.random.randn(n, m).astype(np.float32)

    with self.session(use_gpu=False):
      sparse, nnz = _sparsify(rand_vals_np)
      dense = constant_op.constant(dense_np, dtype=dtypes.float32)
      s = sparse_ops.sparse_add(sparse, dense)

      err = gradient_checker.compute_gradient_error([sparse.values, dense],
                                                    [(nnz,), (n, m)], s, (n, m))
      self.assertLess(err, 1e-3)

  @test_util.run_deprecated_v1
  def testInvalidSparseTensor(self):
    with test_util.force_cpu():
      shape = [2, 2]
      val = [0]
      dense = constant_op.constant(np.zeros(shape, dtype=np.int32))

      for bad_idx in [
          [[-1, 0]],  # -1 is invalid.
          [[1, 3]],  # ...so is 3.
      ]:
        sparse = sparse_tensor.SparseTensorValue(bad_idx, val, shape)
        s = sparse_ops.sparse_add(sparse, dense)

        with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                     "invalid index"):
          self.evaluate(s)

######################## Benchmarking code


def _s2d_add_vs_sparse_add(sparsity, n, m, num_iters=50):
  np.random.seed(1618)

  with session.Session(graph=ops.Graph()) as sess:
    sp_vals = np.random.rand(n, m).astype(np.float32)
    sp_t, unused_nnz = _sparsify(sp_vals, thresh=sparsity, index_dtype=np.int32)
    vals = np.random.rand(n, m).astype(np.float32)

    s2d = math_ops.add(
        sparse_ops.sparse_tensor_to_dense(sp_t), constant_op.constant(vals))
    sa = sparse_ops.sparse_add(sp_t, constant_op.constant(vals))

    timeit.timeit(lambda: sess.run(s2d), number=3)
    timeit.timeit(lambda: sess.run(sa), number=3)

    s2d_total = timeit.timeit(lambda: sess.run(s2d), number=num_iters)
    sa_total = timeit.timeit(lambda: sess.run(sa), number=num_iters)

  # per-iter latency; secs to millis
  return s2d_total * 1e3 / num_iters, sa_total * 1e3 / num_iters


class SparseAddBenchmark(test.Benchmark):

  def benchmarkSparseAddDense(self):

    print("SparseAddDense: add with sparse_to_dense vs. sparse_add")
    print("%nnz \t n \t m \t millis(s2d) \t millis(sparse_add) \t speedup")

    for sparsity in [0.99, 0.5, 0.01]:
      for n in [1, 256, 50000]:
        for m in [100, 1000]:
          s2d_dt, sa_dt = _s2d_add_vs_sparse_add(sparsity, n, m)
          print("%.2f \t %d \t %d \t %.4f \t %.4f \t %.2f" % (sparsity, n, m,
                                                              s2d_dt, sa_dt,
                                                              s2d_dt / sa_dt))


if __name__ == "__main__":
  test.main()
