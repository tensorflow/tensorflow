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
"""Tests for sparse_ops.sparse_tensor_dense_matmul."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import app
from tensorflow.python.platform import test


def _maybe_complex(x):
  if x.dtype.kind == "c":  # complex
    return (x + 1j * x) / 2
  return x


class SparseTensorDenseMatMulTest(test.TestCase):

  def _testMatmul(self, x, y, adjoint_a=False, adjoint_b=False):
    x_mat = np.matrix(x)
    if adjoint_a:
      x_mat = x_mat.H
    y_mat = np.matrix(y)
    if adjoint_b:
      y_mat = y_mat.H

    np_ans = x_mat * y_mat

    x_indices = np.vstack(np.where(x)).astype(np.int64).T
    x_values = x[np.where(x)]
    x_shape = x.shape

    with self.test_session(use_gpu=True):
      sp_x_value = sparse_tensor.SparseTensorValue(
          indices=x_indices, values=x_values, dense_shape=x_shape)
      tf_value_ans = sparse_ops.sparse_tensor_dense_matmul(
          sp_x_value, y, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
      tf_tensor_ans = sparse_ops.sparse_tensor_dense_matmul(
          sparse_tensor.SparseTensor.from_value(sp_x_value),
          y,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b)

      # Ensure that the RHS shape is known at least.
      self.assertEqual(tf_value_ans.get_shape()[1], np_ans.shape[1])
      self.assertEqual(tf_tensor_ans.get_shape()[1], np_ans.shape[1])

      for out in (tf_value_ans.eval(), tf_tensor_ans.eval()):
        if x.dtype == np.float32:
          self.assertAllClose(np_ans, out, rtol=1e-4, atol=1e-4)
        elif x.dtype == np.float64:
          self.assertAllClose(np_ans, out, rtol=1e-6, atol=1e-6)
        else:
          self.assertAllClose(np_ans, out, rtol=1e-4, atol=1e-4)

  def _testBasic(self, np_dtype):
    x = _maybe_complex(np.random.rand(10, 10).astype(np_dtype))
    x[np.abs(x) < 0.5] = 0  # Make it sparse

    y = _maybe_complex(np.random.randn(10, 20).astype(np_dtype))

    self._testMatmul(x, y)

  def testBasic(self):
    np.random.seed(127)  # Repeatable results
    self._testBasic(np.int32)
    self._testBasic(np.float32)
    self._testBasic(np.float64)
    self._testBasic(np.complex64)
    self._testBasic(np.complex128)

  def testShapeInference(self):
    x = np.random.rand(10, 10)
    x[np.abs(x) < 0.5] = 0  # Make it sparse
    y = np.random.randn(10, 20)
    x_indices = np.vstack(np.where(x)).astype(np.int64).T
    x_values = x[np.where(x)]
    x_shape = x.shape
    x_st = sparse_tensor.SparseTensor(x_indices, x_values, x_shape)
    result = sparse_ops.sparse_tensor_dense_matmul(x_st, y)
    self.assertEqual(result.get_shape(), (10, 20))

    x_shape_unknown = array_ops.placeholder(dtype=dtypes.int64, shape=None)
    x_st_shape_unknown = sparse_tensor.SparseTensor(x_indices, x_values,
                                                    x_shape_unknown)
    result_left_shape_unknown = sparse_ops.sparse_tensor_dense_matmul(
        x_st_shape_unknown, y)
    self.assertEqual(result_left_shape_unknown.get_shape().as_list(),
                     [None, 20])

    x_shape_inconsistent = [10, 15]
    x_st_shape_inconsistent = sparse_tensor.SparseTensor(x_indices, x_values,
                                                         x_shape_inconsistent)
    with self.assertRaisesRegexp(ValueError, "Dimensions must be equal"):
      sparse_ops.sparse_tensor_dense_matmul(x_st_shape_inconsistent, y)

  # Tests setting one dimension to be a high value.
  def _testLarge(self, np_dtype):
    r1 = np.random.randint(6000, 20000)
    r2 = np.random.randint(1, 10)
    r3 = np.random.randint(1, 10)

    for m, k, n in [(r1, r2, r3),
                    (r2, r1, r3),
                    (r2, r3, r1)]:
      x = _maybe_complex(np.random.rand(m, k).astype(np_dtype))
      x[np.abs(x) < 0.8] = 0

      y = _maybe_complex(np.random.randn(k, n).astype(np_dtype))

      self._testMatmul(x, y)

  def testLarge(self):
    np.random.seed(127)  # Repeatable results
    self._testLarge(np.float32)
    self._testLarge(np.float64)
    self._testLarge(np.complex64)
    self._testLarge(np.complex128)

  # Tests random sized matrices.
  def testFloatRandom(self):
    np.random.seed(127)  # Repeatable results
    for _ in range(8):
      for adjoint_a in [True, False]:
        for adjoint_b in [True, False]:
          for thresh in [0.0, 0.2, 0.8, 1.0]:
            n, k, m = np.random.randint(1, 100, size=3)
            x = np.random.rand(n, k).astype(np.float32)
            x[x < thresh] = 0  # Make it sparse
            y = np.random.randn(k, m).astype(np.float32)
            x = x.transpose() if adjoint_a else x
            y = y.transpose() if adjoint_b else y
            self._testMatmul(x, y, adjoint_a, adjoint_b)


def _sparse_tensor_dense_vs_dense_matmul_benchmark_dense(x, y, adjoint_a,
                                                         adjoint_b):

  def body(t, prev):
    with ops.control_dependencies([prev]):
      return (t + 1, math_ops.matmul(
          x,
          y,
          transpose_a=adjoint_a,
          transpose_b=adjoint_b,
          a_is_sparse=True,
          b_is_sparse=False))

  t0 = constant_op.constant(0)
  v0 = constant_op.constant(0.0)

  def _timeit(iterations, _):
    (_, final) = control_flow_ops.while_loop(
        lambda t, _: t < iterations,
        body, (t0, v0),
        parallel_iterations=1,
        back_prop=False)
    return [final]

  return _timeit


def _sparse_tensor_dense_vs_dense_matmul_benchmark_sparse(x_ind, x_val, x_shape,
                                                          y, adjoint_a,
                                                          adjoint_b):
  sp_x = sparse_tensor.SparseTensor(
      indices=x_ind, values=x_val, dense_shape=x_shape)

  def body(t, prev):
    with ops.control_dependencies([prev]):
      return (t + 1, sparse_ops.sparse_tensor_dense_matmul(
          sp_x, y, adjoint_a=adjoint_a, adjoint_b=adjoint_b))

  t0 = constant_op.constant(0)
  v0 = constant_op.constant(0.0)

  def _timeit(iterations, _):
    (_, final) = control_flow_ops.while_loop(
        lambda t, _: t < iterations,
        body, (t0, v0),
        parallel_iterations=1,
        back_prop=False)
    return [final]

  return _timeit


def sparse_tensor_dense_vs_dense_matmul_benchmark(thresh,
                                                  m,
                                                  k,
                                                  n,
                                                  adjoint_a,
                                                  adjoint_b,
                                                  use_gpu,
                                                  skip_dense=False):
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  # Configurable for benchmarking:
  # config.intra_op_parallelism_threads = 100
  # config.gpu_options.per_process_gpu_memory_fraction = 0.3

  np.random.seed([6, 117])  # Reproducibility
  x = np.random.rand(m, k).astype(np.float32)
  x[x < thresh] = 0
  y = np.random.randn(k, n).astype(np.float32)
  if adjoint_a:
    x = x.T
  if adjoint_b:
    y = y.T

  def _timer(sess, ops_fn, iterations):
    # Warm in
    sess.run(ops_fn(10, sess))

    # Timing run
    start = time.time()
    sess.run(ops_fn(iterations, sess))
    end = time.time()

    return (end - start) / (1.0 * iterations)  # Average runtime per iteration

  # Using regular matmul, marking one of the matrices as dense.
  if skip_dense:
    delta_dense = float("nan")
  else:
    with session.Session("", config=config, graph=ops.Graph()) as sess:
      if not use_gpu:
        with ops.device("/cpu:0"):
          x_t = constant_op.constant(x)
          y_t = constant_op.constant(y)
          ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_dense(
              x_t, y_t, adjoint_a, adjoint_b)
      else:
        x_t = constant_op.constant(x)
        y_t = constant_op.constant(y)
        ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_dense(x_t, y_t,
                                                                      adjoint_a,
                                                                      adjoint_b)
      delta_dense = _timer(sess, ops_fn, 1000)

  # Using sparse_tensor_dense_matmul.
  with session.Session("", config=config, graph=ops.Graph()) as sess:
    if not use_gpu:
      with ops.device("/cpu:0"):
        x_ind = constant_op.constant(np.vstack(np.where(x)).astype(np.int64).T)
        x_val = constant_op.constant(x[np.where(x)])
        x_shape = constant_op.constant(np.array(x.shape).astype(np.int64))
        y_t = constant_op.constant(y)
        ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_sparse(
            x_ind, x_val, x_shape, y_t, adjoint_a, adjoint_b)
    else:
      x_ind = constant_op.constant(np.vstack(np.where(x)).astype(np.int64).T)
      x_val = constant_op.constant(x[np.where(x)])
      x_shape = constant_op.constant(np.array(x.shape).astype(np.int64))
      y_t = constant_op.constant(y)
      ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_sparse(
          x_ind, x_val, x_shape, y_t, adjoint_a, adjoint_b)
    delta_sparse = _timer(sess, ops_fn, 1000)

  print("%g \t %d \t %s \t %d \t %d \t %g \t %g \t %g" %
        (1 - thresh, n, use_gpu, m, k, delta_dense, delta_sparse,
         delta_sparse / delta_dense))


def main(_):
  print("DenseDense MatMul (w/ Sparse Flag) vs. SparseTensorDense MatMul")
  print("Matrix sizes:")
  print("  A sparse [m, k] with % nonzero values between 1% and 80%")
  print("  B dense [k, n]")
  print("")
  print("% nnz \t n \t gpu \t m \t k \t dt(dense) \t dt(sparse) "
        "\t dt(sparse)/dt(dense)")

  for thresh in (0.99, 0.8, 0.5, 0.2):
    for n in (1, 10, 25):
      for use_gpu in (True, False):
        for m in (100, 1000):
          for k in (100, 1000):
            sparse_tensor_dense_vs_dense_matmul_benchmark(
                thresh, m, k, n, False, False, use_gpu=use_gpu)

  # Enable for large scale benchmarks, these ones take a long time to run.
  #
  # for use_gpu in (True, False):
  #   sparse_tensor_dense_vs_dense_matmul_benchmark(
  #       thresh=0.99, m=1000000, k=1000, n=100, adjoint_a=False,
  #       adjoint_b=False, use_gpu=use_gpu, skip_dense=True)


if __name__ == "__main__":
  if "--benchmarks" in sys.argv:
    sys.argv.remove("--benchmarks")
    app.run()
  else:
    test.main()
