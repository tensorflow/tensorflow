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
"""Tests for tensorflow.ops.math_ops.matrix_solve."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test as test_lib


def _AddTest(test, op_name, testcase_name, fn):
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


def _GenerateTestData(matrix_shape, num_rhs):
  batch_shape = matrix_shape[:-2]
  matrix_shape = matrix_shape[-2:]
  m = matrix_shape[-2]
  np.random.seed(1)
  matrix = np.random.uniform(
      low=-1.0, high=1.0,
      size=np.prod(matrix_shape)).reshape(matrix_shape).astype(np.float32)
  rhs = np.ones([m, num_rhs]).astype(np.float32)
  matrix = variables.Variable(
      np.tile(matrix, batch_shape + (1, 1)), trainable=False)
  rhs = variables.Variable(np.tile(rhs, batch_shape + (1, 1)), trainable=False)
  return matrix, rhs


def _SolveWithNumpy(matrix, rhs, l2_regularizer=0):
  if l2_regularizer == 0:
    np_ans, _, _, _ = np.linalg.lstsq(matrix, rhs)
    return np_ans
  else:
    rows = matrix.shape[-2]
    cols = matrix.shape[-1]
    if rows >= cols:
      preconditioner = l2_regularizer * np.identity(cols)
      gramian = np.dot(np.conj(matrix.T), matrix) + preconditioner
      rhs = np.dot(np.conj(matrix.T), rhs)
      return np.linalg.solve(gramian, rhs)
    else:
      preconditioner = l2_regularizer * np.identity(rows)
      gramian = np.dot(matrix, np.conj(matrix.T)) + preconditioner
      z = np.linalg.solve(gramian, rhs)
      return np.dot(np.conj(matrix.T), z)


class MatrixSolveLsOpTest(test_lib.TestCase):

  def _verifySolve(self,
                   x,
                   y,
                   dtype,
                   use_placeholder,
                   fast,
                   l2_regularizer,
                   batch_shape=()):
    if not fast and l2_regularizer != 0:
      # The slow path does not support regularization.
      return
    if use_placeholder and context.executing_eagerly():
      return
    maxdim = np.max(x.shape)
    if dtype == np.float32 or dtype == np.complex64:
      tol = maxdim * 5e-4
    else:
      tol = maxdim * 5e-7
      a = x.astype(dtype)
      b = y.astype(dtype)
      if dtype in [np.complex64, np.complex128]:
        a.imag = a.real
        b.imag = b.real
      # numpy.linalg.lstqr does not batching, so we just solve a single system
      # and replicate the solution. and residual norm.
      np_ans = _SolveWithNumpy(x, y, l2_regularizer=l2_regularizer)
      np_r = np.dot(np.conj(a.T), b - np.dot(a, np_ans))
      np_r_norm = np.sqrt(np.sum(np.conj(np_r) * np_r))
      if batch_shape != ():
        a = np.tile(a, batch_shape + (1, 1))
        b = np.tile(b, batch_shape + (1, 1))
        np_ans = np.tile(np_ans, batch_shape + (1, 1))
        np_r_norm = np.tile(np_r_norm, batch_shape)
      if use_placeholder:
        a_ph = array_ops.placeholder(dtypes.as_dtype(dtype))
        b_ph = array_ops.placeholder(dtypes.as_dtype(dtype))
        feed_dict = {a_ph: a, b_ph: b}
        tf_ans = linalg_ops.matrix_solve_ls(
            a_ph, b_ph, fast=fast, l2_regularizer=l2_regularizer)
      else:
        tf_ans = linalg_ops.matrix_solve_ls(
            a, b, fast=fast, l2_regularizer=l2_regularizer)
        feed_dict = None
        self.assertEqual(np_ans.shape, tf_ans.get_shape())
      if feed_dict:
        with self.session() as sess:
          tf_ans_val = sess.run(tf_ans, feed_dict=feed_dict)
      else:
        tf_ans_val = self.evaluate(tf_ans)
      self.assertEqual(np_ans.shape, tf_ans_val.shape)
      self.assertAllClose(np_ans, tf_ans_val, atol=2 * tol, rtol=2 * tol)

      if l2_regularizer == 0:
        # The least squares solution should satisfy A^H * (b - A*x) = 0.
        tf_r = b - math_ops.matmul(a, tf_ans)
        tf_r = math_ops.matmul(a, tf_r, adjoint_a=True)
        tf_r_norm = linalg_ops.norm(tf_r, ord="fro", axis=[-2, -1])
        if feed_dict:
          with self.session() as sess:
            tf_ans_val, tf_r_norm_val = sess.run([tf_ans, tf_r_norm],
                                                 feed_dict=feed_dict)
        else:
          tf_ans_val, tf_r_norm_val = self.evaluate([tf_ans, tf_r_norm])
        self.assertAllClose(np_r_norm, tf_r_norm_val, atol=tol, rtol=tol)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testWrongDimensions(self):
    # The matrix and right-hand sides should have the same number of rows.
    with self.session():
      matrix = constant_op.constant([[1., 0.], [0., 1.]])
      rhs = constant_op.constant([[1., 0.]])
      with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
        linalg_ops.matrix_solve_ls(matrix, rhs)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testEmpty(self):
    full = np.array([[1., 2.], [3., 4.], [5., 6.]])
    empty0 = np.empty([3, 0])
    empty1 = np.empty([0, 2])
    for fast in [True, False]:
      tf_ans = self.evaluate(
          linalg_ops.matrix_solve_ls(empty0, empty0, fast=fast))
      self.assertEqual(tf_ans.shape, (0, 0))
      tf_ans = self.evaluate(
          linalg_ops.matrix_solve_ls(empty0, full, fast=fast))
      self.assertEqual(tf_ans.shape, (0, 2))
      tf_ans = self.evaluate(
          linalg_ops.matrix_solve_ls(full, empty0, fast=fast))
      self.assertEqual(tf_ans.shape, (2, 0))
      tf_ans = self.evaluate(
          linalg_ops.matrix_solve_ls(empty1, empty1, fast=fast))
      self.assertEqual(tf_ans.shape, (2, 2))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testBatchResultSize(self):
    # 3x3x3 matrices, 3x3x1 right-hand sides.
    matrix = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.] * 3).reshape(3, 3, 3)  # pylint: disable=too-many-function-args
    rhs = np.array([1., 2., 3.] * 3).reshape(3, 3, 1)  # pylint: disable=too-many-function-args
    answer = linalg_ops.matrix_solve(matrix, rhs)
    ls_answer = linalg_ops.matrix_solve_ls(matrix, rhs)
    self.assertEqual(ls_answer.get_shape(), [3, 3, 1])
    self.assertEqual(answer.get_shape(), [3, 3, 1])


def _GetSmallMatrixSolveLsOpTests(dtype, use_placeholder, fast, l2_regularizer):

  def Square(self):
    # 2x2 matrices, 2x3 right-hand sides.
    matrix = np.array([[1., 2.], [3., 4.]])
    rhs = np.array([[1., 0., 1.], [0., 1., 1.]])
    for batch_shape in (), (2, 3):
      self._verifySolve(
          matrix,
          rhs,
          dtype,
          use_placeholder,
          fast,
          l2_regularizer,
          batch_shape=batch_shape)

  def Overdetermined(self):
    # 2x2 matrices, 2x3 right-hand sides.
    matrix = np.array([[1., 2.], [3., 4.], [5., 6.]])
    rhs = np.array([[1., 0., 1.], [0., 1., 1.], [1., 1., 0.]])
    for batch_shape in (), (2, 3):
      self._verifySolve(
          matrix,
          rhs,
          dtype,
          use_placeholder,
          fast,
          l2_regularizer,
          batch_shape=batch_shape)

  def Underdetermined(self):
    # 2x2 matrices, 2x3 right-hand sides.
    matrix = np.array([[1., 2., 3], [4., 5., 6.]])
    rhs = np.array([[1., 0., 1.], [0., 1., 1.]])
    for batch_shape in (), (2, 3):
      self._verifySolve(
          matrix,
          rhs,
          dtype,
          use_placeholder,
          fast,
          l2_regularizer,
          batch_shape=batch_shape)

  return (Square, Overdetermined, Underdetermined)


def _GetLargeMatrixSolveLsOpTests(dtype, use_placeholder, fast, l2_regularizer):

  def LargeBatchSquare(self):
    np.random.seed(1)
    num_rhs = 1
    matrix_shape = (127, 127)
    matrix = np.random.uniform(
        low=-1.0, high=1.0,
        size=np.prod(matrix_shape)).reshape(matrix_shape).astype(np.float32)
    rhs = np.ones([matrix_shape[0], num_rhs]).astype(np.float32)
    self._verifySolve(
        matrix,
        rhs,
        dtype,
        use_placeholder,
        fast,
        l2_regularizer,
        batch_shape=(16, 8))

  def LargeBatchOverdetermined(self):
    np.random.seed(1)
    num_rhs = 1
    matrix_shape = (127, 64)
    matrix = np.random.uniform(
        low=-1.0, high=1.0,
        size=np.prod(matrix_shape)).reshape(matrix_shape).astype(np.float32)
    rhs = np.ones([matrix_shape[0], num_rhs]).astype(np.float32)
    self._verifySolve(
        matrix,
        rhs,
        dtype,
        use_placeholder,
        fast,
        l2_regularizer,
        batch_shape=(16, 8))

  def LargeBatchUnderdetermined(self):
    np.random.seed(1)
    num_rhs = 1
    matrix_shape = (64, 127)
    matrix = np.random.uniform(
        low=-1.0, high=1.0,
        size=np.prod(matrix_shape)).reshape(matrix_shape).astype(np.float32)
    rhs = np.ones([matrix_shape[0], num_rhs]).astype(np.float32)
    self._verifySolve(
        matrix,
        rhs,
        dtype,
        use_placeholder,
        fast,
        l2_regularizer,
        batch_shape=(16, 8))

  return (LargeBatchSquare, LargeBatchOverdetermined, LargeBatchUnderdetermined)


class MatrixSolveLsBenchmark(test_lib.Benchmark):

  matrix_shapes = [
      (4, 4),
      (8, 4),
      (4, 8),
      (10, 10),
      (10, 8),
      (8, 10),
      (16, 16),
      (16, 10),
      (10, 16),
      (101, 101),
      (101, 31),
      (31, 101),
      (256, 256),
      (256, 200),
      (200, 256),
      (1001, 1001),
      (1001, 501),
      (501, 1001),
      (1024, 1024),
      (1024, 128),
      (128, 1024),
      (2048, 2048),
      (2048, 64),
      (64, 2048),
      (513, 4, 4),
      (513, 4, 2),
      (513, 2, 4),
      (513, 16, 16),
      (513, 16, 10),
      (513, 10, 16),
      (513, 256, 256),
      (513, 256, 128),
      (513, 128, 256),
  ]

  def benchmarkMatrixSolveLsOp(self):
    run_gpu_test = test_lib.is_gpu_available(True)
    regularizer = 1.0
    for matrix_shape in self.matrix_shapes:
      for num_rhs in 1, 2, matrix_shape[-1]:

        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device("/cpu:0"):
          matrix, rhs = _GenerateTestData(matrix_shape, num_rhs)
          x = linalg_ops.matrix_solve_ls(matrix, rhs, regularizer)
          self.evaluate(variables.global_variables_initializer())
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(x),
              min_iters=25,
              store_memory_usage=False,
              name=("matrix_solve_ls_cpu_shape_{matrix_shape}_num_rhs_{num_rhs}"
                   ).format(matrix_shape=matrix_shape, num_rhs=num_rhs))

        if run_gpu_test and (len(matrix_shape) < 3 or matrix_shape[0] < 513):
          with ops.Graph().as_default(), \
                session.Session(config=benchmark.benchmark_config()) as sess, \
                ops.device("/gpu:0"):
            matrix, rhs = _GenerateTestData(matrix_shape, num_rhs)
            x = linalg_ops.matrix_solve_ls(matrix, rhs, regularizer)
            self.evaluate(variables.global_variables_initializer())
            self.run_op_benchmark(
                sess,
                control_flow_ops.group(x),
                min_iters=25,
                store_memory_usage=False,
                name=("matrix_solve_ls_gpu_shape_{matrix_shape}_num_rhs_"
                      "{num_rhs}").format(
                          matrix_shape=matrix_shape, num_rhs=num_rhs))


if __name__ == "__main__":
  dtypes_to_test = [np.float32, np.float64, np.complex64, np.complex128]
  for dtype_ in dtypes_to_test:
    for use_placeholder_ in set([False, True]):
      for fast_ in [True, False]:
        l2_regularizers = [0] if dtype_ == np.complex128 else [0, 0.1]
        for l2_regularizer_ in l2_regularizers:
          for test_case in _GetSmallMatrixSolveLsOpTests(
              dtype_, use_placeholder_, fast_, l2_regularizer_):
            name = "%s_%s_placeholder_%s_fast_%s_regu_%s" % (test_case.__name__,
                                                             dtype_.__name__,
                                                             use_placeholder_,
                                                             fast_,
                                                             l2_regularizer_)
            _AddTest(MatrixSolveLsOpTest, "MatrixSolveLsOpTest", name,
                     test_case)
  for dtype_ in dtypes_to_test:
    for test_case in _GetLargeMatrixSolveLsOpTests(dtype_, False, True, 0.0):
      name = "%s_%s" % (test_case.__name__, dtype_.__name__)
      _AddTest(MatrixSolveLsOpTest, "MatrixSolveLsOpTest", name, test_case)

  test_lib.main()
