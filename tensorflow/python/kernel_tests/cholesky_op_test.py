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
"""Tests for tensorflow.ops.tf.Cholesky."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


# Different gradient implementations for benchmark purposes
def SpecializedGrad(l, grad):
  return gen_linalg_ops.cholesky_grad(l, grad)


def _GradWithInverseL(l, l_inverse, grad):
  middle = math_ops.matmul(l, grad, adjoint_a=True)
  middle = array_ops.matrix_set_diag(middle,
                                     0.5 * array_ops.matrix_diag_part(middle))
  middle = array_ops.matrix_band_part(middle, -1, 0)
  grad_a = math_ops.matmul(
      math_ops.matmul(l_inverse, middle, adjoint_a=True), l_inverse)
  grad_a += math_ops.conj(array_ops.matrix_transpose(grad_a))
  return grad_a * 0.5


def TriAngSolveCompositeGrad(l, grad):
  # Gradient is l^{-H} @ ((l^{H} @ grad) * (tril(ones)-1/2*eye)) @ l^{-1}

  # Compute ((l^{H} @ grad) * (tril(ones)-1/2*eye)) = middle
  middle = math_ops.matmul(l, grad, adjoint_a=True)
  middle = array_ops.matrix_set_diag(middle,
                                     0.5 * array_ops.matrix_diag_part(middle))
  middle = array_ops.matrix_band_part(middle, -1, 0)

  # Compute l^{-H} @ middle = z
  l_inverse_middle = linalg_ops.matrix_triangular_solve(l, middle, adjoint=True)

  # We need to compute z @ l^{-1}. With matrix_triangular_solve we
  # actually compute l^{-H} @ z^{H} = grad. Since we later add grad^{H}
  # we can ommit the conjugate transpose here.
  z_h = math_ops.conj(array_ops.matrix_transpose(l_inverse_middle))
  grad_a = linalg_ops.matrix_triangular_solve(l, z_h, adjoint=True)
  grad_a += linalg.adjoint(grad_a)
  return grad_a * 0.5


def MatrixInverseCompositeGrad(l, grad):
  l_inverse = linalg_ops.matrix_inverse(l)
  return _GradWithInverseL(l, l_inverse, grad)


def TriAngInvCompositeGrad(l, grad):
  num_rows = array_ops.shape(l)[-1]
  batch_shape = array_ops.shape(l)[:-2]
  l_inverse = linalg_ops.matrix_triangular_solve(l,
                                                 linalg_ops.eye(
                                                     num_rows,
                                                     batch_shape=batch_shape,
                                                     dtype=l.dtype))
  return _GradWithInverseL(l, l_inverse, grad)


class CholeskyOpTest(test.TestCase):

  def _verifyCholeskyBase(self, sess, x, chol, verification):
    chol_np, verification_np = sess.run([chol, verification])
    self.assertAllClose(x, verification_np)
    self.assertShapeEqual(x, chol)
    # Check that the cholesky is lower triangular, and has positive diagonal
    # elements.
    if chol_np.shape[-1] > 0:
      chol_reshaped = np.reshape(chol_np, (-1, chol_np.shape[-2],
                                           chol_np.shape[-1]))
      for chol_matrix in chol_reshaped:
        self.assertAllClose(chol_matrix, np.tril(chol_matrix))
        self.assertTrue((np.diag(chol_matrix) > 0.0).all())

  def _verifyCholesky(self, x):
    # Verify that LL^T == x.
    with self.cached_session(use_gpu=True) as sess:
      chol = linalg_ops.cholesky(x)
      verification = math_ops.matmul(chol, chol, adjoint_b=True)
      self._verifyCholeskyBase(sess, x, chol, verification)

  def testBasic(self):
    data = np.array([[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]])
    for dtype in (np.float32, np.float64):
      self._verifyCholesky(data.astype(dtype))
    for dtype in (np.complex64, np.complex128):
      complex_data = np.tril(1j * data, -1).astype(dtype)
      complex_data += np.triu(-1j * data, 1).astype(dtype)
      complex_data += data
      self._verifyCholesky(complex_data)

  def testBatch(self):
    simple_array = np.array([[[1., 0.], [0., 5.]]])  # shape (1, 2, 2)
    self._verifyCholesky(simple_array)
    self._verifyCholesky(np.vstack((simple_array, simple_array)))
    odd_sized_array = np.array([[[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]]])
    self._verifyCholesky(np.vstack((odd_sized_array, odd_sized_array)))

    # Generate random positive-definite matrices.
    matrices = np.random.rand(10, 5, 5)
    for i in xrange(10):
      matrices[i] = np.dot(matrices[i].T, matrices[i])
    self._verifyCholesky(matrices)

    # Generate random complex valued positive-definite matrices.
    matrices = np.random.rand(10, 5, 5) + 1j * np.random.rand(10, 5, 5)
    for i in xrange(10):
      matrices[i] = np.dot(matrices[i].T.conj(), matrices[i])
    self._verifyCholesky(matrices)

  def testNonSquareMatrix(self):
    with self.assertRaises(ValueError):
      linalg_ops.cholesky(np.array([[1., 2., 3.], [3., 4., 5.]]))
    with self.assertRaises(ValueError):
      linalg_ops.cholesky(
          np.array([[[1., 2., 3.], [3., 4., 5.]], [[1., 2., 3.], [3., 4., 5.]]
                   ]))

  def testWrongDimensions(self):
    tensor3 = constant_op.constant([1., 2.])
    with self.assertRaises(ValueError):
      linalg_ops.cholesky(tensor3)
    with self.assertRaises(ValueError):
      linalg_ops.cholesky(tensor3)

  def testNotInvertibleCPU(self):
    # The input should be invertible.
    with self.session(use_gpu=True):
      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          "Cholesky decomposition was not successful. The"
          " input might not be valid."):
        # All rows of the matrix below add to zero
        self._verifyCholesky(
            np.array([[1., -1., 0.], [-1., 1., -1.], [0., -1., 1.]]))

  def testEmpty(self):
    self._verifyCholesky(np.empty([0, 2, 2]))
    self._verifyCholesky(np.empty([2, 0, 0]))

  def testConcurrentExecutesWithoutError(self):
    with self.session(use_gpu=True) as sess:
      matrix1 = random_ops.random_normal([5, 5], seed=42)
      matrix2 = random_ops.random_normal([5, 5], seed=42)
      matrix1 = math_ops.matmul(matrix1, matrix1, adjoint_a=True)
      matrix2 = math_ops.matmul(matrix2, matrix2, adjoint_a=True)
      c1 = linalg_ops.cholesky(matrix1)
      c2 = linalg_ops.cholesky(matrix2)
      c1_val, c2_val = sess.run([c1, c2])
      self.assertAllEqual(c1_val, c2_val)


class CholeskyGradTest(test.TestCase):
  _backprop_block_size = 32

  def getShapes(self, shapeList):
    return ((elem, int(np.floor(1.2 * elem))) for elem in shapeList)

  def testSmallMatrices(self):
    np.random.seed(0)
    shapes = self.getShapes([1, 2, 10])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.float32, dtypes_lib.float64))

  def testSmallMatricesComplex(self):
    np.random.seed(0)
    shapes = self.getShapes([1, 2, 10])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.complex64, dtypes_lib.complex128))

  def testOneBlockMatrices(self):
    np.random.seed(0)
    shapes = self.getShapes([self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes,
        dtypes=(dtypes_lib.float32, dtypes_lib.float64),
        scalarTest=True)

  def testTwoBlockMatrixFloat(self):
    np.random.seed(0)
    shapes = self.getShapes([2 * self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.float32,), scalarTest=True)

  def testTwoBlockMatrixDouble(self):
    np.random.seed(0)
    shapes = self.getShapes([2 * self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.float64,), scalarTest=True)

  def testTwoBlockMatrixComplexFloat(self):
    np.random.seed(0)
    shapes = self.getShapes([2 * self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.complex64,), scalarTest=True)

  def testTwoBlockMatrixComplexDouble(self):
    np.random.seed(0)
    shapes = self.getShapes([2 * self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.complex128,), scalarTest=True)

  def testAgainstSpecialized(self):
    np.random.seed(0)
    data = np.random.randn(33, 33).astype(np.float32)
    data = np.matmul(data, data.T)
    grad_data = np.random.randn(*data.shape).astype(np.float32)

    with ops.Graph().as_default(), self.session(use_gpu=False) as s:
      x = constant_op.constant(data, dtypes_lib.float32)
      chol = linalg_ops.cholesky(x)
      composite_grad = gradients_impl.gradients(chol, x, grad_data)[0]
      specialized_grad = SpecializedGrad(chol, grad_data)
      reference, actual = s.run([specialized_grad, composite_grad])
    self.assertAllClose(reference, actual)

  def runFiniteDifferences(self,
                           shapes,
                           dtypes=(dtypes_lib.float32, dtypes_lib.float64,
                                   dtypes_lib.complex64, dtypes_lib.complex128),
                           scalarTest=False):
    with self.session(use_gpu=True):
      for shape in shapes:
        for batch in False, True:
          for dtype in dtypes:
            if not scalarTest:
              data = np.random.randn(shape[0], shape[1])
              if dtype.is_complex:
                data = data.astype(np.complex64)
                data += 1j * np.random.randn(shape[0], shape[1])
              x = constant_op.constant(data, dtype)
              tensor = math_ops.matmul(
                  x, math_ops.conj(array_ops.transpose(x))) / shape[0]
            else:
              # This is designed to be a faster test for larger matrices.
              data = np.random.randn()
              if dtype.is_complex:
                data = np.complex64(data)
                data += 1j * np.random.randn()
              x = constant_op.constant(data, dtype)
              R = constant_op.constant(
                  np.random.randn(shape[0], shape[1]), dtype)
              e = math_ops.multiply(R, x)
              tensor = math_ops.matmul(
                  e, math_ops.conj(array_ops.transpose(e))) / shape[0]

            # Inner-most matrices in tensor are positive definite.
            if batch:
              tensor = array_ops.tile(
                  array_ops.expand_dims(tensor, 0), [4, 1, 1])
            y = linalg_ops.cholesky(tensor)
            if scalarTest:
              y = math_ops.reduce_mean(y)
            error = gradient_checker.compute_gradient_error(
                x, x._shape_as_list(), y, y._shape_as_list())
            tf_logging.info("error = %f", error)
            if dtype == dtypes_lib.float64:
              self.assertLess(error, 1e-5)
            elif dtype == dtypes_lib.complex128:
              self.assertLess(error, 5e-5)
            else:
              self.assertLess(error, 5e-3)


class CholeskyBenchmark(test.Benchmark):

  shapes = [
      (4, 4),
      (10, 10),
      (16, 16),
      (101, 101),
      (256, 256),
      (1000, 1000),
      (1024, 1024),
      (2048, 2048),
      (513, 2, 2),
      (513, 8, 8),
      (513, 256, 256),
      (4, 513, 2, 2),
  ]

  def _GenerateMatrix(self, shape):
    batch_shape = shape[:-2]
    shape = shape[-2:]
    assert shape[0] == shape[1]
    n = shape[0]
    matrix = np.ones(shape).astype(np.float32) / (
        2.0 * n) + np.diag(np.ones(n).astype(np.float32))
    return np.tile(matrix, batch_shape + (1, 1))

  def benchmarkCholeskyOp(self):
    for shape in self.shapes:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device("/cpu:0"):
        matrix = variables.Variable(self._GenerateMatrix(shape))
        l = linalg_ops.cholesky(matrix)
        variables.global_variables_initializer().run()
        self.run_op_benchmark(
            sess,
            control_flow_ops.group(
                l,),
            min_iters=25,
            name="cholesky_cpu_{shape}".format(shape=shape))

      if test.is_gpu_available(True):
        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device("/device:GPU:0"):
          matrix = variables.Variable(self._GenerateMatrix(shape))
          l = linalg_ops.cholesky(matrix)
          variables.global_variables_initializer().run()
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(
                  l,),
              min_iters=25,
              name="cholesky_gpu_{shape}".format(shape=shape))

  def benchmarkGradVariants(self):

    def _BenchmarkGrad(grad_fn, name, device):
      for shape in self.shapes:
        matrix = self._GenerateMatrix(shape)
        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device(device):
          l = variables.Variable(np.linalg.cholesky(matrix))
          grad_matrix = variables.Variable(
              np.random.randn(*matrix.shape).astype(np.float32))
          grad = grad_fn(l, grad_matrix)
          variables.global_variables_initializer().run()
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(
                  grad,),
              min_iters=25,
              name="{name}_{dev}_{shape}".format(
                  name=name, dev=grad.device, shape=shape))

    if test.is_gpu_available(True):
      _BenchmarkGrad(MatrixInverseCompositeGrad, "composite_matrix_inverse",
                     "/device:GPU:0")
      _BenchmarkGrad(TriAngInvCompositeGrad, "composite_tri_ang_inverse",
                     "/device:GPU:0")
      _BenchmarkGrad(TriAngSolveCompositeGrad, "composite_triangular_solve",
                     "/device:GPU:0")

    _BenchmarkGrad(MatrixInverseCompositeGrad, "composite_matrix_inverse",
                   "/cpu:0")
    _BenchmarkGrad(TriAngInvCompositeGrad, "composite_tri_ang_inverse",
                   "/cpu:0")
    _BenchmarkGrad(TriAngSolveCompositeGrad, "composite_triangular_solve",
                   "/cpu:0")
    _BenchmarkGrad(SpecializedGrad, "specialized", "/cpu:0")


if __name__ == "__main__":
  test.main()
