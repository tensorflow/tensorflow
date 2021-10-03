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

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


# Different gradient implementations for benchmark purposes
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

  def _verifyCholeskyBase(self, x, chol, verification):
    chol_np, verification_np = self.evaluate([chol, verification])
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
    chol = linalg_ops.cholesky(x)
    verification = test_util.matmul_without_tf32(chol, chol, adjoint_b=True)
    self._verifyCholeskyBase(x, chol, verification)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testBasic(self):
    data = np.array([[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]])
    for dtype in (np.float32, np.float64):
      with self.subTest(dtype=dtype):
        self._verifyCholesky(data.astype(dtype))
    for dtype in (np.complex64, np.complex128):
      with self.subTest(dtype=dtype):
        complex_data = np.tril(1j * data, -1).astype(dtype)
        complex_data += np.triu(-1j * data, 1).astype(dtype)
        complex_data += data
        self._verifyCholesky(complex_data)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testBatch(self):
    simple_array = np.array([[[1., 0.], [0., 5.]]])  # shape (1, 2, 2)
    self._verifyCholesky(simple_array)
    self._verifyCholesky(np.vstack((simple_array, simple_array)))
    odd_sized_array = np.array([[[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]]])
    self._verifyCholesky(np.vstack((odd_sized_array, odd_sized_array)))

    # Generate random positive-definite matrices.
    matrices = np.random.rand(10, 5, 5)
    for i in xrange(10):
      with self.subTest(i=i):
        matrices[i] = np.dot(matrices[i].T, matrices[i])
    self._verifyCholesky(matrices)

    # Generate random complex valued positive-definite matrices.
    matrices = np.random.rand(10, 5, 5) + 1j * np.random.rand(10, 5, 5)
    for i in xrange(10):
      with self.subTest(i=i):
        matrices[i] = np.dot(matrices[i].T.conj(), matrices[i])
    self._verifyCholesky(matrices)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testNonSquareMatrix(self):
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      linalg_ops.cholesky(np.array([[1., 2., 3.], [3., 4., 5.]]))
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      linalg_ops.cholesky(
          np.array([[[1., 2., 3.], [3., 4., 5.]], [[1., 2., 3.], [3., 4., 5.]]
                   ]))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testWrongDimensions(self):
    tensor3 = constant_op.constant([1., 2.])
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      linalg_ops.cholesky(tensor3)
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      linalg_ops.cholesky(tensor3)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testNotInvertibleCpu(self):
    # Non-invertible inputs result in lower-triangular NaNs.
    x = constant_op.constant([[1., -1., 0.], [-1., 1., -1.], [0., -1., 1.]])
    chol = linalg_ops.cholesky(x)
    # Extract the lower-triangular elements.
    lower_mask = array_ops.matrix_band_part(
        constant_op.constant(True, shape=x.shape), -1, 0)
    chol_lower = array_ops.boolean_mask(chol, lower_mask)
    # Assert all NaN.
    all_nan = self.evaluate(
        math_ops.reduce_all(math_ops.reduce_all(math_ops.is_nan(chol_lower))))
    self.assertTrue(all_nan)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testEmpty(self):
    self._verifyCholesky(np.empty([0, 2, 2]))
    self._verifyCholesky(np.empty([2, 0, 0]))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testConcurrentExecutesWithoutError(self):
    seed = [42, 24]
    matrix_shape = [5, 5]
    matrix1 = stateless_random_ops.stateless_random_normal(matrix_shape, seed)
    matrix2 = stateless_random_ops.stateless_random_normal(matrix_shape, seed)
    matrix1 = math_ops.matmul(matrix1, matrix1, adjoint_a=True)
    matrix2 = math_ops.matmul(matrix2, matrix2, adjoint_a=True)
    c1 = linalg_ops.cholesky(matrix1)
    c2 = linalg_ops.cholesky(matrix2)
    c1_val, c2_val = self.evaluate([c1, c2])
    self.assertAllClose(c1_val, c2_val)


class CholeskyGradTest(test.TestCase):
  _backprop_block_size = 16

  def getShapes(self, shapeList):
    return ((elem, int(np.floor(1.2 * elem))) for elem in shapeList)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testSmallMatrices(self):
    np.random.seed(0)
    shapes = self.getShapes([1, 2, 10])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.float32, dtypes_lib.float64))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testSmallMatricesComplex(self):
    np.random.seed(0)
    shapes = self.getShapes([1, 2, 10])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.complex64, dtypes_lib.complex128))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testOneBlockMatrices(self):
    np.random.seed(0)
    shapes = self.getShapes([self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes,
        dtypes=(dtypes_lib.float32, dtypes_lib.float64),
        scalar_test=True)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testTwoBlockMatrixFloat(self):
    np.random.seed(0)
    shapes = self.getShapes([2 * self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.float32,), scalar_test=True)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testTwoBlockMatrixDouble(self):
    np.random.seed(0)
    shapes = self.getShapes([2 * self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.float64,), scalar_test=True)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testTwoBlockMatrixComplexFloat(self):
    np.random.seed(0)
    shapes = self.getShapes([2 * self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.complex64,), scalar_test=True)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testTwoBlockMatrixComplexDouble(self):
    np.random.seed(0)
    shapes = self.getShapes([2 * self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes, dtypes=(dtypes_lib.complex128,), scalar_test=True)

  def _runOneTest(self, shape, dtype, batch, scalar_test):
    if dtype == dtypes_lib.float64:
      tol = 1e-5
    elif dtype == dtypes_lib.complex128:
      tol = 5e-5
    else:
      tol = 5e-3
    epsilon = np.finfo(dtype.as_numpy_dtype).eps
    delta = epsilon**(1.0 / 3.0)

    def RandomInput():
      a = np.random.randn(shape[0], shape[1]).astype(dtype.as_numpy_dtype)
      if dtype.is_complex:
        a += 1j * np.random.randn(shape[0], shape[1]).astype(
            dtype.as_numpy_dtype)
      return a

    def Compute(x):
      # Turn the random matrix x into a Hermitian matrix by
      # computing the quadratic form x * x^H.
      a = test_util.matmul_without_tf32(
          x, math_ops.conj(array_ops.matrix_transpose(x))) / shape[0]
      if batch:
        a = array_ops.tile(array_ops.expand_dims(a, 0), [2, 1, 1])
      # Finally take the cholesky decomposition of the Hermitian matrix.
      c = linalg_ops.cholesky(a)
      if scalar_test:
        # Reduce to a single scalar output to speed up test.
        c = math_ops.reduce_mean(c)
      return c

    theoretical, numerical = gradient_checker_v2.compute_gradient(
        Compute, [RandomInput()], delta=delta)
    self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

  def runFiniteDifferences(self,
                           shapes,
                           dtypes=(dtypes_lib.float32, dtypes_lib.float64,
                                   dtypes_lib.complex64, dtypes_lib.complex128),
                           scalar_test=False):
    for shape_ in shapes:
      for dtype_ in dtypes:
        for batch_ in False, True:
          self._runOneTest(shape_, dtype_, batch_, scalar_test)


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
        self.evaluate(variables.global_variables_initializer())
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
          self.evaluate(variables.global_variables_initializer())
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
          self.evaluate(variables.global_variables_initializer())
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


if __name__ == "__main__":
  test.main()
