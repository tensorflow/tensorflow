# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.linalg_ops.eig."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


def _AddTest(test_class, op_name, testcase_name, fn):
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test_class, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test_class, test_name, fn)


class EigTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testWrongDimensions(self):
    # The input to self_adjoint_eig should be a tensor of
    # at least rank 2.
    scalar = constant_op.constant(1.)
    with self.assertRaises(ValueError):
      linalg_ops.eig(scalar)
    vector = constant_op.constant([1., 2.])
    with self.assertRaises(ValueError):
      linalg_ops.eig(vector)

  @test_util.run_deprecated_v1
  def testConcurrentExecutesWithoutError(self):
    all_ops = []
    with self.session(use_gpu=True) as sess:
      for compute_v_ in True, False:
        matrix1 = random_ops.random_normal([5, 5], seed=42)
        matrix2 = random_ops.random_normal([5, 5], seed=42)
        if compute_v_:
          e1, v1 = linalg_ops.eig(matrix1)
          e2, v2 = linalg_ops.eig(matrix2)
          all_ops += [e1, v1, e2, v2]
        else:
          e1 = linalg_ops.eigvals(matrix1)
          e2 = linalg_ops.eigvals(matrix2)
          all_ops += [e1, e2]
      val = self.evaluate(all_ops)
      self.assertAllEqual(val[0], val[2])
      # The algorithm is slightly different for compute_v being True and False,
      # so require approximate equality only here.
      self.assertAllClose(val[2], val[4])
      self.assertAllEqual(val[4], val[5])
      self.assertAllEqual(val[1], val[3])

  def testMatrixThatFailsWhenFlushingDenormsToZero(self):
    # Test a 32x32 matrix which is known to fail if denorm floats are flushed to
    # zero.
    matrix = np.genfromtxt(
        test.test_src_dir_path(
            "python/kernel_tests/testdata/"
            "self_adjoint_eig_fail_if_denorms_flushed.txt")).astype(np.float32)
    self.assertEqual(matrix.shape, (32, 32))
    matrix_tensor = constant_op.constant(matrix)
    with self.session(use_gpu=True) as sess:
      (e, v) = self.evaluate(linalg_ops.self_adjoint_eig(matrix_tensor))
      self.assertEqual(e.size, 32)
      self.assertAllClose(
          np.matmul(v, v.transpose()), np.eye(32, dtype=np.float32), atol=2e-3)
      self.assertAllClose(matrix,
                          np.matmul(np.matmul(v, np.diag(e)), v.transpose()))


def SortEigenValues(e):
  perm = np.argsort(e.real + e.imag, -1)
  return np.take(e, perm, -1)


def SortEigenDecomposition(e, v):
  if v.ndim < 2:
    return e, v
  else:
    perm = np.argsort(e.real + e.imag, -1)
    return np.take(e, perm, -1), np.take(v, perm, -1)


def EquilibrateEigenVectorPhases(x, y):
  """Equilibrate the phase of the Eigenvectors in the columns of `x` and `y`.

  Eigenvectors are only unique up to an arbitrary phase. This function rotates x
  such that it matches y. Precondition: The columns of x and y differ by a
  multiplicative complex phase factor only.

  Args:
    x: `np.ndarray` with Eigenvectors
    y: `np.ndarray` with Eigenvectors

  Returns:
    `np.ndarray` containing an equilibrated version of x.
  """
  phases = np.sum(np.conj(x) * y, -2, keepdims=True)
  phases /= np.abs(phases)
  return phases * x


def _GetEigTest(dtype_, shape_, compute_v_):

  def CompareEigenVectors(self, x, y, tol):
    x = EquilibrateEigenVectorPhases(x, y)
    self.assertAllClose(x, y, atol=tol)

  def CompareEigenDecompositions(self, x_e, x_v, y_e, y_v, tol):
    num_batches = int(np.prod(x_e.shape[:-1]))
    n = x_e.shape[-1]
    x_e = np.reshape(x_e, [num_batches] + [n])
    x_v = np.reshape(x_v, [num_batches] + [n, n])
    y_e = np.reshape(y_e, [num_batches] + [n])
    y_v = np.reshape(y_v, [num_batches] + [n, n])
    for i in range(num_batches):
      x_ei, x_vi = SortEigenDecomposition(x_e[i, :], x_v[i, :, :])
      y_ei, y_vi = SortEigenDecomposition(y_e[i, :], y_v[i, :, :])
      self.assertAllClose(x_ei, y_ei, atol=tol, rtol=tol)
      CompareEigenVectors(self, x_vi, y_vi, tol)

  def Test(self):
    np.random.seed(1)
    n = shape_[-1]
    batch_shape = shape_[:-2]
    np_dtype = dtype_.as_numpy_dtype
    # most of matrices are diagonalizable # TODO
    a = np.random.uniform(
        low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(np_dtype)
    if dtype_.is_complex:
      a += 1j * np.random.uniform(
          low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(np_dtype)
    a = np.tile(a, batch_shape + (1, 1))
    if dtype_ in (dtypes_lib.float32, dtypes_lib.complex64):
      atol = 1e-4
    else:
      atol = 1e-12
    np_e, np_v = np.linalg.eig(a)
    with self.session(use_gpu=True):
      if compute_v_:
        tf_e, tf_v = linalg_ops.eig(constant_op.constant(a))

        # Check that V*diag(E)*V^(-1) is close to A.
        a_ev = math_ops.matmul(
            math_ops.matmul(tf_v, array_ops.matrix_diag(tf_e)),
            linalg_ops.matrix_inverse(tf_v))
        self.assertAllClose(self.evaluate(a_ev), a, atol=atol)

        # Compare to numpy.linalg.eig.
        CompareEigenDecompositions(self, np_e, np_v, self.evaluate(tf_e),
                                   self.evaluate(tf_v), atol)
      else:
        tf_e = linalg_ops.eigvals(constant_op.constant(a))
        self.assertAllClose(
            SortEigenValues(np_e),
            SortEigenValues(self.evaluate(tf_e)),
            atol=atol)

  return Test


if __name__ == "__main__":
  dtypes_to_test = [dtypes_lib.float32, dtypes_lib.float64]
  if not test.is_built_with_rocm():
    # ROCm does not support BLAS operations for complex types
    dtypes_to_test += [dtypes_lib.complex64, dtypes_lib.complex128]
  for compute_v in True, False:
    for dtype in dtypes_to_test:
      for size in 1, 2, 5, 10:
        for batch_dims in [(), (3,)] + [(3, 2)] * (max(size, size) < 10):
          shape = batch_dims + (size, size)
          name = "%s_%s_%s" % (dtype.name, "_".join(map(str, shape)), compute_v)
          _AddTest(EigTest, "Eig", name, _GetEigTest(dtype, shape, compute_v))
          # No gradient yet
  test.main()
