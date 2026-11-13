# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.linalg.eig under XLA (DEVICE_CPU_XLA_JIT)."""

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test


class XlaEigOpTest(xla_test.XLATestCase, parameterized.TestCase):

  def _get_output_dtype(self, input_dtype):
    if input_dtype in (
        dtypes.float32, dtypes.complex64, np.float32, np.complex64
    ):
      return dtypes.complex64, np.complex64
    elif input_dtype in (
        dtypes.float64, dtypes.complex128, np.float64, np.complex128
    ):
      return dtypes.complex128, np.complex128
    else:
      raise ValueError(f"Unsupported dtype: {input_dtype}")

  def _sort_eigenvalues(self, e):
    """Sorts eigenvalues by real then imaginary parts for deterministic

    comparison across backends.
    """
    orig_shape = e.shape
    e_flat = e.reshape(-1, orig_shape[-1])
    sorted_e = np.empty_like(e_flat)
    for b in range(e_flat.shape[0]):
      order = np.lexsort((np.imag(e_flat[b]), np.real(e_flat[b])))
      sorted_e[b] = e_flat[b, order]
    return sorted_e.reshape(orig_shape)

  def _verify_eig(self, x_np, compute_v=True):
    tf_in_dtype = dtypes.as_dtype(x_np.dtype)
    tf_out_dtype, np_out_dtype = self._get_output_dtype(tf_in_dtype)
    n = x_np.shape[-1]
    is_double = (tf_out_dtype == dtypes.complex128)
    atol = 1e-9 if is_double else 1e-4
    rtol = 1e-9 if is_double else 1e-4

    # 1. Non-XLA eager CPU baseline
    with ops.device("/device:CPU:0"):
      expected_e, expected_v = gen_linalg_ops.eig(
          constant_op.constant(x_np, dtype=tf_in_dtype),
          Tout=tf_out_dtype,
          compute_v=compute_v,
      )
      expected_e_np = expected_e.numpy()
      expected_v_np = expected_v.numpy()

    # 2. XLA session graph execution
    with self.session() as sess:
      x_ph = array_ops.placeholder(tf_in_dtype, shape=x_np.shape)
      with self.device_scope():
        out_e, out_v = gen_linalg_ops.eig(
            x_ph,
            Tout=tf_out_dtype,
            compute_v=compute_v,
        )
      res_e, res_v = sess.run([out_e, out_v], feed_dict={x_ph: x_np})

      # Eigenvalue validation
      self.assertEqual(res_e.shape, x_np.shape[:-1])
      sorted_res_e = self._sort_eigenvalues(res_e)
      sorted_exp_e = self._sort_eigenvalues(expected_e_np)
      self.assertAllClose(sorted_exp_e, sorted_res_e, atol=atol, rtol=rtol)

      if compute_v:
        self.assertEqual(res_v.shape, x_np.shape)
        if n > 0:
          # Mathematical identity: A * v[:, i] = lambda_i * v[:, i]
          # Matrix product: A @ v == v * e[..., None, :]
          av = np.matmul(x_np.astype(np_out_dtype), res_v)
          lv = res_v * res_e[..., np.newaxis, :]
          self.assertAllClose(av, lv, atol=atol, rtol=rtol)
      else:
        self.assertEqual(res_v.shape, (0,))

    # 3. tf.function(jit_compile=True)
    if context.executing_eagerly():
      @def_function.function(jit_compile=True)
      def jitted_eig(inp):
        return gen_linalg_ops.eig(
            inp,
            Tout=tf_out_dtype,
            compute_v=compute_v,
        )

      jit_e, jit_v = jitted_eig(constant_op.constant(x_np, dtype=tf_in_dtype))
      jit_e_np = jit_e.numpy()
      jit_v_np = jit_v.numpy()

      sorted_jit_e = self._sort_eigenvalues(jit_e_np)
      self.assertAllClose(sorted_exp_e, sorted_jit_e, atol=atol, rtol=rtol)

      if compute_v:
        self.assertEqual(jit_v_np.shape, x_np.shape)
        if n > 0:
          av = np.matmul(x_np.astype(np_out_dtype), jit_v_np)
          lv = jit_v_np * jit_e_np[..., np.newaxis, :]
          self.assertAllClose(av, lv, atol=atol, rtol=rtol)
      else:
        self.assertEqual(jit_v_np.shape, (0,))

  # -------------------------------------------------------------------------
  # All 4 supported dtypes
  # -------------------------------------------------------------------------

  @parameterized.parameters(
      {"dtype": np.float32},
      {"dtype": np.float64},
      {"dtype": np.complex64},
      {"dtype": np.complex128},
  )
  def testDtypes2x2RealAndComplex(self, dtype):
    if np.issubdtype(dtype, np.complexfloating):
      x = np.array(
          [[1.0 + 2.0j, 2.0 - 1.0j], [0.5 + 1.0j, 3.0 + 0.0j]], dtype=dtype
      )
    else:
      x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)

    self._verify_eig(x, compute_v=True)
    self._verify_eig(x, compute_v=False)

  # -------------------------------------------------------------------------
  # High-level tf.linalg.eig and tf.linalg.eigvals wrappers
  # -------------------------------------------------------------------------

  def testHighLevelEigAndEigvals(self):
    x_np = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)

    if context.executing_eagerly():
      @def_function.function(jit_compile=True)
      def jitted_eig_high_level(inp):
        return linalg_ops.eig(inp)

      @def_function.function(jit_compile=True)
      def jitted_eigvals_high_level(inp):
        return linalg_ops.eigvals(inp)

      e_val, v_val = jitted_eig_high_level(constant_op.constant(x_np))
      e_only = jitted_eigvals_high_level(constant_op.constant(x_np))

      self.assertAllClose(
          self._sort_eigenvalues(e_val.numpy()),
          self._sort_eigenvalues(e_only.numpy()),
      )
      self.assertEqual(v_val.shape, (2, 2))

  # -------------------------------------------------------------------------
  # Complex conjugate eigenvalues from real non-symmetric matrix
  # -------------------------------------------------------------------------

  def testComplexConjugateEigenvalues(self):
    # 90-degree 2D rotation matrix: eigenvalues are +i, -i
    x = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    self._verify_eig(x, compute_v=True)
    self._verify_eig(x, compute_v=False)

    x64 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    self._verify_eig(x64, compute_v=True)
    self._verify_eig(x64, compute_v=False)

  # -------------------------------------------------------------------------
  # Symmetric and Hermitian matrices
  # -------------------------------------------------------------------------

  def testSymmetricAndHermitian(self):
    # Real symmetric
    sym = np.array([[4.0, 1.0, -2.0],
                    [1.0, 2.0, 0.0],
                    [-2.0, 0.0, 3.0]], dtype=np.float32)
    self._verify_eig(sym, compute_v=True)
    self._verify_eig(sym, compute_v=False)

    # Complex Hermitian
    herm = np.array(
        [[2.0 + 0.0j, 1.0 - 1.0j],
         [1.0 + 1.0j, 3.0 + 0.0j]],
        dtype=np.complex64,
    )
    self._verify_eig(herm, compute_v=True)
    self._verify_eig(herm, compute_v=False)

  # -------------------------------------------------------------------------
  # Batched and Higher Rank inputs
  # -------------------------------------------------------------------------

  def testBatched2Dand3D(self):
    np.random.seed(42)
    # Shape: (4, 3, 3)
    batch_3d = np.random.randn(4, 3, 3).astype(np.float32)
    self._verify_eig(batch_3d, compute_v=True)
    self._verify_eig(batch_3d, compute_v=False)

    # Multi-batch shape: (2, 3, 4, 4)
    batch_4d = np.random.randn(2, 3, 4, 4).astype(np.float64)
    self._verify_eig(batch_4d, compute_v=True)
    self._verify_eig(batch_4d, compute_v=False)

    # Complex batched: (3, 2, 2)
    cbatch = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(
        np.complex64
    )
    self._verify_eig(cbatch, compute_v=True)
    self._verify_eig(cbatch, compute_v=False)

  # -------------------------------------------------------------------------
  # Edge cases: 0x0 empty matrix, 1x1 scalar, Identity, Repeated eigenvalues
  # -------------------------------------------------------------------------

  def testEmptyMatrix(self):
    x_empty = np.zeros((0, 0), dtype=np.float32)
    self._verify_eig(x_empty, compute_v=True)
    self._verify_eig(x_empty, compute_v=False)

    x_empty_batch = np.zeros((3, 0, 0), dtype=np.float32)
    self._verify_eig(x_empty_batch, compute_v=True)
    self._verify_eig(x_empty_batch, compute_v=False)

  def test1x1Matrix(self):
    x_1x1 = np.array([[5.5]], dtype=np.float32)
    self._verify_eig(x_1x1, compute_v=True)
    self._verify_eig(x_1x1, compute_v=False)

    c_1x1 = np.array([[3.0 - 4.0j]], dtype=np.complex128)
    self._verify_eig(c_1x1, compute_v=True)
    self._verify_eig(c_1x1, compute_v=False)

  def testIdentityAndRepeatedEigenvalues(self):
    eye = np.eye(4, dtype=np.float32)
    self._verify_eig(eye, compute_v=True)
    self._verify_eig(eye, compute_v=False)

    # Defective matrix (Jordan block with repeated eigenvalue 2.0)
    jordan = np.array([[2.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    self._verify_eig(jordan, compute_v=True)
    self._verify_eig(jordan, compute_v=False)

  # -------------------------------------------------------------------------
  # Error handling: invalid rank or non-square matrix
  # -------------------------------------------------------------------------

  def testRankLessThanTwoRaisesError(self):
    with self.session() as sess:
      x_ph = array_ops.placeholder(dtypes.float32)
      with self.device_scope():
        out_e, out_v = gen_linalg_ops.eig(
            x_ph,
            Tout=dtypes.complex64,
            compute_v=True,
        )
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run([out_e, out_v], feed_dict={x_ph: np.array([1.0, 2.0, 3.0])})

  def testNonSquareMatrixRaisesError(self):
    with self.session() as sess:
      x_ph = array_ops.placeholder(dtypes.float32)
      with self.device_scope():
        out_e, out_v = gen_linalg_ops.eig(
            x_ph,
            Tout=dtypes.complex64,
            compute_v=True,
        )
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(
            [out_e, out_v],
            feed_dict={x_ph: np.ones((3, 4), dtype=np.float32)},
        )


if __name__ == "__main__":
  test.main()
