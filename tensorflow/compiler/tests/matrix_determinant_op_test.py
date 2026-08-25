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
"""Tests for XLA implementations of matrix determinant ops."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import googletest


class DeterminantOpTest(xla_test.XLATestCase):

  def _verifyDeterminant(self, x, np_type):
    y = x.astype(np_type)
    if y.shape[-1] == 0 and y.shape[-2] == 0:
      np_det = np.ones(y.shape[:-2], dtype=np_type)
      np_sign = np.ones(y.shape[:-2], dtype=np_type)
      np_log_abs = np.zeros(y.shape[:-2], dtype=np_type)
    else:
      np_det = np.array(np.linalg.det(y)).astype(np_type)
      np_sign, np_log_abs = np.linalg.slogdet(y)
      np_sign = np.array(np_sign).astype(np_type)
      np_log_abs = np.array(np_log_abs).astype(np_type)

    with self.session() as sess:
      p = array_ops.placeholder(dtypes.as_dtype(y.dtype), y.shape, name="x")
      with self.test_scope():
        det = linalg_ops.matrix_determinant(p)
        sign, log_abs = gen_linalg_ops.log_matrix_determinant(p)
      det_out, sign_out, log_abs_out = sess.run(
          [det, sign, log_abs], feed_dict={p: y})

    self.assertAllClose(np_det, det_out, rtol=1e-3, atol=1e-3)
    self.assertShapeEqual(np_det, det)
    # Compare reconstructed determinants so a QR-vs-LU split of sign/log does
    # not fail the test when the product still matches. Guard exp() so a
    # singular matrix's -inf log-abs-det does not warn or overflow.
    with np.errstate(over="ignore", invalid="ignore"):
      np_recon = np_sign * np.exp(np_log_abs)
      tf_recon = sign_out * np.exp(log_abs_out)
    self.assertAllClose(np_recon, tf_recon, rtol=1e-3, atol=1e-3)
    self.assertShapeEqual(np_sign, sign)
    self.assertShapeEqual(np_log_abs, log_abs)

  def _verifyDeterminantReal(self, x):
    for np_type in self.float_types & {np.float32, np.float64}:
      self._verifyDeterminant(x, np_type)

  def testBasic(self):
    # 1x1
    self._verifyDeterminantReal(np.array([[7.]]))
    # 2x2 with negative determinant: det([[1, 2], [3, 4]]) == -2.
    # This is the case xla::LogDet() gets wrong (returns NaN).
    self._verifyDeterminantReal(np.array([[1., 2.], [3., 4.]]))
    # 2x2 with positive determinant (the motivating jit_compile example).
    self._verifyDeterminantReal(np.array([[4., 7.], [2., 6.]]))
    # Singular.
    self._verifyDeterminantReal(np.array([[0., 0.], [0., 0.]]))
    # 3x3 with negative determinant.
    self._verifyDeterminantReal(
        np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]))
    # Well-conditioned 5x5 triangular matrix; det = 2*3*4*5*6 = 720.
    self._verifyDeterminantReal(
        np.array([[2., 0., 0., 0., 0.], [1., 3., 0., 0., 0.],
                  [0., 1., 4., 0., 0.], [0., 0., 1., 5., 0.],
                  [0., 0., 0., 1., 6.]]))

  def testBatch(self):
    # Mixed signs in the batch: dets are -2 and 3.
    self._verifyDeterminantReal(
        np.array([[[1., 2.], [3., 4.]], [[2., 1.], [1., 2.]]]))
    matrix1 = np.array([[1., 2.], [3., 4.]])
    matrix2 = np.array([[1., 3.], [3., 5.]])
    batch = np.concatenate(
        [np.expand_dims(matrix1, 0),
         np.expand_dims(matrix2, 0)])
    batch = np.tile(batch, [2, 3, 1, 1])
    self._verifyDeterminantReal(batch)

  def testEmpty(self):
    self._verifyDeterminantReal(np.empty([0, 2, 2]))
    self._verifyDeterminantReal(np.empty([2, 0, 0]))


if __name__ == "__main__":
  googletest.main()
