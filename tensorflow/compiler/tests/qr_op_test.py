# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.math_ops.matrix_inverse."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class QrOpTest(xla_test.XLATestCase, parameterized.TestCase):

  def AdjustedNorm(self, x):
    """Computes the norm of matrices in 'x', adjusted for dimension and type."""
    norm = np.linalg.norm(x, axis=(-2, -1))
    return norm / (max(x.shape[-2:]) * np.finfo(x.dtype).eps)

  def CompareOrthogonal(self, x, y, rank):
    # We only compare the first 'rank' orthogonal vectors since the
    # remainder form an arbitrary orthonormal basis for the
    # (row- or column-) null space, whose exact value depends on
    # implementation details. Notice that since we check that the
    # matrices of singular vectors are unitary elsewhere, we do
    # implicitly test that the trailing vectors of x and y span the
    # same space.
    x = x[..., 0:rank]
    y = y[..., 0:rank]
    # Q is only unique up to sign (complex phase factor for complex matrices),
    # so we normalize the sign first.
    sum_of_ratios = np.sum(np.divide(y, x), -2, keepdims=True)
    phases = np.divide(sum_of_ratios, np.abs(sum_of_ratios))
    x *= phases
    self.assertTrue(np.all(self.AdjustedNorm(x - y) < 30.0))

  def CheckApproximation(self, a, q, r):
    # Tests that a ~= q*r.
    precision = self.AdjustedNorm(a - np.matmul(q, r))
    self.assertTrue(np.all(precision < 5.0))

  def CheckUnitary(self, x):
    # Tests that x[...,:,:]^H * x[...,:,:] is close to the identity.
    xx = math_ops.matmul(x, x, adjoint_a=True)
    identity = array_ops.matrix_band_part(array_ops.ones_like(xx), 0, 0)
    precision = self.AdjustedNorm(xx.eval() - identity.eval())
    self.assertTrue(np.all(precision < 5.0))

  def _test(self, dtype, shape, full_matrices):
    np.random.seed(1)
    x_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape)).reshape(shape).astype(dtype)

    with self.test_session() as sess:
      x_tf = array_ops.placeholder(dtype)
      with self.test_scope():
        q_tf, r_tf = linalg_ops.qr(x_tf, full_matrices=full_matrices)
      q_tf_val, r_tf_val = sess.run([q_tf, r_tf], feed_dict={x_tf: x_np})

      q_dims = q_tf_val.shape
      np_q = np.ndarray(q_dims, dtype)
      np_q_reshape = np.reshape(np_q, (-1, q_dims[-2], q_dims[-1]))
      new_first_dim = np_q_reshape.shape[0]

      x_reshape = np.reshape(x_np, (-1, x_np.shape[-2], x_np.shape[-1]))
      for i in range(new_first_dim):
        if full_matrices:
          np_q_reshape[i, :, :], _ = np.linalg.qr(
              x_reshape[i, :, :], mode="complete")
        else:
          np_q_reshape[i, :, :], _ = np.linalg.qr(
              x_reshape[i, :, :], mode="reduced")
      np_q = np.reshape(np_q_reshape, q_dims)
      self.CompareOrthogonal(np_q, q_tf_val, min(shape[-2:]))
      self.CheckApproximation(x_np, q_tf_val, r_tf_val)
      self.CheckUnitary(q_tf_val)

  SIZES = [1, 2, 5, 10, 32, 100, 300]
  DTYPES = [np.float32]
  PARAMS = itertools.product(SIZES, SIZES, DTYPES)

  @parameterized.parameters(*PARAMS)
  def testQR(self, rows, cols, dtype):
    # TODO(b/111317468): implement full_matrices=False, test other types.
    for full_matrices in [True]:
      # Only tests the (3, 2) case for small numbers of rows/columns.
      for batch_dims in [(), (3,)] + [(3, 2)] * (max(rows, cols) < 10):
        self._test(dtype, batch_dims + (rows, cols), full_matrices)


if __name__ == "__main__":
  test.main()
