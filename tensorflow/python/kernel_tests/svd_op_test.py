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
"""Tests for tensorflow.ops.math_ops.matrix_inverse."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class SvdOpTest(test.TestCase):

  def testWrongDimensions(self):
    # The input to svd should be a tensor of at least rank 2.
    scalar = constant_op.constant(1.)
    with self.assertRaisesRegexp(ValueError,
                                 "Shape must be at least rank 2 but is rank 0"):
      linalg_ops.svd(scalar)
    vector = constant_op.constant([1., 2.])
    with self.assertRaisesRegexp(ValueError,
                                 "Shape must be at least rank 2 but is rank 1"):
      linalg_ops.svd(vector)


def _GetSvdOpTest(dtype_, shape_, use_static_shape_):

  is_complex = dtype_ in (np.complex64, np.complex128)
  is_single = dtype_ in (np.float32, np.complex64)

  def CompareSingularValues(self, x, y):
    if is_single:
      tol = 5e-5
    else:
      tol = 1e-14
    self.assertAllClose(x, y, atol=(x[0] + y[0]) * tol)

  def CompareSingularVectors(self, x, y, rank):
    if is_single:
      atol = 5e-4
    else:
      atol = 5e-14
    # We only compare the first 'rank' singular vectors since the
    # remainder form an arbitrary orthonormal basis for the
    # (row- or column-) null space, whose exact value depends on
    # implementation details. Notice that since we check that the
    # matrices of singular vectors are unitary elsewhere, we do
    # implicitly test that the trailing vectors of x and y span the
    # same space.
    x = x[..., 0:rank]
    y = y[..., 0:rank]
    # Singular vectors are only unique up to sign (complex phase factor for
    # complex matrices), so we normalize the sign first.
    sum_of_ratios = np.sum(np.divide(y, x), -2, keepdims=True)
    phases = np.divide(sum_of_ratios, np.abs(sum_of_ratios))
    x *= phases
    self.assertAllClose(x, y, atol=atol)

  def CheckApproximation(self, a, u, s, v, full_matrices):
    if is_single:
      tol = 1e-5
    else:
      tol = 1e-14
    # Tests that a ~= u*diag(s)*transpose(v).
    batch_shape = a.shape[:-2]
    m = a.shape[-2]
    n = a.shape[-1]
    diag_s = math_ops.cast(array_ops.matrix_diag(s), dtype=dtype_)
    if full_matrices:
      if m > n:
        zeros = array_ops.zeros(batch_shape + (m - n, n), dtype=dtype_)
        diag_s = array_ops.concat([diag_s, zeros], a.ndim - 2)
      elif n > m:
        zeros = array_ops.zeros(batch_shape + (m, n - m), dtype=dtype_)
        diag_s = array_ops.concat([diag_s, zeros], a.ndim - 1)
    a_recon = math_ops.matmul(u, diag_s)
    a_recon = math_ops.matmul(a_recon, v, adjoint_b=True)
    self.assertAllClose(a_recon.eval(), a, rtol=tol, atol=tol)

  def CheckUnitary(self, x):
    # Tests that x[...,:,:]^H * x[...,:,:] is close to the identity.
    xx = math_ops.matmul(x, x, adjoint_a=True)
    identity = array_ops.matrix_band_part(array_ops.ones_like(xx), 0, 0)
    if is_single:
      tol = 1e-5
    else:
      tol = 1e-14
    self.assertAllClose(identity.eval(), xx.eval(), atol=tol)

  def Test(self):
    np.random.seed(1)
    x_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape_)).reshape(shape_).astype(dtype_)
    if is_complex:
      x_np += 1j * np.random.uniform(
          low=-1.0, high=1.0,
          size=np.prod(shape_)).reshape(shape_).astype(dtype_)

    for compute_uv in False, True:
      for full_matrices in False, True:
        with self.test_session() as sess:
          if use_static_shape_:
            x_tf = constant_op.constant(x_np)
          else:
            x_tf = array_ops.placeholder(dtype_)

          if compute_uv:
            s_tf, u_tf, v_tf = linalg_ops.svd(x_tf,
                                              compute_uv=compute_uv,
                                              full_matrices=full_matrices)
            if use_static_shape_:
              s_tf_val, u_tf_val, v_tf_val = sess.run([s_tf, u_tf, v_tf])
            else:
              s_tf_val, u_tf_val, v_tf_val = sess.run([s_tf, u_tf, v_tf],
                                                      feed_dict={x_tf: x_np})
          else:
            s_tf = linalg_ops.svd(x_tf,
                                  compute_uv=compute_uv,
                                  full_matrices=full_matrices)
            if use_static_shape_:
              s_tf_val = sess.run(s_tf)
            else:
              s_tf_val = sess.run(s_tf, feed_dict={x_tf: x_np})

          if compute_uv:
            u_np, s_np, v_np = np.linalg.svd(x_np,
                                             compute_uv=compute_uv,
                                             full_matrices=full_matrices)
          else:
            s_np = np.linalg.svd(x_np,
                                 compute_uv=compute_uv,
                                 full_matrices=full_matrices)
          # We explicitly avoid the situation where numpy eliminates a first
          # dimension that is equal to one
          s_np = np.reshape(s_np, s_tf_val.shape)

          CompareSingularValues(self, s_np, s_tf_val)
          if compute_uv:
            CompareSingularVectors(self, u_np, u_tf_val, min(shape_[-2:]))
            CompareSingularVectors(self,
                                   np.conj(np.swapaxes(v_np, -2, -1)), v_tf_val,
                                   min(shape_[-2:]))
            CheckApproximation(self, x_np, u_tf_val, s_tf_val, v_tf_val,
                               full_matrices)
            CheckUnitary(self, u_tf_val)
            CheckUnitary(self, v_tf_val)

  return Test


if __name__ == "__main__":
  for dtype in np.float32, np.float64, np.complex64, np.complex128:
    for rows in 1, 2, 5, 10, 32, 100:
      for cols in 1, 2, 5, 10, 32, 100:
        for batch_dims in [(), (3,)] + [(3, 2)] * (max(rows, cols) < 10):
          shape = batch_dims + (rows, cols)
          for use_static_shape in True, False:
            name = "%s_%s_%s" % (dtype.__name__, "_".join(map(str, shape)),
                                 use_static_shape)
            setattr(SvdOpTest, "testSvd_" + name,
                    _GetSvdOpTest(dtype, shape, use_static_shape))
  test.main()
