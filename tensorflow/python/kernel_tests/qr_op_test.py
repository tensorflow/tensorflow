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
import tensorflow as tf


class QrOpTest(tf.test.TestCase):

  def testWrongDimensions(self):
    # The input to qr should be a tensor of at least rank 2.
    scalar = tf.constant(1.)
    with self.assertRaisesRegexp(ValueError,
                                 "Shape must be at least rank 2 but is rank 0"):
      tf.qr(scalar)
    vector = tf.constant([1., 2.])
    with self.assertRaisesRegexp(ValueError,
                                 "Shape must be at least rank 2 but is rank 1"):
      tf.qr(vector)


def _GetQrOpTest(dtype_, shape_, use_static_shape_):

  is_complex = dtype_ in (np.complex64, np.complex128)
  is_single = dtype_ in (np.float32, np.complex64)

  def CompareOrthogonal(self, x, y, rank):
    if is_single:
      atol = 5e-4
    else:
      atol = 5e-14
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
    self.assertAllClose(x, y, atol=atol)

  def CheckApproximation(self, a, q, r):
    if is_single:
      tol = 1e-5
    else:
      tol = 1e-14
    # Tests that a ~= q*r.
    a_recon = tf.matmul(q, r)
    self.assertAllClose(a_recon.eval(), a, rtol=tol, atol=tol)

  def CheckUnitary(self, x):
    # Tests that x[...,:,:]^H * x[...,:,:] is close to the identity.
    xx = tf.matmul(tf.conj(x), x, transpose_a=True)
    identity = tf.matrix_band_part(tf.ones_like(xx), 0, 0)
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

    for full_matrices in False, True:
      with self.test_session() as sess:
        if use_static_shape_:
          x_tf = tf.constant(x_np)
        else:
          x_tf = tf.placeholder(dtype_)
        q_tf, r_tf = tf.qr(x_tf, full_matrices=full_matrices)

        if use_static_shape_:
          q_tf_val, r_tf_val = sess.run([q_tf, r_tf])
        else:
          q_tf_val, r_tf_val = sess.run([q_tf, r_tf], feed_dict={x_tf: x_np})

        q_dims = q_tf_val.shape
        np_q = np.ndarray(q_dims, dtype_)
        np_q_reshape = np.reshape(np_q, (-1, q_dims[-2], q_dims[-1]))
        new_first_dim = np_q_reshape.shape[0]

        x_reshape = np.reshape(x_np, (-1, x_np.shape[-2], x_np.shape[-1]))
        for i in range(new_first_dim):
          if full_matrices:
            np_q_reshape[i,:,:], _ = \
                np.linalg.qr(x_reshape[i,:,:], mode="complete")
          else:
            np_q_reshape[i,:,:], _ = \
                np.linalg.qr(x_reshape[i,:,:], mode="reduced")
        np_q = np.reshape(np_q_reshape, q_dims)
        CompareOrthogonal(self, np_q, q_tf_val, min(shape_[-2:]))
        CheckApproximation(self, x_np, q_tf_val, r_tf_val)
        CheckUnitary(self, q_tf_val)

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
            setattr(QrOpTest, "testQr_" + name,
                    _GetQrOpTest(dtype, shape, use_static_shape))
  tf.test.main()
