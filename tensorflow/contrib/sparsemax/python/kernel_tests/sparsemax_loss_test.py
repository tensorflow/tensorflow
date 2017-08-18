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
"""Tests for SparsemaxLossOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.sparsemax import sparsemax, sparsemax_loss
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test

test_obs = 10


class SparsemaxLossTest(test.TestCase):

  def _np_sparsemax(self, z):
    z = z - np.mean(z, axis=1)[:, np.newaxis]

    # sort z
    z_sorted = np.sort(z, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, z.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    # use argmax to get the index by row as .nonzero() doesn't
    # take an axis argument. np.argmax return the first index, but the last
    # index is required here, use np.flip to get the last index and
    # `z.shape[axis]` to compensate for np.flip afterwards.
    k_z = z.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(z)
    tau_sum = z_cumsum[np.arange(0, z.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    # calculate p
    return np.maximum(0, z - tau_z)

  def _np_sparsemax_loss(self, z, q):
    z = z - np.mean(z, axis=1)[:, np.newaxis]

    # Calculate q^T * z
    z_k = np.sum(q * z, axis=1)

    # calculate sum over S(z)
    p = self._np_sparsemax(z)
    s = p > 0
    # z_i^2 - tau(z)^2 = p_i (2 * z_i - p_i) for i \in S(z)
    S_sum = np.sum(s * p * (2 * z - p), axis=1)

    # because q is binary, sum([q_1^2, q_2^2, ...]) is just sum(q)
    q_norm = np.sum(q, axis=1)

    return -z_k + 0.5 * S_sum + 0.5 * q_norm

  def _np_sparsemax_loss_grad(self, z, q):
    # chain rule
    grad = 1

    return grad * (-q + self._np_sparsemax(z))

  def _tf_sparsemax(self, z, dtype, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      tf_sparsemax_op = sparsemax(z.astype(dtype))
      tf_sparsemax_out = tf_sparsemax_op.eval()

    return tf_sparsemax_op, tf_sparsemax_out

  def _tf_sparsemax_loss(self, z, q, dtype, use_gpu):
    z = z.astype(dtype)
    q = q.astype(dtype)

    with self.test_session(use_gpu=use_gpu):
      tf_sparsemax_op = sparsemax(z)
      tf_loss_op = sparsemax_loss(z, tf_sparsemax_op, q)
      tf_loss_out = tf_loss_op.eval()

    return tf_loss_op, tf_loss_out

  def _test_sparsemax_loss_against_numpy(self, dtype, random, use_gpu):
    """check sparsemax-loss kernel against numpy"""
    z = random.uniform(low=-3, high=3, size=(test_obs, 10))
    q = np.zeros((test_obs, 10))
    q[np.arange(0, test_obs), random.randint(0, 10, size=test_obs)] = 1

    tf_loss_op, tf_loss_out = self._tf_sparsemax_loss(z, q, dtype, use_gpu)
    np_loss = self._np_sparsemax_loss(z, q).astype(dtype)

    self.assertAllCloseAccordingToType(np_loss, tf_loss_out,
                                       half_atol=1e-2, half_rtol=5e-3)
    self.assertShapeEqual(np_loss, tf_loss_op)

  def _test_constant_add(self, dtype, random, use_gpu):
    """check sparsemax-loss proposition 3"""
    z = random.uniform(low=-3, high=3, size=(test_obs, 10))
    c = random.uniform(low=-3, high=3, size=(test_obs, 1))
    q = np.zeros((test_obs, 10))
    q[np.arange(0, test_obs), np.random.randint(0, 10, size=test_obs)] = 1

    _, tf_loss_zpc = self._tf_sparsemax_loss(
        z + c, q, dtype, use_gpu
    )

    _, tf_loss_z = self._tf_sparsemax_loss(
        z, q, dtype, use_gpu
    )

    self.assertAllCloseAccordingToType(tf_loss_zpc, tf_loss_z,
                                       float_atol=5e-6, float_rtol=5e-6,
                                       half_atol=1e-2, half_rtol=1e-2)

  def _test_sparsemax_loss_positive(self, dtype, random, use_gpu):
    """check sparsemax-loss proposition 4"""
    z = random.uniform(low=-3, high=3, size=(test_obs, 10))
    q = np.zeros((test_obs, 10))
    q[np.arange(0, test_obs), random.randint(0, 10, size=test_obs)] = 1

    tf_loss_op, tf_loss_out = self._tf_sparsemax_loss(z, q, dtype, use_gpu)

    self.assertAllCloseAccordingToType(np.abs(tf_loss_out), tf_loss_out)
    self.assertShapeEqual(np.zeros(test_obs), tf_loss_op)

  def _test_sparsemax_loss_zero(self, dtype, random, use_gpu):
    """check sparsemax-loss proposition 5"""
    # construct z and q, such that z_k >= 1 + max_{j!=k} z_k holds for
    # delta_0 = 1.
    z = random.uniform(low=-3, high=3, size=(test_obs, 10))
    z[:, 0] = np.max(z, axis=1) + 1.05

    q = np.zeros((test_obs, 10))
    q[:, 0] = 1

    tf_loss_op, tf_loss_out = self._tf_sparsemax_loss(z, q, dtype, use_gpu)
    tf_sparsemax_op, tf_sparsemax_out = self._tf_sparsemax(z, dtype, use_gpu)

    self.assertAllCloseAccordingToType(np.zeros(test_obs), tf_loss_out)
    self.assertShapeEqual(np.zeros(test_obs), tf_loss_op)

    self.assertAllCloseAccordingToType(q, tf_sparsemax_out)
    self.assertShapeEqual(q, tf_sparsemax_op)

  def _test_gradient_against_estimate(self, dtype, random, use_gpu):
    """check sparsemax-loss Rop, against estimated-loss Rop"""
    z = random.uniform(low=-3, high=3, size=(test_obs, 10)).astype(dtype)
    q = np.zeros((test_obs, 10)).astype(dtype)
    q[np.arange(0, test_obs), np.random.randint(0, 10, size=test_obs)] = 1

    logits = array_ops.placeholder(dtype, name='z')
    sparsemax_op = sparsemax(logits)
    loss_op = sparsemax_loss(logits, sparsemax_op, q)

    with self.test_session(use_gpu=use_gpu):
      err = gradient_checker.compute_gradient_error(
        logits, z.shape,
        loss_op, (test_obs, ),
        x_init_value=z, delta=1e-9
      )

    self.assertLess(err, 1e-4)

  def _test_gradient_against_numpy(self, dtype, random, use_gpu):
    """check sparsemax-loss Rop, against numpy Rop"""
    z = random.uniform(low=-3, high=3, size=(test_obs, 10))
    q = np.zeros((test_obs, 10))
    q[np.arange(0, test_obs), np.random.randint(0, 10, size=test_obs)] = 1

    logits = constant_op.constant(z.astype(dtype), name='z')
    sparsemax_op = sparsemax(logits)
    loss_op = sparsemax_loss(logits, sparsemax_op, q.astype(dtype))
    loss_grad_op = gradients_impl.gradients(loss_op, [logits])[0]

    with self.test_session(use_gpu=use_gpu):
      tf_grad = loss_grad_op.eval()
      np_grad = self._np_sparsemax_loss_grad(z, q).astype(dtype)

      self.assertAllCloseAccordingToType(np_grad, tf_grad,
                                         half_atol=1e-2, half_rtol=5e-3)
      self.assertShapeEqual(np_grad, loss_grad_op)

  def _test_dtype(self, dtype):
    random = np.random.RandomState(1)

    self._test_sparsemax_loss_against_numpy(dtype, random, use_gpu=False)

    self._test_constant_add(dtype, random, use_gpu=False)

    self._test_sparsemax_loss_positive(dtype, random, use_gpu=False)

    self._test_sparsemax_loss_zero(dtype, random, use_gpu=False)

    # sparsemax is not a smooth function so gradient estimation is only
    # possibol for float64.
    if dtype == 'float64':
      self._test_gradient_against_estimate(dtype, random, use_gpu=False)

    self._test_gradient_against_numpy(dtype, random, use_gpu=False)

  def testFloat(self):
    self._test_dtype('float32')

  def testDouble(self):
    self._test_dtype('float64')

if __name__ == "__main__":
  test.main()
