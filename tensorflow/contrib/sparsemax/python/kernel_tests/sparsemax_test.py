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
"""Tests for SparsemaxOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.sparsemax import sparsemax
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test

test_obs = 10


class SparsemaxTest(test.TestCase):

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

  def _np_sparsemax_grad(self, z):
    # chain rule
    grad = np.ones_like(z)

    # Construct S(z)
    probability = self._np_sparsemax(z)
    support = probability > 0

    # Calculate \hat{v}, which will be a vector (scalar for each z)
    v_hat = np.sum(grad * support, axis=1) / np.sum(support, axis=1)

    # Calculates J(z) * v
    return support * (grad - v_hat[:, np.newaxis])

  def _tf_sparsemax(self, z, dtype, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      tf_sparsemax_op = sparsemax(z.astype(dtype))
      tf_sparsemax_out = tf_sparsemax_op.eval()

    return tf_sparsemax_op, tf_sparsemax_out

  def _test_sparsemax_against_numpy(self, dtype, random, use_gpu):
    """check sparsemax kernel against numpy"""
    z = random.uniform(low=-3, high=3, size=(test_obs, 10))

    tf_sparsemax_op, tf_sparsemax_out = self._tf_sparsemax(z, dtype, use_gpu)
    p_sparemax = self._np_sparsemax(z).astype(dtype)

    self.assertAllCloseAccordingToType(p_sparemax, tf_sparsemax_out,
                                       half_atol=5e-3)
    self.assertShapeEqual(p_sparemax, tf_sparsemax_op)

  def _test_sparsemax_of_zero(self, dtype, random, use_gpu):
    """check sparsemax proposition 1, part 1"""
    z = np.zeros((1, 10))

    tf_sparsemax_op, tf_sparsemax_out = self._tf_sparsemax(z, dtype, use_gpu)
    p_sparemax = np.ones_like(z, dtype=dtype) / z.size

    self.assertAllCloseAccordingToType(p_sparemax, tf_sparsemax_out)
    self.assertShapeEqual(p_sparemax, tf_sparsemax_op)

  def _test_sparsemax_of_inf(self, dtype, random, use_gpu):
    """check sparsemax proposition 1, part 2"""
    z = random.uniform(low=-3, high=3, size=(test_obs, 10))

    # assume |A(z)| = 1, as z is continues random
    z_sort_arg = np.argsort(z, axis=1)[:, ::-1]
    z_sort = np.sort(z, axis=-1)[:, ::-1]
    gamma_z = z_sort[:, 0] - z_sort[:, 1]
    epsilon = (0.99 * gamma_z * 1).reshape(-1, 1)

    # construct the expected 1_A(z) array
    p_expected = np.zeros((test_obs, 10), dtype=dtype)
    p_expected[np.arange(0, test_obs), z_sort_arg[:, 0]] = 1

    tf_sparsemax_op, tf_sparsemax_out = self._tf_sparsemax(
        (1 / epsilon) * z, dtype, use_gpu
    )

    self.assertAllCloseAccordingToType(p_expected, tf_sparsemax_out)
    self.assertShapeEqual(p_expected, tf_sparsemax_op)

  def _test_constant_add(self, dtype, random, use_gpu):
    """check sparsemax proposition 2"""
    z = random.uniform(low=-3, high=3, size=(test_obs, 10)).astype(dtype)
    c = random.uniform(low=-3, high=3, size=(test_obs, 1)).astype(dtype)

    _, tf_sparsemax_zpc = self._tf_sparsemax(
        z + c, dtype, use_gpu
    )

    _, tf_sparsemax_z = self._tf_sparsemax(
        z, dtype, use_gpu
    )

    self.assertAllCloseAccordingToType(tf_sparsemax_zpc, tf_sparsemax_z,
                                       half_atol=5e-3)

  def _test_permutation(self, dtype, random, use_gpu):
    """check sparsemax proposition 3"""
    z = random.uniform(low=-3, high=3, size=(test_obs, 10))
    _, p = self._tf_sparsemax(z, dtype, use_gpu)

    for i in range(test_obs):
      per = random.permutation(10)

      tf_sparsemax_op, tf_sparsemax_out = self._tf_sparsemax(
        z[i, per].reshape(1, -1), dtype, use_gpu
      )
      p_expected = p[i, per].reshape(1, -1)

      self.assertAllCloseAccordingToType(p_expected, tf_sparsemax_out,
                                         half_atol=5e-3)
      self.assertShapeEqual(p_expected, tf_sparsemax_op)

  def _test_diffrence(self, dtype, random, use_gpu):
    """check sparsemax proposition 4"""
    z = random.uniform(low=-3, high=3, size=(test_obs, 10))
    _, p = self._tf_sparsemax(z, dtype, use_gpu)

    etol = {'float16': 1e-2, 'float32': 1e-6, 'float64': 1e-9}[dtype]

    for val in range(0, test_obs):
      for i in range(0, 10):
        for j in range(0, 10):
          # check condition, the obesite pair will be checked anyway
          if z[val, i] > z[val, j]:
            continue

          self.assertTrue(
            0 <= p[val, j] - p[val, i] <= z[val, j] - z[val, i] + etol,
            "0 <= %.10f <= %.10f" % (
              p[val, j] - p[val, i], z[val, j] - z[val, i] + etol
            )
          )

  def _test_two_dimentional(self, dtype, random, use_gpu):
    """check two dimentation sparsemax case"""
    t = np.linspace(-2, 2, test_obs, dtype=dtype)
    z = np.vstack([
      t, np.zeros(test_obs, dtype=dtype)
    ]).T

    tf_sparsemax_op, tf_sparsemax_out = self._tf_sparsemax(z, dtype, use_gpu)

    p0_expected = np.select([t < -1, t <= 1, t > 1], [0, (t + 1) / 2, 1])

    self.assertAllCloseAccordingToType(p0_expected, tf_sparsemax_out[:, 0])
    self.assertAllCloseAccordingToType(1 - p0_expected, tf_sparsemax_out[:, 1])
    self.assertShapeEqual(z, tf_sparsemax_op)

  def _test_gradient_against_estimate(self, dtype, random, use_gpu):
    """check sparsemax Rop, aginst estimated Rop"""
    z = random.uniform(low=-3, high=3, size=(test_obs, 10)).astype(dtype)

    logits = array_ops.placeholder(dtype, name='z')
    sparsemax_op = sparsemax(logits)

    with self.test_session(use_gpu=use_gpu):
      err = gradient_checker.compute_gradient_error(
        logits, z.shape,
        sparsemax_op, z.shape,
        x_init_value=z, delta=1e-9
      )

    self.assertLess(err, 1e-4)

  def _test_gradient_against_numpy(self, dtype, random, use_gpu):
    """check sparsemax Rop, aginst numpy Rop"""
    z = random.uniform(low=-3, high=3, size=(test_obs, 10)).astype(dtype)

    logits = constant_op.constant(z, name='z')
    sparsemax_op = sparsemax(logits)
    sparsemax_grad_op = gradients_impl.gradients(sparsemax_op, [logits])[0]

    with self.test_session(use_gpu=use_gpu):
      tf_grad = sparsemax_grad_op.eval()
      np_grad = self._np_sparsemax_grad(z)

      self.assertAllCloseAccordingToType(np_grad, tf_grad)
      self.assertShapeEqual(np_grad, sparsemax_grad_op)

  def _test_dtype(self, dtype):
    random = np.random.RandomState(1)

    self._test_sparsemax_against_numpy(dtype, random, use_gpu=False)

    self._test_sparsemax_of_zero(dtype, random, use_gpu=False)

    self._test_sparsemax_of_inf(dtype, random, use_gpu=False)

    self._test_constant_add(dtype, random, use_gpu=False)

    self._test_permutation(dtype, random, use_gpu=False)

    self._test_diffrence(dtype, random, use_gpu=False)

    self._test_two_dimentional(dtype, random, use_gpu=False)

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
