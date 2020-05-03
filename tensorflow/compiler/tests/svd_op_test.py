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
"""Tests for tensorflow.ops.svd."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test


class SvdOpTest(xla_test.XLATestCase, parameterized.TestCase):

  def _compute_usvt(self, s, u, v):
    m = u.shape[-1]
    n = v.shape[-1]
    if m <= n:
      v = v[..., :m]
    else:
      u = u[..., :n]

    return np.matmul(u * s[..., None, :], np.swapaxes(v, -1, -2))

  def _testSvdCorrectness(self, dtype, shape):
    np.random.seed(1)
    x_np = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(dtype)
    m, n = shape[-2], shape[-1]
    _, s_np, _ = np.linalg.svd(x_np)
    with self.session() as sess:
      x_tf = array_ops.placeholder(dtype)
      with self.test_scope():
        s, u, v = linalg_ops.svd(x_tf, full_matrices=True)
      s_val, u_val, v_val = sess.run([s, u, v], feed_dict={x_tf: x_np})
      u_diff = np.matmul(u_val, np.swapaxes(u_val, -1, -2)) - np.eye(m)
      v_diff = np.matmul(v_val, np.swapaxes(v_val, -1, -2)) - np.eye(n)
      # Check u_val and v_val are orthogonal matrices.
      self.assertLess(np.linalg.norm(u_diff), 1e-2)
      self.assertLess(np.linalg.norm(v_diff), 1e-2)
      # Check that the singular values are correct, i.e., close to the ones from
      # numpy.lingal.svd.
      self.assertLess(np.linalg.norm(s_val - s_np), 1e-2)
      # The tolerance is set based on our tests on numpy's svd. As our tests
      # have batch dimensions and all our operations are on float32, we set the
      # tolerance a bit larger. Numpy's svd calls LAPACK's svd, which operates
      # on double precision.
      self.assertLess(
          np.linalg.norm(self._compute_usvt(s_val, u_val, v_val) - x_np), 2e-2)

      # Check behavior with compute_uv=False.  We expect to still see 3 outputs,
      # with a sentinel scalar 0 in the last two outputs.
      with self.test_scope():
        no_uv_s, no_uv_u, no_uv_v = gen_linalg_ops.svd(
            x_tf, full_matrices=True, compute_uv=False)
      no_uv_s_val, no_uv_u_val, no_uv_v_val = sess.run(
          [no_uv_s, no_uv_u, no_uv_v], feed_dict={x_tf: x_np})
      self.assertAllClose(no_uv_s_val, s_val, atol=1e-4, rtol=1e-4)
      self.assertEqual(no_uv_u_val, 0.0)
      self.assertEqual(no_uv_v_val, 0.0)

  SIZES = [1, 2, 5, 10, 32, 64]
  DTYPES = [np.float32]
  PARAMS = itertools.product(SIZES, DTYPES)

  @parameterized.parameters(*PARAMS)
  def testSvd(self, n, dtype):
    for batch_dims in [(), (3,)] + [(3, 2)] * (n < 10):
      self._testSvdCorrectness(dtype, batch_dims + (n, n))
      self._testSvdCorrectness(dtype, batch_dims + (2 * n, n))
      self._testSvdCorrectness(dtype, batch_dims + (n, 2 * n))


if __name__ == "__main__":
  test.main()
