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
"""Tests for tensorflow.ops.self_adjoint_eig."""

import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test


class SelfAdjointEigOpTest(xla_test.XLATestCase, parameterized.TestCase):

  def _test(self, dtype, shape):
    np.random.seed(1)
    x_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape)).reshape(shape).astype(dtype)
    x_np = x_np + np.swapaxes(x_np, -1, -2)
    n = shape[-1]

    e_np, _ = np.linalg.eigh(x_np)
    with self.session() as sess:
      x_tf = array_ops.placeholder(dtype)
      with self.test_scope():
        e, v = linalg_ops.self_adjoint_eig(x_tf)
      e_val, v_val = sess.run([e, v], feed_dict={x_tf: x_np})

      v_diff = np.matmul(v_val, np.swapaxes(v_val, -1, -2)) - np.eye(n)
      self.assertAlmostEqual(np.mean(v_diff**2), 0.0, delta=1e-6)
      self.assertAlmostEqual(np.mean((e_val - e_np)**2), 0.0, delta=1e-6)

  SIZES = [1, 2, 5, 10, 32]
  DTYPES = [np.float32]
  PARAMS = itertools.product(SIZES, DTYPES)

  @parameterized.parameters(*PARAMS)
  def testSelfAdjointEig(self, n, dtype):
    for batch_dims in [(), (3,)] + [(3, 2)] * (n < 10):
      self._test(dtype, batch_dims + (n, n))


if __name__ == "__main__":
  test.main()
