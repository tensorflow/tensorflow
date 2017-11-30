# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for scaled_softplus.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.nn.python.ops.scaled_softplus import scaled_softplus
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test


class ScaledSoftplusTest(test.TestCase):

  def test(self):
    np.random.seed(1)  # Make it reproducible.
    x = np.random.randn(3, 4).astype(np.float32)
    x64 = np.random.randn(3, 4).astype(np.float64)
    alpha = np.random.rand() + 0.01
    y = alpha * np.log(1. + np.exp(x / alpha))
    y64 = alpha * np.log(1. + np.exp(x64 / alpha))
    with self.test_session(use_gpu=True) as sess:
      z = scaled_softplus(constant_op.constant(x), alpha)
      z64 = scaled_softplus(constant_op.constant(x64), alpha)
      z, z64 = sess.run([z, z64])
      eps = 1e-6
      self.assertAllClose(y, z, eps)
      self.assertAllClose(y64, z64, eps)

  def testGradient(self):
    np.random.seed(1)  # Make it reproducible.
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)
    alpha_np = np.float32(np.random.rand() + 0.01)
    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np)
      alpha_tf = constant_op.constant(alpha_np)
      y_tf = scaled_softplus(x_tf, alpha_tf)
      err = gradient_checker.compute_gradient_error([x_tf, alpha_tf],
                                                    [x_shape, []],
                                                    y_tf, x_shape,
                                                    [x_np, alpha_np],
                                                    delta=1e-2)
    eps = 1e-4
    self.assertLess(err, eps)


if __name__ == '__main__':
  test.main()


