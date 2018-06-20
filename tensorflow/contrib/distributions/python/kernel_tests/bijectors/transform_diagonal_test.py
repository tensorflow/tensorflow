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
"""Tests for TransformDiagonal bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import bijectors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class TransformDiagonalBijectorTest(test.TestCase):
  """Tests correctness of the TransformDiagonal bijector."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  @test_util.run_in_graph_and_eager_modes()
  def testBijector(self):
    x = np.float32(np.random.randn(3, 4, 4))

    y = x.copy()
    for i in range(x.shape[0]):
      np.fill_diagonal(y[i, :, :], np.exp(np.diag(x[i, :, :])))

    exp = bijectors.Exp()
    b = bijectors.TransformDiagonal(diag_bijector=exp)

    y_ = self.evaluate(b.forward(x))
    self.assertAllClose(y, y_)

    x_ = self.evaluate(b.inverse(y))
    self.assertAllClose(x, x_)

    fldj = self.evaluate(b.forward_log_det_jacobian(x, event_ndims=2))
    ildj = self.evaluate(b.inverse_log_det_jacobian(y, event_ndims=2))
    self.assertAllEqual(
        fldj,
        self.evaluate(exp.forward_log_det_jacobian(
            np.array([np.diag(x_mat) for x_mat in x]),
            event_ndims=1)))
    self.assertAllEqual(
        ildj,
        self.evaluate(exp.inverse_log_det_jacobian(
            np.array([np.diag(y_mat) for y_mat in y]),
            event_ndims=1)))


if __name__ == "__main__":
  test.main()
