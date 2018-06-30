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
"""Tests for ScaleTriL bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import bijectors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class ScaleTriLBijectorTest(test.TestCase):
  """Tests the correctness of the ScaleTriL bijector."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testComputesCorrectValues(self):
    shift = 1.61803398875
    x = np.float32(np.array([-1, .5, 2]))
    y = np.float32(np.array([[np.exp(2) + shift, 0.],
                             [.5, np.exp(-1) + shift]]))

    b = bijectors.ScaleTriL(diag_bijector=bijectors.Exp(),
                            diag_shift=shift)

    y_ = self.evaluate(b.forward(x))
    self.assertAllClose(y, y_)

    x_ = self.evaluate(b.inverse(y))
    self.assertAllClose(x, x_)

  @test_util.run_in_graph_and_eager_modes
  def testInvertible(self):

    # Generate random inputs from an unconstrained space, with
    # event size 6 to specify 3x3 triangular matrices.
    batch_shape = [2, 1]
    x = np.float32(np.random.randn(*(batch_shape + [6])))
    b = bijectors.ScaleTriL(diag_bijector=bijectors.Softplus(),
                            diag_shift=3.14159)
    y = self.evaluate(b.forward(x))
    self.assertAllEqual(y.shape, batch_shape + [3, 3])

    x_ = self.evaluate(b.inverse(y))
    self.assertAllClose(x, x_)

    fldj = self.evaluate(b.forward_log_det_jacobian(x, event_ndims=1))
    ildj = self.evaluate(b.inverse_log_det_jacobian(y, event_ndims=2))
    self.assertAllClose(fldj, -ildj)

if __name__ == "__main__":
  test.main()
