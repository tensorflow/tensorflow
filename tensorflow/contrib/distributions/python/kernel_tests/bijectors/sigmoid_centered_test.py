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
"""Tests for Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops.bijectors import sigmoid_centered as sigmoid_centered_lib
from tensorflow.python.platform import test


class SigmoidCenteredBijectorTest(test.TestCase):
  """Tests correctness of the Y = g(X) = (1 + exp(-X))^-1 transformation."""

  def testBijector(self):
    with self.test_session():
      sigmoid = sigmoid_centered_lib.SigmoidCentered()
      self.assertEqual("sigmoid_centered", sigmoid.name)
      x = np.log([[2., 3, 4],
                  [4., 8, 12]])
      y = [[[2. / 3, 1. / 3],
            [3. / 4, 1. / 4],
            [4. / 5, 1. / 5]],
           [[4. / 5, 1. / 5],
            [8. / 9, 1. / 9],
            [12. / 13, 1. / 13]]]
      self.assertAllClose(y, sigmoid.forward(x).eval())
      self.assertAllClose(x, sigmoid.inverse(y).eval())
      self.assertAllClose(
          -np.sum(np.log(y), axis=2),
          sigmoid.inverse_log_det_jacobian(y).eval(),
          atol=0.,
          rtol=1e-7)
      self.assertAllClose(
          -sigmoid.inverse_log_det_jacobian(y).eval(),
          sigmoid.forward_log_det_jacobian(x).eval(),
          atol=0.,
          rtol=1e-7)


if __name__ == "__main__":
  test.main()
