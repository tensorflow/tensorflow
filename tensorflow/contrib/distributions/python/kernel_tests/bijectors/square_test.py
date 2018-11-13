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

from tensorflow.contrib.distributions.python.ops import bijectors
from tensorflow.python.ops.distributions.bijector_test_util import assert_scalar_congruency
from tensorflow.python.platform import test


class SquareBijectorTest(test.TestCase):
  """Tests the correctness of the Y = X ** 2 transformation."""

  def testBijectorScalar(self):
    with self.cached_session():
      bijector = bijectors.Square(validate_args=True)
      self.assertEqual("square", bijector.name)
      x = [[[1., 5],
            [2, 1]],
           [[np.sqrt(2.), 3],
            [np.sqrt(8.), 1]]]
      y = np.square(x)
      ildj = -np.log(2.) - np.log(x)
      self.assertAllClose(y, bijector.forward(x).eval())
      self.assertAllClose(x, bijector.inverse(y).eval())
      self.assertAllClose(
          ildj, bijector.inverse_log_det_jacobian(
              y, event_ndims=0).eval(), atol=0., rtol=1e-7)
      self.assertAllClose(
          -bijector.inverse_log_det_jacobian(y, event_ndims=0).eval(),
          bijector.forward_log_det_jacobian(x, event_ndims=0).eval(),
          atol=0.,
          rtol=1e-7)

  def testScalarCongruency(self):
    with self.cached_session():
      bijector = bijectors.Square(validate_args=True)
      assert_scalar_congruency(bijector, lower_x=1e-3, upper_x=1.5, rtol=0.05)


if __name__ == "__main__":
  test.main()
