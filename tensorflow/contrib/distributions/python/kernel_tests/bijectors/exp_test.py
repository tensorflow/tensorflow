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
"""Exp Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops.bijectors.exp import Exp
from tensorflow.python.ops.distributions.bijector_test_util import assert_bijective_and_finite
from tensorflow.python.ops.distributions.bijector_test_util import assert_scalar_congruency
from tensorflow.python.platform import test


class ExpBijectorTest(test.TestCase):
  """Tests correctness of the Y = g(X) = exp(X) transformation."""

  def testBijector(self):
    with self.test_session():
      bijector = Exp()
      self.assertEqual("exp", bijector.name)
      x = [[[1.], [2.]]]
      y = np.exp(x)
      self.assertAllClose(y, bijector.forward(x).eval())
      self.assertAllClose(x, bijector.inverse(y).eval())
      self.assertAllClose(
          -np.squeeze(np.log(y), axis=-1),
          bijector.inverse_log_det_jacobian(
              y, event_ndims=1).eval())
      self.assertAllClose(
          -bijector.inverse_log_det_jacobian(
              np.exp(x), event_ndims=1).eval(),
          bijector.forward_log_det_jacobian(
              x, event_ndims=1).eval())

  def testScalarCongruency(self):
    with self.test_session():
      bijector = Exp()
      assert_scalar_congruency(
          bijector, lower_x=-2., upper_x=1.5, rtol=0.05)

  def testBijectiveAndFinite(self):
    with self.test_session():
      bijector = Exp()
      x = np.linspace(-10, 10, num=10).astype(np.float32)
      y = np.logspace(-10, 10, num=10).astype(np.float32)
      assert_bijective_and_finite(bijector, x, y, event_ndims=0)


if __name__ == "__main__":
  test.main()
