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
"""Identity Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops.bijectors import bijector_test_util
from tensorflow.contrib.distributions.python.ops.bijectors import identity as identity_lib
from tensorflow.python.platform import test


class IdentityBijectorTest(test.TestCase):
  """Tests correctness of the Y = g(X) = X transformation."""

  def testBijector(self):
    with self.test_session():
      bijector = identity_lib.Identity()
      self.assertEqual("identity", bijector.name)
      x = [[[0.], [1.]]]
      self.assertAllEqual(x, bijector.forward(x).eval())
      self.assertAllEqual(x, bijector.inverse(x).eval())
      self.assertAllEqual(0., bijector.inverse_log_det_jacobian(x).eval())
      self.assertAllEqual(0., bijector.forward_log_det_jacobian(x).eval())

  def testScalarCongruency(self):
    with self.test_session():
      bijector = identity_lib.Identity()
      bijector_test_util.assert_scalar_congruency(
          bijector, lower_x=-2., upper_x=2.)


if __name__ == "__main__":
  test.main()
