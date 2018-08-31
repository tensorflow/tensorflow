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
"""Tests for AbsoluteValue Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-importing-member
from tensorflow.contrib.distributions.python.ops.bijectors.absolute_value import AbsoluteValue
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

# pylint: enable=g-importing-member


class AbsoluteValueTest(test.TestCase):
  """Tests correctness of the absolute value bijector."""

  def testBijectorVersusNumpyRewriteOfBasicFunctionsEventNdims0(self):
    with self.cached_session() as sess:
      bijector = AbsoluteValue(validate_args=True)
      self.assertEqual("absolute_value", bijector.name)
      x = array_ops.constant([[0., 1., -1], [0., -5., 3.]])  # Shape [2, 3]
      y = math_ops.abs(x)

      y_ = y.eval()

      self.assertAllClose(y_, bijector.forward(x).eval())
      self.assertAllClose((-y_, y_), sess.run(bijector.inverse(y)))
      self.assertAllClose((0., 0.),
                          sess.run(bijector.inverse_log_det_jacobian(
                              y, event_ndims=0)))

      # Run things twice to make sure there are no issues in caching the tuples
      # returned by .inverse*
      self.assertAllClose(y_, bijector.forward(x).eval())
      self.assertAllClose((-y_, y_), sess.run(bijector.inverse(y)))
      self.assertAllClose((0., 0.),
                          sess.run(bijector.inverse_log_det_jacobian(
                              y, event_ndims=0)))

  def testNegativeYRaisesForInverseIfValidateArgs(self):
    with self.cached_session() as sess:
      bijector = AbsoluteValue(validate_args=True)
      with self.assertRaisesOpError("y was negative"):
        sess.run(bijector.inverse(-1.))

  def testNegativeYRaisesForILDJIfValidateArgs(self):
    with self.cached_session() as sess:
      bijector = AbsoluteValue(validate_args=True)
      with self.assertRaisesOpError("y was negative"):
        sess.run(bijector.inverse_log_det_jacobian(-1., event_ndims=0))


if __name__ == "__main__":
  test.main()
