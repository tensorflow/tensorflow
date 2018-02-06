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

import numpy as np

# pylint: disable=g-importing-member
from tensorflow.contrib.distributions.python.ops.bijectors.absolute_value import AbsoluteValue
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

# pylint: enable=g-importing-member


class AbsoluteValueTest(test.TestCase):
  """Tests correctness of the absolute value bijector."""

  def testBijectorVersusNumpyRewriteOfBasicFunctionsEventNdims0(self):
    with self.test_session() as sess:
      bijector = AbsoluteValue(event_ndims=0, validate_args=True)
      self.assertEqual("absolute_value", bijector.name)
      x = array_ops.constant([[0., 1., -1], [0., -5., 3.]])  # Shape [2, 3]
      y = math_ops.abs(x)

      y_ = y.eval()
      zeros = np.zeros((2, 3))

      self.assertAllClose(y_, bijector.forward(x).eval())
      self.assertAllClose((-y_, y_), sess.run(bijector.inverse(y)))
      self.assertAllClose((zeros, zeros),
                          sess.run(bijector.inverse_log_det_jacobian(y)))

      # Run things twice to make sure there are no issues in caching the tuples
      # returned by .inverse*
      self.assertAllClose(y_, bijector.forward(x).eval())
      self.assertAllClose((-y_, y_), sess.run(bijector.inverse(y)))
      self.assertAllClose((zeros, zeros),
                          sess.run(bijector.inverse_log_det_jacobian(y)))

  def testEventNdimsMustBeZeroOrRaiseStatic(self):
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "event_ndims.*was not 0"):
        AbsoluteValue(event_ndims=1)

  def testEventNdimsMustBeZeroOrRaiseDynamic(self):
    with self.test_session() as sess:
      event_ndims = array_ops.placeholder(dtypes.int32)
      abs_bijector = AbsoluteValue(event_ndims=event_ndims, validate_args=True)
      with self.assertRaisesOpError("event_ndims was not 0"):
        sess.run(abs_bijector.inverse_log_det_jacobian([1.]),
                 feed_dict={event_ndims: 1})

  def testNegativeYRaisesForInverseIfValidateArgs(self):
    with self.test_session() as sess:
      bijector = AbsoluteValue(event_ndims=0, validate_args=True)
      with self.assertRaisesOpError("y was negative"):
        sess.run(bijector.inverse(-1.))

  def testNegativeYRaisesForILDJIfValidateArgs(self):
    with self.test_session() as sess:
      bijector = AbsoluteValue(event_ndims=0, validate_args=True)
      with self.assertRaisesOpError("y was negative"):
        sess.run(bijector.inverse_log_det_jacobian(-1.))


if __name__ == "__main__":
  test.main()
