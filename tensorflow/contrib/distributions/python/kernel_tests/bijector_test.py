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

import math

import tensorflow as tf

from tensorflow.contrib.distributions.python.ops.bijector import _Exp  # pylint: disable=line-too-long
from tensorflow.contrib.distributions.python.ops.bijector import _Identity  # pylint: disable=line-too-long
from tensorflow.contrib.distributions.python.ops.shape import _ShapeUtil  # pylint: disable=line-too-long


class IdentityBijectorTest(tf.test.TestCase):
  """Tests the correctness of the Y = g(X) = X transformation."""

  def testBijector(self):
    with self.test_session():
      bijector = _Identity(_ShapeUtil(batch_ndims=1, event_ndims=1))
      self.assertEqual(bijector.name, 'Identity')
      x = [[[0.], [1]]]
      self.assertAllEqual(bijector.forward(x).eval(), x)
      self.assertAllEqual(bijector.inverse(x).eval(), x)
      self.assertAllEqual(bijector.inverse_log_det_jacobian(x).eval(),
                          [[0., 0]])
      rev, jac = bijector.inverse_and_inverse_log_det_jacobian(x)
      self.assertAllEqual(rev.eval(), x)
      self.assertAllEqual(jac.eval(), [[0., 0]])


class ExpBijectorTest(tf.test.TestCase):
  """Tests the correctness of the Y = g(X) = exp(X) transformation."""

  def testBijector(self):
    with self.test_session():
      bijector = _Exp(_ShapeUtil(batch_ndims=1, event_ndims=1))
      self.assertEqual(bijector.name, 'Exp')
      x = [[[1.], [2]]]
      self.assertAllClose(bijector.forward(x).eval(),
                          [[[math.exp(1.)], [math.exp(2.)]]])
      self.assertAllClose(bijector.inverse(x).eval(),
                          [[[math.log(1.)], [math.log(2.)]]])
      self.assertAllClose(bijector.inverse_log_det_jacobian(x).eval(),
                          [[0., -math.log(2.)]])
      rev, jac = bijector.inverse_and_inverse_log_det_jacobian(x)
      self.assertAllClose(rev.eval(), [[[math.log(1.)], [math.log(2.)]]])
      self.assertAllClose(jac.eval(), [[0., -math.log(2.)]])


if __name__ == '__main__':
  tf.test.main()
