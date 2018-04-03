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
"""Affine Scalar Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops.bijectors.affine_scalar import AffineScalar
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.distributions.bijector_test_util import assert_scalar_congruency
from tensorflow.python.platform import test


class AffineScalarBijectorTest(test.TestCase):
  """Tests correctness of the Y = scale @ x + shift transformation."""

  def testProperties(self):
    with self.test_session():
      mu = -1.
      # scale corresponds to 1.
      bijector = AffineScalar(shift=mu)
      self.assertEqual("affine_scalar", bijector.name)

  def testNoBatchScalar(self):
    with self.test_session() as sess:

      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = array_ops.placeholder(dtypes.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = -1.
        # Corresponds to scale = 2
        bijector = AffineScalar(shift=mu, scale=2.)
        x = [1., 2, 3]  # Three scalar samples (no batches).
        self.assertAllClose([1., 3, 5], run(bijector.forward, x))
        self.assertAllClose([1., 1.5, 2.], run(bijector.inverse, x))
        self.assertAllClose([-np.log(2.)] * 3,
                            run(bijector.inverse_log_det_jacobian, x))

  def testOneBatchScalarViaIdentityIn64BitUserProvidesShiftOnly(self):
    with self.test_session() as sess:

      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value).astype(np.float64)
        x = array_ops.placeholder(dtypes.float64, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = np.float64([1.])
        # One batch, scalar.
        # Corresponds to scale = 1.
        bijector = AffineScalar(shift=mu)
        x = np.float64([1.])  # One sample from one batches.
        self.assertAllClose([2.], run(bijector.forward, x))
        self.assertAllClose([0.], run(bijector.inverse, x))
        self.assertAllClose([0.], run(bijector.inverse_log_det_jacobian, x))

  def testOneBatchScalarViaIdentityIn64BitUserProvidesScaleOnly(self):
    with self.test_session() as sess:

      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value).astype(np.float64)
        x = array_ops.placeholder(dtypes.float64, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        multiplier = np.float64([2.])
        # One batch, scalar.
        # Corresponds to scale = 2, shift = 0.
        bijector = AffineScalar(scale=multiplier)
        x = np.float64([1.])  # One sample from one batches.
        self.assertAllClose([2.], run(bijector.forward, x))
        self.assertAllClose([0.5], run(bijector.inverse, x))
        self.assertAllClose([np.log(0.5)],
                            run(bijector.inverse_log_det_jacobian, x))

  def testTwoBatchScalarIdentityViaIdentity(self):
    with self.test_session() as sess:

      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = array_ops.placeholder(dtypes.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = [1., -1]
        # Univariate, two batches.
        # Corresponds to scale = 1.
        bijector = AffineScalar(shift=mu)
        x = [1., 1]  # One sample from each of two batches.
        self.assertAllClose([2., 0], run(bijector.forward, x))
        self.assertAllClose([0., 2], run(bijector.inverse, x))
        self.assertAllClose([0., 0.], run(bijector.inverse_log_det_jacobian, x))

  def testTwoBatchScalarIdentityViaScale(self):
    with self.test_session() as sess:

      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = array_ops.placeholder(dtypes.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = [1., -1]
        # Univariate, two batches.
        # Corresponds to scale = 1.
        bijector = AffineScalar(shift=mu, scale=[2., 1])
        x = [1., 1]  # One sample from each of two batches.
        self.assertAllClose([3., 0], run(bijector.forward, x))
        self.assertAllClose([0., 2], run(bijector.inverse, x))
        self.assertAllClose(
            [-np.log(2), 0.], run(bijector.inverse_log_det_jacobian, x))

  def testScalarCongruency(self):
    with self.test_session():
      bijector = AffineScalar(shift=3.6, scale=0.42)
      assert_scalar_congruency(bijector, lower_x=-2., upper_x=2.)

if __name__ == "__main__":
  test.main()
