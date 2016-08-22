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

import numpy as np
import tensorflow as tf

from tensorflow.contrib.distributions.python.ops.bijector import _Exp
from tensorflow.contrib.distributions.python.ops.bijector import _Identity
from tensorflow.contrib.distributions.python.ops.bijector import _ShiftAndScale


class IdentityBijectorTest(tf.test.TestCase):
  """Tests the correctness of the Y = g(X) = X transformation."""

  def testBijector(self):
    with self.test_session():
      bijector = _Identity()
      self.assertEqual("Identity", bijector.name)
      x = [[[0.],
            [1.]]]
      self.assertAllEqual(x, bijector.forward(x).eval())
      self.assertAllEqual(x, bijector.inverse(x).eval())
      self.assertAllEqual(0., bijector.inverse_log_det_jacobian(x).eval())
      rev, jac = bijector.inverse_and_inverse_log_det_jacobian(x)
      self.assertAllEqual(x, rev.eval())
      self.assertAllEqual(0., jac.eval())


class ExpBijectorTest(tf.test.TestCase):
  """Tests the correctness of the Y = g(X) = exp(X) transformation."""

  def testBijector(self):
    with self.test_session():
      bijector = _Exp(event_ndims=1)
      self.assertEqual("Exp", bijector.name)
      x = [[[1.],
            [2.]]]
      self.assertAllClose(np.exp(x), bijector.forward(x).eval())
      self.assertAllClose(np.log(x), bijector.inverse(x).eval())
      self.assertAllClose([[0., -math.log(2.)]],
                          bijector.inverse_log_det_jacobian(x).eval())
      rev, jac = bijector.inverse_and_inverse_log_det_jacobian(x)
      self.assertAllClose(np.log(x), rev.eval())
      self.assertAllClose([[0., -math.log(2.)]], jac.eval())


class _ShiftAndScaleBijectorTest(tf.test.TestCase):

  def testProperties(self):
    with self.test_session():
      mu = -1.
      sigma = 2.
      bijector = _ShiftAndScale(loc=mu, scale=sigma)
      self.assertEqual("ShiftAndScale", bijector.name)

  def testNoBatchScalar(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = -1.
        sigma = 2.  # Scalar.
        bijector = _ShiftAndScale(loc=mu, scale=sigma)
        self.assertEqual(0, bijector.shaper.batch_ndims.eval())  # "no batches"
        self.assertEqual(0, bijector.shaper.event_ndims.eval())  # "is scalar"
        x = [1., 2, 3]  # Three scalar samples (no batches).
        self.assertAllClose([1., 3, 5], run(bijector.forward, x))
        self.assertAllClose([1., 1.5, 2.], run(bijector.inverse, x))
        self.assertAllClose([-math.log(2.)],
                            run(bijector.inverse_log_det_jacobian, x))

  def testWeirdSampleNoBatchScalar(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = -1.
        sigma = 2.  # Scalar.
        bijector = _ShiftAndScale(loc=mu, scale=sigma)
        self.assertEqual(0, bijector.shaper.batch_ndims.eval())  # "no batches"
        self.assertEqual(0, bijector.shaper.event_ndims.eval())  # "is scalar"
        x = [[1., 2, 3],
             [4, 5, 6]]  # Weird sample shape.
        self.assertAllClose([[1., 3, 5],
                             [7, 9, 11]],
                            run(bijector.forward, x))
        self.assertAllClose([[1., 1.5, 2.],
                             [2.5, 3, 3.5]],
                            run(bijector.inverse, x))
        self.assertAllClose([-math.log(2.)],
                            run(bijector.inverse_log_det_jacobian, x))

  def testOneBatchScalar(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = [1.]
        sigma = [1.]  # One batch, scalar.
        bijector = _ShiftAndScale(loc=mu, scale=sigma)
        self.assertEqual(
            1, bijector.shaper.batch_ndims.eval())  # "one batch dim"
        self.assertEqual(
            0, bijector.shaper.event_ndims.eval())  # "is scalar"
        x = [1.]  # One sample from one batches.
        self.assertAllClose([2.], run(bijector.forward, x))
        self.assertAllClose([0.], run(bijector.inverse, x))
        self.assertAllClose([0.],
                            run(bijector.inverse_log_det_jacobian, x))

  def testTwoBatchScalar(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = [1., -1]
        sigma = [1., 1]  # Univariate, two batches.
        bijector = _ShiftAndScale(loc=mu, scale=sigma)
        self.assertEqual(
            1, bijector.shaper.batch_ndims.eval())  # "one batch dim"
        self.assertEqual(
            0, bijector.shaper.event_ndims.eval())  # "is scalar"
        x = [1., 1]  # One sample from each of two batches.
        self.assertAllClose([2., 0], run(bijector.forward, x))
        self.assertAllClose([0., 2], run(bijector.inverse, x))
        self.assertAllClose([0., 0],
                            run(bijector.inverse_log_det_jacobian, x))

  def testNoBatchMultivariate(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = [1., -1]
        sigma = np.eye(2, dtype=np.float32)
        bijector = _ShiftAndScale(loc=mu, scale=sigma, event_ndims=1)
        self.assertEqual(0, bijector.shaper.batch_ndims.eval())  # "no batches"
        self.assertEqual(1, bijector.shaper.event_ndims.eval())  # "is vector"
        x = [1., 1]
        self.assertAllClose([2., 0], run(bijector.forward, x))
        self.assertAllClose([0., 2], run(bijector.inverse, x))
        self.assertAllClose([0.], run(bijector.inverse_log_det_jacobian, x))

        x = [[1., 1],
             [-1., -1]]
        self.assertAllClose([[2., 0],
                             [0, -2]],
                            run(bijector.forward, x))
        self.assertAllClose([[0., 2],
                             [-2., 0]],
                            run(bijector.inverse, x))
        self.assertAllClose([0.], run(bijector.inverse_log_det_jacobian, x))

      # When mu is a scalar and x is multivariate then the location is
      # broadcast.
      for run in (static_run, dynamic_run):
        mu = 1.
        sigma = np.eye(2, dtype=np.float32)
        bijector = _ShiftAndScale(loc=mu, scale=sigma, event_ndims=1)
        self.assertEqual(0, bijector.shaper.batch_ndims.eval())  # "no batches"
        self.assertEqual(1, bijector.shaper.event_ndims.eval())  # "is vector"
        x = [1., 1]
        self.assertAllClose([2., 2], run(bijector.forward, x))
        self.assertAllClose([0., 0], run(bijector.inverse, x))
        self.assertAllClose([0.], run(bijector.inverse_log_det_jacobian, x))
        x = [[1., 1]]
        self.assertAllClose([[2., 2]], run(bijector.forward, x))
        self.assertAllClose([[0., 0]], run(bijector.inverse, x))
        self.assertAllClose([0.], run(bijector.inverse_log_det_jacobian, x))

  def testNoBatchMultivariateFullDynamic(self):
    with self.test_session() as sess:
      x = tf.placeholder(tf.float32, name="x")
      mu = tf.placeholder(tf.float32, name="mu")
      sigma = tf.placeholder(tf.float32, name="sigma")
      event_ndims = tf.placeholder(tf.int32, name="event_ndims")

      x_value = np.array([[1., 1]], dtype=np.float32)
      mu_value = np.array([1., -1], dtype=np.float32)
      sigma_value = np.eye(2, dtype=np.float32)
      event_ndims_value = np.array(1, dtype=np.int32)
      feed_dict = {x: x_value, mu: mu_value, sigma: sigma_value, event_ndims:
                   event_ndims_value}

      bijector = _ShiftAndScale(loc=mu, scale=sigma, event_ndims=event_ndims)
      self.assertEqual(0, sess.run(bijector.shaper.batch_ndims, feed_dict))
      self.assertEqual(1, sess.run(bijector.shaper.event_ndims, feed_dict))
      self.assertAllClose([[2., 0]], sess.run(bijector.forward(x), feed_dict))
      self.assertAllClose([[0., 2]], sess.run(bijector.inverse(x), feed_dict))
      self.assertAllClose(
          [0.], sess.run(bijector.inverse_log_det_jacobian(x), feed_dict))

  def testBatchMultivariate(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value, dtype=np.float32)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = [[1., -1]]
        sigma = np.array([np.eye(2, dtype=np.float32)])
        bijector = _ShiftAndScale(loc=mu, scale=sigma, event_ndims=1)
        self.assertEqual(
            1, bijector.shaper.batch_ndims.eval())  # "one batch dim"
        self.assertEqual(
            1, bijector.shaper.event_ndims.eval())  # "is vector"
        x = [[[1., 1]]]
        self.assertAllClose([[[2., 0]]], run(bijector.forward, x))
        self.assertAllClose([[[0., 2]]], run(bijector.inverse, x))
        self.assertAllClose([0.], run(bijector.inverse_log_det_jacobian, x))

  def testBatchMultivariateFullDynamic(self):
    with self.test_session() as sess:
      x = tf.placeholder(tf.float32, name="x")
      mu = tf.placeholder(tf.float32, name="mu")
      sigma = tf.placeholder(tf.float32, name="sigma")
      event_ndims = tf.placeholder(tf.int32, name="event_ndims")

      x_value = np.array([[[1., 1]]], dtype=np.float32)
      mu_value = np.array([[1., -1]], dtype=np.float32)
      sigma_value = np.array([np.eye(2, dtype=np.float32)])
      event_ndims_value = np.array(1, dtype=np.int32)
      feed_dict = {x: x_value, mu: mu_value, sigma: sigma_value,
                   event_ndims: event_ndims_value}

      bijector = _ShiftAndScale(loc=mu, scale=sigma, event_ndims=event_ndims)
      self.assertEqual(1, sess.run(bijector.shaper.batch_ndims, feed_dict))
      self.assertEqual(1, sess.run(bijector.shaper.event_ndims, feed_dict))
      self.assertAllClose([[[2., 0]]], sess.run(bijector.forward(x), feed_dict))
      self.assertAllClose([[[0., 2]]], sess.run(bijector.inverse(x), feed_dict))
      self.assertAllClose(
          [0.], sess.run(bijector.inverse_log_det_jacobian(x), feed_dict))


if __name__ == "__main__":
  tf.test.main()
