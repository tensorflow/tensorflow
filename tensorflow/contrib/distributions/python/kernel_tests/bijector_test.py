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

import abc
import itertools
import math

import numpy as np
import six
import tensorflow as tf

bijectors = tf.contrib.distributions.bijector
distributions = tf.contrib.distributions
linalg = tf.contrib.linalg
rng = np.random.RandomState(42)


def assert_finite(array):
  if not np.isfinite(array).all():
    raise AssertionError("array was not all finite. %s" % array[:15])


def assert_strictly_increasing(array):
  np.testing.assert_array_less(0.0, np.diff(array))


def assert_strictly_decreasing(array):
  np.testing.assert_array_less(np.diff(array), 0.0)


def assert_strictly_monotonic(array):
  if array[0] < array[-1]:
    assert_strictly_increasing(array)
  else:
    assert_strictly_decreasing(array)


def assert_scalar_congruency(
    bijector, lower_x, upper_x, n=10000, rtol=0.01, sess=None):
  """Assert `bijector`'s forward/inverse/inverse_log_det_jacobian are congruent.

  We draw samples `X ~ U(lower_x, upper_x)`, then feed these through the
  `bijector` in order to check that:

  1. the forward is strictly monotonic.
  2. the forward/inverse methods are inverses of each other.
  3. the jacobian is the correct change of measure.

  This can only be used for a Bijector mapping open subsets of the real line
  to themselves.  This is due to the fact that this test compares the pdf
  before/after transformation with the Lebesgue measure on the line.

  Args:
    bijector:  Instance of Bijector
    lower_x:  Python scalar.
    upper_x:  Python scalar.  Must have `lower_x < upper_x`, and both must be in
      the domain of the `bijector`.  The `bijector` should probably not produce
      huge variation in values in the interval `(lower_x, upper_x)`, or else
      the variance based check of the Jacobian will require small `rtol` or
      huge `n`.
    n:  Number of samples to draw for the checks.
    rtol:  Positive number.  Used for the Jacobian check.
    sess:  `tf.Session`.  Defaults to the default session.

  Raises:
    AssertionError:  If tests fail.
  """

  # Checks and defaults.
  assert bijector.shaper is None or bijector.shaper.event_ndims.eval() == 0
  if sess is None:
    sess = tf.get_default_session()

  # Should be monotonic over this interval
  ten_x_pts = np.linspace(lower_x, upper_x, num=10).astype(np.float32)
  if bijector.dtype is not None:
    ten_x_pts = ten_x_pts.astype(bijector.dtype.as_numpy_dtype)
  forward_on_10_pts = bijector.forward(ten_x_pts)

  # Set the lower/upper limits in the range of the bijector.
  lower_y, upper_y = sess.run(
      [bijector.forward(lower_x), bijector.forward(upper_x)])
  if upper_y < lower_y:  # If bijector.forward is a decreasing function.
    lower_y, upper_y = upper_y, lower_y

  # Uniform samples from the domain, range.
  uniform_x_samps = distributions.Uniform(a=lower_x, b=upper_x).sample(n)
  uniform_y_samps = distributions.Uniform(a=lower_y, b=upper_y).sample(n)

  # These compositions should be the identity.
  inverse_forward_x = bijector.inverse(bijector.forward(uniform_x_samps))
  forward_inverse_y = bijector.forward(bijector.inverse(uniform_y_samps))

  # For a < b, and transformation y = y(x),
  # (b - a) = \int_a^b dx = \int_{y(a)}^{y(b)} |dx/dy| dy
  # "change_measure_dy_dx" below is a Monte Carlo approximation to the right
  # hand side, which should then be close to the left, which is (b - a).
  dy_dx = tf.exp(bijector.inverse_log_det_jacobian(uniform_y_samps))
  # E[|dx/dy|] under Uniform[lower_y, upper_y]
  # = \int_{y(a)}^{y(b)} |dx/dy| dP(u), where dP(u) is the uniform measure
  expectation_of_dy_dx_under_uniform = tf.reduce_mean(dy_dx)
  # dy = dP(u) * (upper_y - lower_y)
  change_measure_dy_dx = ((upper_y - lower_y) *
                          expectation_of_dy_dx_under_uniform)

  # We'll also check that dy_dx = 1 / dx_dy.
  dx_dy = tf.exp(bijector.forward_log_det_jacobian(
      bijector.inverse(uniform_y_samps)))

  (
      forward_on_10_pts_v,
      dy_dx_v,
      dx_dy_v,
      change_measure_dy_dx_v,
      uniform_x_samps_v,
      uniform_y_samps_v,
      inverse_forward_x_v,
      forward_inverse_y_v,
  ) = sess.run(
      [
          forward_on_10_pts,
          dy_dx,
          dx_dy,
          change_measure_dy_dx,
          uniform_x_samps,
          uniform_y_samps,
          inverse_forward_x,
          forward_inverse_y,
      ])

  assert_strictly_monotonic(forward_on_10_pts_v)
  # Composition of forward/inverse should be the identity.
  np.testing.assert_allclose(
      inverse_forward_x_v, uniform_x_samps_v, atol=1e-5, rtol=1e-3)
  np.testing.assert_allclose(
      forward_inverse_y_v, uniform_y_samps_v, atol=1e-5, rtol=1e-3)
  # Change of measure should be correct.
  np.testing.assert_allclose(upper_x - lower_x,
                             change_measure_dy_dx_v,
                             atol=0, rtol=rtol)
  # Inverse Jacobian should be equivalent to the reciprocal of the forward
  # Jacobian.
  np.testing.assert_allclose(dy_dx_v, np.divide(1., dx_dy_v),
                             atol=1e-5, rtol=1e-3)


def assert_bijective_and_finite(bijector, x, y, atol=0, rtol=1e-5, sess=None):
  """Assert that forward/inverse (along with jacobians) are inverses and finite.

  It is recommended to use x and y values that are very very close to the edge
  of the Bijector's domain.

  Args:
    bijector:  A Bijector instance.
    x:  np.array of values in the domain of bijector.forward.
    y:  np.array of values in the domain of bijector.inverse.
    atol:  Absolute tolerance.
    rtol:  Relative tolerance.
    sess:  TensorFlow session.  Defaults to the default session.

  Raises:
    AssertionError:  If tests fail.
  """
  sess = sess or tf.get_default_session()

  # These are the incoming points, but people often create a crazy range of
  # values for which these end up being bad, especially in 16bit.
  assert_finite(x)
  assert_finite(y)
  np.testing.assert_array_less(0, y)

  f_x = bijector.forward(x)
  g_y = bijector.inverse(y)

  (
      x_from_x, y_from_y, ildj_f_x, fldj_x, ildj_y, fldj_g_y, f_x_v, g_y_v,
  ) = sess.run(
      [bijector.inverse(f_x),
       bijector.forward(g_y),
       bijector.inverse_log_det_jacobian(f_x),
       bijector.forward_log_det_jacobian(x),
       bijector.inverse_log_det_jacobian(y),
       bijector.forward_log_det_jacobian(g_y),
       f_x,
       g_y,
      ])

  assert_finite(x_from_x)
  assert_finite(y_from_y)
  assert_finite(ildj_f_x)
  assert_finite(fldj_x)
  assert_finite(ildj_y)
  assert_finite(fldj_g_y)
  assert_finite(f_x_v)
  assert_finite(g_y_v)

  np.testing.assert_allclose(x_from_x, x, atol=atol, rtol=rtol)
  np.testing.assert_allclose(y_from_y, y, atol=atol, rtol=rtol)
  np.testing.assert_allclose(-ildj_f_x, fldj_x, atol=atol, rtol=rtol)
  np.testing.assert_allclose(-ildj_y, fldj_g_y, atol=atol, rtol=rtol)


class BaseBijectorTest(tf.test.TestCase):
  """Tests properties of the Bijector base-class."""

  def testBijector(self):
    with self.test_session():
      with self.assertRaisesRegexp(
          TypeError,
          ("Can't instantiate abstract class Bijector "
           "with abstract methods __init__")):
        bijectors.Bijector()


class IntentionallyMissingError(Exception):
  pass


class BrokenBijectorWithInverseAndInverseLogDetJacobian(bijectors.Bijector):
  """Bijector with broken directions.

  This BrokenBijector implements _inverse_and_inverse_log_det_jacobian.
  """

  def __init__(self, forward_missing=False, inverse_missing=False):
    super(BrokenBijectorWithInverseAndInverseLogDetJacobian, self).__init__(
        batch_ndims=0,
        event_ndims=0,
        validate_args=False,
        name="BrokenBijectorDual")
    self._forward_missing = forward_missing
    self._inverse_missing = inverse_missing

  def _forward(self, x):
    if self._forward_missing:
      raise IntentionallyMissingError
    return 2. * x

  def _inverse_and_inverse_log_det_jacobian(self, y):
    if self._inverse_missing:
      raise IntentionallyMissingError
    return y / 2., -tf.log(2.)

  def _forward_log_det_jacobian(self, x):  # pylint:disable=unused-argument
    if self._forward_missing:
      raise IntentionallyMissingError
    return tf.log(2.)


class BrokenBijectorSeparateInverseAndInverseLogDetJacobian(bijectors.Bijector):
  """Forward and inverse are not inverses of each other.

  This BrokenBijector implements _inverse and _inverse_log_det_jacobian as
  separate functions.
  """

  def __init__(self, forward_missing=False, inverse_missing=False):
    super(BrokenBijectorSeparateInverseAndInverseLogDetJacobian, self).__init__(
        batch_ndims=0,
        event_ndims=0,
        validate_args=False,
        name="broken")
    self._forward_missing = forward_missing
    self._inverse_missing = inverse_missing

  def _forward(self, x):
    if self._forward_missing:
      raise IntentionallyMissingError
    return 2 * x

  def _inverse(self, y):
    if self._inverse_missing:
      raise IntentionallyMissingError
    return y / 2.

  def _inverse_log_det_jacobian(self, y):  # pylint:disable=unused-argument
    if self._inverse_missing:
      raise IntentionallyMissingError
    return -tf.log(2.)

  def _forward_log_det_jacobian(self, x):  # pylint:disable=unused-argument
    if self._forward_missing:
      raise IntentionallyMissingError
    return tf.log(2.)


@six.add_metaclass(abc.ABCMeta)
class BijectorCachingTest(object):

  @abc.abstractproperty
  def broken_bijector_cls(self):
    # return a BrokenBijector type Bijector, since this will test the caching.
    raise IntentionallyMissingError("Not implemented")

  def testCachingOfForwardResultsWhenCalledOneByOne(self):
    broken_bijector = self.broken_bijector_cls(inverse_missing=True)
    with self.test_session():
      x = tf.constant(1.1)

      # Call forward and forward_log_det_jacobian one-by-one (not together).
      y = broken_bijector.forward(x)
      _ = broken_bijector.forward_log_det_jacobian(x)

      # Now, everything should be cached if the argument is y.
      try:
        broken_bijector.inverse(y)
        broken_bijector.inverse_log_det_jacobian(y)
        broken_bijector.inverse_and_inverse_log_det_jacobian(y)
      except IntentionallyMissingError:
        raise AssertionError("Tests failed!  Cached values not used.")

  def testCachingOfInverseResultsWhenCalledOneByOne(self):
    broken_bijector = self.broken_bijector_cls(forward_missing=True)
    with self.test_session():
      y = tf.constant(1.1)

      # Call inverse and inverse_log_det_jacobian one-by-one (not together).
      x = broken_bijector.inverse(y)
      _ = broken_bijector.inverse_log_det_jacobian(y)

      # Now, everything should be cached if the argument is x.
      try:
        broken_bijector.forward(x)
        broken_bijector.forward_log_det_jacobian(x)
      except IntentionallyMissingError:
        raise AssertionError("Tests failed!  Cached values not used.")

  def testCachingOfInverseResultsWhenCalledTogether(self):
    broken_bijector = self.broken_bijector_cls(forward_missing=True)
    with self.test_session():
      y = tf.constant(1.1)

      # Call inverse and inverse_log_det_jacobian one-by-one (not together).
      x, _ = broken_bijector.inverse_and_inverse_log_det_jacobian(y)

      # Now, everything should be cached if the argument is x.
      try:
        broken_bijector.forward(x)
        broken_bijector.forward_log_det_jacobian(x)
      except IntentionallyMissingError:
        raise AssertionError("Tests failed!  Cached values not used.")


class SeparateCallsBijectorCachingTest(BijectorCachingTest, tf.test.TestCase):
  """Test caching with BrokenBijectorSeparateInverseAndInverseLogDetJacobian.

  These bijectors implement forward, inverse,... all as separate functions.
  """

  @property
  def broken_bijector_cls(self):
    return BrokenBijectorSeparateInverseAndInverseLogDetJacobian


class JointCallsBijectorCachingTest(BijectorCachingTest, tf.test.TestCase):
  """Test caching with BrokenBijectorWithInverseAndInverseLogDetJacobian.

  These bijectors implement _inverse_and_inverse_log_det_jacobian, which is two
  functionalities together.
  """

  @property
  def broken_bijector_cls(self):
    return BrokenBijectorWithInverseAndInverseLogDetJacobian


class IdentityBijectorTest(tf.test.TestCase):
  """Tests correctness of the Y = g(X) = X transformation."""

  def testBijector(self):
    with self.test_session():
      bijector = bijectors.Identity()
      self.assertEqual("identity", bijector.name)
      x = [[[0.],
            [1.]]]
      self.assertAllEqual(x, bijector.forward(x).eval())
      self.assertAllEqual(x, bijector.inverse(x).eval())
      self.assertAllEqual(0., bijector.inverse_log_det_jacobian(x).eval())
      self.assertAllEqual(0., bijector.forward_log_det_jacobian(x).eval())
      rev, jac = bijector.inverse_and_inverse_log_det_jacobian(x)
      self.assertAllEqual(x, rev.eval())
      self.assertAllEqual(0., jac.eval())

  def testScalarCongruency(self):
    with self.test_session():
      bijector = bijectors.Identity()
      assert_scalar_congruency(bijector, lower_x=-2., upper_x=2.)


class ExpBijectorTest(tf.test.TestCase):
  """Tests correctness of the Y = g(X) = exp(X) transformation."""

  def testBijector(self):
    with self.test_session():
      bijector = bijectors.Exp(event_ndims=1)
      self.assertEqual("exp", bijector.name)
      x = [[[1.],
            [2.]]]
      y = np.exp(x)
      self.assertAllClose(y, bijector.forward(x).eval())
      self.assertAllClose(x, bijector.inverse(y).eval())
      self.assertAllClose(-np.sum(np.log(y), axis=-1),
                          bijector.inverse_log_det_jacobian(y).eval())
      self.assertAllClose(-bijector.inverse_log_det_jacobian(np.exp(x)).eval(),
                          bijector.forward_log_det_jacobian(x).eval())
      rev, jac = bijector.inverse_and_inverse_log_det_jacobian(y)
      self.assertAllClose(x, rev.eval())
      self.assertAllClose(-np.sum(np.log(y), axis=-1), jac.eval())

  def testScalarCongruency(self):
    with self.test_session():
      bijector = bijectors.Exp()
      assert_scalar_congruency(bijector, lower_x=-2., upper_x=1.5, rtol=0.05)

  def testBijectiveAndFinite(self):
    with self.test_session():
      bijector = bijectors.Exp(event_ndims=0)
      x = np.linspace(-10, 10, num=10).astype(np.float32)
      y = np.logspace(-10, 10, num=10).astype(np.float32)
      assert_bijective_and_finite(bijector, x, y)


class PowerTransformBijectorTest(tf.test.TestCase):
  """Tests correctness of the power transformation."""

  def testBijector(self):
    with self.test_session():
      c = 0.2
      bijector = bijectors.PowerTransform(power=c, event_ndims=1,
                                          validate_args=True)
      self.assertEqual("power_transform", bijector.name)
      x = np.array([[[-1.],
                     [2.],
                     [-5.+1e-4]]])
      y = (1. + x * c)**(1. / c)
      self.assertAllClose(y, bijector.forward(x).eval())
      self.assertAllClose(x, bijector.inverse(y).eval())
      self.assertAllClose((c - 1.) * np.sum(np.log(y), axis=-1),
                          bijector.inverse_log_det_jacobian(y).eval())
      self.assertAllClose(-bijector.inverse_log_det_jacobian(y).eval(),
                          bijector.forward_log_det_jacobian(x).eval(),
                          rtol=1e-4, atol=0.)
      rev, jac = bijector.inverse_and_inverse_log_det_jacobian(y)
      self.assertAllClose(x, rev.eval())
      self.assertAllClose((c - 1.) * np.sum(np.log(y), axis=-1), jac.eval())

  def testScalarCongruency(self):
    with self.test_session():
      bijector = bijectors.PowerTransform(power=0.2, validate_args=True)
      assert_scalar_congruency(bijector, lower_x=-2., upper_x=1.5, rtol=0.05)

  def testBijectiveAndFinite(self):
    with self.test_session():
      bijector = bijectors.PowerTransform(power=0.2, event_ndims=0,
                                          validate_args=True)
      x = np.linspace(-4.999, 10, num=10).astype(np.float32)
      y = np.logspace(0.001, 10, num=10).astype(np.float32)
      assert_bijective_and_finite(bijector, x, y, rtol=1e-3)


class InlineBijectorTest(tf.test.TestCase):
  """Tests correctness of the inline constructed bijector."""

  def testBijector(self):
    with self.test_session():
      exp = bijectors.Exp(event_ndims=1)
      inline = bijectors.Inline(
          forward_fn=tf.exp,
          inverse_fn=tf.log,
          inverse_log_det_jacobian_fn=(
              lambda y: -tf.reduce_sum(tf.log(y), reduction_indices=-1)),
          forward_log_det_jacobian_fn=(
              lambda x: tf.reduce_sum(x, reduction_indices=-1)),
          name="exp")

      self.assertEqual(exp.name, inline.name)
      x = [[[1., 2.],
            [3., 4.],
            [5., 6.]]]
      y = np.exp(x)
      self.assertAllClose(y, inline.forward(x).eval())
      self.assertAllClose(x, inline.inverse(y).eval())
      self.assertAllClose(-np.sum(np.log(y), axis=-1),
                          inline.inverse_log_det_jacobian(y).eval())
      self.assertAllClose(-inline.inverse_log_det_jacobian(y).eval(),
                          inline.forward_log_det_jacobian(x).eval())
      rev, jac = inline.inverse_and_inverse_log_det_jacobian(y)
      self.assertAllClose(x, rev.eval())
      self.assertAllClose(-np.sum(np.log(y), axis=-1), jac.eval())

  def testShapeGetters(self):
    with self.test_session():
      bijector = bijectors.Inline(
          forward_event_shape_fn=lambda x: tf.concat_v2((x, [1]), 0),
          get_forward_event_shape_fn=lambda x: x.as_list() + [1],
          inverse_event_shape_fn=lambda x: x[:-1],
          get_inverse_event_shape_fn=lambda x: x[:-1],
          name="shape_only")
      x = tf.TensorShape([1, 2, 3])
      y = tf.TensorShape([1, 2, 3, 1])
      self.assertAllEqual(y, bijector.get_forward_event_shape(x))
      self.assertAllEqual(y.as_list(),
                          bijector.forward_event_shape(x.as_list()).eval())
      self.assertAllEqual(x, bijector.get_inverse_event_shape(y))
      self.assertAllEqual(x.as_list(),
                          bijector.inverse_event_shape(y.as_list()).eval())


class AffineLinearOperatorTest(tf.test.TestCase):

  def testIdentity(self):
    with self.test_session():
      affine = bijectors.AffineLinearOperator(validate_args=True)
      x = np.array([[1, 0, -1],
                    [2, 3, 4]], dtype=np.float32)
      y = x
      ildj = 0.

      self.assertEqual(affine.name, "affine_linear_operator")
      self.assertAllClose(y, affine.forward(x).eval())
      self.assertAllClose(x, affine.inverse(y).eval())
      self.assertAllClose(ildj, affine.inverse_log_det_jacobian(y).eval())
      self.assertAllClose(-affine.inverse_log_det_jacobian(y).eval(),
                          affine.forward_log_det_jacobian(x).eval())
      rev, actual_ildj = affine.inverse_and_inverse_log_det_jacobian(y)
      self.assertAllClose(x, rev.eval())
      self.assertAllClose(ildj, actual_ildj.eval())

  def testDiag(self):
    with self.test_session():
      shift = np.array([-1, 0, 1], dtype=np.float32)
      diag = np.array([[1, 2, 3],
                       [2, 5, 6]], dtype=np.float32)
      scale = linalg.LinearOperatorDiag(diag, is_non_singular=True)
      affine = bijectors.AffineLinearOperator(shift=shift, scale=scale,
                                              validate_args=True)

      x = np.array([[1, 0, -1],
                    [2, 3, 4]], dtype=np.float32)
      y = diag * x + shift
      ildj = -np.sum(np.log(np.abs(diag)), axis=-1)

      self.assertEqual(affine.name, "affine_linear_operator")
      self.assertAllClose(y, affine.forward(x).eval())
      self.assertAllClose(x, affine.inverse(y).eval())
      self.assertAllClose(ildj, affine.inverse_log_det_jacobian(y).eval())
      self.assertAllClose(-affine.inverse_log_det_jacobian(y).eval(),
                          affine.forward_log_det_jacobian(x).eval())
      rev, actual_ildj = affine.inverse_and_inverse_log_det_jacobian(y)
      self.assertAllClose(x, rev.eval())
      self.assertAllClose(ildj, actual_ildj.eval())

  def testTriL(self):
    with self.test_session():
      shift = np.array([-1, 0, 1], dtype=np.float32)
      tril = np.array([[[1, 0, 0],
                        [2, -1, 0],
                        [3, 2, 1]],
                       [[2, 0, 0],
                        [3, -2, 0],
                        [4, 3, 2]]], dtype=np.float32)
      scale = linalg.LinearOperatorTriL(tril, is_non_singular=True)
      affine = bijectors.AffineLinearOperator(shift=shift, scale=scale,
                                              validate_args=True)

      x = np.array([[[1, 0, -1],
                     [2, 3, 4]],
                    [[4, 1, -7],
                     [6, 9, 8]]], dtype=np.float32)
      # If we made the bijector do x*A+b then this would be simplified to:
      # y = np.matmul(x, tril) + shift.
      y = np.squeeze(np.matmul(tril, np.expand_dims(x, -1)), -1) + shift
      ildj = -np.sum(np.log(np.abs(np.diagonal(tril, axis1=-2, axis2=-1))),
                     axis=-1)

      self.assertEqual(affine.name, "affine_linear_operator")
      self.assertAllClose(y, affine.forward(x).eval())
      self.assertAllClose(x, affine.inverse(y).eval())
      self.assertAllClose(ildj, affine.inverse_log_det_jacobian(y).eval())
      self.assertAllClose(-affine.inverse_log_det_jacobian(y).eval(),
                          affine.forward_log_det_jacobian(x).eval())
      rev, actual_ildj = affine.inverse_and_inverse_log_det_jacobian(y)
      self.assertAllClose(x, rev.eval())
      self.assertAllClose(ildj, actual_ildj.eval())


class AffineBijectorTest(tf.test.TestCase):
  """Tests correctness of the Y = scale @ x + shift transformation."""

  def testProperties(self):
    with self.test_session():
      mu = -1.
      # scale corresponds to 1.
      bijector = bijectors.Affine(shift=mu, event_ndims=0)
      self.assertEqual("affine", bijector.name)

  def testNoBatchScalarViaIdentity(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = -1.
        # Corresponds to scale = 2
        bijector = bijectors.Affine(
            shift=mu, scale_identity_multiplier=2., event_ndims=0)
        self.assertEqual(0, bijector.shaper.event_ndims.eval())  # "is scalar"
        x = [1., 2, 3]  # Three scalar samples (no batches).
        self.assertAllClose([1., 3, 5], run(bijector.forward, x))
        self.assertAllClose([1., 1.5, 2.], run(bijector.inverse, x))
        self.assertAllClose(-math.log(2.),
                            run(bijector.inverse_log_det_jacobian, x))

  def testNoBatchScalarViaDiag(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = -1.
        # Corresponds to scale = 2
        bijector = bijectors.Affine(shift=mu, scale_diag=[2.], event_ndims=0)
        self.assertEqual(0, bijector.shaper.event_ndims.eval())  # "is scalar"
        x = [1., 2, 3]  # Three scalar samples (no batches).
        self.assertAllClose([1., 3, 5], run(bijector.forward, x))
        self.assertAllClose([1., 1.5, 2.], run(bijector.inverse, x))
        self.assertAllClose(-math.log(2.),
                            run(bijector.inverse_log_det_jacobian, x))

  def testWeirdSampleNoBatchScalarViaIdentity(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = -1.
        # Corresponds to scale = 2.
        bijector = bijectors.Affine(shift=mu,
                                    scale_identity_multiplier=2.,
                                    event_ndims=0)
        self.assertEqual(0, bijector.shaper.event_ndims.eval())  # "is scalar"
        x = [[1., 2, 3],
             [4, 5, 6]]  # Weird sample shape.
        self.assertAllClose([[1., 3, 5],
                             [7, 9, 11]],
                            run(bijector.forward, x))
        self.assertAllClose([[1., 1.5, 2.],
                             [2.5, 3, 3.5]],
                            run(bijector.inverse, x))
        self.assertAllClose(-math.log(2.),
                            run(bijector.inverse_log_det_jacobian, x))

  def testOneBatchScalarViaIdentity(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = [1.]
        # One batch, scalar.
        # Corresponds to scale = 1.
        bijector = bijectors.Affine(shift=mu, event_ndims=0)
        self.assertEqual(
            0, bijector.shaper.event_ndims.eval())  # "is scalar"
        x = [1.]  # One sample from one batches.
        self.assertAllClose([2.], run(bijector.forward, x))
        self.assertAllClose([0.], run(bijector.inverse, x))
        self.assertAllClose(0.,
                            run(bijector.inverse_log_det_jacobian, x))

  def testOneBatchScalarViaDiag(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = [1.]
        # One batch, scalar.
        # Corresponds to scale = 1.
        bijector = bijectors.Affine(shift=mu, scale_diag=[1.], event_ndims=0)
        self.assertEqual(
            0, bijector.shaper.event_ndims.eval())  # "is scalar"
        x = [1.]  # One sample from one batches.
        self.assertAllClose([2.], run(bijector.forward, x))
        self.assertAllClose([0.], run(bijector.inverse, x))
        self.assertAllClose(0.,
                            run(bijector.inverse_log_det_jacobian, x))

  def testTwoBatchScalarIdentityViaIdentity(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = [1., -1]
        # Univariate, two batches.
        # Corresponds to scale = 1.
        bijector = bijectors.Affine(shift=mu, event_ndims=0)
        self.assertEqual(
            0, bijector.shaper.event_ndims.eval())  # "is scalar"
        x = [1., 1]  # One sample from each of two batches.
        self.assertAllClose([2., 0], run(bijector.forward, x))
        self.assertAllClose([0., 2], run(bijector.inverse, x))
        self.assertAllClose(0.,
                            run(bijector.inverse_log_det_jacobian, x))

  def testTwoBatchScalarIdentityViaDiag(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = [1., -1]
        # Univariate, two batches.
        # Corresponds to scale = 1.
        bijector = bijectors.Affine(shift=mu, scale_diag=[1.], event_ndims=0)
        self.assertEqual(
            0, bijector.shaper.event_ndims.eval())  # "is scalar"
        x = [1., 1]  # One sample from each of two batches.
        self.assertAllClose([2., 0], run(bijector.forward, x))
        self.assertAllClose([0., 2], run(bijector.inverse, x))
        self.assertAllClose(0.,
                            run(bijector.inverse_log_det_jacobian, x))

  def testNoBatchMultivariateIdentity(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = [1., -1]
        # Multivariate
        # Corresponds to scale = [[1., 0], [0, 1.]]
        bijector = bijectors.Affine(shift=mu)
        self.assertEqual(1, bijector.shaper.event_ndims.eval())  # "is vector"
        x = [1., 1]
        # matmul(sigma, x) + shift
        # = [-1, -1] + [1, -1]
        self.assertAllClose([2., 0], run(bijector.forward, x))
        self.assertAllClose([0., 2], run(bijector.inverse, x))

        # x is a 2-batch of 2-vectors.
        # The first vector is [1, 1], the second is [-1, -1].
        # Each undergoes matmul(sigma, x) + shift.
        x = [[1., 1],
             [-1., -1]]
        self.assertAllClose([[2., 0],
                             [0., -2]],
                            run(bijector.forward, x))
        self.assertAllClose([[0., 2],
                             [-2., 0]],
                            run(bijector.inverse, x))
        self.assertAllClose(0., run(bijector.inverse_log_det_jacobian, x))

  def testNoBatchMultivariateDiag(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = [1., -1]
        # Multivariate
        # Corresponds to scale = [[2., 0], [0, 1.]]
        bijector = bijectors.Affine(shift=mu, scale_diag=[2., 1])
        self.assertEqual(1, bijector.shaper.event_ndims.eval())  # "is vector"
        x = [1., 1]
        # matmul(sigma, x) + shift
        # = [-1, -1] + [1, -1]
        self.assertAllClose([3., 0], run(bijector.forward, x))
        self.assertAllClose([0., 2], run(bijector.inverse, x))
        self.assertAllClose(
            -math.log(2.), run(bijector.inverse_log_det_jacobian, x))

        # x is a 2-batch of 2-vectors.
        # The first vector is [1, 1], the second is [-1, -1].
        # Each undergoes matmul(sigma, x) + shift.
        x = [[1., 1],
             [-1., -1]]
        self.assertAllClose([[3., 0],
                             [-1., -2]],
                            run(bijector.forward, x))
        self.assertAllClose([[0., 2],
                             [-1., 0]],
                            run(bijector.inverse, x))
        self.assertAllClose(-math.log(2.),
                            run(bijector.inverse_log_det_jacobian, x))

  def testNoBatchMultivariateFullDynamic(self):
    with self.test_session() as sess:
      x = tf.placeholder(tf.float32, name="x")
      mu = tf.placeholder(tf.float32, name="mu")
      scale_diag = tf.placeholder(tf.float32, name="scale_diag")
      event_ndims = tf.placeholder(tf.int32, name="event_ndims")

      x_value = np.array([[1., 1]], dtype=np.float32)
      mu_value = np.array([1., -1], dtype=np.float32)
      scale_diag_value = np.array([2., 2], dtype=np.float32)
      event_ndims_value = np.array(1, dtype=np.int32)
      feed_dict = {x: x_value, mu: mu_value, scale_diag: scale_diag_value,
                   event_ndims: event_ndims_value}

      bijector = bijectors.Affine(
          shift=mu,
          scale_diag=scale_diag,
          event_ndims=event_ndims)
      self.assertEqual(1, sess.run(bijector.shaper.event_ndims, feed_dict))
      self.assertAllClose([[3., 1]], sess.run(bijector.forward(x), feed_dict))
      self.assertAllClose([[0., 1]], sess.run(bijector.inverse(x), feed_dict))
      self.assertAllClose(
          -math.log(4),
          sess.run(bijector.inverse_log_det_jacobian(x), feed_dict))

  def testBatchMultivariateIdentity(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value, dtype=np.float32)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = [[1., -1]]
        # Corresponds to 1 2x2 matrix, with twos on the diagonal.
        scale = 2.
        bijector = bijectors.Affine(shift=mu, scale_identity_multiplier=scale)
        self.assertEqual(
            1, bijector.shaper.event_ndims.eval())  # "is vector"
        x = [[[1., 1]]]
        self.assertAllClose([[[3., 1]]], run(bijector.forward, x))
        self.assertAllClose([[[0., 1]]], run(bijector.inverse, x))
        self.assertAllClose(
            -math.log(4), run(bijector.inverse_log_det_jacobian, x))

  def testBatchMultivariateDiag(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value, dtype=np.float32)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = [[1., -1]]
        # Corresponds to 1 2x2 matrix, with twos on the diagonal.
        scale_diag = [[2., 2]]
        bijector = bijectors.Affine(shift=mu, scale_diag=scale_diag)
        self.assertEqual(
            1, bijector.shaper.event_ndims.eval())  # "is vector"
        x = [[[1., 1]]]
        self.assertAllClose([[[3., 1]]], run(bijector.forward, x))
        self.assertAllClose([[[0., 1]]], run(bijector.inverse, x))
        self.assertAllClose(
            [-math.log(4)], run(bijector.inverse_log_det_jacobian, x))

  def testBatchMultivariateFullDynamic(self):
    with self.test_session() as sess:
      x = tf.placeholder(tf.float32, name="x")
      mu = tf.placeholder(tf.float32, name="mu")
      scale_diag = tf.placeholder(tf.float32, name="scale_diag")
      event_ndims = tf.placeholder(tf.int32, name="event_ndims")

      x_value = np.array([[[1., 1]]], dtype=np.float32)
      mu_value = np.array([[1., -1]], dtype=np.float32)
      scale_diag_value = np.array([[2., 2]], dtype=np.float32)
      event_ndims_value = 1

      feed_dict = {x: x_value, mu: mu_value, scale_diag:
                   scale_diag_value, event_ndims: event_ndims_value}

      bijector = bijectors.Affine(shift=mu, scale_diag=scale_diag,
                                  event_ndims=event_ndims)
      self.assertEqual(1, sess.run(bijector.shaper.event_ndims, feed_dict))
      self.assertAllClose([[[3., 1]]], sess.run(bijector.forward(x), feed_dict))
      self.assertAllClose([[[0., 1]]], sess.run(bijector.inverse(x), feed_dict))
      self.assertAllClose([-math.log(4)],
                          sess.run(
                              bijector.inverse_log_det_jacobian(x), feed_dict))

  def testIdentityWithDiagUpdate(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = -1.
        # Corresponds to scale = 2
        bijector = bijectors.Affine(
            shift=mu,
            scale_identity_multiplier=1.,
            scale_diag=[1.],
            event_ndims=0)
        self.assertEqual(0, bijector.shaper.event_ndims.eval())  # "is vector"
        x = [1., 2, 3]  # Three scalar samples (no batches).
        self.assertAllClose([1., 3, 5], run(bijector.forward, x))
        self.assertAllClose([1., 1.5, 2.], run(bijector.inverse, x))
        self.assertAllClose(-math.log(2.),
                            run(bijector.inverse_log_det_jacobian, x))

  def testIdentityWithTriL(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = -1.
        # scale = [[2., 0], [2, 2]]
        bijector = bijectors.Affine(
            shift=mu,
            scale_identity_multiplier=1.,
            scale_tril=[[1., 0], [2., 1]])
        self.assertEqual(1, bijector.shaper.event_ndims.eval())  # "is vector"
        x = [[1., 2]]  # One multivariate sample.
        self.assertAllClose([[1., 5]], run(bijector.forward, x))
        self.assertAllClose([[1., 0.5]], run(bijector.inverse, x))
        self.assertAllClose(-math.log(4.),
                            run(bijector.inverse_log_det_jacobian, x))

  def testDiagWithTriL(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = -1.
        # scale = [[2., 0], [2, 3]]
        bijector = bijectors.Affine(
            shift=mu,
            scale_diag=[1., 2.],
            scale_tril=[[1., 0], [2., 1]])
        self.assertEqual(1, bijector.shaper.event_ndims.eval())  # "is vector"
        x = [[1., 2]]  # One multivariate sample.
        self.assertAllClose([[1., 7]], run(bijector.forward, x))
        self.assertAllClose([[1., 1 / 3.]], run(bijector.inverse, x))
        self.assertAllClose(-math.log(6.),
                            run(bijector.inverse_log_det_jacobian, x))

  def testIdentityAndDiagWithTriL(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = -1.
        # scale = [[3., 0], [2, 4]]
        bijector = bijectors.Affine(
            shift=mu,
            scale_identity_multiplier=1.0,
            scale_diag=[1., 2.],
            scale_tril=[[1., 0], [2., 1]])
        self.assertEqual(1, bijector.shaper.event_ndims.eval())  # "is vector"
        x = [[1., 2]]  # One multivariate sample.
        self.assertAllClose([[2., 9]], run(bijector.forward, x))
        self.assertAllClose([[2 / 3., 5 / 12.]], run(bijector.inverse, x))
        self.assertAllClose(-math.log(12.),
                            run(bijector.inverse_log_det_jacobian, x))

  def testIdentityWithVDVTUpdate(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = -1.
        # Corresponds to scale = [[10, 0, 0], [0, 2, 0], [0, 0, 3]]
        bijector = bijectors.Affine(
            shift=mu,
            scale_identity_multiplier=2.,
            scale_perturb_diag=[2., 1],
            scale_perturb_factor=[[2., 0],
                                  [0., 0],
                                  [0, 1]])
        bijector_ref = bijectors.Affine(shift=mu, scale_diag=[10., 2, 3])

        self.assertEqual(1, bijector.shaper.event_ndims.eval())  # "is vector"
        x = [1., 2, 3]  # Vector.
        self.assertAllClose([9., 3, 8], run(bijector.forward, x))
        self.assertAllClose(
            run(bijector_ref.forward, x), run(bijector.forward, x))

        self.assertAllClose([0.2, 1.5, 4/3.], run(bijector.inverse, x))
        self.assertAllClose(
            run(bijector_ref.inverse, x), run(bijector.inverse, x))
        self.assertAllClose(-math.log(60.),
                            run(bijector.inverse_log_det_jacobian, x))
        self.assertAllClose(
            run(bijector.inverse_log_det_jacobian, x),
            run(bijector_ref.inverse_log_det_jacobian, x))

  def testDiagWithVDVTUpdate(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = -1.
        # Corresponds to scale = [[10, 0, 0], [0, 3, 0], [0, 0, 5]]
        bijector = bijectors.Affine(
            shift=mu,
            scale_diag=[2., 3, 4],
            scale_perturb_diag=[2., 1],
            scale_perturb_factor=[[2., 0],
                                  [0., 0],
                                  [0, 1]])
        bijector_ref = bijectors.Affine(shift=mu, scale_diag=[10., 3, 5])

        self.assertEqual(1, bijector.shaper.event_ndims.eval())  # "is vector"
        x = [1., 2, 3]  # Vector.
        self.assertAllClose([9., 5, 14], run(bijector.forward, x))
        self.assertAllClose(
            run(bijector_ref.forward, x), run(bijector.forward, x))
        self.assertAllClose([0.2, 1., 0.8], run(bijector.inverse, x))
        self.assertAllClose(
            run(bijector_ref.inverse, x), run(bijector.inverse, x))
        self.assertAllClose(-math.log(150.),
                            run(bijector.inverse_log_det_jacobian, x))
        self.assertAllClose(
            run(bijector.inverse_log_det_jacobian, x),
            run(bijector_ref.inverse_log_det_jacobian, x))

  def testTriLWithVDVTUpdate(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = -1.
        # Corresponds to scale = [[10, 0, 0], [1, 3, 0], [2, 3, 5]]
        bijector = bijectors.Affine(
            shift=mu,
            scale_tril=[[2., 0, 0],
                        [1, 3, 0],
                        [2, 3, 4]],
            scale_perturb_diag=[2., 1],
            scale_perturb_factor=[[2., 0],
                                  [0., 0],
                                  [0, 1]])
        bijector_ref = bijectors.Affine(
            shift=mu,
            scale_tril=[[10., 0, 0],
                        [1, 3, 0],
                        [2, 3, 5]])

        self.assertEqual(1, bijector.shaper.event_ndims.eval())  # "is vector"
        x = [1., 2, 3]  # Vector.
        self.assertAllClose([9., 6, 22], run(bijector.forward, x))
        self.assertAllClose(
            run(bijector_ref.forward, x), run(bijector.forward, x))
        self.assertAllClose([0.2, 14 / 15., 4 / 25.], run(bijector.inverse, x))
        self.assertAllClose(
            run(bijector_ref.inverse, x), run(bijector.inverse, x))
        self.assertAllClose(-math.log(150.),
                            run(bijector.inverse_log_det_jacobian, x))
        self.assertAllClose(
            run(bijector.inverse_log_det_jacobian, x),
            run(bijector_ref.inverse_log_det_jacobian, x))

  def testTriLWithVDVTUpdateNoDiagonal(self):
    with self.test_session() as sess:
      def static_run(fun, x):
        return fun(x).eval()

      def dynamic_run(fun, x_value):
        x_value = np.array(x_value)
        x = tf.placeholder(tf.float32, name="x")
        return sess.run(fun(x), feed_dict={x: x_value})

      for run in (static_run, dynamic_run):
        mu = -1.
        # Corresponds to scale = [[6, 0, 0], [1, 3, 0], [2, 3, 5]]
        bijector = bijectors.Affine(
            shift=mu,
            scale_tril=[[2., 0, 0],
                        [1, 3, 0],
                        [2, 3, 4]],
            scale_perturb_diag=None,
            scale_perturb_factor=[[2., 0],
                                  [0., 0],
                                  [0, 1]])
        bijector_ref = bijectors.Affine(
            shift=mu,
            scale_tril=[[6., 0, 0],
                        [1, 3, 0],
                        [2, 3, 5]])

        self.assertEqual(1, bijector.shaper.event_ndims.eval())  # "is vector"
        x = [1., 2, 3]  # Vector.
        self.assertAllClose([5., 6, 22], run(bijector.forward, x))
        self.assertAllClose(
            run(bijector_ref.forward, x), run(bijector.forward, x))
        self.assertAllClose([1 / 3., 8 / 9., 4 / 30.], run(bijector.inverse, x))
        self.assertAllClose(
            run(bijector_ref.inverse, x), run(bijector.inverse, x))
        self.assertAllClose(-math.log(90.),
                            run(bijector.inverse_log_det_jacobian, x))
        self.assertAllClose(
            run(bijector.inverse_log_det_jacobian, x),
            run(bijector_ref.inverse_log_det_jacobian, x))

  def testNoBatchMultivariateRaisesWhenSingular(self):
    with self.test_session():
      mu = [1., -1]
      bijector = bijectors.Affine(
          shift=mu,
          # Has zero on the diagonal.
          scale_diag=[0., 1],
          validate_args=True)
      with self.assertRaisesOpError("Condition x > 0"):
        bijector.forward([1., 1.]).eval()

  def testEventNdimsLargerThanOneRaises(self):
    with self.test_session():
      mu = [1., -1]
      # Scale corresponds to 2x2 identity matrix.
      bijector = bijectors.Affine(
          shift=mu, event_ndims=2, validate_args=True)
      bijector.forward([1., 1.]).eval()

  def testScaleZeroScalarRaises(self):
    with self.test_session():
      mu = -1.
      # Check Identity matrix with zero scaling.
      bijector = bijectors.Affine(
          shift=mu,
          scale_identity_multiplier=0.0,
          event_ndims=0,
          validate_args=True)
      with self.assertRaisesOpError("Condition x > 0"):
        bijector.forward(1.).eval()

      # Check Diag matrix with zero scaling.
      bijector = bijectors.Affine(
          shift=mu,
          scale_diag=[0.0],
          event_ndims=0,
          validate_args=True)
      with self.assertRaisesOpError("Condition x > 0"):
        bijector.forward(1.).eval()

  def testScalarCongruency(self):
    with self.test_session():
      bijector = bijectors.Affine(
          shift=3.6,
          scale_identity_multiplier=0.42,
          event_ndims=0)
      assert_scalar_congruency(bijector, lower_x=-2., upper_x=2.)

  def _makeScale(
      self,
      x,
      scale_identity_multiplier=None,
      scale_diag=None,
      scale_tril=None,
      scale_perturb_factor=None,
      scale_perturb_diag=None):
    """Create a scale matrix. Return None if it can not be created."""
    c = scale_identity_multiplier
    d1 = scale_diag
    tril = scale_tril
    v = scale_perturb_factor
    d2 = scale_perturb_diag

    # Ambiguous low rank update.
    if v is None and d2 is not None:
      return None

    if c is None and d1 is None and tril is None:
      # Special case when no scale args are passed in. This means use an
      # identity matrix.
      if v is None and d2 is None:
        c = 1.
      # No scale.
      else:
        return None

    matrix = np.float32(0.)
    if c is not None:
      # Infer the dimension from x.
      matrix += c * self._matrix_diag(np.ones_like(x))
    if d1 is not None:
      matrix += self._matrix_diag(np.array(d1, dtype=np.float32))
    if tril is not None:
      matrix += np.array(tril, dtype=np.float32)
    if v is not None:
      v = np.array(v, dtype=np.float32)
      if v.ndim < 2:
        vt = v.T
      else:
        vt = np.swapaxes(v, axis1=v.ndim - 2, axis2=v.ndim - 1)
      if d2 is not None:
        d2 = self._matrix_diag(np.array(d2, dtype=np.float32))
        right = np.matmul(d2, vt)
      else:
        right = vt
      matrix += np.matmul(v, right)
    return matrix

  def _matrix_diag(self, d):
    """Batch version of np.diag."""
    orig_shape = d.shape
    d = np.reshape(d, (np.prod(d.shape[:-1]), d.shape[-1]))
    diag_list = []
    for i in range(d.shape[0]):
      diag_list.append(np.diag(d[i, ...]))
    return np.reshape(diag_list, orig_shape + (d.shape[-1],))

  def _testLegalInputs(self, shift=None, scale_params=None, x=None):
    def _powerset(x):
      s = list(x)
      return itertools.chain.from_iterable(
          itertools.combinations(s, r) for r in range(len(s) + 1))

    with self.test_session():
      for args in _powerset(scale_params.items()):
        args = dict(args)

        scale_args = dict({"x": x}, **args)
        scale = self._makeScale(**scale_args)

        bijector_args = dict({"event_ndims": 1}, **args)

        # We haven't specified enough information for the scale.
        if scale is None:
          with self.assertRaisesRegexp(ValueError, ("must be specified.")):
            bijector = bijectors.Affine(shift=shift, **bijector_args)
        else:
          bijector = bijectors.Affine(shift=shift, **bijector_args)
          np_x = x
          # For the case a vector is passed in, we need to make the shape
          # match the matrix for matmul to work.
          if x.ndim == scale.ndim - 1:
            np_x = np.expand_dims(x, axis=-1)

          forward = np.matmul(scale, np_x) + shift
          if x.ndim == scale.ndim - 1:
            forward = np.squeeze(forward, axis=-1)
          self.assertAllClose(forward, bijector.forward(x).eval())

          backward = np.linalg.solve(scale, np_x - shift)
          if x.ndim == scale.ndim - 1:
            backward = np.squeeze(backward, axis=-1)
          self.assertAllClose(backward, bijector.inverse(x).eval())

          ildj = -np.log(np.abs(np.linalg.det(scale)))
          # TODO(jvdillon): We need to make it so the scale_identity_multiplier
          # case does not deviate in expected shape. Fixing this will get rid of
          # these special cases.
          if (ildj.ndim > 0 and (
              len(scale_args) == 1 or
              (len(scale_args) == 2 and
               scale_args.get("scale_identity_multiplier", None) is not None))):
            ildj = np.squeeze(ildj[0])
          elif ildj.ndim < scale.ndim - 2:
            ildj = np.reshape(ildj, scale.shape[0:-2])
          self.assertAllClose(
              ildj, bijector.inverse_log_det_jacobian(x).eval())

  def testLegalInputs(self):
    self._testLegalInputs(
        shift=np.float32(-1),
        scale_params={
            "scale_identity_multiplier": 2.,
            "scale_diag": [2., 3.],
            "scale_tril": [[1., 0.], [-3., 3.]],
            "scale_perturb_factor": [[1., 0], [1.5, 3.]],
            "scale_perturb_diag": [3., 1.]},
        x=np.array([1., 2], dtype=np.float32))

  def testLegalInputsWithBatch(self):
    # Shape of scale is [2, 1, 2, 2]
    self._testLegalInputs(
        shift=np.float32(-1),
        scale_params={
            "scale_identity_multiplier": 2.,
            "scale_diag": [[[2., 3.]], [[1., 2]]],
            "scale_tril": [[[[1., 0.], [-3., 3.]]], [[[0.5, 0.], [1., 1.]]]],
            "scale_perturb_factor": [
                [[[1., 0], [1.5, 3.]]], [[[1., 0], [1., 1.]]]],
            "scale_perturb_diag": [[[3., 1.]], [[0.5, 1.]]]},
        x=np.array([[[1., 2]], [[3., 4]]], dtype=np.float32))

  def testNegativeDetTrilPlusVDVT(self):
    # scale = [[3.7, 2.7],
    #          [-0.3, -1.3]]
    # inv(scale) = [[0.325, 0.675],
    #               [-0.075, -0.925]]
    # eig(scale) = [3.5324, -1.1324]
    self._testLegalInputs(
        shift=np.float32(-1),
        scale_params={
            "scale_tril": [[1., 0], [-3, -4]],
            "scale_perturb_factor": [[0.1, 0], [0.5, 0.3]],
            "scale_perturb_diag": [3., 1]},
        x=np.array([1., 2], dtype=np.float32))

  def testScalePropertyAssertsCorrectly(self):
    with self.test_session():
      with self.assertRaises(NotImplementedError):
        scale = bijectors.Affine(  # pylint:disable=unused-variable
            scale_tril=[[1., 0], [2, 1]],
            scale_perturb_factor=[2., 1.]).scale


class SoftplusBijectorTest(tf.test.TestCase):
  """Tests the correctness of the Y = g(X) = Log[1 + exp(X)] transformation."""

  def _softplus(self, x):
    return np.log(1 + np.exp(x))

  def _softplus_inverse(self, y):
    return np.log(np.exp(y) - 1)

  def _softplus_ildj_before_reduction(self, y):
    """Inverse log det jacobian, before being reduced."""
    return -np.log(1 - np.exp(-y))

  def testBijectorForwardInverseEventDimsZero(self):
    with self.test_session():
      bijector = bijectors.Softplus(event_ndims=0)
      self.assertEqual("softplus", bijector.name)
      x = 2 * rng.randn(2, 10)
      y = self._softplus(x)

      self.assertAllClose(y, bijector.forward(x).eval())
      self.assertAllClose(x, bijector.inverse(y).eval())
      self.assertAllClose(
          x, bijector.inverse_and_inverse_log_det_jacobian(y)[0].eval())

  def testBijectorLogDetJacobianEventDimsZero(self):
    with self.test_session():
      bijector = bijectors.Softplus(event_ndims=0)
      y = 2 * rng.rand(2, 10)
      # No reduction needed if event_dims = 0.
      ildj = self._softplus_ildj_before_reduction(y)

      self.assertAllClose(ildj, bijector.inverse_log_det_jacobian(y).eval())
      self.assertAllClose(
          ildj, bijector.inverse_and_inverse_log_det_jacobian(y)[1].eval())

  def testBijectorForwardInverseEventDimsOne(self):
    with self.test_session():
      bijector = bijectors.Softplus(event_ndims=1)
      self.assertEqual("softplus", bijector.name)
      x = 2 * rng.randn(2, 10)
      y = self._softplus(x)

      self.assertAllClose(y, bijector.forward(x).eval())
      self.assertAllClose(x, bijector.inverse(y).eval())
      self.assertAllClose(
          x, bijector.inverse_and_inverse_log_det_jacobian(y)[0].eval())

  def testBijectorLogDetJacobianEventDimsOne(self):
    with self.test_session():
      bijector = bijectors.Softplus(event_ndims=1)
      y = 2 * rng.rand(2, 10)
      ildj_before = self._softplus_ildj_before_reduction(y)
      ildj = np.sum(ildj_before, axis=1)

      self.assertAllClose(ildj, bijector.inverse_log_det_jacobian(y).eval())
      self.assertAllClose(
          ildj, bijector.inverse_and_inverse_log_det_jacobian(y)[1].eval())

  def testScalarCongruency(self):
    with self.test_session():
      bijector = bijectors.Softplus(event_ndims=0)
      assert_scalar_congruency(bijector, lower_x=-2., upper_x=2.)

  def testBijectiveAndFinite32bit(self):
    with self.test_session():
      bijector = bijectors.Softplus(event_ndims=0)
      x = np.linspace(-20., 20., 100).astype(np.float32)
      y = np.logspace(-10, 10, 100).astype(np.float32)
      assert_bijective_and_finite(bijector, x, y, rtol=1e-2, atol=1e-2)

  def testBijectiveAndFinite16bit(self):
    with self.test_session():
      bijector = bijectors.Softplus(event_ndims=0)
      # softplus(-20) is zero, so we can't use such a large range as in 32bit.
      x = np.linspace(-10., 20., 100).astype(np.float16)
      # Note that float16 is only in the open set (0, inf) for a smaller
      # logspace range.  The actual range was (-7, 4), so use something smaller
      # for the test.
      y = np.logspace(-6, 3, 100).astype(np.float16)
      assert_bijective_and_finite(bijector, x, y, rtol=1e-1, atol=1e-3)


class SoftmaxCenteredBijectorTest(tf.test.TestCase):
  """Tests correctness of the Y = g(X) = exp(X) / sum(exp(X)) transformation."""

  def testBijectorScalar(self):
    with self.test_session():
      softmax = bijectors.SoftmaxCentered()  # scalar by default
      self.assertEqual("softmax_centered", softmax.name)
      x = np.log([[2., 3, 4],
                  [4., 8, 12]])
      y = [[[2./3, 1./3],
            [3./4, 1./4],
            [4./5, 1./5]],
           [[4./5, 1./5],
            [8./9, 1./9],
            [12./13, 1./13]]]
      self.assertAllClose(y, softmax.forward(x).eval())
      self.assertAllClose(x, softmax.inverse(y).eval())
      self.assertAllClose(-np.sum(np.log(y), axis=2),
                          softmax.inverse_log_det_jacobian(y).eval(),
                          atol=0., rtol=1e-7)
      self.assertAllClose(-softmax.inverse_log_det_jacobian(y).eval(),
                          softmax.forward_log_det_jacobian(x).eval(),
                          atol=0., rtol=1e-7)

  def testBijectorVector(self):
    with self.test_session():
      softmax = bijectors.SoftmaxCentered(event_ndims=1)
      self.assertEqual("softmax_centered", softmax.name)
      x = np.log([[2., 3, 4],
                  [4., 8, 12]])
      y = [[0.2, 0.3, 0.4, 0.1],
           [0.16, 0.32, 0.48, 0.04]]
      self.assertAllClose(y, softmax.forward(x).eval())
      self.assertAllClose(x, softmax.inverse(y).eval())
      self.assertAllClose(-np.sum(np.log(y), axis=1),
                          softmax.inverse_log_det_jacobian(y).eval(),
                          atol=0., rtol=1e-7)
      self.assertAllClose(-softmax.inverse_log_det_jacobian(y).eval(),
                          softmax.forward_log_det_jacobian(x).eval(),
                          atol=0., rtol=1e-7)

  def testShapeGetters(self):
    with self.test_session():
      for x, y, b in (
          (tf.TensorShape([]),
           tf.TensorShape([2]),
           bijectors.SoftmaxCentered(event_ndims=0, validate_args=True)),
          (tf.TensorShape([4]),
           tf.TensorShape([5]),
           bijectors.SoftmaxCentered(event_ndims=1, validate_args=True))):
        self.assertAllEqual(y, b.get_forward_event_shape(x))
        self.assertAllEqual(y.as_list(),
                            b.forward_event_shape(x.as_list()).eval())
        self.assertAllEqual(x, b.get_inverse_event_shape(y))
        self.assertAllEqual(x.as_list(),
                            b.inverse_event_shape(y.as_list()).eval())

  def testBijectiveAndFinite(self):
    with self.test_session():
      softmax = bijectors.SoftmaxCentered(event_ndims=1)
      x = np.linspace(-50, 50, num=10).reshape(5, 2).astype(np.float32)
      # Make y values on the simplex with a wide range.
      y_0 = np.ones(5).astype(np.float32)
      y_1 = (1e-5 * rng.rand(5)).astype(np.float32)
      y_2 = (1e1 * rng.rand(5)).astype(np.float32)
      y = np.array([y_0, y_1, y_2])
      y /= y.sum(axis=0)
      y = y.T  # y.shape = [5, 3]
      assert_bijective_and_finite(softmax, x, y)


class SigmoidCenteredBijectorTest(tf.test.TestCase):
  """Tests correctness of the Y = g(X) = (1 + exp(-X))^-1 transformation."""

  def testBijector(self):
    with self.test_session():
      sigmoid = bijectors.SigmoidCentered()
      self.assertEqual("sigmoid_centered", sigmoid.name)
      x = np.log([[2., 3, 4],
                  [4., 8, 12]])
      y = [[[2./3, 1./3],
            [3./4, 1./4],
            [4./5, 1./5]],
           [[4./5, 1./5],
            [8./9, 1./9],
            [12./13, 1./13]]]
      self.assertAllClose(y, sigmoid.forward(x).eval())
      self.assertAllClose(x, sigmoid.inverse(y).eval())
      self.assertAllClose(-np.sum(np.log(y), axis=2),
                          sigmoid.inverse_log_det_jacobian(y).eval(),
                          atol=0., rtol=1e-7)
      self.assertAllClose(-sigmoid.inverse_log_det_jacobian(y).eval(),
                          sigmoid.forward_log_det_jacobian(x).eval(),
                          atol=0., rtol=1e-7)


class CholeskyOuterProductBijectorTest(tf.test.TestCase):
  """Tests the correctness of the Y = X * X^T transformation."""

  def testBijectorMatrix(self):
    with self.test_session():
      bijector = bijectors.CholeskyOuterProduct(event_ndims=2,
                                                validate_args=True)
      self.assertEqual("cholesky_outer_product", bijector.name)
      x = [[[1., 0],
            [2, 1]],
           [[math.sqrt(2.), 0],
            [math.sqrt(8.), 1]]]
      y = np.matmul(x, np.transpose(x, axes=(0, 2, 1)))
      # Fairly easy to compute differentials since we have 2x2.
      dx_dy = [[[2.*1, 0, 0],
                [2, 1, 0],
                [0, 2*2, 2*1]],
               [[2*math.sqrt(2.), 0, 0],
                [math.sqrt(8.), math.sqrt(2.), 0],
                [0, 2*math.sqrt(8.), 2*1]]]
      ildj = -np.sum(
          np.log(np.asarray(dx_dy).diagonal(offset=0, axis1=1, axis2=2)),
          axis=1)
      self.assertAllEqual((2, 2, 2), bijector.forward(x).get_shape())
      self.assertAllEqual((2, 2, 2), bijector.inverse(y).get_shape())
      self.assertAllClose(y, bijector.forward(x).eval())
      self.assertAllClose(x, bijector.inverse(y).eval())
      self.assertAllClose(ildj,
                          bijector.inverse_log_det_jacobian(y).eval(),
                          atol=0., rtol=1e-7)
      self.assertAllClose(-bijector.inverse_log_det_jacobian(y).eval(),
                          bijector.forward_log_det_jacobian(x).eval(),
                          atol=0., rtol=1e-7)

  def testBijectorScalar(self):
    with self.test_session():
      bijector = bijectors.CholeskyOuterProduct(event_ndims=0,
                                                validate_args=True)
      self.assertEqual("cholesky_outer_product", bijector.name)
      x = [[[1., 5],
            [2, 1]],
           [[math.sqrt(2.), 3],
            [math.sqrt(8.), 1]]]
      y = np.square(x)
      ildj = -math.log(2.) - np.log(x)
      self.assertAllClose(y, bijector.forward(x).eval())
      self.assertAllClose(x, bijector.inverse(y).eval())
      self.assertAllClose(ildj,
                          bijector.inverse_log_det_jacobian(y).eval(),
                          atol=0., rtol=1e-7)
      self.assertAllClose(-bijector.inverse_log_det_jacobian(y).eval(),
                          bijector.forward_log_det_jacobian(x).eval(),
                          atol=0., rtol=1e-7)

  def testScalarCongruency(self):
    with self.test_session():
      bijector = bijectors.CholeskyOuterProduct(event_ndims=0,
                                                validate_args=True)
      assert_scalar_congruency(bijector, lower_x=1e-3, upper_x=1.5, rtol=0.05)


class ChainBijectorTest(tf.test.TestCase):
  """Tests the correctness of the Y = Chain(bij1, bij2, bij3) transformation."""

  def testBijector(self):
    with self.test_session():
      chain = bijectors.Chain((bijectors.Exp(event_ndims=1),
                               bijectors.Softplus(event_ndims=1)))
      self.assertEqual("chain_of_exp_of_softplus", chain.name)
      x = np.asarray([[[1., 2.],
                       [2., 3.]]])
      self.assertAllClose(1. + np.exp(x), chain.forward(x).eval())
      self.assertAllClose(np.log(x - 1.), chain.inverse(x).eval())
      self.assertAllClose(-np.sum(np.log(x - 1.), axis=2),
                          chain.inverse_log_det_jacobian(x).eval())
      self.assertAllClose(np.sum(x, axis=2),
                          chain.forward_log_det_jacobian(x).eval())

  def testBijectorIdentity(self):
    with self.test_session():
      chain = bijectors.Chain()
      self.assertEqual("identity", chain.name)
      x = np.asarray([[[1., 2.],
                       [2., 3.]]])
      self.assertAllClose(x, chain.forward(x).eval())
      self.assertAllClose(x, chain.inverse(x).eval())
      self.assertAllClose(0., chain.inverse_log_det_jacobian(x).eval())
      self.assertAllClose(0., chain.forward_log_det_jacobian(x).eval())

  def testScalarCongruency(self):
    with self.test_session():
      bijector = bijectors.Chain((bijectors.Exp(),
                                  bijectors.Softplus()))
      assert_scalar_congruency(bijector, lower_x=1e-3, upper_x=1.5, rtol=0.05)

  def testShapeGetters(self):
    with self.test_session():
      bijector = bijectors.Chain((
          bijectors.SoftmaxCentered(event_ndims=1, validate_args=True),
          bijectors.SoftmaxCentered(event_ndims=0, validate_args=True)))
      x = tf.TensorShape([])
      y = tf.TensorShape([2+1])
      self.assertAllEqual(y, bijector.get_forward_event_shape(x))
      self.assertAllEqual(y.as_list(),
                          bijector.forward_event_shape(x.as_list()).eval())
      self.assertAllEqual(x, bijector.get_inverse_event_shape(y))
      self.assertAllEqual(x.as_list(),
                          bijector.inverse_event_shape(y.as_list()).eval())


class InvertBijectorTest(tf.test.TestCase):
  """Tests the correctness of the Y = Invert(bij) transformation."""

  def testBijector(self):
    with self.test_session():
      for fwd in [
          bijectors.Identity(),
          bijectors.Exp(event_ndims=1),
          bijectors.Affine(
              shift=[0., 1.],
              scale_diag=[2., 3.], event_ndims=1),
          bijectors.Softplus(event_ndims=1),
          bijectors.SoftmaxCentered(event_ndims=1),
          bijectors.SigmoidCentered(),
      ]:
        rev = bijectors.Invert(fwd)
        self.assertEqual("_".join(["invert", fwd.name]), rev.name)
        x = [[[1., 2.],
              [2., 3.]]]
        self.assertAllClose(fwd.inverse(x).eval(),
                            rev.forward(x).eval())
        self.assertAllClose(fwd.forward(x).eval(),
                            rev.inverse(x).eval())
        self.assertAllClose(fwd.forward_log_det_jacobian(x).eval(),
                            rev.inverse_log_det_jacobian(x).eval())
        self.assertAllClose(fwd.inverse_log_det_jacobian(x).eval(),
                            rev.forward_log_det_jacobian(x).eval())
        inv, jac = rev.inverse_and_inverse_log_det_jacobian(x)
        self.assertAllClose(fwd.forward(x).eval(),
                            inv.eval())
        self.assertAllClose(fwd.forward_log_det_jacobian(x).eval(),
                            jac.eval())

  def testScalarCongruency(self):
    with self.test_session():
      bijector = bijectors.Invert(bijectors.Exp())
      assert_scalar_congruency(bijector, lower_x=1e-3, upper_x=1.5, rtol=0.05)

  def testShapeGetters(self):
    with self.test_session():
      bijector = bijectors.Invert(bijectors.SigmoidCentered(
          validate_args=True))
      x = tf.TensorShape([2])
      y = tf.TensorShape([])
      self.assertAllEqual(y, bijector.get_forward_event_shape(x))
      self.assertAllEqual(y.as_list(),
                          bijector.forward_event_shape(x.as_list()).eval())
      self.assertAllEqual(x, bijector.get_inverse_event_shape(y))
      self.assertAllEqual(x.as_list(),
                          bijector.inverse_event_shape(y.as_list()).eval())


if __name__ == "__main__":
  tf.test.main()
