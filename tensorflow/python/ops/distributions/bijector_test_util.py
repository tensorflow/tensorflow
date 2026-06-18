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
"""Bijector unit-test utilities."""

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import uniform as uniform_lib


def assert_finite(array):
  if not np.isfinite(array).all():
    raise AssertionError("array was not all finite. %s" % array[:15])


def assert_strictly_increasing(array):
  np.testing.assert_array_less(0., np.diff(array))


def assert_strictly_decreasing(array):
  np.testing.assert_array_less(np.diff(array), 0.)


def assert_strictly_monotonic(array):
  if array[0] < array[-1]:
    assert_strictly_increasing(array)
  else:
    assert_strictly_decreasing(array)


def assert_scalar_congruency(bijector,
                             lower_x,
                             upper_x,
                             n=int(10e3),
                             rtol=0.01,
                             sess=None):
  """Assert `bijector`'s forward/inverse/inverse_log_det_jacobian are congruent.

  We draw samples `X ~ U(lower_x, upper_x)`, then feed these through the
  `bijector` in order to check that:

  1. the forward is strictly monotonic.
  2. the forward/inverse methods are inverses of each other.
  3. the jacobian is the correct change of measure.

  This can only be used for a Bijector mapping open subsets of the real line
  to themselves.  This is due to the fact that this test compares the `prob`
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
    sess:  `tf.compat.v1.Session`.  Defaults to the default session.

  Raises:
    AssertionError:  If tests fail.
  """
  # Checks and defaults.
  if sess is None:
    sess = ops.get_default_session()

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
  uniform_x_samps = uniform_lib.Uniform(
      low=lower_x, high=upper_x).sample(n, seed=0)
  uniform_y_samps = uniform_lib.Uniform(
      low=lower_y, high=upper_y).sample(n, seed=1)

  # These compositions should be the identity.
  inverse_forward_x = bijector.inverse(bijector.forward(uniform_x_samps))
  forward_inverse_y = bijector.forward(bijector.inverse(uniform_y_samps))

  # For a < b, and transformation y = y(x),
  # (b - a) = \int_a^b dx = \int_{y(a)}^{y(b)} |dx/dy| dy
  # "change_measure_dy_dx" below is a Monte Carlo approximation to the right
  # hand side, which should then be close to the left, which is (b - a).
  # We assume event_ndims=0 because we assume scalar -> scalar. The log_det
  # methods will handle whether they expect event_ndims > 0.
  dy_dx = math_ops.exp(bijector.inverse_log_det_jacobian(
      uniform_y_samps, event_ndims=0))
  # E[|dx/dy|] under Uniform[lower_y, upper_y]
  # = \int_{y(a)}^{y(b)} |dx/dy| dP(u), where dP(u) is the uniform measure
  expectation_of_dy_dx_under_uniform = math_ops.reduce_mean(dy_dx)
  # dy = dP(u) * (upper_y - lower_y)
  change_measure_dy_dx = (
      (upper_y - lower_y) * expectation_of_dy_dx_under_uniform)

  # We'll also check that dy_dx = 1 / dx_dy.
  dx_dy = math_ops.exp(
      bijector.forward_log_det_jacobian(
          bijector.inverse(uniform_y_samps), event_ndims=0))

  [
      forward_on_10_pts_v,
      dy_dx_v,
      dx_dy_v,
      change_measure_dy_dx_v,
      uniform_x_samps_v,
      uniform_y_samps_v,
      inverse_forward_x_v,
      forward_inverse_y_v,
  ] = sess.run([
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
  np.testing.assert_allclose(
      upper_x - lower_x, change_measure_dy_dx_v, atol=0, rtol=rtol)
  # Inverse Jacobian should be equivalent to the reciprocal of the forward
  # Jacobian.
  np.testing.assert_allclose(
      dy_dx_v, np.divide(1., dx_dy_v), atol=1e-5, rtol=1e-3)


def assert_bijective_and_finite(
    bijector, x, y, event_ndims, atol=0, rtol=1e-5, sess=None):
  """Assert that forward/inverse (along with jacobians) are inverses and finite.

  It is recommended to use x and y values that are very very close to the edge
  of the Bijector's domain.

  Args:
    bijector:  A Bijector instance.
    x:  np.array of values in the domain of bijector.forward.
    y:  np.array of values in the domain of bijector.inverse.
    event_ndims: Integer describing the number of event dimensions this bijector
      operates on.
    atol:  Absolute tolerance.
    rtol:  Relative tolerance.
    sess:  TensorFlow session.  Defaults to the default session.

  Raises:
    AssertionError:  If tests fail.
  """
  sess = sess or ops.get_default_session()

  # These are the incoming points, but people often create a crazy range of
  # values for which these end up being bad, especially in 16bit.
  assert_finite(x)
  assert_finite(y)

  f_x = bijector.forward(x)
  g_y = bijector.inverse(y)

  [
      x_from_x,
      y_from_y,
      ildj_f_x,
      fldj_x,
      ildj_y,
      fldj_g_y,
      f_x_v,
      g_y_v,
  ] = sess.run([
      bijector.inverse(f_x),
      bijector.forward(g_y),
      bijector.inverse_log_det_jacobian(f_x, event_ndims=event_ndims),
      bijector.forward_log_det_jacobian(x, event_ndims=event_ndims),
      bijector.inverse_log_det_jacobian(y, event_ndims=event_ndims),
      bijector.forward_log_det_jacobian(g_y, event_ndims=event_ndims),
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
