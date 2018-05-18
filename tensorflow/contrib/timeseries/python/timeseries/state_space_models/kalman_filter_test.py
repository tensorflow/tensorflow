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
"""Tests for Kalman filtering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from tensorflow.contrib.timeseries.python.timeseries import math_utils
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import kalman_filter

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


# Two-dimensional state model with "slope" and "level" components.
STATE_TRANSITION = [
    [1., 1.],  # Add slope to level
    [0., 1.]   # Maintain slope
]
# Independent noise for each component
STATE_TRANSITION_NOISE = [[0.1, 0.0], [0.0, 0.2]]
OBSERVATION_MODEL = [[[0.5, 0.0], [0.0, 1.0]]]
OBSERVATION_NOISE = [[0.0001, 0.], [0., 0.0002]]
STATE_NOISE_TRANSFORM = [[1.0, 0.0], [0.0, 1.0]]


def _powers_and_sums_from_transition_matrix(
    state_transition, state_transition_noise_covariance,
    state_noise_transform, max_gap=1):
  def _transition_matrix_powers(powers):
    return math_utils.matrix_to_powers(state_transition, powers)
  def _power_sums(num_steps):
    power_sums_tensor = math_utils.power_sums_tensor(
        max_gap + 1, state_transition,
        math_ops.matmul(state_noise_transform,
                        math_ops.matmul(
                            state_transition_noise_covariance,
                            state_noise_transform,
                            adjoint_b=True)))
    return array_ops.gather(power_sums_tensor, indices=num_steps)
  return (_transition_matrix_powers, _power_sums)


class MultivariateTests(test.TestCase):

  def _multivariate_symmetric_covariance_test_template(
      self, dtype, simplified_posterior_variance_computation):
    """Check that errors aren't building up asymmetries in covariances."""
    kf = kalman_filter.KalmanFilter(dtype=dtype)
    observation_noise_covariance = constant_op.constant(
        [[1., 0.5], [0.5, 1.]], dtype=dtype)
    observation_model = constant_op.constant(
        [[[1., 0., 0., 0.], [0., 0., 1., 0.]]], dtype=dtype)
    state = array_ops.placeholder(shape=[1, 4], dtype=dtype)
    state_var = array_ops.placeholder(shape=[1, 4, 4], dtype=dtype)
    observation = array_ops.placeholder(shape=[1, 2], dtype=dtype)
    transition_fn, power_sum_fn = _powers_and_sums_from_transition_matrix(
        state_transition=constant_op.constant(
            [[1., 1., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 1.],
             [0., 0., 0., 1.]],
            dtype=dtype),
        state_noise_transform=linalg_ops.eye(4, dtype=dtype),
        state_transition_noise_covariance=constant_op.constant(
            [[1., 0., 0.5, 0.], [0., 1., 0., 0.5], [0.5, 0., 1., 0.],
             [0., 0.5, 0., 1.]],
            dtype=dtype))
    pred_state = kf.predict_state_mean(
        prior_state=state, transition_matrices=transition_fn([1]))
    pred_state_var = kf.predict_state_var(
        prior_state_var=state_var, transition_matrices=transition_fn([1]),
        transition_noise_sums=power_sum_fn([1]))
    observed_mean, observed_var = kf.observed_from_state(
        state_mean=pred_state, state_var=pred_state_var,
        observation_model=observation_model,
        observation_noise=observation_noise_covariance)
    post_state, post_state_var = kf.posterior_from_prior_state(
        prior_state=pred_state, prior_state_var=pred_state_var,
        observation=observation,
        observation_model=observation_model,
        predicted_observations=(observed_mean, observed_var),
        observation_noise=observation_noise_covariance)
    with self.test_session() as session:
      evaled_state = numpy.array([[1., 1., 1., 1.]])
      evaled_state_var = numpy.eye(4)[None]
      for i in range(500):
        evaled_state, evaled_state_var, evaled_observed_var = session.run(
            [post_state, post_state_var, observed_var],
            feed_dict={state: evaled_state,
                       state_var: evaled_state_var,
                       observation: [[float(i), float(i)]]})
        self.assertAllClose(evaled_observed_var[0],
                            evaled_observed_var[0].T)
        self.assertAllClose(evaled_state_var[0],
                            evaled_state_var[0].T)

  def test_multivariate_symmetric_covariance_float32(self):
    self._multivariate_symmetric_covariance_test_template(
        dtypes.float32, simplified_posterior_variance_computation=False)

  def test_multivariate_symmetric_covariance_float64(self):
    self._multivariate_symmetric_covariance_test_template(
        dtypes.float64, simplified_posterior_variance_computation=True)


class KalmanFilterNonBatchTest(test.TestCase):
  """Single-batch KalmanFilter tests."""

  def setUp(self):
    """The basic model defined above, with unit batches."""
    self.kalman_filter = kalman_filter.KalmanFilter()
    self.transition_fn, self.power_sum_fn = (
        _powers_and_sums_from_transition_matrix(
            state_transition=STATE_TRANSITION,
            state_transition_noise_covariance=STATE_TRANSITION_NOISE,
            state_noise_transform=STATE_NOISE_TRANSFORM,
            max_gap=5))

  def test_observed_from_state(self):
    """Compare observation mean and noise to hand-computed values."""
    with self.test_session():
      state = constant_op.constant([[2., 1.]])
      state_var = constant_op.constant([[[4., 0.], [0., 3.]]])
      observed_mean, observed_var = self.kalman_filter.observed_from_state(
          state, state_var,
          observation_model=OBSERVATION_MODEL,
          observation_noise=OBSERVATION_NOISE)
      observed_mean_override, observed_var_override = (
          self.kalman_filter.observed_from_state(
              state, state_var,
              observation_model=OBSERVATION_MODEL,
              observation_noise=100 * constant_op.constant(
                  OBSERVATION_NOISE)[None]))
      self.assertAllClose(numpy.array([[1., 1.]]),
                          observed_mean.eval())
      self.assertAllClose(numpy.array([[1., 1.]]),
                          observed_mean_override.eval())
      self.assertAllClose(numpy.array([[[1.0001, 0.], [0., 3.0002]]]),
                          observed_var.eval())
      self.assertAllClose(numpy.array([[[1.01, 0.], [0., 3.02]]]),
                          observed_var_override.eval())

  def _posterior_from_prior_state_test_template(
      self, state, state_var, observation, observation_model, observation_noise,
      expected_state, expected_state_var):
    """Test that repeated observations converge to the expected value."""
    predicted_observations = self.kalman_filter.observed_from_state(
        state, state_var, observation_model,
        observation_noise=observation_noise)
    state_update, state_var_update = (
        self.kalman_filter.posterior_from_prior_state(
            state, state_var, observation,
            observation_model=observation_model,
            predicted_observations=predicted_observations,
            observation_noise=observation_noise))
    with self.test_session() as session:
      evaled_state, evaled_state_var = session.run([state, state_var])
      for _ in range(300):
        evaled_state, evaled_state_var = session.run(
            [state_update, state_var_update],
            feed_dict={state: evaled_state, state_var: evaled_state_var})
    self.assertAllClose(expected_state,
                        evaled_state,
                        atol=1e-5)
    self.assertAllClose(
        expected_state_var,
        evaled_state_var,
        atol=1e-5)

  def test_posterior_from_prior_state_univariate(self):
    self._posterior_from_prior_state_test_template(
        state=constant_op.constant([[0.3]]),
        state_var=constant_op.constant([[[1.]]]),
        observation=constant_op.constant([[1.]]),
        observation_model=[[[2.]]],
        observation_noise=[[[0.01]]],
        expected_state=numpy.array([[0.5]]),
        expected_state_var=[[[0.]]])

  def test_posterior_from_prior_state_univariate_unit_noise(self):
    self._posterior_from_prior_state_test_template(
        state=constant_op.constant([[0.3]]),
        state_var=constant_op.constant([[[1e10]]]),
        observation=constant_op.constant([[1.]]),
        observation_model=[[[2.]]],
        observation_noise=[[[1.0]]],
        expected_state=numpy.array([[0.5]]),
        expected_state_var=[[[1. / (300. * 2. ** 2)]]])

  def test_posterior_from_prior_state_multivariate_2d(self):
    self._posterior_from_prior_state_test_template(
        state=constant_op.constant([[1.9, 1.]]),
        state_var=constant_op.constant([[[1., 0.], [0., 2.]]]),
        observation=constant_op.constant([[1., 1.]]),
        observation_model=OBSERVATION_MODEL,
        observation_noise=OBSERVATION_NOISE,
        expected_state=numpy.array([[2., 1.]]),
        expected_state_var=[[[0., 0.], [0., 0.]]])

  def test_posterior_from_prior_state_multivariate_3d(self):
    self._posterior_from_prior_state_test_template(
        state=constant_op.constant([[1.9, 1., 5.]]),
        state_var=constant_op.constant(
            [[[200., 0., 1.], [0., 2000., 0.], [1., 0., 40000.]]]),
        observation=constant_op.constant([[1., 1., 3.]]),
        observation_model=constant_op.constant(
            [[[0.5, 0., 0.],
              [0., 10., 0.],
              [0., 0., 100.]]]),
        observation_noise=linalg_ops.eye(3) / 10000.,
        expected_state=numpy.array([[2., .1, .03]]),
        expected_state_var=numpy.zeros([1, 3, 3]))

  def test_predict_state_mean(self):
    """Compare state mean transitions with simple hand-computed values."""
    with self.test_session():
      state = constant_op.constant([[4., 2.]])
      state = self.kalman_filter.predict_state_mean(
          state, self.transition_fn([1]))
      for _ in range(2):
        state = self.kalman_filter.predict_state_mean(
            state, self.transition_fn([1]))
      self.assertAllClose(
          numpy.array([[2. * 3. + 4.,  # Slope * time + base
                        2.]]),
          state.eval())

  def test_predict_state_var(self):
    """Compare a variance transition with simple hand-computed values."""
    with self.test_session():
      state_var = constant_op.constant([[[1., 0.], [0., 2.]]])
      state_var = self.kalman_filter.predict_state_var(
          state_var, self.transition_fn([1]), self.power_sum_fn([1]))
      self.assertAllClose(
          numpy.array([[[3.1, 2.0], [2.0, 2.2]]]),
          state_var.eval())

  def test_do_filter(self):
    """Tests do_filter.

    Tests that correct values have high probability and incorrect values
    have low probability when there is low uncertainty.
    """
    with self.test_session():
      state = constant_op.constant([[4., 2.]])
      state_var = constant_op.constant([[[0.0001, 0.], [0., 0.0001]]])
      observation = constant_op.constant([[
          .5 * (
              4.  # Base
              + 2.),  # State transition
          2.
      ]])
      estimated_state = self.kalman_filter.predict_state_mean(
          state, self.transition_fn([1]))
      estimated_state_covariance = self.kalman_filter.predict_state_var(
          state_var, self.transition_fn([1]), self.power_sum_fn([1]))
      (predicted_observation,
       predicted_observation_covariance) = (
           self.kalman_filter.observed_from_state(
               estimated_state, estimated_state_covariance,
               observation_model=OBSERVATION_MODEL,
               observation_noise=OBSERVATION_NOISE))
      (_, _, first_log_prob) = self.kalman_filter.do_filter(
          estimated_state=estimated_state,
          estimated_state_covariance=estimated_state_covariance,
          predicted_observation=predicted_observation,
          predicted_observation_covariance=predicted_observation_covariance,
          observation=observation,
          observation_model=OBSERVATION_MODEL,
          observation_noise=OBSERVATION_NOISE)
      self.assertGreater(first_log_prob.eval()[0], numpy.log(0.99))

  def test_predict_n_ahead_mean(self):
    with self.test_session():
      original_state = constant_op.constant([[4., 2.]])
      n = 5
      iterative_state = original_state
      for i in range(n):
        self.assertAllClose(
            iterative_state.eval(),
            self.kalman_filter.predict_state_mean(
                original_state,
                self.transition_fn([i])).eval())
        iterative_state = self.kalman_filter.predict_state_mean(
            iterative_state,
            self.transition_fn([1]))

  def test_predict_n_ahead_var(self):
    with self.test_session():
      original_var = constant_op.constant([[[2., 3.], [4., 5.]]])
      n = 5
      iterative_var = original_var
      for i in range(n):
        self.assertAllClose(
            iterative_var.eval(),
            self.kalman_filter.predict_state_var(
                original_var,
                self.transition_fn([i]),
                self.power_sum_fn([i])).eval())
        iterative_var = self.kalman_filter.predict_state_var(
            iterative_var,
            self.transition_fn([1]),
            self.power_sum_fn([1]))


class KalmanFilterBatchTest(test.TestCase):
  """KalmanFilter tests with more than one element batches."""

  def test_do_filter_batch(self):
    """Tests do_filter, in batch mode.

    Tests that correct values have high probability and incorrect values
    have low probability when there is low uncertainty.
    """
    with self.test_session():
      state = constant_op.constant([[4., 2.], [5., 3.], [6., 4.]])
      state_var = constant_op.constant(3 * [[[0.0001, 0.], [0., 0.0001]]])
      observation = constant_op.constant([
          [
              .5 * (
                  4.  # Base
                  + 2.),  # State transition
              2.
          ],
          [
              .5 * (
                  5.  # Base
                  + 3.),  # State transition
              3.
          ],
          [3.14, 2.71]
      ])  # Low probability observation
      kf = kalman_filter.KalmanFilter()
      transition_fn, power_sum_fn = _powers_and_sums_from_transition_matrix(
          state_transition=STATE_TRANSITION,
          state_transition_noise_covariance=STATE_TRANSITION_NOISE,
          state_noise_transform=STATE_NOISE_TRANSFORM,
          max_gap=2)
      estimated_state = kf.predict_state_mean(state, transition_fn(3*[1]))
      estimated_state_covariance = kf.predict_state_var(
          state_var, transition_fn(3*[1]), power_sum_fn(3*[1]))
      observation_model = array_ops.tile(OBSERVATION_MODEL, [3, 1, 1])
      (predicted_observation,
       predicted_observation_covariance) = (
           kf.observed_from_state(
               estimated_state, estimated_state_covariance,
               observation_model=observation_model,
               observation_noise=OBSERVATION_NOISE))
      (state, state_var, log_prob) = kf.do_filter(
          estimated_state=estimated_state,
          estimated_state_covariance=estimated_state_covariance,
          predicted_observation=predicted_observation,
          predicted_observation_covariance=predicted_observation_covariance,
          observation=observation,
          observation_model=observation_model,
          observation_noise=OBSERVATION_NOISE)
      first_log_prob, second_log_prob, third_log_prob = log_prob.eval()
      self.assertGreater(first_log_prob.sum(), numpy.log(0.99))
      self.assertGreater(second_log_prob.sum(), numpy.log(0.99))
      self.assertLess(third_log_prob.sum(), numpy.log(0.01))

  def test_predict_n_ahead_mean(self):
    with self.test_session():
      kf = kalman_filter.KalmanFilter()
      transition_fn, _ = _powers_and_sums_from_transition_matrix(
          state_transition=STATE_TRANSITION,
          state_transition_noise_covariance=STATE_TRANSITION_NOISE,
          state_noise_transform=STATE_NOISE_TRANSFORM,
          max_gap=2)
      original_state = constant_op.constant([[4., 2.], [3., 1.], [6., 2.]])
      state0 = original_state
      state1 = kf.predict_state_mean(state0, transition_fn(3 * [1]))
      state2 = kf.predict_state_mean(state1, transition_fn(3 * [1]))
      batch_eval = kf.predict_state_mean(
          original_state, transition_fn([1, 0, 2])).eval()
      self.assertAllClose(state0.eval()[1], batch_eval[1])
      self.assertAllClose(state1.eval()[0], batch_eval[0])
      self.assertAllClose(state2.eval()[2], batch_eval[2])

  def test_predict_n_ahead_var(self):
    with self.test_session():
      kf = kalman_filter.KalmanFilter()
      transition_fn, power_sum_fn = _powers_and_sums_from_transition_matrix(
          state_transition=STATE_TRANSITION,
          state_transition_noise_covariance=STATE_TRANSITION_NOISE,
          state_noise_transform=STATE_NOISE_TRANSFORM,
          max_gap=2)
      base_var = 2.0 * numpy.identity(2) + numpy.ones([2, 2])
      original_var = constant_op.constant(
          numpy.array(
              [base_var, 2.0 * base_var, 3.0 * base_var], dtype=numpy.float32))
      var0 = original_var
      var1 = kf.predict_state_var(
          var0, transition_fn(3 * [1]), power_sum_fn(3 * [1]))
      var2 = kf.predict_state_var(
          var1, transition_fn(3 * [1]), power_sum_fn(3 * [1]))
      batch_eval = kf.predict_state_var(
          original_var,
          transition_fn([1, 0, 2]),
          power_sum_fn([1, 0, 2])).eval()
      self.assertAllClose(var0.eval()[1], batch_eval[1])
      self.assertAllClose(var1.eval()[0], batch_eval[0])
      self.assertAllClose(var2.eval()[2], batch_eval[2])


if __name__ == "__main__":
  test.main()
