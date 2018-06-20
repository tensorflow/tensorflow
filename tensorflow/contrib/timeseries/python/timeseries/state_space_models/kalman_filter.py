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
"""Implements Kalman filtering for linear state space models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import distributions

from tensorflow.contrib.timeseries.python.timeseries import math_utils

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import numerics


# TODO(allenl): support for always-factored covariance matrices
class KalmanFilter(object):
  """Inference on linear state models.

  The model for observations in a given state is:
    observation(t) = observation_model * state(t)
        + Gaussian(0, observation_noise_covariance)

  State updates take the following form:
    state(t) = state_transition * state(t-1)
        + state_noise_transform * Gaussian(0, state_transition_noise_covariance)

  This is a real-valued analog to hidden Markov models, with linear transitions
  and a Gaussian noise model. Given initial conditions, noise, and state
  transition, Kalman filtering recursively estimates states and observations,
  along with their associated uncertainty. When fed observations, future state
  and uncertainty estimates are conditioned on those observations (in a Bayesian
  sense).

  Typically some "given"s mentioned above (noises) will be unknown, and so
  optimizing the Kalman filter's probabilistic predictions with respect to these
  parameters is a good approach. The state transition and observation models are
  usually known a priori as a modeling decision.

  """

  def __init__(self, dtype=dtypes.float32,
               simplified_posterior_covariance_computation=False):
    """Initialize the Kalman filter.

    Args:
      dtype: The data type to use for floating point tensors.
      simplified_posterior_covariance_computation: If True, uses an algebraic
        simplification of the Kalman filtering posterior covariance update,
        which is slightly faster at the cost of numerical stability. The
        simplified update is often stable when using double precision on small
        models or with fixed transition matrices.
    """
    self._simplified_posterior_covariance_computation = (
        simplified_posterior_covariance_computation)
    self.dtype = dtype

  def do_filter(
      self, estimated_state, estimated_state_covariance,
      predicted_observation, predicted_observation_covariance,
      observation, observation_model, observation_noise):
    """Convenience function for scoring predictions.

    Scores a prediction against an observation, and computes the updated
    posterior over states.

    Shapes given below for arguments are for single-model Kalman filtering
    (e.g. KalmanFilter). For ensembles, prior_state and prior_state_var are
    same-length tuples of values corresponding to each model.

    Args:
      estimated_state: A prior mean over states [batch size x state dimension]
      estimated_state_covariance: Covariance of state prior [batch size x D x
          D], with D depending on the Kalman filter implementation (typically
          the state dimension).
      predicted_observation: A prediction for the observed value, such as that
          returned by observed_from_state. A [batch size x num features] Tensor.
      predicted_observation_covariance: A covariance matrix corresponding to
          `predicted_observation`, a [batch size x num features x num features]
          Tensor.
      observation: The observed value corresponding to the predictions
          given [batch size x observation dimension]
      observation_model: The [batch size x observation dimension x model state
          dimension] Tensor indicating how a particular state is mapped to
          (pre-noise) observations for each part of the batch.
      observation_noise: A [batch size x observation dimension x observation
          dimension] Tensor or [observation dimension x observation dimension]
          Tensor with covariance matrices to use for each part of the batch (a
          two-dimensional input will be broadcast).
    Returns:
      posterior_state, posterior_state_var: Posterior mean and
          covariance, updated versions of prior_state and
          prior_state_var.
      log_prediction_prob: Log probability of the observations under
          the priors, suitable for optimization (should be maximized).

    """
    symmetrized_observation_covariance = 0.5 * (
        predicted_observation_covariance + array_ops.matrix_transpose(
            predicted_observation_covariance))
    instability_message = (
        "This may occur due to numerically unstable filtering when there is "
        "a large difference in posterior variances, or when inferences are "
        "near-deterministic. Considering tuning the "
        "'filtering_maximum_posterior_variance_ratio' or "
        "'filtering_minimum_posterior_variance' parameters in your "
        "StateSpaceModelConfiguration, or tuning the transition matrix.")
    symmetrized_observation_covariance = numerics.verify_tensor_all_finite(
        symmetrized_observation_covariance,
        "Predicted observation covariance was not finite. {}".format(
            instability_message))
    diag = array_ops.matrix_diag_part(symmetrized_observation_covariance)
    min_diag = math_ops.reduce_min(diag)
    non_negative_assert = control_flow_ops.Assert(
        min_diag >= 0.,
        [("The predicted observation covariance "
          "has a negative diagonal entry. {}").format(instability_message),
         min_diag])
    with ops.control_dependencies([non_negative_assert]):
      observation_covariance_cholesky = linalg_ops.cholesky(
          symmetrized_observation_covariance)
    log_prediction_prob = distributions.MultivariateNormalTriL(
        predicted_observation, observation_covariance_cholesky).log_prob(
            observation)
    (posterior_state,
     posterior_state_var) = self.posterior_from_prior_state(
         prior_state=estimated_state,
         prior_state_var=estimated_state_covariance,
         observation=observation,
         observation_model=observation_model,
         predicted_observations=(predicted_observation,
                                 predicted_observation_covariance),
         observation_noise=observation_noise)
    return (posterior_state, posterior_state_var, log_prediction_prob)

  def predict_state_mean(self, prior_state, transition_matrices):
    """Compute state transitions.

    Args:
      prior_state: Current estimated state mean [batch_size x state_dimension]
      transition_matrices: A [batch size, state dimension, state dimension]
        batch of matrices (dtype matching the `dtype` argument to the
        constructor) with the transition matrix raised to the power of the
        number of steps to be taken (not element-wise; use
        math_utils.matrix_to_powers if there is no efficient special case) if
        more than one step is desired.
    Returns:
      State mean advanced based on `transition_matrices` (dimensions matching
      first argument).
    """
    advanced_state = array_ops.squeeze(
        math_ops.matmul(
            transition_matrices,
            prior_state[..., None]),
        axis=[-1])
    return advanced_state

  def predict_state_var(
      self, prior_state_var, transition_matrices, transition_noise_sums):
    r"""Compute variance for state transitions.

    Computes a noise estimate corresponding to the value returned by
    predict_state_mean.

    Args:
      prior_state_var: Covariance matrix specifying uncertainty of current state
          estimate [batch size x state dimension x state dimension]
      transition_matrices: A [batch size, state dimension, state dimension]
        batch of matrices (dtype matching the `dtype` argument to the
        constructor) with the transition matrix raised to the power of the
        number of steps to be taken (not element-wise; use
        math_utils.matrix_to_powers if there is no efficient special case).
      transition_noise_sums: A [batch size, state dimension, state dimension]
        Tensor (dtype matching the `dtype` argument to the constructor) with:

          \sum_{i=0}^{num_steps - 1} (
             state_transition_to_powers_fn(i)
             * state_transition_noise_covariance
             * state_transition_to_powers_fn(i)^T
          )

        for the number of steps to be taken in each part of the batch (this
        should match `transition_matrices`). Use math_utils.power_sums_tensor
        with `tf.gather` if there is no efficient special case.
    Returns:
      State variance advanced based on `transition_matrices` and
      `transition_noise_sums` (dimensions matching first argument).
    """
    prior_variance_transitioned = math_ops.matmul(
        math_ops.matmul(transition_matrices, prior_state_var),
        transition_matrices,
        adjoint_b=True)
    return prior_variance_transitioned + transition_noise_sums

  def posterior_from_prior_state(self, prior_state, prior_state_var,
                                 observation, observation_model,
                                 predicted_observations,
                                 observation_noise):
    """Compute a posterior over states given an observation.

    Args:
      prior_state: Prior state mean [batch size x state dimension]
      prior_state_var: Prior state covariance [batch size x state dimension x
          state dimension]
      observation: The observed value corresponding to the predictions given
          [batch size x observation dimension]
      observation_model: The [batch size x observation dimension x model state
          dimension] Tensor indicating how a particular state is mapped to
          (pre-noise) observations for each part of the batch.
      predicted_observations: An (observation mean, observation variance) tuple
          computed based on the current state, usually the output of
          observed_from_state.
      observation_noise: A [batch size x observation dimension x observation
          dimension] or [observation dimension x observation dimension] Tensor
          with covariance matrices to use for each part of the batch (a
          two-dimensional input will be broadcast).
    Returns:
      Posterior mean and covariance (dimensions matching the first two
      arguments).

    """
    observed_mean, observed_var = predicted_observations
    residual = observation - observed_mean
    # TODO(allenl): Can more of this be done using matrix_solve_ls?
    kalman_solve_rhs = math_ops.matmul(
        observation_model, prior_state_var, adjoint_b=True)
    # This matrix_solve adjoint doesn't make a difference symbolically (since
    # observed_var is a covariance matrix, and should be symmetric), but
    # filtering on multivariate series is unstable without it. See
    # test_multivariate_symmetric_covariance_float64 in kalman_filter_test.py
    # for an example of the instability (fails with adjoint=False).
    kalman_gain_transposed = linalg_ops.matrix_solve(
        matrix=observed_var, rhs=kalman_solve_rhs, adjoint=True)
    posterior_state = prior_state + array_ops.squeeze(
        math_ops.matmul(
            kalman_gain_transposed,
            array_ops.expand_dims(residual, -1),
            adjoint_a=True),
        axis=[-1])
    gain_obs = math_ops.matmul(
        kalman_gain_transposed, observation_model, adjoint_a=True)
    identity_extradim = linalg_ops.eye(
        array_ops.shape(gain_obs)[1], dtype=gain_obs.dtype)[None]
    identity_minus_factor = identity_extradim - gain_obs
    if self._simplified_posterior_covariance_computation:
      # posterior covariance =
      #   (I - kalman_gain * observation_model) * prior_state_var
      posterior_state_var = math_ops.matmul(identity_minus_factor,
                                            prior_state_var)
    else:
      observation_noise = ops.convert_to_tensor(observation_noise)
      # A Joseph form update, which provides better numeric stability than the
      # simplified optimal Kalman gain update, at the cost of a few extra
      # operations. Joseph form updates are valid for any gain (not just the
      # optimal Kalman gain), and so are more forgiving of numerical errors in
      # computing the optimal Kalman gain.
      #
      # posterior covariance =
      #   (I - kalman_gain * observation_model) * prior_state_var
      #     * (I - kalman_gain * observation_model)^T
      #   + kalman_gain * observation_noise * kalman_gain^T
      left_multiplied_state_var = math_ops.matmul(identity_minus_factor,
                                                  prior_state_var)
      multiplied_state_var = math_ops.matmul(
          identity_minus_factor, left_multiplied_state_var, adjoint_b=True)
      def _batch_observation_noise_update():
        return (multiplied_state_var + math_ops.matmul(
            math_ops.matmul(
                kalman_gain_transposed, observation_noise, adjoint_a=True),
            kalman_gain_transposed))
      def _matrix_observation_noise_update():
        return (multiplied_state_var + math_ops.matmul(
            math_utils.batch_times_matrix(
                kalman_gain_transposed, observation_noise, adj_x=True),
            kalman_gain_transposed))
      if observation_noise.get_shape().ndims is None:
        posterior_state_var = control_flow_ops.cond(
            math_ops.equal(array_ops.rank(observation_noise), 2),
            _matrix_observation_noise_update, _batch_observation_noise_update)
      else:
        # If static shape information exists, it gets checked in each cond()
        # branch, so we need a special case to avoid graph-build-time
        # exceptions.
        if observation_noise.get_shape().ndims == 2:
          posterior_state_var = _matrix_observation_noise_update()
        else:
          posterior_state_var = _batch_observation_noise_update()
    return posterior_state, posterior_state_var

  def observed_from_state(self, state_mean, state_var, observation_model,
                          observation_noise):
    """Compute an observation distribution given a state distribution.

    Args:
      state_mean: State mean vector [batch size x state dimension]
      state_var: State covariance [batch size x state dimension x state
          dimension]
      observation_model: The [batch size x observation dimension x model state
          dimension] Tensor indicating how a particular state is mapped to
          (pre-noise) observations for each part of the batch.
      observation_noise: A [batch size x observation dimension x observation
          dimension] Tensor with covariance matrices to use for each part of the
          batch. To remove observation noise, pass a Tensor of zeros (or simply
          0, which will broadcast).
    Returns:
      observed_mean: Observation mean vector [batch size x observation
          dimension]
      observed_var: Observation covariance [batch size x observation dimension x
          observation dimension]

    """
    observed_mean = array_ops.squeeze(
        math_ops.matmul(
            array_ops.expand_dims(state_mean, 1),
            observation_model,
            adjoint_b=True),
        axis=[1])
    observed_var = math_ops.matmul(
        math_ops.matmul(observation_model, state_var),
        observation_model,
        adjoint_b=True)
    observed_var += observation_noise
    return observed_mean, observed_var
