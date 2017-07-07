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
"""Abstract base for state space models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import numpy

from tensorflow.contrib.layers.python.layers import layers

from tensorflow.contrib.timeseries.python.timeseries import math_utils
from tensorflow.contrib.timeseries.python.timeseries import model
from tensorflow.contrib.timeseries.python.timeseries import model_utils
from tensorflow.contrib.timeseries.python.timeseries.feature_keys import TrainEvalFeatures
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import kalman_filter

from tensorflow.python.estimator import estimator_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope


class StateSpaceModelConfiguration(
    collections.namedtuple(
        typename="StateSpaceModelConfiguration",
        field_names=[
            "num_features", "use_observation_noise", "dtype",
            "covariance_prior_fn", "bayesian_prior_weighting",
            "filtering_postprocessor", "trainable_start_state",
            "exogenous_noise_increases", "exogenous_noise_decreases",
            "exogenous_feature_columns", "exogenous_update_condition",
            "filtering_maximum_posterior_variance_ratio",
            "filtering_minimum_posterior_variance",
            "transition_covariance_initial_log_scale_bias",
            "static_unrolling_window_size_threshold"])):
  """Configuration options for StateSpaceModels."""

  def __new__(
      cls,
      num_features=1,
      use_observation_noise=True,
      dtype=dtypes.float32,
      covariance_prior_fn=math_utils.log_noninformative_covariance_prior,
      bayesian_prior_weighting=True,
      filtering_postprocessor=None,
      trainable_start_state=False,
      exogenous_noise_increases=True,
      exogenous_noise_decreases=False,
      exogenous_feature_columns=None,
      exogenous_update_condition=None,
      filtering_maximum_posterior_variance_ratio=1e6,
      filtering_minimum_posterior_variance=0.,
      transition_covariance_initial_log_scale_bias=-5.,
      static_unrolling_window_size_threshold=None):
    """Configuration options for StateSpaceModels.

    Args:
      num_features: Output dimension for model
      use_observation_noise: If true, observations are modeled as noisy
        functions of the current state. If false, observations are a
        deterministic function of the current state. Only applicable to the
        top-level model in an ensemble. Consider also changing the
        transition_covariance_initial_log_scale_bias when disabling observation
        noise, as its default setting assumes that observation noise is part of
        the model.
      dtype: The float dtype to use when defining the model.
      covariance_prior_fn: A function mapping from a covariance matrix to a
          scalar value (e.g. log likelihood) which can be summed across
          matrices. Defaults to an independent Jeffreys prior on the diagonal
          elements (regularizing as log(1. / variance)). To use a flat prior
          (i.e. no regularization), set to `lambda _: 0.`.  Defaults to
          relatively uninformative priors on state transition and observation
          noise, which have the effect of encouraging low-noise solutions which
          provide confident predictions when possible. Without regularization,
          transition noise tends to remain high, and multi-step predictions are
          under-confident.
      bayesian_prior_weighting: If True, weights the prior (covariance_prior_fn)
          based on an estimate of the full dataset size. If False, weights it
          based on the mini-batch window size, which (while statistically
          improper) can lead to more desirable low-noise solutions in cases
          where the full dataset is large enough to overwhelm the prior.
      filtering_postprocessor: A FilteringStepPostprocessor object to use,
          useful for ignoring anomalies in training data.
      trainable_start_state: If True, start state may depend on trainable
          Variables. If False, it will not.
      exogenous_noise_increases: If True, exogenous regressors can add to model
          state, increasing uncertainty. If both this parameter and
          exogenous_noise_decreases are False, exogenous regressors are ignored.
      exogenous_noise_decreases: If True, exogenous regressors can "set" model
          state, decreasing uncertainty. If both this parameter and
          exogenous_noise_increases are False, exogenous regressors are ignored.
      exogenous_feature_columns: A list of tf.contrib.layers.FeatureColumn
          objects (for example tf.contrib.layers.embedding_column) corresponding
          to exogenous features which provide extra information to the model but
          are not part of the series to be predicted. Passed to
          tf.contrib.layers.input_from_feature_columns.
      exogenous_update_condition: A function taking two Tensor arguments `times`
          (shape [batch size]) and `features` (a dictionary mapping exogenous
          feature keys to Tensors with shapes [batch size, ...]) and returning a
          boolean Tensor with shape [batch size] indicating whether state should
          be updated using exogenous features for each part of the batch. Where
          it is False, no exogenous update is performed. If None (default),
          exogenous updates are always performed. Useful for avoiding "leaky"
          frequent exogenous updates when sparse updates are desired. Called
          only during graph construction.
      filtering_maximum_posterior_variance_ratio: The maximum allowed ratio of
          two diagonal entries in a state covariance matrix just prior to
          filtering. Lower values mean that filtering will be more numerically
          stable, at the cost of artificially increasing estimated uncertainty
          in some cases. This parameter can be important when learning a
          transition matrix.
      filtering_minimum_posterior_variance: The minimum diagonal value in a
          state covariance matrix just prior to filtering, preventing numerical
          instability due to deterministic beliefs (sometimes an issue when
          learning transition matrices). This value should be set several orders
          of magnitude below any expected minimum state uncertainty.
      transition_covariance_initial_log_scale_bias: Controls the initial
          tradeoff between the transition noise covariance matrix and the
          observation noise covariance matrix, on a log scale (the elements of
          the transition noise covariance matrix are proportional to `e^{X +
          transition_covariance_initial_log_scale_bias}` where `X` is learned
          and may depend on input statistics, observation noise covariance is
          proportional to `e^{Y -
          transition_covariance_initial_log_scale_bias}`). For models *with*
          observation noise, -5 is a reasonable value. Models which do not use
          observation noise, and are not part of an ensemble which does use
          observation noise, should have this set to 0 or more to avoid
          numerical issues due to filtering with too little noise.
      static_unrolling_window_size_threshold: Only relevant for the top-level
          StateSpaceModel in an ensemble; enables switching between static and
          dynamic looping (if not None, default, meaning that no static
          unrolling is performed) based on the window size (windows with this
          size and smaller will have their graphs unrolled statically). See the
          SequentialTimeSeriesModel constructor for details.
    Returns:
      A StateSpaceModelConfiguration object.
    """
    if exogenous_feature_columns is None:
      exogenous_feature_columns = []
    return super(StateSpaceModelConfiguration, cls).__new__(
        cls, num_features, use_observation_noise, dtype,
        covariance_prior_fn, bayesian_prior_weighting,
        filtering_postprocessor, trainable_start_state,
        exogenous_noise_increases, exogenous_noise_decreases,
        exogenous_feature_columns, exogenous_update_condition,
        filtering_maximum_posterior_variance_ratio,
        filtering_minimum_posterior_variance,
        transition_covariance_initial_log_scale_bias,
        static_unrolling_window_size_threshold)


class StateSpaceModel(model.SequentialTimeSeriesModel):
  """Base class for linear state space models.

  Sub-classes can specify the model to be learned by overriding
  get_state_transition, get_noise_transform, and get_observation_model.

  See kalman_filter.py for a detailed description of the class of models covered
  by StateSpaceModel.

  Briefly, state space models are defined by a state transition equation:

  state[t] = StateTransition * state[t-1] + NoiseTransform * StateNoise[t]
             + ExogenousNoiseIncreasing[t]
  StateNoise[t] ~ Gaussian(0, StateNoiseCovariance)
  ExogenousNoiseIncreasing[t] ~ Gaussian(ExogenousNoiseIncreasingMean[t],
                                         ExogenousNoiseIncreasingCovariance[t])

  And an observation model:

  observation[t] = ObservationModel * state[t] + ObservationNoise[t]
  ObservationNoise[t] ~ Gaussian(0, ObservationNoiseCovariance)

  Additionally, exogenous regressors can act as observations, decreasing
  uncertainty:

  ExogenousNoiseDecreasingObservation[t] ~ Gaussian(
      ExogenousNoiseDecreasingMean[t], ExogenousNoiseDecreasingCovariance[t])

  Attributes:
    kalman_filter: If initialize_graph has been called, the initialized
        KalmanFilter to use for inference. None otherwise.
    prior_state_mean: If initialize_graph has been called, a
        Variable-parameterized Tensor with shape [state dimension];
        the initial prior mean for one or more time series. None otherwise.
    prior_state_var: If initialize_graph has been called, a
        Variable-parameterized Tensor with shape [state dimension x state
        dimension]; the initial prior covariance. None otherwise.
    state_transition_noise_covariance: If initialize_graph has been called, a
        Variable-parameterized Tensor with shape [state noise dimension x state
        noise dimension] indicating the amount of noise added at each
        transition.
  """

  def __init__(self, configuration):
    """Initialize a state space model.

    Args:
      configuration: A StateSpaceModelConfiguration object.
    """
    self._configuration = configuration
    if configuration.filtering_postprocessor is not None:
      filtering_postprocessor_names = (
          configuration.filtering_postprocessor.output_names)
    else:
      filtering_postprocessor_names = []
    super(StateSpaceModel, self).__init__(
        train_output_names=(["mean", "covariance", "log_likelihood"]
                            + filtering_postprocessor_names),
        predict_output_names=["mean", "covariance"],
        num_features=configuration.num_features,
        dtype=configuration.dtype,
        exogenous_feature_columns=configuration.exogenous_feature_columns,
        exogenous_update_condition=configuration.exogenous_update_condition,
        static_unrolling_window_size_threshold=
        configuration.static_unrolling_window_size_threshold)
    self._kalman_filter = None
    self.prior_state_mean = None
    self.prior_state_var = None
    self.state_transition_noise_covariance = None
    self._total_observation_count = None
    self._observation_noise_covariance = None
    # Capture the current variable scope and use it to define all model
    # variables. Especially useful for ensembles, where variables may be defined
    # for every component model in one function call, which would otherwise
    # prevent the user from separating variables from different models into
    # different scopes.
    self._variable_scope = variable_scope.get_variable_scope()

  def transition_power_noise_accumulator(self, num_steps):
    r"""Sum a transitioned covariance matrix over a number of steps.

    Computes

      \sum_{i=0}^{num_steps - 1} (
        state_transition^i
        * state_transition_noise_covariance
        * (state_transition^i)^T)

    If special cases are available, overriding this function can lead to more
    efficient inferences.

    Args:
      num_steps: A [...] shape integer Tensor with numbers of steps to compute
        power sums for.
    Returns:
      The computed power sum, with shape [..., state dimension, state
      dimension].
    """
    # TODO(allenl): This general case should use cumsum if transition_to_powers
    # can be computed in constant time (important for correlated ensembles,
    # where transition_power_noise_accumulator special cases cannot be
    # aggregated from member models).
    noise_transform = ops.convert_to_tensor(self.get_noise_transform(),
                                            self.dtype)
    noise_transformed = math_ops.matmul(
        math_ops.matmul(noise_transform,
                        self.state_transition_noise_covariance),
        noise_transform,
        transpose_b=True)
    noise_additions = math_utils.power_sums_tensor(
        math_ops.reduce_max(num_steps) + 1,
        ops.convert_to_tensor(self.get_state_transition(), dtype=self.dtype),
        noise_transformed)
    return array_ops.gather(noise_additions, indices=num_steps)

  def transition_to_powers(self, powers):
    """Raise the transition matrix to a batch of powers.

    Computes state_transition^powers. If special cases are available, overriding
    this function can lead to more efficient inferences.

    Args:
      powers: A [...] shape integer Tensor with powers to raise the transition
        matrix to.
    Returns:
      The computed matrix powers, with shape [..., state dimension, state
      dimension].
    """
    return math_utils.matrix_to_powers(
        ops.convert_to_tensor(self.get_state_transition(), dtype=self.dtype),
        powers)

  def _window_initializer(self, times, state):
    """Prepare to impute across the gaps in a window."""
    _, _, priors_from_time = state
    times = ops.convert_to_tensor(times)
    priors_from_time = ops.convert_to_tensor(priors_from_time)
    with ops.control_dependencies([
        control_flow_ops.Assert(
            math_ops.reduce_all(priors_from_time <= times[:, 0]),
            [priors_from_time, times[:, 0]],
            summarize=100)
    ]):
      times = array_ops.identity(times)
    intra_batch_gaps = array_ops.reshape(times[:, 1:] - times[:, :-1], [-1])
    starting_gaps = times[:, 0] - priors_from_time
    # Pre-define transition matrices raised to powers (and their sums) for every
    # gap in this window. This avoids duplicate computation (for example many
    # steps will use the transition matrix raised to the first power) and
    # batches the computation rather than doing it inside the per-step loop.
    unique_gaps, _ = array_ops.unique(
        array_ops.concat([intra_batch_gaps, starting_gaps], axis=0))
    self._window_power_sums = self.transition_power_noise_accumulator(
        unique_gaps)
    self._window_transition_powers = self.transition_to_powers(unique_gaps)
    self._window_gap_sizes = unique_gaps

  def _lookup_window_caches(self, caches, indices):
    _, window_power_ids = array_ops.unique(
        array_ops.concat(
            [
                self._window_gap_sizes, math_ops.cast(
                    indices, self._window_gap_sizes.dtype)
            ],
            axis=0))
    all_gathered_indices = []
    for cache in caches:
      gathered_indices = array_ops.gather(
          cache, window_power_ids[-array_ops.shape(indices)[0]:])
      gathered_indices.set_shape(indices.get_shape().concatenate(
          gathered_indices.get_shape()[-2:]))
      all_gathered_indices.append(gathered_indices)
    return all_gathered_indices

  def _cached_transition_powers_and_sums(self, num_steps):
    return self._lookup_window_caches(
        caches=[self._window_transition_powers, self._window_power_sums],
        indices=num_steps)

  def _imputation_step(self, current_times, state):
    """Add state transition noise to catch `state` up to `current_times`.

    State space models are inherently sequential, so we need to "predict
    through" any missing time steps to catch up each element of the batch to its
    next observation/prediction time.

    Args:
      current_times: A [batch size] Tensor of times to impute up to, not
          inclusive.
      state: A tuple of (mean, covariance, previous_times) having shapes
          mean; [batch size x state dimension]
          covariance; [batch size x state dimension x state dimension]
          previous_times; [batch size]
    Returns:
      Imputed model state corresponding to the `state` argument.
    """
    estimated_state, estimated_state_var, previous_times = state
    catchup_times = current_times - previous_times
    non_negative_assertion = control_flow_ops.Assert(
        math_ops.reduce_all(catchup_times >= 0), [
            "Negative imputation interval", catchup_times, current_times,
            previous_times
        ],
        summarize=100)
    with ops.control_dependencies([non_negative_assertion]):
      transition_matrices, transition_noise_sums = (  # pylint: disable=unbalanced-tuple-unpacking
          self._cached_transition_powers_and_sums(catchup_times))
      estimated_state = self._kalman_filter.predict_state_mean(
          estimated_state, transition_matrices)
      estimated_state_var = self._kalman_filter.predict_state_var(
          estimated_state_var, transition_matrices, transition_noise_sums)
    return (estimated_state, estimated_state_var,
            previous_times + catchup_times)

  def _filtering_step(self, current_times, current_values, state, predictions):
    """Compute posteriors and accumulate one-step-ahead predictions.

    Args:
      current_times: A [batch size] Tensor for times for each observation.
      current_values: A [batch size] Tensor of values for each observaiton.
      state: A tuple of (mean, covariance, previous_times) having shapes
          mean; [batch size x state dimension]
          covariance; [batch size x state dimension x state dimension]
          previous_times; [batch size]
      predictions: A dictionary containing mean and covariance Tensors, the
          output of _prediction_step.
    Returns:
      A tuple of (posteriors, outputs):
        posteriors: Model state updated to take `current_values` into account.
        outputs: The `predictions` dictionary updated to include "loss" and
            "log_likelihood" entries (loss simply being negative log
            likelihood).
    """
    estimated_state, estimated_state_covariance, previous_times = state
    observation_model = self.get_broadcasted_observation_model(current_times)
    imputed_to_current_step_assert = control_flow_ops.Assert(
        math_ops.reduce_all(math_ops.equal(current_times, previous_times)),
        ["Attempted to perform filtering without imputation/prediction"])
    with ops.control_dependencies([imputed_to_current_step_assert]):
      estimated_state_covariance = math_utils.clip_covariance(
          estimated_state_covariance,
          self._configuration.filtering_maximum_posterior_variance_ratio,
          self._configuration.filtering_minimum_posterior_variance)
      (filtered_state, filtered_state_covariance,
       log_prob) = self._kalman_filter.do_filter(
           estimated_state=estimated_state,
           estimated_state_covariance=estimated_state_covariance,
           predicted_observation=predictions["mean"],
           predicted_observation_covariance=predictions["covariance"],
           observation=current_values,
           observation_model=observation_model,
           observation_noise=self._observation_noise_covariance)
    filtered_state = (filtered_state, filtered_state_covariance, current_times)
    log_prob.set_shape(current_times.get_shape())
    predictions["loss"] = -log_prob
    predictions["log_likelihood"] = log_prob
    if self._configuration.filtering_postprocessor is not None:
      return self._configuration.filtering_postprocessor.process_filtering_step(
          current_times=current_times,
          current_values=current_values,
          predicted_state=state,
          filtered_state=filtered_state,
          outputs=predictions)
    return (filtered_state, predictions)

  def _prediction_step(self, current_times, state):
    """Make a prediction based on `state`.

    Computes predictions based on the current `state`, checking that it has
    already been updated (in `_imputation_step`) to `current_times`.

    Args:
      current_times: A [batch size] Tensor for times to make predictions for.
      state: A tuple of (mean, covariance, previous_times) having shapes
          mean; [batch size x state dimension]
          covariance; [batch size x state dimension x state dimension]
          previous_times; [batch size]
    Returns:
      A tuple of (updated state, predictions):
        updated state: Model state with added transition noise.
        predictions: A dictionary with "mean" and "covariance", having shapes
            "mean": [batch size x num features]
            "covariance: [batch size x num features x num features]
    """
    estimated_state, estimated_state_var, previous_times = state
    advanced_to_current_assert = control_flow_ops.Assert(
        math_ops.reduce_all(math_ops.equal(current_times, previous_times)),
        ["Attempted to predict without imputation"])
    with ops.control_dependencies([advanced_to_current_assert]):
      observation_model = self.get_broadcasted_observation_model(current_times)
      predicted_obs, predicted_obs_var = (
          self._kalman_filter.observed_from_state(
              state_mean=estimated_state,
              state_var=estimated_state_var,
              observation_model=observation_model,
              observation_noise=self._observation_noise_covariance))
      predicted_obs_var.set_shape(
          ops.convert_to_tensor(current_times).get_shape()
          .concatenate([self.num_features, self.num_features]))
    predicted_obs.set_shape(current_times.get_shape().concatenate(
        (self.num_features,)))
    predicted_obs_var.set_shape(current_times.get_shape().concatenate(
        (self.num_features, self.num_features)))
    predictions = {
        "mean": predicted_obs,
        "covariance": predicted_obs_var}
    state = (estimated_state, estimated_state_var, current_times)
    return (state, predictions)

  def _exogenous_noise_decreasing(self, current_times, exogenous_values, state):
    """Update state with exogenous regressors, decreasing uncertainty.

    Constructs a mean and covariance based on transformations of
    `exogenous_values`, then performs Bayesian inference on the constructed
    observation. This has the effect of lowering uncertainty.

    This update refines or overrides previous inferences, useful for modeling
    exogenous inputs which "set" state, e.g. we dumped boiling water on the
    thermometer so we're pretty sure it's 100 degrees C.

    Args:
      current_times: A [batch size] Tensor of times for the exogenous values
          being input.
      exogenous_values: A [batch size x exogenous input dimension] Tensor of
          exogenous values for each part of the batch.
      state: A tuple of (mean, covariance, previous_times) having shapes
          mean; [batch size x state dimension]
          covariance; [batch size x state dimension x state dimension]
          previous_times; [batch size]
    Returns:
      Updated state taking the exogenous regressors into account (with lower
      uncertainty than the input state).

    """
    estimated_state, estimated_state_covariance, previous_times = state
    state_transition = ops.convert_to_tensor(
        self.get_state_transition(), dtype=self.dtype)
    state_dimension = state_transition.get_shape()[0].value
    # Learning the observation model would be redundant since we transform
    # `exogenous_values` to the state space via a linear transformation anyway.
    observation_model = linalg_ops.eye(
        state_dimension,
        batch_shape=array_ops.shape(exogenous_values)[:-1],
        dtype=self.dtype)
    with variable_scope.variable_scope("exogenous_noise_decreasing_covariance"):
      observation_noise = math_utils.transform_to_covariance_matrices(
          exogenous_values, state_dimension)
    with variable_scope.variable_scope(
        "exogenous_noise_decreasing_observation"):
      observation = layers.fully_connected(
          exogenous_values, state_dimension, activation_fn=None)
    # Pretend that we are making an observation with an observation model equal
    # to the identity matrix (i.e. a direct observation of the latent state),
    # with learned observation noise.
    posterior_state, posterior_state_var = (
        self._kalman_filter.posterior_from_prior_state(
            prior_state=estimated_state,
            prior_state_var=estimated_state_covariance,
            observation=observation,
            observation_model=observation_model,
            predicted_observations=(
                estimated_state,
                # The predicted noise covariance is noise due to current state
                # uncertainty plus noise learned based on the exogenous
                # observation (a somewhat trivial call to
                # self._kalman_filter.observed_from_state has been omitted).
                observation_noise + estimated_state_covariance),
            observation_noise=observation_noise))
    return (posterior_state, posterior_state_var, previous_times)

  def _exogenous_noise_increasing(self, current_times, exogenous_values, state):
    """Update state with exogenous regressors, increasing uncertainty.

    Adds to the state mean a linear transformation of `exogenous_values`, and
    increases uncertainty by constructing a covariance matrix based on
    `exogenous_values` and adding it to the state covariance.

    This update is useful for modeling changes relative to current state,
    e.g. the furnace turned on so the temperature will be increasing at an
    additional 1 degree per minute with some uncertainty, this uncertainty being
    added to our current uncertainty in the per-minute change in temperature.

    Args:
      current_times: A [batch size] Tensor of times for the exogenous values
          being input.
      exogenous_values: A [batch size x exogenous input dimension] Tensor of
          exogenous values for each part of the batch.
      state: A tuple of (mean, covariance, previous_times) having shapes
          mean; [batch size x state dimension]
          covariance; [batch size x state dimension x state dimension]
          previous_times; [batch size]
    Returns:
      Updated state taking the exogenous regressors into account (with higher
      uncertainty than the input state).

    """
    start_mean, start_covariance, previous_times = state
    with variable_scope.variable_scope("exogenous_noise_increasing_mean"):
      mean_addition = layers.fully_connected(
          exogenous_values, start_mean.get_shape()[1].value, activation_fn=None)
    state_dimension = start_covariance.get_shape()[1].value
    with variable_scope.variable_scope("exogenous_noise_increasing_covariance"):
      covariance_addition = (
          math_utils.transform_to_covariance_matrices(
              exogenous_values, state_dimension))
    return (start_mean + mean_addition,
            start_covariance + covariance_addition,
            previous_times)

  def _exogenous_input_step(
      self, current_times, current_exogenous_regressors, state):
    """Update state with exogenous regressors.

    Allows both increases and decreases in uncertainty.

    Args:
      current_times: A [batch size] Tensor of times for the exogenous values
          being input.
      current_exogenous_regressors: A [batch size x exogenous input dimension]
          Tensor of exogenous values for each part of the batch.
      state: A tuple of (mean, covariance, previous_times) having shapes
          mean; [batch size x state dimension]
          covariance; [batch size x state dimension x state dimension]
          previous_times; [batch size]
    Returns:
      Updated state taking the exogenous regressors into account.
    """
    if self._configuration.exogenous_noise_decreases:
      state = self._exogenous_noise_decreasing(
          current_times, current_exogenous_regressors, state)
    if self._configuration.exogenous_noise_increases:
      state = self._exogenous_noise_increasing(
          current_times, current_exogenous_regressors, state)
    return state

  def _loss_additions(self, times, values, mode):
    """Add regularization during training."""
    if mode == estimator_lib.ModeKeys.TRAIN:
      if (self._input_statistics is not None
          and self._configuration.bayesian_prior_weighting):
        normalization = 1. / math_ops.cast(
            self._input_statistics.total_observation_count, self.dtype)
      else:
        # If there is no total observation count recorded, or if we are not
        # doing a Bayesian prior weighting, assumes/pretends that the full
        # dataset size is the window size.
        normalization = 1. / math_ops.cast(
            array_ops.shape(times)[1], self.dtype)
      transition_contribution = ops.convert_to_tensor(
          self._configuration.covariance_prior_fn(
              self.state_transition_noise_covariance),
          dtype=self.dtype)
      if (self._configuration.use_observation_noise
          and self._observation_noise_covariance is not None):
        observation_contribution = ops.convert_to_tensor(
            self._configuration.covariance_prior_fn(
                self._observation_noise_covariance),
            dtype=self.dtype)
        regularization_sum = transition_contribution + observation_contribution
      else:
        regularization_sum = transition_contribution
      return -normalization * regularization_sum
    else:
      return array_ops.zeros([], dtype=self.dtype)

  def _variable_observation_transition_tradeoff_log(self):
    """Define a variable to trade off observation and transition noise."""
    return variable_scope.get_variable(
        name="observation_transition_tradeoff_log_scale",
        initializer=constant_op.constant(
            -self._configuration.transition_covariance_initial_log_scale_bias,
            dtype=self.dtype),
        dtype=self.dtype)

  def _define_parameters(self, observation_transition_tradeoff_log=None):
    """Define extra model-specific parameters.

    Models should wrap any variables defined here in the model's variable scope.

    Args:
      observation_transition_tradeoff_log: An ensemble-global parameter
        controlling the tradeoff between observation noise and transition
        noise. If its value is not None, component transition noise should scale
        with e^-observation_transition_tradeoff_log.
    """
    with variable_scope.variable_scope(self._variable_scope):
      # A scalar which allows the optimizer to quickly shift from observation
      # noise to transition noise (this value is subtracted from log transition
      # noise and added to log observation noise).
      if observation_transition_tradeoff_log is None:
        self._observation_transition_tradeoff_log_scale = (
            self._variable_observation_transition_tradeoff_log())
      else:
        self._observation_transition_tradeoff_log_scale = (
            observation_transition_tradeoff_log)
      self.state_transition_noise_covariance = (
          self.get_state_transition_noise_covariance())

  def _set_input_statistics(self, input_statistics=None):
    super(StateSpaceModel, self).initialize_graph(
        input_statistics=input_statistics)

  def initialize_graph(self, input_statistics=None):
    """Define variables and ops relevant to the top-level model in an ensemble.

    For generic model parameters, _define_parameters() is called recursively on
    all members of an ensemble.

    Args:
      input_statistics: A math_utils.InputStatistics object containing input
          statistics. If None, data-independent defaults are used, which may
          result in longer or unstable training.
    """
    self._set_input_statistics(input_statistics=input_statistics)
    self._define_parameters()
    with variable_scope.variable_scope(self._variable_scope):
      self._observation_noise_covariance = ops.convert_to_tensor(
          self.get_observation_noise_covariance(), dtype=self.dtype)
    self._kalman_filter = kalman_filter.KalmanFilter(dtype=self.dtype)
    (self.prior_state_mean,
     self.prior_state_var) = self._make_priors()

  def _make_priors(self):
    """Creates and returns model priors."""
    prior_state_covariance = self.get_prior_covariance()
    prior_state_mean = self.get_prior_mean()
    return (prior_state_mean, prior_state_covariance)

  def get_prior_covariance(self):
    """Constructs a variable prior covariance with data-based initialization.

    Models should wrap any variables defined here in the model's variable scope.

    Returns:
      A two-dimensional [state dimension, state dimension] floating point Tensor
      with a (positive definite) prior state covariance matrix.
    """
    with variable_scope.variable_scope(self._variable_scope):
      state_dimension = ops.convert_to_tensor(
          self.get_state_transition()).get_shape()[0].value
      if self._configuration.trainable_start_state:
        base_covariance = math_utils.variable_covariance_matrix(
            state_dimension, "prior_state_var",
            dtype=self.dtype)
      else:
        return linalg_ops.eye(state_dimension, dtype=self.dtype)
      if self._input_statistics is not None:
        # Make sure initial latent value uncertainty is at least on the same
        # scale as noise in the data.
        covariance_multiplier = math_ops.reduce_max(
            self._input_statistics.series_start_moments.variance)
        return base_covariance * gen_math_ops.maximum(
            covariance_multiplier, 1.0)
      else:
        return base_covariance

  def get_prior_mean(self):
    """Constructs a Variable-parameterized prior mean.

    Models should wrap any variables defined here in the model's variable scope.

    Returns:
      A one-dimensional floating point Tensor with shape [state dimension]
      indicating the prior mean.
    """
    with variable_scope.variable_scope(self._variable_scope):
      state_transition = ops.convert_to_tensor(
          self.get_state_transition(), dtype=self.dtype)
      state_dimension = state_transition.get_shape()[0].value
      return variable_scope.get_variable(
          name="prior_state_mean",
          shape=[state_dimension],
          dtype=self.dtype,
          trainable=self._configuration.trainable_start_state)

  # TODO(allenl): It would be nice if the generation were done with TensorFlow
  # ops, and if the model parameters were somehow set instead of being passed
  # around in a dictionary. Maybe unconditional generation should be through a
  # special set of initializers?
  def random_model_parameters(self, seed=None):
    if self.num_features != 1:
      raise NotImplementedError("Generation for multivariate state space models"
                                " is not currently implemented.")
    if seed:
      numpy.random.seed(seed)
    state_dimension, noise_dimension = ops.convert_to_tensor(
        self.get_noise_transform()).get_shape().as_list()
    transition_var = 1.0 / numpy.random.gamma(shape=10., scale=10.,
                                              size=[noise_dimension])
    initial_state = numpy.random.normal(size=[state_dimension])
    params_dict = {}
    if self.prior_state_mean is not None:
      params_dict[self.prior_state_mean] = initial_state
    if self.state_transition_noise_covariance is not None:
      params_dict[self.state_transition_noise_covariance] = numpy.diag(
          transition_var)
    if self.prior_state_var is not None:
      params_dict[self.prior_state_var] = numpy.zeros(
          [state_dimension, state_dimension])
    if self._configuration.use_observation_noise:
      observation_var = 1.0 / numpy.random.gamma(shape=4, scale=4)
      params_dict[self._observation_noise_covariance] = [[observation_var]]
    return params_dict

  def generate(self, number_of_series, series_length,
               model_parameters=None, seed=None, add_observation_noise=None):
    if seed is not None:
      numpy.random.seed(seed)
    if self.num_features != 1:
      raise NotImplementedError("Generation for multivariate state space models"
                                " is not currently implemented.")
    if add_observation_noise is None:
      add_observation_noise = self._configuration.use_observation_noise
    if model_parameters is None:
      model_parameters = {}
    transitions = ops.convert_to_tensor(
        self.get_state_transition(), dtype=self.dtype).eval(
            feed_dict=model_parameters)
    noise_transform = ops.convert_to_tensor(self.get_noise_transform()).eval(
        feed_dict=model_parameters)

    noise_dimension = noise_transform.shape[1]
    get_passed_or_trained_value = model_utils.parameter_switch(model_parameters)
    transition_var = numpy.diag(get_passed_or_trained_value(
        self.state_transition_noise_covariance))
    transition_std = numpy.sqrt(transition_var)
    if add_observation_noise:
      observation_var = get_passed_or_trained_value(
          self._observation_noise_covariance)[0][0]
      observation_std = numpy.sqrt(observation_var)
    initial_state = get_passed_or_trained_value(self.prior_state_mean)
    current_state = numpy.tile(numpy.expand_dims(initial_state, 0),
                               [number_of_series, 1])
    observations = numpy.zeros([number_of_series, series_length])
    observation_models = self.get_broadcasted_observation_model(
        times=math_ops.range(series_length)).eval(feed_dict=model_parameters)
    for timestep, observation_model in enumerate(observation_models):
      current_state = numpy.dot(current_state, transitions.T)
      current_state += numpy.dot(
          numpy.random.normal(
              loc=numpy.zeros([number_of_series, noise_dimension]),
              scale=numpy.tile(numpy.expand_dims(transition_std, 0),
                               [number_of_series, 1])),
          noise_transform.T)
      observation_mean = numpy.dot(current_state, observation_model[0].T)
      if add_observation_noise:
        observations[:, timestep] = numpy.random.normal(loc=observation_mean,
                                                        scale=observation_std)
      else:
        observations[:, timestep] = observation_mean
    observations = numpy.expand_dims(observations, -1)
    times = numpy.tile(
        numpy.expand_dims(numpy.arange(observations.shape[1]), 0),
        [observations.shape[0], 1])
    return {TrainEvalFeatures.TIMES: times,
            TrainEvalFeatures.VALUES: observations}

  @abc.abstractmethod
  def get_state_transition(self):
    """Specifies the state transition model to use.

    Returns:
      A [state dimension x state dimension] Tensor specifying how states
      transition from one timestep to the next.
    """
    pass

  @abc.abstractmethod
  def get_noise_transform(self):
    """Specifies the noise transition model to use.

    Returns:
      A [state dimension x state noise dimension] Tensor specifying how noise
      (generated with shape [state noise dimension]) affects the model's state.
    """
    pass

  @abc.abstractmethod
  def get_observation_model(self, times):
    """Specifies the observation model to use.

    Args:
      times: A [batch dimension] int32 Tensor with times for each part of the
          batch, on which the observation model can depend.
    Returns:
      This function, when overridden, has three possible return values:
        - A [state dimension] Tensor with a static, univariate observation
          model.
        - A [self.num_features x state dimension] static, multivariate model.
        - A [batch dimension x self.num_features x state dimension] observation
          model, which may depend on `times`.
      See get_broadcasted_observation_model for details of the broadcasting.
    """
    pass

  def get_broadcasted_observation_model(self, times):
    """Broadcast this model's observation model if necessary.

    The model can define a univariate observation model which will be broadcast
    over both self.num_features and the batch dimension of `times`.

    The model can define a multi-variate observation model which does not depend
    on `times`, and it will be broadcast over the batch dimension of `times`.

    Finally, the model can define a multi-variate observation model with a batch
    dimension, which will not be broadcast.

    Args:
      times: A [batch dimension] int32 Tensor with times for each part of the
          batch, on which the observation model can depend.
    Returns:
      A [batch dimension x self.num_features x state dimension] Tensor
      specifying the observation model to use for each time in `times` and each
      feature.
    """
    unbroadcasted_model = ops.convert_to_tensor(
        self.get_observation_model(times), dtype=self.dtype)
    unbroadcasted_shape = (unbroadcasted_model.get_shape()
                           .with_rank_at_least(1).with_rank_at_most(3))
    if unbroadcasted_shape.ndims is None:
      # Pass through fully undefined shapes, but make sure they're rank 3 at
      # graph eval time
      assert_op = control_flow_ops.Assert(
          math_ops.equal(array_ops.rank(unbroadcasted_model), 3),
          [array_ops.shape(unbroadcasted_model)])
      with ops.control_dependencies([assert_op]):
        return array_ops.identity(unbroadcasted_model)
    if unbroadcasted_shape.ndims == 1:
      # Unbroadcasted shape [state dimension]
      broadcasted_model = array_ops.tile(
          array_ops.reshape(tensor=unbroadcasted_model, shape=[1, 1, -1]),
          [array_ops.shape(times)[0], self.num_features, 1])
    elif unbroadcasted_shape.ndims == 2:
      # Unbroadcasted shape [num features x state dimension]
      broadcasted_model = array_ops.tile(
          array_ops.expand_dims(unbroadcasted_model, dim=0),
          [array_ops.shape(times)[0], 1, 1])
    elif unbroadcasted_shape.ndims == 3:
      broadcasted_model = unbroadcasted_model
    broadcasted_model.get_shape().assert_has_rank(3)
    return broadcasted_model

  def get_state_transition_noise_covariance(
      self, minimum_initial_variance=1e-5):
    state_noise_transform = ops.convert_to_tensor(
        self.get_noise_transform(), dtype=self.dtype)
    state_noise_dimension = state_noise_transform.get_shape()[1].value
    if self._input_statistics is not None:
      feature_variance = self._input_statistics.series_start_moments.variance
      initial_transition_noise_scale = math_ops.log(
          gen_math_ops.maximum(
              math_ops.reduce_mean(feature_variance) / math_ops.cast(
                  self._input_statistics.total_observation_count, self.dtype),
              minimum_initial_variance))
    else:
      initial_transition_noise_scale = 0.
    # Generally high transition noise is undesirable; we want to set it quite
    # low to start so that we don't need too much training to get to good
    # solutions (i.e. with confident predictions into the future if possible),
    # but not so low that training can't yield a high transition noise if the
    # data demands it.
    initial_transition_noise_scale -= (
        self._observation_transition_tradeoff_log_scale)
    return math_utils.variable_covariance_matrix(
        state_noise_dimension, "state_transition_noise",
        dtype=self.dtype,
        initial_overall_scale_log=initial_transition_noise_scale)

  def get_observation_noise_covariance(self, minimum_initial_variance=1e-5):
    if self._configuration.use_observation_noise:
      if self._input_statistics is not None:
        # Get variance across the first few values in each batch for each
        # feature, for an initial observation noise (over-)estimate.
        feature_variance = self._input_statistics.series_start_moments.variance
      else:
        feature_variance = None
      if feature_variance is not None:
        feature_variance = gen_math_ops.maximum(feature_variance,
                                                minimum_initial_variance)
      return math_utils.variable_covariance_matrix(
          size=self.num_features,
          dtype=self.dtype,
          name="observation_noise_covariance",
          initial_diagonal_values=feature_variance,
          initial_overall_scale_log=(
              self._observation_transition_tradeoff_log_scale))
    else:
      return array_ops.zeros(
          shape=[self.num_features, self.num_features],
          name="observation_noise_covariance",
          dtype=self.dtype)

  def get_start_state(self):
    """Defines and returns a non-batched prior state and covariance."""
    # TODO(allenl,vitalyk): Add an option for non-Gaussian priors once extended
    # Kalman filtering is implemented (ideally any Distribution object).
    if self._input_statistics is not None:
      start_time = self._input_statistics.start_time
    else:
      start_time = array_ops.zeros([], dtype=dtypes.int64)
    return (self.prior_state_mean,
            self.prior_state_var,
            start_time - 1)

  def get_features_for_timesteps(self, timesteps):
    """Get features for a batch of timesteps. Default to no features."""
    return array_ops.zeros([array_ops.shape(timesteps)[0], 0], dtype=self.dtype)


class StateSpaceEnsemble(StateSpaceModel):
  """Base class for combinations of state space models."""

  def __init__(self, ensemble_members, configuration):
    """Initialize the ensemble by specifying its members.

    Args:
      ensemble_members: A list of StateSpaceModel objects which will be included
          in this ensemble.
      configuration: A StateSpaceModelConfiguration object.
    """
    self._ensemble_members = ensemble_members
    super(StateSpaceEnsemble, self).__init__(configuration=configuration)

  def _set_input_statistics(self, input_statistics):
    super(StateSpaceEnsemble, self)._set_input_statistics(input_statistics)
    for member in self._ensemble_members:
      member._set_input_statistics(input_statistics)  # pylint: disable=protected-access

  def _loss_additions(self, times, values, mode):
    # Allow sub-models to regularize
    return (super(StateSpaceEnsemble, self)._loss_additions(
        times, values, mode) + math_ops.add_n([
            member._loss_additions(times, values, mode)  # pylint: disable=protected-access
            for member in self._ensemble_members
        ]))

  def _compute_blocked(self, member_fn, name):
    with variable_scope.variable_scope(self._variable_scope):
      return math_utils.block_diagonal(
          [member_fn(member)
           for member in self._ensemble_members],
          dtype=self.dtype,
          name=name)

  def transition_to_powers(self, powers):
    return self._compute_blocked(
        member_fn=lambda member: member.transition_to_powers(powers),
        name="ensemble_transition_to_powers")

  def _define_parameters(self, observation_transition_tradeoff_log=None):
    with variable_scope.variable_scope(self._variable_scope):
      if observation_transition_tradeoff_log is None:
        # Define the tradeoff parameter between observation and transition noise
        # once for the whole ensemble, and pass it down to members.
        observation_transition_tradeoff_log = (
            self._variable_observation_transition_tradeoff_log())
      for member in self._ensemble_members:
        member._define_parameters(observation_transition_tradeoff_log=(  # pylint: disable=protected-access
            observation_transition_tradeoff_log))
      super(StateSpaceEnsemble, self)._define_parameters(
          observation_transition_tradeoff_log
          =observation_transition_tradeoff_log)

  def random_model_parameters(self, seed=None):
    param_union = {}
    for i, member in enumerate(self._ensemble_members):
      member_params = member.random_model_parameters(
          seed=seed + i if seed else None)
      param_union.update(member_params)
    param_union.update(
        super(StateSpaceEnsemble, self).random_model_parameters(seed=seed))
    return param_union

  def get_prior_mean(self):
    return array_ops.concat(
        values=[member.get_prior_mean() for member in self._ensemble_members],
        axis=0,
        name="ensemble_prior_state_mean")

  def get_state_transition(self):
    return self._compute_blocked(
        member_fn=
        lambda member: member.get_state_transition(),
        name="ensemble_state_transition")

  def get_noise_transform(self):
    return self._compute_blocked(
        member_fn=
        lambda member: member.get_noise_transform(),
        name="ensemble_noise_transform")

  def get_observation_model(self, times):
    raise NotImplementedError("No un-broadcasted observation model defined for"
                              " ensembles.")

  def get_broadcasted_observation_model(self, times):
    """Computes a combined observation model based on member models.

    The effect is that predicted observations from each model are summed.

    Args:
      times: A [batch dimension] int32 Tensor with times for each part of the
          batch, on which member observation models can depend.
    Returns:
      A [batch dimension x num features x combined state dimension] Tensor with
      the combined observation model.
    """
    member_observation_models = [
        ops.convert_to_tensor(
            member.get_broadcasted_observation_model(times), dtype=self.dtype)
        for member in self._ensemble_members
    ]
    return array_ops.concat(values=member_observation_models, axis=2)


class StateSpaceIndependentEnsemble(StateSpaceEnsemble):
  """Implements ensembles of independent state space models.

  Useful for fitting multiple independent state space models together while
  keeping their specifications decoupled. The "ensemble" is simply a state space
  model with the observation models of its members concatenated, and the
  transition matrices and noise transforms stacked in block-diagonal
  matrices. This means that the dimensionality of the ensemble's state is the
  sum of those of its components, which can lead to slow and memory-intensive
  training and inference as the posterior (shape [state dimension x state
  dimension]) gets large.

  Each individual model j's state at time t is defined by:

  state[t, j] = StateTransition[j] * state[t-1, j]
      + NoiseTransform[j] * StateNoise[t, j]
  StateNoise[t, j] ~ Gaussian(0, StateNoiseCovariance[j])

  and the ensemble observation model is:

  observation[t] = Sum { ObservationModel[j] * state[t, j] }
      + ObservationNoise[t]
  ObservationNoise[t] ~ Gaussian(0, ObservationNoiseCovariance)
  """

  def transition_power_noise_accumulator(self, num_steps):
    return self._compute_blocked(
        member_fn=lambda m: m.transition_power_noise_accumulator(num_steps),
        name="ensemble_power_noise_accumulator")

  def get_prior_covariance(self):
    """Construct the ensemble prior covariance based on component models."""
    return self._compute_blocked(
        member_fn=
        lambda member: member.get_prior_covariance(),
        name="ensemble_prior_state_covariance")

  def get_state_transition_noise_covariance(self):
    """Construct the ensemble transition noise covariance from components."""
    return self._compute_blocked(
        member_fn=
        lambda member: member.state_transition_noise_covariance,
        name="ensemble_state_transition_noise")


# TODO(allenl): It would be nice to have replicated feature models which are
# identical batched together to reduce the graph size.
# TODO(allenl): Support for sharing M independent models across N features, with
# N > M.
# TODO(allenl): Stack component prior covariances while allowing cross-model
# correlations to be learned (currently a full covariance prior is learned, but
# custom component model covariances are not used).
class StateSpaceCorrelatedFeaturesEnsemble(StateSpaceEnsemble):
  """An correlated ensemble where each model represents a feature.

  Unlike `StateSpaceIndependentEnsemble`, a full state transition noise
  covariance matrix is learned for this ensemble; the models are not assumed to
  be independent. Rather than concatenating observation models (i.e. summing the
  contributions of each model to each feature),
  StateSpaceCorrelatedFeaturesEnsemble stacks observation models diagonally,
  meaning that each model corresponds to one feature of the series.

  Behaves like (and is) a single state space model where:

  StateTransition = Diag(StateTransition[j] for models j)
  ObservationModel = Diag(ObservationModel[j] for models j)

  Note that each ObservationModel[j] is a [1 x S_j] matrix (S_j being the state
  dimension of model j), i.e. a univariate model. The combined model is
  multivariate, the number of features of the series being equal to the number
  of component models in the ensemble.
  """

  def __init__(self, ensemble_members, configuration):
    """Specify the ensemble's configuration and component models.

    Args:
      ensemble_members: A list of `StateSpaceModel` objects, with length equal
        to `configuration.num_features`. Each of these models, which must be
        univariate, corresponds to a single feature of the time series.
      configuration: A StateSpaceModelConfiguration object.
    Raises:
      ValueError: If the length of `ensemble_members` does not equal the number
        of features in the series, or any component is not univariate.
    """
    if len(ensemble_members) != configuration.num_features:
      raise ValueError(
          "The number of members in a StateSpaceCorrelatedFeaturesEnsemble "
          "must equal the number of features in the time series.")
    for member in ensemble_members:
      if member.num_features != 1:
        raise ValueError(
            "StateSpaceCorrelatedFeaturesEnsemble components must be "
            "univariate.")
    super(StateSpaceCorrelatedFeaturesEnsemble, self).__init__(
        ensemble_members=ensemble_members, configuration=configuration)

  def transition_power_noise_accumulator(self, num_steps):
    """Use a noise accumulator special case when possible."""
    if len(self._ensemble_members) == 1:
      # If this is a univariate series, we should use the special casing built
      # into the single component model.
      return self._ensemble_members[0].transition_power_noise_accumulator(
          num_steps=num_steps)
    # If we have multiple features, and therefore multiple models, we have
    # introduced correlations which make noise accumulation more
    # complicated. Here we fall back to the general case, since we can't just
    # aggregate member special cases.
    return super(StateSpaceCorrelatedFeaturesEnsemble,
                 self).transition_power_noise_accumulator(num_steps=num_steps)

  def get_broadcasted_observation_model(self, times):
    """Stack observation models diagonally."""
    def _member_observation_model(member):
      return ops.convert_to_tensor(
          member.get_broadcasted_observation_model(times), dtype=self.dtype)
    return self._compute_blocked(member_fn=_member_observation_model,
                                 name="feature_ensemble_observation_model")
