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
"""Implements a time series model with seasonality, trends, and transients."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.timeseries.python.timeseries.state_space_models import level_trend
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import periodic
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import state_space_model
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import varma

from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


def _replicate_level_trend_models(multivariate_configuration,
                                  univariate_configuration):
  """Helper function to construct a multivariate level/trend component."""
  with variable_scope.variable_scope("adder"):
    # Construct a level and trend model for each feature, with correlated
    # transition noise.
    adder_features = []
    for feature in range(multivariate_configuration.num_features):
      with variable_scope.variable_scope("feature{}".format(feature)):
        adder_features.append(level_trend.AdderStateSpaceModel(
            configuration=univariate_configuration))
    adder_part = state_space_model.StateSpaceCorrelatedFeaturesEnsemble(
        ensemble_members=adder_features,
        configuration=multivariate_configuration)
  return adder_part


class StructuralEnsemble(state_space_model.StateSpaceIndependentEnsemble):
  r"""A structural state space time series model.

  In the spirit of:

  Scott, Steven L., and Hal R. Varian. "Predicting the present with bayesian
    structural time series." International Journal of Mathematical Modelling and
    Numerical Optimisation 5.1-2 (2014): 4-23.

  Without the spike-and-slab prior, and with point estimates of parameters
  instead of sampling.

  The model includes level, trend, seasonality, and a transient moving average.

  An observation at time t is drawn according to:
    observation_t = level_t + seasonality_t + moving_average_t
        + observation_noise_t
    level_t = level_{t-1} + trend_{t-1} + level_noise_t
    trend_t = trend_{t-1} + trend_noise_t
    seasonality_t = -\sum_{n=1}^{num_seasons-1} seasonality_{t-n} +
        seasonality_noise_t
    moving_average_t = transient_t
        + \sum_{j=1}^{moving_average_order} ma_coefs_j * transient_{t - j}

  `observation_noise`, `level_noise`, `trend noise`, `seasonality_noise`, and
  `transient` are (typically scalar) Gaussian random variables whose variance is
  learned from data, and that variance is not time dependent in this
  implementation. Level noise is optional due to its similarity with observation
  noise in some cases. Seasonality is enforced by constraining a full cycle of
  seasonal variables to have zero expectation, allowing seasonality to adapt
  over time. The moving average coefficients `ma_coefs` are learned.

  When presented with a multivariate series (more than one "feature", here
  referring to endogenous features of the series), the model is replicated
  across these features (one copy per feature of each periodic component, and
  one level/trend model per feature), and correlations in transition noise are
  learned between these replicated components (see
  StateSpaceCorrelatedFeaturesEnsemble). This is in addition to the learned
  correlations in observation noise between features. While this is often the
  most expressive thing to do with multiple features, it does mean that the
  model grows quite quickly, creating and computing with square matrices with
  each dimension equal to num_features * (sum(periodicities) +
  moving_average_order + 3), meaning that some operations are approximately
  cubic in this value.
  """
  # TODO(allenl): Implement partial model replication/sharing for multivariate
  # series (to save time/memory when the series presented can be modeled as a
  # smaller number of underlying series). Likely just a modification of the
  # observation model so that each feature of the series is a learned linear
  # combination of the replicated models.

  def __init__(self,
               periodicities,
               moving_average_order,
               autoregressive_order,
               use_level_noise=True,
               configuration=state_space_model.StateSpaceModelConfiguration()):
    """Initialize the Basic Structural Time Series model.

    Args:
      periodicities: Number of time steps for cyclic behavior. May be a list, in
          which case one periodic component is created for each element.
      moving_average_order: The number of moving average coefficients to use,
          which also defines the number of steps after which transient
          deviations revert to the mean defined by periodic and level/trend
          components.
      autoregressive_order: The number of steps back for autoregression.
      use_level_noise: Whether to model the time series as having level
          noise. See level_noise in the model description above.
      configuration: A StateSpaceModelConfiguration object.
    """
    component_model_configuration = configuration._replace(
        use_observation_noise=False)
    univariate_component_model_configuration = (
        component_model_configuration._replace(
            num_features=1))

    adder_part = _replicate_level_trend_models(
        multivariate_configuration=component_model_configuration,
        univariate_configuration=univariate_component_model_configuration)
    with variable_scope.variable_scope("varma"):
      varma_part = varma.VARMA(
          autoregressive_order=autoregressive_order,
          moving_average_order=moving_average_order,
          configuration=component_model_configuration)

    cycle_parts = []
    periodicity_list = nest.flatten(periodicities)
    for cycle_number, cycle_periodicity in enumerate(periodicity_list):
      # For each specified periodicity, construct models for each feature with
      # correlated noise.
      with variable_scope.variable_scope("cycle{}".format(cycle_number)):
        cycle_features = []
        for feature in range(configuration.num_features):
          with variable_scope.variable_scope("feature{}".format(feature)):
            cycle_features.append(periodic.CycleStateSpaceModel(
                periodicity=cycle_periodicity,
                configuration=univariate_component_model_configuration))
        cycle_parts.append(
            state_space_model.StateSpaceCorrelatedFeaturesEnsemble(
                ensemble_members=cycle_features,
                configuration=component_model_configuration))

    super(StructuralEnsemble, self).__init__(
        ensemble_members=[adder_part, varma_part] + cycle_parts,
        configuration=configuration)


# TODO(allenl): Implement a multi-resolution moving average component to
# decouple model size from the length of transient deviations.
class MultiResolutionStructuralEnsemble(
    state_space_model.StateSpaceIndependentEnsemble):
  """A structural ensemble modeling arbitrary periods with a fixed model size.

  See periodic.ResolutionCycleModel, which allows a fixed number of latent
  values to cycle at multiple/variable resolutions, for more details on the
  difference between MultiResolutionStructuralEnsemble and
  StructuralEnsemble. With `cycle_num_latent_values` (controlling model size)
  equal to `periodicities` (controlling the time over which these values
  complete a full cycle), the models are
  equivalent. MultiResolutionStructuralEnsemble allows `periodicities` to vary
  while the model size remains fixed. Note that high `periodicities` without a
  correspondingly high `cycle_num_latent_values` means that the modeled series
  must have a relatively smooth periodic component.

  Multiple features are handled the same way as in StructuralEnsemble (one
  replication per feature, with correlations learned between the replicated
  models). This strategy produces a very flexible model, but means that series
  with many features may be slow to train.

  Model size (the state dimension) is:
    num_features * (sum(cycle_num_latent_values)
      + max(moving_average_order + 1, autoregressive_order) + 2)
  """

  def __init__(self,
               cycle_num_latent_values,
               moving_average_order,
               autoregressive_order,
               periodicities,
               use_level_noise=True,
               configuration=state_space_model.StateSpaceModelConfiguration()):
    """Initialize the multi-resolution structural ensemble.

    Args:
      cycle_num_latent_values: Controls the model size and the number of latent
          values cycled between (but not the periods over which they cycle).
          Reducing this parameter can save significant amounts of memory, but
          the tradeoff is with resolution: cycling between a smaller number of
          latent values means that only smoother functions can be modeled. For
          multivariate series, may either be a scalar integer (in which case it
          is applied to all periodic components) or a list with length matching
          `periodicities`.
      moving_average_order: The number of moving average coefficients to use,
          which also defines the number of steps after which transient
          deviations revert to the mean defined by periodic and level/trend
          components. Adds to model size.
      autoregressive_order: The number of steps back for
          autoregression. Learning autoregressive coefficients typically
          requires more steps and a smaller step size than other components.
      periodicities: Same meaning as for StructuralEnsemble: number of steps for
          cyclic behavior. Floating point and Tensor values are supported. May
          be a list of values, in which case one component is created for each
          periodicity. If `periodicities` is a list while
          `cycle_num_latent_values` is a scalar, its value is broadcast to each
          periodic component. Otherwise they should be lists of the same length,
          in which case they are paired.
      use_level_noise: See StructuralEnsemble.
      configuration: A StateSpaceModelConfiguration object.
    Raises:
      ValueError: If `cycle_num_latent_values` is neither a scalar nor agrees in
          size with `periodicities`.
    """
    component_model_configuration = configuration._replace(
        use_observation_noise=False)
    univariate_component_model_configuration = (
        component_model_configuration._replace(
            num_features=1))

    adder_part = _replicate_level_trend_models(
        multivariate_configuration=component_model_configuration,
        univariate_configuration=univariate_component_model_configuration)
    with variable_scope.variable_scope("varma"):
      varma_part = varma.VARMA(
          autoregressive_order=autoregressive_order,
          moving_average_order=moving_average_order,
          configuration=component_model_configuration)

    cycle_parts = []
    if periodicities is None:
      periodicities = []
    periodicity_list = nest.flatten(periodicities)
    latent_values_list = nest.flatten(cycle_num_latent_values)
    if len(periodicity_list) != len(latent_values_list):
      if len(latent_values_list) != 1:
        raise ValueError(
            ("`cycle_num_latent_values` must either be a list with the same "
             "size as `periodicity` or a scalar. Received length {} "
             "`cycle_num_latent_values`, while `periodicities` has length {}.")
            .format(len(latent_values_list), len(periodicity_list)))
      latent_values_list *= len(periodicity_list)
    for cycle_number, (cycle_periodicity, num_latent_values) in enumerate(
        zip(periodicity_list, latent_values_list)):
      with variable_scope.variable_scope("cycle{}".format(cycle_number)):
        cycle_features = []
        for feature in range(configuration.num_features):
          with variable_scope.variable_scope("feature{}".format(feature)):
            cycle_features.append(
                periodic.ResolutionCycleModel(
                    num_latent_values=num_latent_values,
                    periodicity=cycle_periodicity,
                    configuration=univariate_component_model_configuration))
        cycle_parts.append(
            state_space_model.StateSpaceCorrelatedFeaturesEnsemble(
                ensemble_members=cycle_features,
                configuration=component_model_configuration))

    super(MultiResolutionStructuralEnsemble, self).__init__(
        ensemble_members=[adder_part, varma_part] + cycle_parts,
        configuration=configuration)
