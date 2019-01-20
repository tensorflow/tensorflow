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
"""Filtering postprocessors for SequentialTimeSeriesModels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.contrib.timeseries.python.timeseries import math_utils

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


@six.add_metaclass(abc.ABCMeta)
class FilteringStepPostprocessor(object):
  """Base class for processors that are applied after each filter step."""

  @abc.abstractmethod
  def process_filtering_step(self, current_times, current_values,
                             predicted_state, filtered_state, outputs):
    """Extends/modifies a filtering step, altering state and loss.

    Args:
      current_times: A [batch size] integer Tensor of times.
      current_values: A [batch size x num features] Tensor of values filtering
          is being performed on.
      predicted_state: A (possibly nested) list of Tensors indicating model
          state which does not take `current_times` and `current_values` into
          account.
      filtered_state: Same structure as predicted_state, but updated to take
          `current_times` and `current_values` into account.
      outputs: A dictionary of outputs produced by model filtering
          (SequentialTimeSeriesModel._process_filtering_step).
    Returns: A tuple of (new_state, updated_outputs);
      new_state: Updated state with the same structure as `filtered_state` and
          `predicted_state`.
      updated_outputs: The `outputs` dictionary, updated with any new outputs
          from this filtering postprocessor.
    """
    pass

  @abc.abstractproperty
  def output_names(self):
    return []


def cauchy_alternative_to_gaussian(current_times, current_values, outputs):
  """A Cauchy anomaly distribution, centered at a Gaussian prediction.

  Performs an entropy-matching approximation of the scale parameters of
  independent Cauchy distributions given the covariance matrix of a multivariate
  Gaussian in outputs["covariance"], and centers the Cauchy distributions at
  outputs["mean"]. This requires that the model that we are creating an
  alternative/anomaly distribution for produces a mean and covariance.

  Args:
    current_times: A [batch size] Tensor of times, unused.
    current_values: A [batch size x num features] Tensor of values to evaluate
        the anomaly distribution at.
    outputs: A dictionary of Tensors with keys "mean" and "covariance"
        describing the Gaussian to construct an anomaly distribution from. The
        value corresponding to "mean" has shape [batch size x num features], and
        the value corresponding to "covariance" has shape [batch size x num
        features x num features].
  Returns:
    A [batch size] Tensor of log likelihoods; the anomaly log PDF evaluated at
    `current_values`.
  """
  del current_times  # unused
  cauchy_scale = math_utils.entropy_matched_cauchy_scale(outputs["covariance"])
  individual_log_pdfs = math_utils.cauchy_log_prob(
      loc=outputs["mean"],
      scale=cauchy_scale,
      x=current_values)
  return math_ops.reduce_sum(individual_log_pdfs, axis=1)


def _interpolate_state_linear(first_state, second_state, first_responsibility):
  """Interpolate between two model states linearly."""
  interpolated_state_flat = []
  for first_state_tensor, second_state_tensor in zip(
      nest.flatten(first_state), nest.flatten(second_state)):
    assert first_state_tensor.dtype == second_state_tensor.dtype
    if first_state_tensor.dtype.is_floating:
      # Pad the responsibility shape with ones up to the state's rank so that it
      # broadcasts
      first_responsibility_padded = array_ops.reshape(
          tensor=first_responsibility,
          shape=array_ops.concat([
              array_ops.shape(first_responsibility), array_ops.ones(
                  [array_ops.rank(first_state_tensor) - 1], dtype=dtypes.int32)
          ], 0))
      interpolated_state = (
          first_responsibility_padded * first_state_tensor
          + (1. - first_responsibility_padded) * second_state_tensor)
      interpolated_state.set_shape(first_state_tensor.get_shape())
      interpolated_state_flat.append(interpolated_state)
    else:
      # Integer dtypes are probably representing times, and don't need
      # interpolation. Make sure they're identical to be sure.
      with ops.control_dependencies(
          [check_ops.assert_equal(first_state_tensor, second_state_tensor)]):
        interpolated_state_flat.append(array_ops.identity(first_state_tensor))
  return nest.pack_sequence_as(first_state, interpolated_state_flat)


class StateInterpolatingAnomalyDetector(FilteringStepPostprocessor):
  """An anomaly detector which guards model state against outliers.

  Smoothly interpolates between a model's predicted and inferred states, based
  on the posterior probability of an anomaly, p(anomaly | data). This is useful
  if anomalies would otherwise lead to model state which is hard to recover
  from (Gaussian state space models suffer from this, for example).

  Relies on (1) an alternative distribution, typically with heavier tails than
  the model's normal predictions, and (2) a prior probability of an anomaly. The
  prior probability acts as a penalty, discouraging the system from marking too
  many points as anomalies. The alternative distribution indicates the
  probability of a datapoint given that it is an anomaly, and is a heavy-tailed
  distribution (Cauchy) centered around the model's predictions by default.

  Specifically, we have:

    p(anomaly | data) = p(data | anomaly) * anomaly_prior_probability
        / (p(data | not anomaly) * (1 - anomaly_prior_probability)
           + p(data | anomaly) * anomaly_prior_probability)

  This is simply Bayes' theorem, where p(data | anomaly) is the
  alternative/anomaly distribution, p(data | not anomaly) is the model's
  predicted distribution, and anomaly_prior_probability is the prior probability
  of an anomaly occurring (user-specified, defaulting to 1%).

  Rather than computing p(anomaly | data) directly, we use the odds ratio:

    odds_ratio = p(data | anomaly) * anomaly_prior_probability
        / (p(data | not anomaly) * (1 - anomaly_prior_probability))

  This has the same information as p(anomaly | data):

    odds_ratio = p(anomaly | data) / p(not anomaly | data)

  A "responsibility" score is computed for the model based on the log odds
  ratio, and state interpolated based on this responsibility:

    model_responsibility = 1 / (1 + exp(-responsibility_scaling
                                        * ln(odds_ratio)))
    model_state = filtered_model_state * model_responsibility
                  + predicted_model_state * (1 - model_responsibility)
    loss = model_responsibility
             * ln(p(data | not anomaly) * (1 - anomaly_prior_probability))
           + (1 - model_responsibility)
             * ln(p(data | anomaly) * anomaly_prior_probability)

  """

  output_names = ["anomaly_score"]

  def __init__(self,
               anomaly_log_likelihood=cauchy_alternative_to_gaussian,
               anomaly_prior_probability=0.01,
               responsibility_scaling=1.0):
    """Configure the anomaly detector.

    Args:
      anomaly_log_likelihood: A function taking `current_times`,
          `current_values`, and `outputs` (same as the corresponding arguments
          to process_filtering_step) and returning a [batch size] Tensor of log
          likelihoods under an anomaly distribution.
      anomaly_prior_probability: A scalar value, between 0 and 1, indicating the
          prior probability of a particular example being an anomaly.
      responsibility_scaling: A positive scalar controlling how fast
          interpolation transitions between not-anomaly and anomaly; lower
          values (closer to 0) create a smoother/slower transition.
    """
    self._anomaly_log_likelihood = anomaly_log_likelihood
    self._responsibility_scaling = responsibility_scaling
    self._anomaly_prior_probability = anomaly_prior_probability

  def process_filtering_step(self, current_times, current_values,
                             predicted_state, filtered_state, outputs):
    """Fall back on `predicted_state` for anomalies.

    Args:
      current_times: A [batch size] integer Tensor of times.
      current_values: A [batch size x num features] Tensor of values filtering
          is being performed on.
      predicted_state: A (possibly nested) list of Tensors indicating model
          state which does not take `current_times` and `current_values` into
          account.
      filtered_state: Same structure as predicted_state, but updated to take
          `current_times` and `current_values` into account.
      outputs: A dictionary of outputs produced by model filtering. Must
          include `log_likelihood`, a [batch size] Tensor indicating the log
          likelihood of the observations under the model's predictions.
    Returns:
      A tuple of (new_state, updated_outputs);
        new_state: Updated state with the same structure as `filtered_state` and
            `predicted_state`; predicted_state for anomalies and filtered_state
            otherwise (per batch element).
        updated_outputs: The `outputs` dictionary, updated with a new "loss"
            (the interpolated negative log likelihoods under the model and
            anomaly distributions) and "anomaly_score" (the log odds ratio of
            each part of the batch being an anomaly).
    """
    anomaly_log_likelihood = self._anomaly_log_likelihood(
        current_times=current_times,
        current_values=current_values,
        outputs=outputs)
    anomaly_prior_probability = ops.convert_to_tensor(
        self._anomaly_prior_probability, dtype=current_values.dtype)
    # p(data | anomaly) * p(anomaly)
    data_and_anomaly_log_probability = (
        anomaly_log_likelihood + math_ops.log(anomaly_prior_probability))
    # p(data | no anomaly) * p(no anomaly)
    data_and_no_anomaly_log_probability = (
        outputs["log_likelihood"] + math_ops.log(1. - anomaly_prior_probability)
    )
    # A log odds ratio is slightly nicer here than computing p(anomaly | data),
    # since it is centered around zero
    anomaly_log_odds_ratio = (
        data_and_anomaly_log_probability
        - data_and_no_anomaly_log_probability)
    model_responsibility = math_ops.sigmoid(-self._responsibility_scaling *
                                            anomaly_log_odds_ratio)
    # Do a linear interpolation between predicted and inferred model state
    # based on the model's "responsibility". If we knew for sure whether
    # this was an anomaly or not (binary responsibility), this would be the
    # correct thing to do, but given that we don't it's just a
    # (differentiable) heuristic.
    interpolated_state = _interpolate_state_linear(
        first_state=filtered_state,
        second_state=predicted_state,
        first_responsibility=model_responsibility)
    # TODO(allenl): Try different responsibility scalings and interpolation
    # methods (e.g. average in probability space rather than log space).
    interpolated_log_likelihood = (
        model_responsibility * data_and_no_anomaly_log_probability
        + (1. - model_responsibility) * data_and_anomaly_log_probability)
    outputs["loss"] = -interpolated_log_likelihood
    outputs["anomaly_score"] = anomaly_log_odds_ratio
    return (interpolated_state, outputs)
