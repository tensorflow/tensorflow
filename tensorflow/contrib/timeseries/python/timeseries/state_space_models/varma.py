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
r"""Multivariate autoregressive model (vector autoregression).

Implements the following model (num_blocks = max(ar_order, ma_order + 1)):

  y(t, 1) = \sum_{i=1}^{ar_order} ar_coefs[i] * y(t - 1, i)
  y(t, i) = y(t - 1, i - 1) + ma_coefs[i - 1] * e(t) for 1 < i < num_blocks
  y(t, num_blocks) = y(t - 1, num_blocks - 1) + e(t)

Where e(t) are Gaussian with zero mean and learned covariance.

Each element of ar_coefs and ma_coefs is a [num_features x num_features]
matrix. Each y(t, i) is a vector of length num_features. Indices in the above
equations are one-based. Initial conditions y(0, i) come from prior state (which
may either be learned or left as a constant with high prior covariance).

If ar_order > ma_order, the observation model is:
  y(t, 1) + observation_noise(t)

If ma_order >= ar_order, it is (to observe the moving average component):
  y(t, 1) + y(t, num_blocks) + observation_noise(t)

Where observation_noise(t) are Gaussian with zero mean and learned covariance.

This implementation uses a formulation which puts all of the autoregressive
coefficients in the transition equation for the observed component, which
enables learning using truncated backpropagation. Noise is not applied directly
to the observed component (with the exception of standard observation noise),
which further aids learning of the autoregressive coefficients when VARMA is in
an ensemble with other models (in which case having an observation noise term is
usually unavoidable).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.timeseries.python.timeseries import math_utils
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import state_space_model

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope


class VARMA(state_space_model.StateSpaceModel):
  """A VARMA model implementation as a special case of the state space model."""

  def __init__(self,
               autoregressive_order,
               moving_average_order,
               configuration=state_space_model.StateSpaceModelConfiguration()):
    """Construct a VARMA model.

    The size of the latent state for this model is:
      num_features * max(autoregressive_order, moving_average_order + 1)
    Square matrices of this size are constructed and multiplied.

    Args:
      autoregressive_order: The maximum autoregressive lag.
      moving_average_order: The maximum moving average lag, after which
        transient deviations are expected to return to their long-term mean.
      configuration: A StateSpaceModelConfiguration object.
    """
    self.ar_order = autoregressive_order
    self.ma_order = moving_average_order
    self.state_num_blocks = max(autoregressive_order, moving_average_order + 1)
    super(VARMA, self).__init__(configuration=configuration)
    self.state_dimension = self.state_num_blocks * self.num_features

  def _define_parameters(self, observation_transition_tradeoff_log=None):
    with variable_scope.variable_scope(self._variable_scope):
      # TODO(allenl): Evaluate parameter transformations for AR/MA coefficients
      # which improve interpretability/stability.
      self.ar_coefs = variable_scope.get_variable(
          name="ar_coefs",
          shape=[self.num_features, self.num_features, self.ar_order],
          dtype=self.dtype,
          initializer=init_ops.zeros_initializer())
      self.ma_coefs = variable_scope.get_variable(
          name="ma_coefs",
          initializer=array_ops.tile(
              linalg_ops.eye(self.num_features, dtype=self.dtype)[None, :, :],
              [self.ma_order, 1, 1]),
          dtype=self.dtype)
    super(VARMA, self)._define_parameters(
        observation_transition_tradeoff_log=observation_transition_tradeoff_log)

  def get_state_transition(self):
    """Construct state transition matrix from VARMA parameters.

    Returns:
      the state transition matrix. It has shape
        [self.state_dimendion, self.state_dimension].
    """
    # Pad any unused AR blocks with zeros. The extra state is necessary if
    # ma_order >= ar_order.
    ar_coefs_padded = array_ops.reshape(
        array_ops.pad(self.ar_coefs,
                      [[0, 0], [0, 0],
                       [0, self.state_num_blocks - self.ar_order]]),
        [self.num_features, self.state_dimension])
    shift_matrix = array_ops.pad(
        linalg_ops.eye(
            (self.state_num_blocks - 1) * self.num_features, dtype=self.dtype),
        [[0, 0], [0, self.num_features]])
    return array_ops.concat([ar_coefs_padded, shift_matrix], axis=0)

  def get_noise_transform(self):
    """Construct state noise transform matrix from VARMA parameters.

    Returns:
      the state noise transform matrix. It has shape
        [self.state_dimendion, self.num_features].
    """
    # Noise is broadcast, through the moving average coefficients, to
    # un-observed parts of the latent state.
    ma_coefs_padded = array_ops.reshape(
        array_ops.pad(self.ma_coefs,
                      [[self.state_num_blocks - 1 - self.ma_order, 0], [0, 0],
                       [0, 0]]),
        [(self.state_num_blocks - 1) * self.num_features, self.num_features],
        name="noise_transform")
    # Deterministically apply noise to the oldest component.
    return array_ops.concat(
        [ma_coefs_padded,
         linalg_ops.eye(self.num_features, dtype=self.dtype)],
        axis=0)

  def get_observation_model(self, times):
    """Construct observation model matrix from VARMA parameters.

    Args:
      times: A [batch size] vector indicating the times observation models are
          requested for. Unused.
    Returns:
      the observation model matrix. It has shape
        [self.num_features, self.state_dimension].
    """
    del times  # StateSpaceModel will broadcast along the batch dimension
    if self.ar_order > self.ma_order or self.state_num_blocks < 2:
      return array_ops.pad(
          linalg_ops.eye(self.num_features, dtype=self.dtype),
          [[0, 0], [0, self.num_features * (self.state_num_blocks - 1)]],
          name="observation_model")
    else:
      # Add a second observed component which "catches" the accumulated moving
      # average errors as they reach the end of the state. If ar_order >
      # ma_order, this is unnecessary, since accumulated errors cycle naturally.
      return array_ops.concat(
          [
              array_ops.pad(
                  linalg_ops.eye(self.num_features, dtype=self.dtype),
                  [[0, 0], [0,
                            self.num_features * (self.state_num_blocks - 2)]]),
              linalg_ops.eye(self.num_features, dtype=self.dtype)
          ],
          axis=1,
          name="observation_model")

  def get_state_transition_noise_covariance(
      self, minimum_initial_variance=1e-5):
    # Most state space models use only an explicit observation noise term to
    # model deviations from expectations, and so a low initial transition noise
    # parameter is helpful there. Since deviations from expectations are also
    # modeled as transition noise in VARMA, we set its initial value based on a
    # slight over-estimate empirical observation noise.
    if self._input_statistics is not None:
      feature_variance = self._input_statistics.series_start_moments.variance
      initial_transition_noise_scale = math_ops.log(
          math_ops.maximum(
              math_ops.reduce_mean(feature_variance), minimum_initial_variance))
    else:
      initial_transition_noise_scale = 0.
    state_noise_transform = ops.convert_to_tensor(
        self.get_noise_transform(), dtype=self.dtype)
    state_noise_dimension = state_noise_transform.get_shape()[1].value
    return math_utils.variable_covariance_matrix(
        state_noise_dimension, "state_transition_noise",
        dtype=self.dtype,
        initial_overall_scale_log=initial_transition_noise_scale)
