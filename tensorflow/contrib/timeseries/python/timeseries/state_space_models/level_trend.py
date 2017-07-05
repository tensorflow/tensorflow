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
"""Implements a state space model with level and local linear trends."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.timeseries.python.timeseries.state_space_models import state_space_model

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope


class AdderStateSpaceModel(state_space_model.StateSpaceModel):
  """A state space model component with level and slope.

  At each timestep, level <- level + slope. Level is observed, slope is not.
  """

  def __init__(
      self,
      use_level_noise=True,
      configuration=state_space_model.StateSpaceModelConfiguration()):
    """Configure the model.

    Args:
      use_level_noise: Whether to model the time series as having level noise.
      configuration: A StateSpaceModelConfiguration object.
    """
    self.use_level_noise = use_level_noise
    super(AdderStateSpaceModel, self).__init__(
        configuration=configuration)

  def get_prior_mean(self):
    """If un-chunked data is available, set initial level to the first value."""
    with variable_scope.variable_scope(self._variable_scope):
      if self._input_statistics is not None:
        # TODO(allenl): Better support for multivariate series here.
        initial_value = array_ops.stack([
            math_ops.reduce_mean(
                self._input_statistics.series_start_moments.mean), 0.
        ])
        return initial_value + variable_scope.get_variable(
            name="prior_state_mean",
            shape=initial_value.get_shape(),
            initializer=init_ops.zeros_initializer(),
            dtype=self.dtype,
            trainable=self._configuration.trainable_start_state)
      else:
        return super(AdderStateSpaceModel, self).get_prior_mean()

  def transition_to_powers(self, powers):
    """Computes powers of the adder transition matrix efficiently.

    Args:
      powers: An integer Tensor, shape [...], with powers to raise the
        transition matrix to.
    Returns:
      A floating point Tensor with shape [..., 2, 2] containing:
        transition^power = [[1., power],
                            [0., 1.]]
    """
    paddings = array_ops.concat(
        [
            array_ops.zeros([array_ops.rank(powers), 2], dtype=dtypes.int32),
            [(0, 1), (1, 0)]
        ],
        axis=0)
    powers_padded = array_ops.pad(powers[..., None, None], paddings=paddings)
    identity_matrices = linalg_ops.eye(
        num_rows=2, batch_shape=array_ops.shape(powers), dtype=self.dtype)
    return identity_matrices + math_ops.cast(powers_padded, self.dtype)

  def transition_power_noise_accumulator(self, num_steps):
    """Computes power sums in closed form."""
    def _pack_and_reshape(*values):
      return array_ops.reshape(
          array_ops.stack(axis=1, values=values),
          array_ops.concat(values=[array_ops.shape(num_steps), [2, 2]], axis=0))

    num_steps = math_ops.cast(num_steps, self.dtype)
    noise_transitions = num_steps - 1
    noise_transform = ops.convert_to_tensor(self.get_noise_transform(),
                                            self.dtype)
    noise_covariance_transformed = math_ops.matmul(
        math_ops.matmul(noise_transform,
                        self.state_transition_noise_covariance),
        noise_transform,
        adjoint_b=True)
    # Un-packing the transformed noise as:
    # [[a b]
    #  [c d]]
    a, b, c, d = array_ops.unstack(
        array_ops.reshape(noise_covariance_transformed, [-1, 4]), axis=1)
    sum_of_first_n = noise_transitions * (noise_transitions + 1) / 2
    sum_of_first_n_squares = sum_of_first_n * (2 * noise_transitions + 1) / 3
    return _pack_and_reshape(
        num_steps * a + sum_of_first_n * (b + c) + sum_of_first_n_squares * d,
        num_steps * b + sum_of_first_n * d,
        num_steps * c + sum_of_first_n * d,
        num_steps * d)

  def get_state_transition(self):
    return [[1., 1.],  # Add slope to level
            [0., 1.]]  # Maintain slope

  def get_noise_transform(self):
    if self.use_level_noise:
      return [[1., 0.],
              [0., 1.]]
    else:
      return [[0.],
              [1.]]

  def get_observation_model(self, times):
    """Observe level but not slope.

    See StateSpaceModel.get_observation_model.

    Args:
      times: Unused. See the parent class for details.
    Returns:
      A static, univariate observation model for later broadcasting.
    """
    del times  # Does not rely on times. Uses broadcasting from the parent.
    return constant_op.constant([1., 0.], dtype=self.dtype)
