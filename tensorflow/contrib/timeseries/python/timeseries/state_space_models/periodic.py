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
"""State space components for modeling seasonality."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from tensorflow.contrib.timeseries.python.timeseries.state_space_models import state_space_model

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops


class CycleStateSpaceModel(state_space_model.StateSpaceModel):
  """A state space model component which cycles between values.

  Stores N values using N - 1 latent values, the Nth being the negative sum of
  those explicitly stored. At any given timestep one of these values is
  observed. Noise is assumed to affect only one of the transitions.
  """

  def __init__(
      self,
      periodicity,
      configuration=state_space_model.StateSpaceModelConfiguration()):
    self._periodicity = periodicity
    super(CycleStateSpaceModel, self).__init__(configuration=configuration)

  def get_state_transition(self):
    return self.transition_to_powers(array_ops.ones([], dtype=dtypes.int32))

  def get_noise_transform(self):
    # transition_power_noise_accumulator makes assumptions about this
    # transformation. If the noise transform is modified or overridden,
    # transition_power_noise_accumulator must be modified as well (or discarded,
    # as it is simply an optimization).
    return array_ops.pad(
        array_ops.ones([1], dtype=self.dtype),
        paddings=[(0, self._periodicity - 2)])[..., None]

  def transition_to_powers(self, powers):
    """Computes powers of the cycle transition matrix efficiently.

    Args:
      powers: An integer Tensor, shape [...], with powers to raise the
        transition matrix to.
    Returns:
      A floating point Tensor with shape [..., self._periodicity - 1,
      self._periodicity - 1] containing:
        (transition^power)_{i, j} = {
           1  if (i - j) % self._periodicity == power % self._periodicity
          -1  if (i + 1) % self._periodicity == power % self._periodicity
           0  otherwise}
    """
    powers %= self._periodicity
    range_shape_padded = array_ops.reshape(
        math_ops.range(self._periodicity - 1, dtype=powers.dtype),
        array_ops.concat(
            [
                array_ops.ones([array_ops.rank(powers)], dtype=dtypes.int32),
                [self._periodicity - 1]
            ],
            axis=0))
    is_row_negative = math_ops.equal(range_shape_padded + 1, powers[..., None])
    row_indicator_shape = array_ops.shape(is_row_negative)
    negative_row_indicator = array_ops.where(is_row_negative, -array_ops.ones(
        shape=row_indicator_shape, dtype=self.dtype),
                                             array_ops.zeros(
                                                 row_indicator_shape,
                                                 dtype=self.dtype))
    coord_diff = (range_shape_padded[..., None]
                  - range_shape_padded[..., None, :])
    is_one = math_ops.equal(coord_diff % self._periodicity,
                            powers[..., None, None])
    positive_ones = array_ops.where(is_one,
                                    array_ops.ones(
                                        array_ops.shape(is_one),
                                        dtype=self.dtype),
                                    array_ops.zeros(
                                        array_ops.shape(is_one),
                                        dtype=self.dtype))
    return math_ops.cast(positive_ones + negative_row_indicator[..., None],
                         self.dtype)

  def transition_power_noise_accumulator(
      self, num_steps, noise_addition_coefficient=1):
    r"""Sum the transitioned covariance matrix over a number of steps.

    Assumes that state_transition_noise_covariance is a matrix with a single
    non-zero value in the upper left.

    Args:
      num_steps: A [...] shape integer Tensor with numbers of steps to compute
        power sums for.
      noise_addition_coefficient: A multiplier for the state transition noise
        covariance (used in ResolutionCycleModel to compute multiples of full
        period sums).
    Returns:
      The computed power sum, with shape [..., state dimension, state
      dimension] containing:

        [\sum_{p=0}^{num_steps - 1} (
           state_transition^p
           * state_transition_noise_covariance
           * (state_transition^p)^T)]_{i, j} = {
          -contribution_{j + 1}                   if j == i - 1
          contribution_{j + 1} + contribution{j}  if j == i
          -contribution_{j}                       if j == i + 1
          0                                        otherwise
        }

        contribution_k = noise_scalar
          * ((num_steps + self._periodicity - 1 - (k % self._periodicity))
             // self._periodicity)

      Where contribution_k is the sum of noise_scalar additions to component k
      of the periodicity.
    """
    noise_addition_scalar = array_ops.squeeze(
        self.state_transition_noise_covariance, axis=[-1, -2])
    period_range_reshaped = array_ops.reshape(
        math_ops.range(self._periodicity, dtype=num_steps.dtype),
        array_ops.concat(
            [
                array_ops.ones([array_ops.rank(num_steps)], dtype=dtypes.int32),
                [self._periodicity]
            ],
            axis=0))
    reversed_remaining_steps = ((period_range_reshaped
                                 - (num_steps[..., None] - 1))
                                % self._periodicity)
    period_additions_reversed = (ops.convert_to_tensor(
        noise_addition_coefficient,
        self.dtype)[..., None] * noise_addition_scalar * math_ops.cast(
            (num_steps[..., None] + reversed_remaining_steps) //
            self._periodicity,
            dtype=self.dtype))
    period_additions_diag = array_ops.matrix_diag(period_additions_reversed)
    upper_band = array_ops.concat(
        [
            array_ops.zeros_like(period_additions_diag[..., :-1, 0:1]),
            -period_additions_diag[..., :-1, 0:-2]
        ],
        axis=-1)
    lower_band = array_ops.concat(
        [
            array_ops.zeros_like(period_additions_diag[..., 0:1, :-1]),
            -period_additions_diag[..., 0:-2, :-1]
        ],
        axis=-2)
    period_additions_rotated = array_ops.concat(
        [
            period_additions_reversed[..., -1:],
            period_additions_reversed[..., :-2]
        ],
        axis=-1)
    diagonal = array_ops.matrix_diag(period_additions_reversed[..., :-1] +
                                     period_additions_rotated)
    return diagonal + lower_band + upper_band

  def get_observation_model(self, times):
    """Observe only the first of the rotating latent values.

    See StateSpaceModel.get_observation_model.
    Args:
      times: Unused. See the parent class for details.
    Returns:
      A static, univariate observation model for later broadcasting.
    """
    del times  # Does not rely on times. Uses broadcasting from the parent.
    return array_ops.concat(
        values=[
            array_ops.ones([1], dtype=self.dtype), array_ops.zeros(
                [self._periodicity - 2], dtype=self.dtype)
        ],
        axis=0)


class ResolutionCycleModel(CycleStateSpaceModel):
  """A version of CycleStateSpaceModel with variable resolution.

  Cycles between "num_latent_values" latent values over a period of
  "periodicity", smoothly interpolating. Simply raises the transition matrix
  from CycleStateSpaceModel to the power (num_latent_values / periodicity).

  Specifically, ResolutionCycleModel uses the following eigendecomposition of
  the CycleStateSpaceModel matrix (there are several parameterizations, others
  leading to roots of the matrix with complex values):

    eigenvectors_{i, j}
        = root_of_unity(floor(j / 2) + 1, i * (-1)^(j + 1))
          - root_of_unity(floor(j / 2) + 1, (i + 1) * (-1)^(j + 1))
    eigenvalues_j = root_of_unity(floor(j / 2) + 1, (-1)^j)
    root_of_unity(root_number, to_power)
        = exp(to_power * 2 * pi * sqrt(-1) * root_number
              / num_latent_values)

  The transition matrix for ResolutionCycleModel is then:

    eigenvectors
    * diag(eigenvalues^(num_latent_values / periodicity))
    * eigenvectors^-1

  Since the eigenvalues are paired with their conjugates (conj(e^(sqrt(-1)*x)) =
  e^(-sqrt(-1)*x)), the resulting matrix has real components (this is why only
  odd numbers of latent values are supported, since the size of the matrix is
  one less than the number of latent values and there must be an even number of
  eigenvalues to pair them off).

  See ./g3doc/periodic_multires_derivation.md for details.
  """

  def __init__(
      self,
      num_latent_values,
      periodicity,
      near_integer_threshold=1e-8,
      configuration=state_space_model.StateSpaceModelConfiguration()):
    """Initialize the ResolutionCycleModel.

    Args:
      num_latent_values: Controls the representational power and memory usage of
        the model. The transition matrix has shape [num_latent_values - 1,
        num_latent_values - 1]. Must be an odd integer (see class docstring for
        why).
      periodicity: The number of steps for cyclic behavior. May be a Tensor, and
        need not be an integer (although integer values greater than
        num_latent_values have more efficient special cases).
      near_integer_threshold: When avoiding singularities, controls how close a
        number should be to that singularity before the special case takes over.
      configuration: A StateSpaceModelConfiguration object.

    Raises:
      ValueError: If num_latent_values is not odd.
    """
    if num_latent_values % 2 != 1:
      raise ValueError("Only odd numbers of latent values are supported.")
    self._num_latent_values = num_latent_values
    self._true_periodicity = periodicity
    self._near_integer_threshold = near_integer_threshold
    super(ResolutionCycleModel, self).__init__(
        periodicity=num_latent_values,
        configuration=configuration)

  def _close_to_integer(self, value):
    value = math_ops.cast(value, self.dtype)
    return math_ops.less(
        math_ops.abs(value - gen_math_ops.round(value)),
        self._near_integer_threshold)

  def transition_to_powers(self, powers):
    """Computes TransitionMatrix^power efficiently.

    For an n x n transition matrix we have:

      (TransitionMatrix**power)_{i, j) = (-1) ** i * sin(pi * power) / (n + 1)
          * ((-1) ** j / sin(pi / (n + 1) * (power - i + j))
             + 1 / sin(pi / (n + 1) * (power - i - 1)))

    The sin(pi * power) term is zero whenever "power" is an integer. However,
    the 1 / sin(x) terms (cosecants) occasionally (when their arguments are
    multiples of pi) cancel out this value. The limit as the argument approaches
    an integer value gives the "correct" result, but computing these separately
    gives 0 * inf = NaN. Instead, there is a special case for near-integer
    values.

    Args:
      powers: A floating point Tensor of powers to raise the transition matrix
        to.
    Returns:
      A [..., self._num_latent_values - 1, self._num_latent_values - 1] floating
        point Tensor with the transition matrix raised to each power in
        `powers`.

    """
    num_latent_values_float = math_ops.cast(self._num_latent_values, self.dtype)
    latent_values_per_period = (num_latent_values_float / math_ops.cast(
        self._true_periodicity, dtype=self.dtype))
    original_matrix_powers = (math_ops.cast(powers, self.dtype) *
                              latent_values_per_period)
    global_coeff = (math_ops.sin(original_matrix_powers * numpy.pi) /
                    num_latent_values_float)[..., None, None]
    matrix_dimension_range = array_ops.reshape(
        math_ops.range(self._num_latent_values - 1),
        array_ops.concat(
            [
                array_ops.ones(
                    [array_ops.rank(original_matrix_powers)],
                    dtype=dtypes.int32), [self._num_latent_values - 1]
            ],
            axis=0))
    matrix_dimension_range_float = math_ops.cast(matrix_dimension_range,
                                                 self.dtype)
    alternating = math_ops.cast(1 - 2 * (matrix_dimension_range % 2),
                                self.dtype)
    row_addend = 1. / math_ops.sin(numpy.pi / num_latent_values_float * (
        original_matrix_powers[..., None] - matrix_dimension_range_float - 1))
    column_minus_row = (matrix_dimension_range_float[..., None, :]
                        - matrix_dimension_range_float[..., None])
    full_matrix_addend = (alternating[..., None, :] / math_ops.sin(
        numpy.pi / num_latent_values_float *
        (original_matrix_powers[..., None, None] + column_minus_row)))
    continuous_construction = global_coeff * alternating[..., None] * (
        row_addend[..., None] + full_matrix_addend)
    # For integer powers, the above formula is only correct in the limit,
    # yielding NaNs as written. We defer to the super-class in such cases, which
    # computes integer powers exactly.
    return array_ops.where(
        self._close_to_integer(original_matrix_powers),
        super(ResolutionCycleModel, self).transition_to_powers(
            math_ops.cast(
                gen_math_ops.round(original_matrix_powers), dtypes.int64)),
        continuous_construction)

  def transition_power_noise_accumulator(self, num_steps):
    """Sum the transitioned covariance matrix over a number of steps.

    Args:
      num_steps: An integer Tensor of any shape [...] indicating the number of
        steps to compute for each part of the batch.

    Returns:
      A [..., self._num_latent_values - 1, self._num_latent_values - 1] floating
      point Tensor corresponding to each requested number of steps, containing:

          sum_{i=1}^{steps} transition^i * noise_covariance
              * (transition^i)^T
    """

    def _whole_periods_folded():
      """A more efficient special casing for integer periods.

      We knock off full periods, leaving at most self._true_periodicity steps to
      compute.

      Returns:
        A tuple of (remaining_whole_steps, current_accumulation):
          remaining_whole_steps: An integer Tensor with the same shape as the
            `num_steps` argument to `transition_power_noise_accumulator`,
            indicating the reduced number of steps which must be computed
            sequentially and added to `current_accumulation`.
          current_accumulation: A [..., self._num_latent_values - 1,
            self._num_latent_values - 1] floating point Tensor corresponding to
            the accumulations for steps which were computed in this function.
      """
      original_transition_noise_addition_coefficient = (math_ops.cast(
          self._true_periodicity, self.dtype) / math_ops.cast(
              self._num_latent_values, self.dtype))
      full_period_accumulation = super(
          ResolutionCycleModel, self).transition_power_noise_accumulator(
              noise_addition_coefficient=
              original_transition_noise_addition_coefficient,
              num_steps=ops.convert_to_tensor(
                  self._num_latent_values, dtype=num_steps.dtype))
      periodicity_integer = math_ops.cast(self._true_periodicity,
                                          num_steps.dtype)
      full_periods = math_ops.cast(num_steps // periodicity_integer, self.dtype)
      current_accumulation = full_periods[..., None, None] * array_ops.reshape(
          full_period_accumulation,
          array_ops.concat(
              [
                  array_ops.ones(
                      [array_ops.rank(full_periods)], dtype=dtypes.int32),
                  array_ops.shape(full_period_accumulation)
              ],
              axis=0))
      remaining_whole_steps = num_steps % periodicity_integer
      return remaining_whole_steps, current_accumulation
    def _no_whole_period_computation():
      """A less efficient special casing for real valued periods.

      This special casing is still preferable to computing using sequential
      matrix multiplies (parallelizable, more numerically stable), but is linear
      in the number of steps.

      Returns:
        Same shapes and types as `_whole_periods_folded`, but no folding is done
        in this function.
      """
      current_accumulation = array_ops.zeros(
          array_ops.concat(
              [
                  array_ops.shape(num_steps),
                  [self._num_latent_values - 1, self._num_latent_values - 1]
              ],
              axis=0),
          dtype=self.dtype)
      remaining_whole_steps = num_steps
      return remaining_whole_steps, current_accumulation
    # Decide whether it's feasible to compute whole periods in closed form,
    # taking advantage of the fact that a sum over self._true_periodicity steps
    # in our transition matrix is proportional to a sum over
    # self._num_latent_values steps in the unmodified matrix (because each
    # latent value gets the same treatment). This is possible for integer
    # self._true_periodicity, since we stay aligned to integer steps. For real
    # valued self._true_periodicity, or when the cyclic behavior is a higher
    # resolution than 1 per step, taking whole periods leads to misalignment
    # with integer steps, which would be difficult to recover from.
    remaining_whole_steps, current_accumulation = control_flow_ops.cond(
        self._whole_period_folding(), _whole_periods_folded,
        _no_whole_period_computation)
    steps_to_compute = math_ops.reduce_max(remaining_whole_steps)
    remaining_step_noise_additions = self._power_sum_array(steps_to_compute)
    noise_addition_scalar = array_ops.squeeze(
        self.state_transition_noise_covariance, axis=[-1, -2])
    return current_accumulation + noise_addition_scalar * array_ops.gather(
        remaining_step_noise_additions, indices=remaining_whole_steps)

  def _whole_period_folding(self):
    """Decides whether computing a whole period maintains alignment."""
    return math_ops.logical_and(
        self._close_to_integer(self._true_periodicity),
        math_ops.greater_equal(self._true_periodicity, self._num_latent_values))

  def _power_sum_array(self, max_remaining_steps):
    r"""Computes \sum_{i=0}^{N-1} A^i B (A^i)^T for N=0..max_remaining_steps.

    A is the transition matrix and B is the noise covariance.

    This is more efficient in practice than math_utils.power_sums_tensor, since
    each A^i B (A^i)^T term has a closed-form expression not depending on i - 1.
    Thus vectorization can replace explicit looping.

    Uses a cumulative sum on the following expression:

      (transition^p * transition_covariance * (transition^p)^T)_{i, j}
        = (-1)^(i + j) * sin^2(pi * p) / num_latent_values^2
          * (1/sin(pi / num_latent_values * (p - i))
             + 1/sin(pi / num_latent_values * (p - i - 1)))
          * (1/sin(pi / num_latent_values * (p - j))
             + 1/sin(pi / num_latent_values * (p - j - 1)))

    The expression being derived from the eigenvectors and eigenvalues given in
    the class docstring (and as with CycleStateSpaceModel taking advantage of
    the sparsity of the transition covariance).

    Args:
      max_remaining_steps: A scalar integer Tensor indicating the number of
        non-trivial values to compute.
    Returns:
      A [max_remaining_steps + 1, self._num_latent_values - 1,
      self._num_latent_values - 1] floating point Tensor S with cumulative power
      sums.

      S[N] = \sum_{i=0}^{N-1} A^i B (A^i)^T
        S[0] is the zero matrix
        S[1] is B
        S[2] is A B A^T + B

    """
    num_latent_values_float = math_ops.cast(self._num_latent_values, self.dtype)
    latent_values_per_period = (num_latent_values_float / math_ops.cast(
        self._true_periodicity, dtype=self.dtype))
    original_matrix_powers = (math_ops.cast(
        math_ops.range(max_remaining_steps),
        self.dtype) * latent_values_per_period)
    matrix_dimension_range = math_ops.range(
        self._num_latent_values - 1)[None, ...]
    matrix_dimension_range_float = math_ops.cast(matrix_dimension_range,
                                                 self.dtype)
    def _cosecant_with_freq(coefficient):
      return 1. / math_ops.sin(numpy.pi / num_latent_values_float * coefficient)
    power_minus_index = (original_matrix_powers[..., None]
                         - matrix_dimension_range_float)
    mesh_values = (_cosecant_with_freq(power_minus_index)
                   + _cosecant_with_freq(power_minus_index - 1.))
    meshed = mesh_values[..., None, :] * mesh_values[..., None]
    full_matrix_alternating = math_ops.cast(1 - 2 * (
        (matrix_dimension_range[..., None, :] +
         matrix_dimension_range[..., None]) % 2), self.dtype)
    def _sine_discontinuity(value):
      """A special case for dealing with discontinuities.

      Decides whether `value`  is close to an integer, and if so computes:

        lim x->n |sin(x * pi)| / sin(x * pi) = sign(sin(n * pi))
                                             = cos(n * pi)

      Args:
        value: The floating point Tensor value which may lead to a
            discontinuity.
      Returns:
        A tuple of (is_discontinuous, sign):
          is_discontinuous: A boolean Tensor of the same shape as `value`,
              indicating whether it is near an integer.
          sign: A floating point Tensor indicating the sign of the discontinuity
            (being near 1 or -1 when `is_discontinuous` is True), of the same
            shape and type as `value`.
      """
      normalized = value / num_latent_values_float
      is_discontinuous = self._close_to_integer(normalized)
      sign = math_ops.cos(normalized * numpy.pi)
      return is_discontinuous, sign
    index_discontinuous, index_sign = _sine_discontinuity(
        original_matrix_powers[..., None]
        - matrix_dimension_range_float)
    index_minus_discontinuous, index_minus_sign = _sine_discontinuity(
        original_matrix_powers[..., None]
        - matrix_dimension_range_float
        - 1)
    ones_mask_vector = math_ops.logical_or(index_discontinuous,
                                           index_minus_discontinuous)
    ones_sign_vector = array_ops.where(index_discontinuous, index_sign,
                                       index_minus_sign)
    ones_mask = math_ops.logical_and(ones_mask_vector[..., None],
                                     ones_mask_vector[..., None, :])
    zeros_mask = self._close_to_integer(original_matrix_powers)
    zeroed = array_ops.where(zeros_mask, array_ops.zeros_like(meshed), meshed)
    global_coefficient = (math_ops.sin(numpy.pi * original_matrix_powers) /
                          num_latent_values_float)
    masked_meshed = array_ops.where(
        ones_mask, ones_sign_vector[..., None] * ones_sign_vector[..., None, :],
        zeroed * global_coefficient[..., None, None]**2)
    powers_above_zero = full_matrix_alternating * masked_meshed
    return array_ops.pad(
        math_ops.cumsum(powers_above_zero), [(1, 0), (0, 0), (0, 0)])
