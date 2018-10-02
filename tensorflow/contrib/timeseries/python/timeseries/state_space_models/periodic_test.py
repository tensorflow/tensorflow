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
"""Tests for periodic state space model components."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.timeseries.python.timeseries.state_space_models import periodic
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import state_space_model
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import test_utils

from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


class SpecialCaseTests(test.TestCase):

  def test_cycle_transition_to_powers(self):
    num_steps = 3
    dtype = dtypes.float64
    periodicity = 3
    cycle = periodic.CycleStateSpaceModel(
        periodicity=periodicity,
        configuration=state_space_model.StateSpaceModelConfiguration(
            dtype=dtype))
    test_utils.transition_power_test_template(
        test_case=self, model=cycle, num_steps=num_steps)

  def test_resolution_cycle_transition_to_powers(self):
    num_steps = 3
    dtype = dtypes.float64
    latent_values = 3
    periodicity = latent_values - 1
    cycle = periodic.ResolutionCycleModel(
        num_latent_values=latent_values,
        periodicity=periodicity,
        configuration=state_space_model.StateSpaceModelConfiguration(
            dtype=dtype))
    test_utils.transition_power_test_template(
        test_case=self, model=cycle, num_steps=num_steps)

  def test_cycle_noise_accumulator(self):
    num_steps = 3
    dtype = dtypes.float64
    periodicity = 3
    cycle = periodic.CycleStateSpaceModel(
        periodicity=periodicity,
        configuration=state_space_model.StateSpaceModelConfiguration(
            dtype=dtype))
    test_utils.noise_accumulator_test_template(
        test_case=self, model=cycle, num_steps=num_steps)

  def test_resolution_cycle_noise_accumulator(self):
    num_steps = 3
    dtype = dtypes.float64
    latent_values = 3
    periodicity = latent_values + 0.1
    cycle = periodic.ResolutionCycleModel(
        num_latent_values=latent_values,
        periodicity=periodicity,
        configuration=state_space_model.StateSpaceModelConfiguration(
            dtype=dtype))
    test_utils.noise_accumulator_test_template(
        test_case=self, model=cycle, num_steps=num_steps)


if __name__ == "__main__":
  test.main()
