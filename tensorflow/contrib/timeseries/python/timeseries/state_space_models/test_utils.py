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
"""Utilities for testing state space models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from tensorflow.contrib.timeseries.python.timeseries import math_utils

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


def transition_power_test_template(test_case, model, num_steps):
  """Tests the transition_to_powers function of a state space model."""
  transition_matrix = ops.convert_to_tensor(
      model.get_state_transition(), dtype=model.dtype)
  step_number = array_ops.placeholder(shape=[], dtype=dtypes.int64)
  state_dimension = transition_matrix.get_shape()[0].value
  previous_matrix = array_ops.placeholder(
      shape=[state_dimension, state_dimension], dtype=transition_matrix.dtype)
  true_single_step_update = math_ops.matmul(previous_matrix,
                                            transition_matrix)
  model_output_tensor = model.transition_to_powers(powers=array_ops.stack(
      [step_number, step_number]))
  with test_case.test_session():
    starting_matrix = linalg_ops.eye(
        state_dimension, batch_shape=array_ops.shape(num_steps)).eval()
    evaled_current_matrix = starting_matrix
    for iteration_number in range(num_steps):
      model_output = model_output_tensor.eval(
          feed_dict={step_number: iteration_number})
      test_case.assertAllClose(
          evaled_current_matrix,
          model_output[0],
          rtol=1e-8 if evaled_current_matrix.dtype == numpy.float64 else 1e-4)
      evaled_current_matrix = true_single_step_update.eval(
          feed_dict={previous_matrix: evaled_current_matrix})


def noise_accumulator_test_template(test_case, model, num_steps):
  """Tests `model`'s transition_power_noise_accumulator."""
  transition_matrix = ops.convert_to_tensor(
      model.get_state_transition(), dtype=model.dtype)
  noise_transform = ops.convert_to_tensor(
      model.get_noise_transform(), dtype=model.dtype)
  state_dimension = transition_matrix.get_shape()[0].value
  state_noise_dimension = noise_transform.get_shape()[1].value
  gen_noise_addition = math_utils.sign_magnitude_positive_definite(
      raw=random_ops.random_normal(
          shape=[state_noise_dimension, state_noise_dimension],
          dtype=model.dtype))
  gen_starting_noise = math_utils.sign_magnitude_positive_definite(
      random_ops.random_normal(
          shape=[state_dimension, state_dimension], dtype=model.dtype))
  starting_noise = array_ops.placeholder(
      shape=[state_dimension, state_dimension], dtype=model.dtype)
  step_number = array_ops.placeholder(shape=[], dtype=dtypes.int64)
  starting_transitioned = math_ops.matmul(
      math_ops.matmul(transition_matrix, starting_noise),
      transition_matrix,
      adjoint_b=True)
  with test_case.test_session():
    evaled_starting_noise = gen_starting_noise.eval()
    current_starting_noise_transitioned = evaled_starting_noise
    current_noise = evaled_starting_noise
    evaled_noise_addition = gen_noise_addition.eval()
    evaled_noise_addition_transformed = math_ops.matmul(
        math_ops.matmul(noise_transform, evaled_noise_addition),
        noise_transform,
        adjoint_b=True).eval()
    model.state_transition_noise_covariance = evaled_noise_addition
    model._window_initializer(  # pylint: disable=protected-access
        times=math_ops.range(num_steps + 1)[..., None], state=(None, None, 0))
    model_update = model.transition_power_noise_accumulator(
        num_steps=step_number)
    for iteration_number in range(num_steps):
      model_new_noise = model_update.eval(
          feed_dict={step_number: iteration_number})
      test_case.assertAllClose(
          current_noise,
          model_new_noise + current_starting_noise_transitioned,
          rtol=1e-8 if current_noise.dtype == numpy.float64 else 1e-3)
      current_starting_noise_transitioned = starting_transitioned.eval(
          feed_dict={starting_noise: current_starting_noise_transitioned})
      current_noise = (
          starting_transitioned.eval(
              feed_dict={starting_noise: current_noise})
          + evaled_noise_addition_transformed)
